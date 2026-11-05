"""Category-only preview/apply. No ingestion or taxonomy replacement calls."""
import copy
import hashlib
import json
import secrets
import threading
import time
from sqlalchemy import text, inspect
from taxonomy_organization import organize_categories, organize_verified_chapters
from source_chunk_alignment import verified_chunk_chapters

_CACHE = {}
_MUTEX = threading.RLock()
TTL = 3600
CAPACITY = 32


def _chapter_chunks(db, project, document):
    # Legacy deployments and documents need no migration/backfill to stay readable.
    inspector = inspect(db.connection())
    if not inspector.has_table("chunks") or "structural_provenance" not in {
            c["name"] for c in inspector.get_columns("chunks")}:
        return []
    return [dict(r) for r in db.execute(text(
        "SELECT id, document_id, chunk_text, structural_provenance FROM chunks "
        "WHERE project_id=:p AND document_id=:d ORDER BY id"),
        {"p": project, "d": document}).mappings()]


def snapshot(db, project, document):
    params = {'p': project, 'd': document}
    doc = db.execute(text('SELECT * FROM documents WHERE project_id=:p AND id=:d'), params).mappings().first()
    if not doc or doc['taxonomy_status'] != 'ready':
        raise ValueError('Document must have completed taxonomy processing')
    topics = [dict(r) for r in db.execute(text('SELECT * FROM topics WHERE project_id=:p AND CAST(document_id AS text)=:d ORDER BY id'), params).mappings()]
    if not topics or any(not t['is_display_topic'] for t in topics):
        raise ValueError('Document requires a complete display-topic taxonomy')
    links = [tuple(r) for r in db.execute(text('SELECT tc.topic_id,tc.chunk_id FROM topic_chunks tc JOIN topics t ON t.id=tc.topic_id WHERE t.project_id=:p AND CAST(t.document_id AS text)=:d ORDER BY tc.topic_id,tc.chunk_id'), params)]
    content = {'document': dict(doc), 'topics': topics, 'links': links,
               'chapter_sources': _chapter_chunks(db, project, document) if doc.get('chapter_structure') else []}
    digest = hashlib.sha256(json.dumps(content, default=str, sort_keys=True).encode()).hexdigest()
    return dict(doc), topics, digest


def preview(db, owner, project, document, authorize, lock_scope, organization, request):
    authorize(db, project, owner)
    doc, topics, digest = snapshot(db, project, document)
    lock_scope(db, project, doc['module_id'], document, require_editable=True)
    settings = organization(db, doc['module_id'])
    if settings.get('organization_mode', 'infer') != 'infer':
        raise ValueError('Category-only reorganization currently supports automatic mode only')
    chapter_structure = doc.get("chapter_structure")
    if isinstance(chapter_structure, str):
        chapter_structure = json.loads(chapter_structure)
    chapter_chunks = _chapter_chunks(db, project, document) if chapter_structure else []
    ownership = verified_chunk_chapters(chapter_structure, chapter_chunks, document)
    evidence = {}
    if ownership is not None:
        for tid, cid in db.execute(text(
                "SELECT tc.topic_id,tc.chunk_id FROM topic_chunks tc JOIN topics t ON t.id=tc.topic_id "
                "WHERE t.project_id=:p AND CAST(t.document_id AS text)=:d"), {"p":project,"d":document}):
            evidence.setdefault(str(tid), []).append(cid)
    db.rollback()  # AI calls never hold a transaction or taxonomy lock.
    data = {'categories': [{'name': '', 'topics': [
        {**t, 'assignment_id': str(t['id']), 'title': t['topic'],
         **({'evidence_ids': evidence.get(str(t['id']), [])} if ownership is not None else {})}
        for t in topics]}]}
    planned = organize_verified_chapters(data, chapter_structure, ownership, chunks=chapter_chunks) if ownership is not None else None
    if planned is None:
        planned = organize_categories(data, 'infer', [], request,
            document_context={'document_title': doc['title'], 'require_hierarchy': True})
    assignments = []
    categories = []
    for ci, category in enumerate(planned['categories']):
        categories.append({'name': category['name'], 'topic_count': len(category['topics']),
                           **({'macrocategory': category['macrocategory']} if category.get('macrocategory') else {})})
        for ti, topic in enumerate(category['topics']):
            assignments.append({'topic_id': topic['assignment_id'], 'topic_title': topic['title'],
                                'original_category': topic['category'], 'category': category['name'],
                                'category_order_index': ci, 'topic_order_index': ti})
            if category.get('macrocategory'):
                assignments[-1]['macrocategory'] = category['macrocategory']
    ids = [a['topic_id'] for a in assignments]
    if len(ids) != len(set(ids)) or set(ids) != {str(t['id']) for t in topics}:
        raise ValueError('Preview must assign every existing topic exactly once')
    result = {'preview_id': secrets.token_urlsafe(32), 'document_id': document,
              'categories': categories, 'assignments': assignments, 'topic_count': len(ids),
              'expires_in_seconds': TTL}
    with _MUTEX:
        for key in list(_CACHE):
            if _CACHE[key]['expires'] <= time.monotonic():
                del _CACHE[key]
        if len(_CACHE) >= CAPACITY:
            del _CACHE[next(iter(_CACHE))]
        _CACHE[result['preview_id']] = {'owner': str(owner), 'project': str(project), 'document': str(document),
            'snapshot': digest, 'settings': copy.deepcopy(settings), 'expires': time.monotonic()+TTL,
            'result': copy.deepcopy(result)}
    return result


def apply(db, owner, project, document, preview_id, approved, authorize, lock_scope, organization):
    if approved is not True:
        raise ValueError('Explicit approval is required')
    with _MUTEX:
        saved = _CACHE.get(preview_id)
        if not saved or saved['expires'] <= time.monotonic() or (saved['owner'],saved['project'],saved['document']) != (str(owner),str(project),str(document)):
            raise ValueError('Preview unavailable, expired or belongs to a different scope')
        try:
            authorize(db, project, owner)
            doc, _, _ = snapshot(db, project, document)
            lock_scope(db, project, doc['module_id'], document, require_editable=True)
            if db.get_bind().dialect.name == 'postgresql':
                # Existing Planner/category writers do not all use the taxonomy
                # lock. Brief table locks prevent dependency/snapshot races.
                db.execute(text('LOCK TABLE planner_weeks, topics, topic_chunks IN SHARE ROW EXCLUSIVE MODE'))
            doc, topics, digest = snapshot(db, project, document)
            if digest != saved['snapshot'] or organization(db, doc['module_id']) != saved['settings']:
                raise ValueError('Preview is stale; topics or settings changed')
            priorities = db.execute(text('SELECT study_priority_categories FROM projects WHERE id=:p'), {'p':project}).scalar()
            if isinstance(priorities, str):
                priorities = json.loads(priorities)
            if priorities or db.execute(text('SELECT id FROM planner_weeks WHERE project_id=:p LIMIT 1'), {'p':project}).first():
                raise ValueError('Saved Planner data or category priorities block reorganization')
            for a in saved['result']['assignments']:
                parent_update = ', source_section=:macrocategory' if a.get('macrocategory') else ''
                updated = db.execute(text('''UPDATE topics SET category=:category,
                    category_order_index=:category_order_index, topic_order_index=:topic_order_index
                    ''' + parent_update + ''' WHERE CAST(id AS text)=:topic_id AND project_id=:p AND CAST(document_id AS text)=:d'''),
                    {**a, 'p':project, 'd':document})
                if updated.rowcount != 1:
                    raise ValueError('Topic changed during application')
            db.commit()
            del _CACHE[preview_id]
            return {'status':'applied', 'document_id':document, 'topic_count':len(topics)}
        except Exception:
            db.rollback()
            raise
