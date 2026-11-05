"""Document processing state; callers own locks and transaction boundaries."""
from sqlalchemy import text
import uuid


def aggregate_status(statuses):
    tracked = [s for s in statuses if s is not None]
    if not tracked:
        return None
    if any(s in ('queued', 'building') for s in tracked):
        return 'building'
    return 'failed' if 'failed' in tracked else 'ready'


def refresh_taxonomy_status(db, project_id, module_id=None):
    """Never advertise completion while another document is outstanding."""
    params = {'project_id': project_id, 'module_id': module_id}
    if module_id:
        status = aggregate_status(r[0] for r in db.execute(text(
            'select taxonomy_status from documents where project_id=:project_id and module_id=:module_id'), params))
        if status:
            db.execute(text("""update study_modules set taxonomy_status=:taxonomy_status,
                status=:status where project_id=:project_id and id=:module_id
                and accepted_for_study=false"""), {**params, 'taxonomy_status': status,
                    'status': 'pending_study' if status == 'ready' else 'building'})
    status = aggregate_status(r[0] for r in db.execute(text(
        'select taxonomy_status from documents where project_id=:project_id'), params))
    if status:
        db.execute(text('update projects set topic_status=:status where id=:project_id'),
                   {**params, 'status': {'building':'processing', 'failed':'error', 'ready':'completed'}[status]})


def queue_document(db, project_id, document_id, module_id):
    row = db.execute(text('''select module_id, taxonomy_status from documents
        where project_id=:project_id and id=:document_id'''),
        {'project_id':project_id, 'document_id':document_id}).fetchone()
    if not row or str(row[0] or '') != str(module_id or ''):
        raise ValueError('Document does not belong to the requested module')
    # Legacy document-owned topics are already valid. Do not regenerate them.
    has_topics = db.execute(text('''select id from topics where project_id=:project_id
        and document_id=:document_id limit 1'''),
        {'project_id':project_id, 'document_id':document_id}).fetchone()
    if row[1] in ('ready', 'building', 'failed') or (row[1] is None and has_topics):
        return False
    db.execute(text("""update documents set taxonomy_status='queued', taxonomy_run_id=coalesce(taxonomy_run_id, :run_id)
        where project_id=:project_id and id=:document_id"""),
        {'project_id':project_id, 'document_id':document_id, 'run_id':str(uuid.uuid4())})
    return True


def replace_document_taxonomy(db, project_id, document_id):
    """Delete only this file's replacement scope; never commit here."""
    if not document_id:
        raise ValueError('Document ID required for taxonomy replacement')
    params = {'project_id':project_id, 'document_id':document_id}
    db.execute(text('''delete from topic_chunks where topic_id in
        (select id from topics where project_id=:project_id and document_id=:document_id)'''), params)
    db.execute(text('delete from topics where project_id=:project_id and document_id=:document_id'), params)


def document_attempt_current(db, project_id, document_id, run_id):
    row = db.execute(text("select taxonomy_run_id from documents where project_id=:p and id=:d"),
                     {"p": project_id, "d": document_id}).fetchone()
    return bool(row and str(row[0]) == str(run_id))
