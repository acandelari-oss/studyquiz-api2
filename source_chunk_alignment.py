"""Offline alignment only. No production imports, database access, or integration.

Offsets are Python Unicode character offsets, half-open, in CanonicalBlock.raw_text.
Callers supply verified extraction-to-canonical bindings; matching text alone does
not establish provenance. Actual cleaner/chunker callbacks guard against drift.
"""
from dataclasses import dataclass, replace
import re
from typing import Callable, Optional, Tuple

from canonical_document import CanonicalBlock, SourcePosition
from document_hierarchy_contracts import CanonicalDocumentStructure, SourceSpan


@dataclass(frozen=True)
class SourceBinding:
    document_id: str
    input_start: int
    input_end: int
    block: CanonicalBlock
    block_start: int = 0
    reliable: bool = True


@dataclass(frozen=True)
class AlignedInterval:
    chunk_start: int
    chunk_end: int
    status: str  # exact, generated, unresolved, ambiguous
    source_spans: Tuple[SourceSpan, ...] = ()


@dataclass(frozen=True)
class AlignedChunk:
    text: str
    intervals: Tuple[AlignedInterval, ...]


@dataclass(frozen=True)
class AlignmentResult:
    document_id: str
    contract_version: int
    chunks: Tuple[AlignedChunk, ...]
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True)
class SectionExtent:
    node_id: str
    parent_id: Optional[str]
    heading_span: SourceSpan
    extent: SourceSpan  # includes heading; ends before the next peer/ancestor


# Each character retains the input character indices it represents. Generated
# separators carry no source indices; collapsed whitespace carries all of them.
def _clean(chars):
    for pattern in (r'([a-z])([A-Z])', r'([a-zA-Z])(\d)',
                    r'(\d)([a-zA-Z])', r'\.(\w)'):
        text = ''.join(c for c, _ in chars)
        out, cursor = [], 0
        for match in re.finditer(pattern, text):
            out.extend(chars[cursor:match.start()])
            out.extend([chars[match.start()], (' ', ()), chars[match.start()+1]])
            cursor = match.end()
        out.extend(chars[cursor:])
        chars = out
    text = ''.join(c for c, _ in chars)
    out, cursor = [], 0
    for match in re.finditer(r'\s+', text):
        out.extend(chars[cursor:match.start()])
        out.append((' ', tuple(i for _, ids in chars[match.start():match.end()] for i in ids)))
        cursor = match.end()
    out.extend(chars[cursor:])
    while out and out[0][0].isspace():
        out.pop(0)
    while out and out[-1][0].isspace():
        out.pop()
    return out


def align_chunks(document_id: str, text: str, bindings, *,
                 clean: Callable, chunk: Callable, max_chars=1000, overlap=200):
    """Observe the existing clean -> chunk -> clean pipeline without changing it.

    Since the current cleaner removes every newline, its chunker receives one
    paragraph. Repeated text is mapped by actual slice position, not substring
    search. A transformation mismatch fails closed with unchanged output.
    """
    if not document_id or not isinstance(text, str):
        raise ValueError('Document identity and text are required')
    if not 0 <= overlap < max_chars:
        raise ValueError('Require 0 <= overlap < max_chars')
    actual_clean = clean(text)
    actual = tuple(c for c in (clean(c) for c in chunk(actual_clean, max_chars=max_chars, overlap=overlap)) if c)
    traced = _clean([(c, (i,)) for i, c in enumerate(text)])
    if len(traced) > max_chars:
        slices = [traced[i:i+max_chars] for i in range(0, len(traced), max_chars-overlap)]
    else:
        slices = [traced] if traced else []
    slices = [_clean(s) for s in slices]
    slices = [s for s in slices if s]
    if actual_clean != ''.join(c for c, _ in traced) or actual != tuple(''.join(c for c, _ in s) for s in slices):
        return AlignmentResult(document_id, 1, tuple(AlignedChunk(c, (AlignedInterval(0, len(c), 'unresolved'),)) for c in actual), ('transformation_mismatch',))
    candidates = [[] for _ in text]
    blocked = set()
    diagnostics = []
    block_identities = {}
    coordinate_identities = {}
    for binding in bindings:
        if binding.document_id != document_id:
            raise ValueError('Cross-document source binding')
        block = binding.block
        identity = (block.raw_text, block.source_position, block.source_order)
        if block.block_id in block_identities and block_identities[block.block_id] != identity:
            raise ValueError('Conflicting canonical block identity')
        block_identities[block.block_id] = identity
        pos = block.source_position
        coordinate = (pos.unit_type, pos.unit_index, pos.block_index)
        if coordinate in coordinate_identities and coordinate_identities[coordinate] != block.block_id:
            raise ValueError('Ambiguous canonical block coordinates')
        coordinate_identities[coordinate] = block.block_id
        start, end = binding.input_start, binding.input_end
        size = end-start
        valid = (0 <= start < end <= len(text) and binding.block_start >= 0
                 and binding.block_start+size <= len(block.raw_text)
                 and block.source_position.start_offset >= 0
                 and text[start:end] == block.raw_text[binding.block_start:binding.block_start+size])
        if not binding.reliable or not valid:
            diagnostics.append('unreliable_binding' if not binding.reliable else 'invalid_binding')
            blocked.update(range(max(0, start), min(len(text), end)))
            continue
        for i in range(start, end):
            offset = block.source_position.start_offset+binding.block_start+i-start
            candidates[i].append((block.block_id, replace(block.source_position, start_offset=offset, end_offset=None)))
    chunks = []
    for chars in slices:
        intervals = []
        for offset, (_, origins) in enumerate(chars):
            status, spans = 'exact', []
            if not origins:
                status = 'generated'
            elif any(i in blocked or not candidates[i] for i in origins):
                status = 'unresolved'
            elif any(len(set(candidates[i])) != 1 for i in origins):
                status = 'ambiguous'
            else:
                for i in origins:
                    _, pos = candidates[i][0]
                    end = replace(pos, start_offset=pos.start_offset+1)
                    if spans and spans[-1].end == pos:
                        spans[-1] = SourceSpan(spans[-1].start, end)
                    else:
                        spans.append(SourceSpan(pos, end))
            intervals.append(AlignedInterval(offset, offset+1, status, tuple(spans)))
        chunks.append(AlignedChunk(''.join(c for c, _ in chars), tuple(intervals)))
    statuses = {i.status for c in chunks for i in c.intervals}
    diagnostics.extend(sorted(statuses & {'unresolved', 'ambiguous'}))
    return AlignmentResult(document_id, 1, tuple(chunks), tuple(sorted(set(diagnostics))))


def section_extents(structure: CanonicalDocumentStructure, blocks, *, ordering_validated=False):
    """Derive extents only from explicitly validated canonical order and parents.

    No major-chapter classification is performed. Minor nodes stay nested.
    Page numbers are never used as ordering evidence. Unknown structure fails
    closed; the caller must retain it as unresolved instead of guessing.
    """
    if not ordering_validated:
        raise ValueError('Canonical ordering must be independently validated')
    if structure.structure_status != 'accepted' or structure.structure_confidence != 'HIGH':
        raise ValueError('Unreliable hierarchy')
    blocks = list(blocks)
    keys = [(b.source_position.unit_type, b.source_position.unit_index, b.source_position.block_index) for b in blocks]
    if not blocks or len(set(keys)) != len(keys) or len({b.source_order for b in blocks}) != len(blocks):
        raise ValueError('Ambiguous canonical block coordinates')
    by_key = dict(zip(keys, blocks))
    def locate(pos):
        block = by_key.get((pos.unit_type, pos.unit_index, pos.block_index))
        if block is None or not block.source_position.start_offset <= pos.start_offset <= block.source_position.start_offset+len(block.raw_text):
            raise ValueError('Unknown source coordinate')
        return (block.source_order, pos.start_offset)
    nodes = list(structure.nodes)
    if len({n.node_id for n in nodes}) != len(nodes):
        raise ValueError('Duplicate hierarchy node')
    by_id = {n.node_id: n for n in nodes}
    ordered = []
    for node in nodes:
        span = node.source_span
        if node.status != 'accepted' or node.confidence != 'HIGH' or not span or not span.end or node.level is None or node.level < 1:
            raise ValueError('Unverified heading span')
        if locate(span.start) >= locate(span.end):
            raise ValueError('Invalid heading range')
        parent = by_id.get(node.parent_id) if node.parent_id else None
        if node.parent_id and (parent is None or parent.level is None or parent.level >= node.level):
            raise ValueError('Invalid parent relationship')
        ordered.append(node)
    ordered.sort(key=lambda n: locate(n.source_span.start))
    if [n.source_order for n in ordered] != sorted({n.source_order for n in ordered}):
        raise ValueError('Conflicting structural order')
    for current, following in zip(ordered, ordered[1:]):
        if locate(current.source_span.end) > locate(following.source_span.start):
            raise ValueError('Overlapping heading spans')
    stack = []
    for node in ordered:
        while stack and stack[-1].level >= node.level:
            stack.pop()
        if node.parent_id != (stack[-1].node_id if stack else None):
            raise ValueError('Non-nested parent relationship')
        stack.append(node)
    last = max(blocks, key=lambda b: b.source_order)
    document_end = replace(last.source_position, start_offset=last.source_position.start_offset+len(last.raw_text), end_offset=None)
    result = []
    for i, node in enumerate(ordered):
        end = next((n.source_span.start for n in ordered[i+1:] if n.level <= node.level), document_end)
        if locate(node.source_span.end) > locate(end):
            raise ValueError('Heading crosses section boundary')
        result.append(SectionExtent(node.node_id, node.parent_id, node.source_span, SourceSpan(node.source_span.start, end)))
    return tuple(result)


def text_digest(value):
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def metadata_revision(metadata):
    import json
    return text_digest(json.dumps({k: v for k, v in metadata.items() if k != "revision"},
                                  sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def verified_chunk_chapters(structure, chunks, document_id):
    """Return complete exclusive chunk ownership, or None for safe AI fallback.

    chunks: dictionaries with id, chunk_text, document_id, structural_provenance.
    Shared initial/recovery/preview gate; never approximates a cross-boundary chunk.
    """
    import json
    try:
        if isinstance(structure, str):
            structure = json.loads(structure)
        if (not structure or structure["version"] != 1 or structure["status"] != "verified"
                or structure["document_id"] != str(document_id)
                or structure["revision"] != metadata_revision(structure)):
            return None
        chapters = structure["chapters"]
        ids = {c["id"] for c in chapters}
        if not ids or len(ids) != len(chapters):
            return None
        def pos(p):
            return p["unit_index"], p["block_index"], p["start_offset"]
        ordered = sorted(chapters, key=lambda c: c["order"])
        for i, c in enumerate(ordered):
            if not c["title"] or pos(c["extent"]["start"]) >= pos(c["extent"]["end"]):
                return None
            if i and pos(ordered[i-1]["extent"]["end"]) != pos(c["extent"]["start"]):
                return None
        ownership = {}
        for chunk in chunks:
            body = chunk["chunk_text"]
            p = chunk["structural_provenance"]
            if isinstance(p, str):
                p = json.loads(p)
            if (str(chunk["document_id"]) != str(document_id) or not p or p["version"] != 1
                    or p["document_id"] != str(document_id) or p["revision"] != structure["revision"]
                    or p["chunk_sha256"] != text_digest(body)):
                return None
            cursor, owners = 0, set()
            for interval in p["intervals"]:
                start, end = interval["start"], interval["end"]
                if type(start) is not int or type(end) is not int or start != cursor or not start < end <= len(body):
                    return None
                cursor = end
                if body[start:end].isspace():
                    continue
                cid = interval["chapter_id"]
                if cid not in ids:
                    return None
                chapter = next(c for c in chapters if c["id"] == cid)
                if interval.get("ownership_basis") == "chapter_page_interval":
                    page = interval.get("source_page")
                    if (type(page) is not int or interval["source_spans"]
                            or not chapter.get("whole_page_start", float("inf")) <= page <= chapter.get("whole_page_end", -1)
                            or page < chapter["extent"]["start"]["unit_index"]
                            or page >= chapter["extent"]["end"]["unit_index"]
                            or (page == chapter["extent"]["start"]["unit_index"] and chapter["extent"]["start"]["block_index"] != 1)):
                        return None
                    owners.add(cid)
                    continue
                if interval["status"] != "exact" or not interval["source_spans"]:
                    return None
                for span in interval["source_spans"]:
                    if not (pos(chapter["extent"]["start"]) <= pos(span["start"])
                            < pos(span["end"]) <= pos(chapter["extent"]["end"])):
                        return None
                owners.add(cid)
            if cursor != len(body) or len(owners) != 1 or chunk["id"] in ownership:
                return None
            ownership[chunk["id"]] = next(iter(owners))
        return ownership or None
    except (ValueError, TypeError, KeyError, IndexError):
        return None


def verified_subsection_ownership(structure, chunks, document_id, chapter_ownership):
    """Optional direct-child scopes inside already verified automatic chapters.

    Heading spans are not content extents. Extents end at the next sibling
    start, or the parent chapter end. Incomplete siblings invalidate that
    chapter's subsection plan; unresolved chunks retain chapter ownership.
    """
    import json
    if verified_chunk_chapters(structure, chunks, document_id) != chapter_ownership:
        raise ValueError('Subsection source snapshot does not match chapter ownership')
    if isinstance(structure, str):
        structure = json.loads(structure)
    def pos(value):
        values = tuple(value[k] for k in ('unit_index', 'block_index', 'start_offset'))
        if any(type(v) is not int for v in values) or min(values[:2]) < 1 or values[2] < 0:
            raise ValueError('Invalid subsection position')
        return values
    nodes = structure.get('nodes', [])
    boundaries = structure.get('index_boundaries', [])
    sections, owners = [], {}
    for chapter in structure['chapters']:
        try:
            start, end = pos(chapter['extent']['start']), pos(chapter['extent']['end'])
            roots = [n for n in nodes if n.get('status') == 'accepted' and any(
                b['id'] == n['node_id'] and pos(b['start']) == start for b in boundaries)]
            if len(roots) != 1:
                continue
            root = roots[0]
            if root.get('provenance', {}).get('conflicting_evidence_ids'):
                continue
            children = [n for n in nodes if n.get('parent_id') == root['node_id']]
            if not children:
                continue
            planned = []
            for node in children:
                matches = [b for b in boundaries if b['id'] == node['node_id']]
                if (node.get('status') != 'accepted' or len(matches) != 1
                        or node.get('provenance', {}).get('conflicting_evidence_ids')
                        or not node.get('source_title')
                        or not start < pos(matches[0]['start']) < end):
                    raise ValueError('Unverified subsection sibling')
                planned.append({'id': node['node_id'], 'chapter_id': chapter['id'],
                                'title': node['source_title'], 'start': pos(matches[0]['start'])})
            planned.sort(key=lambda s: s['start'])
            if len({s['start'] for s in planned}) != len(planned) or len({s['id'] for s in planned}) != len(planned):
                raise ValueError('Ambiguous subsection starts')
            for i, section in enumerate(planned):
                section['end'] = planned[i+1]['start'] if i+1 < len(planned) else end
            node_by_id = {n['node_id']: n for n in nodes}
            if len(node_by_id) != len(nodes):
                raise ValueError('Duplicate canonical identity')
            for boundary in boundaries:
                location = pos(boundary['start'])
                if not start < location < end:
                    continue
                current, visited = boundary['id'], set()
                while current != root['node_id']:
                    if current in visited or current not in node_by_id:
                        raise ValueError('Unresolved subsection ancestry')
                    visited.add(current)
                    parent = node_by_id[current].get('parent_id')
                    if parent == root['node_id']:
                        scope = next((s for s in planned if s['id'] == current), None)
                        if scope is None or not scope['start'] <= location < scope['end']:
                            raise ValueError('Conflicting nested subsection position')
                    current = parent
            sections.extend(planned)
        except (KeyError, TypeError, ValueError):
            continue
    for chunk in chunks:
        provenance = chunk['structural_provenance']
        if isinstance(provenance, str):
            provenance = json.loads(provenance)
        candidates = [s for s in sections if s['chapter_id'] == chapter_ownership[chunk['id']]]
        matches = []
        for section in candidates:
            content = [i for i in provenance['intervals']
                       if chunk['chunk_text'][i['start']:i['end']].strip()]
            if content and all(i['status'] == 'exact' and i.get('source_spans') and all(
                    section['start'] <= pos(span['start']) < pos(span['end']) <= section['end']
                    for span in i['source_spans']) for i in content):
                matches.append(section['id'])
        if len(matches) == 1:
            owners[chunk['id']] = matches[0]
    return sections, owners


def _index_page_ownership(structure, chunks, document_id, items):
    """Student page ranges authorize exclusive pages, verified anchors shared pages.

    This is an ephemeral processing plan, never stored as verified PDF hierarchy.
    Review chunks get separate scopes so consolidation cannot erase uncertainty.
    """
    import json
    import unicodedata
    from taxonomy_organization import REVIEW_CATEGORY

    def norm(value):
        return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', value).casefold()))

    def pos(value):
        result = tuple(value[k] for k in ('unit_index', 'block_index', 'start_offset'))
        if any(type(v) is not int for v in result) or min(result[:2]) < 1 or result[2] < 0:
            raise ValueError('invalid_index_source_position')
        return result

    try:
        if (structure['version'] != 1 or structure['document_id'] != str(document_id)
                or structure['revision'] != metadata_revision(structure)):
            raise ValueError('invalid_index_source_revision')
        last_page = pos(structure['source_end'])[0] - 1
        chapters, ranges, names = [], [], set()
        for i, item in enumerate(items):
            pages = item.get('page_range')
            if (not isinstance(pages, list) or len(pages) != 2
                    or any(type(p) is not int for p in pages)
                    or not 1 <= pages[0] <= pages[1] <= last_page):
                raise ValueError('missing_or_invalid_index_range')
            if ranges and pages[0] < ranges[-1][1]:
                raise ValueError('overlapping_or_unordered_index_ranges')
            title = item['display_title'].strip()
            if not title or title.casefold() in names or title.casefold() == REVIEW_CATEGORY.casefold():
                raise ValueError('invalid_index_category_title')
            names.add(title.casefold())
            ranges.append(pages)
            chapters.append({'id': f'index-{i}', 'title': title, 'order': i,
                             'page_range': list(pages), 'ownership_basis': 'student_index'})
        if not chapters:
            raise ValueError('empty_index')
        pages_to_sections = {page: [i for i, (start, end) in enumerate(ranges) if start <= page <= end]
                             for page in range(1, last_page + 1)}
        cuts, boundary_report = {}, []
        for page, indices in pages_to_sections.items():
            if len(indices) < 2:
                continue
            starts = []
            for i in indices[1:]:
                anchor = items[i].get('source_heading') or items[i]['display_title']
                matches = [b for b in structure.get('index_boundaries', [])
                           if b['start']['unit_index'] == page and norm(b['title']) == norm(anchor)]
                start = pos(matches[0]['start']) if len(matches) == 1 else None
                starts.append(start)
                boundary_report.append({'page': page, 'category': chapters[i]['title'],
                                        'source_heading': anchor,
                                        'status': 'verified' if start else 'unresolved'})
            if all(starts):
                page_cuts = [(page, 1, 0)] + starts + [(page + 1, 1, 0)]
                if all(a < b for a, b in zip(page_cuts, page_cuts[1:])):
                    cuts[page] = page_cuts
                else:
                    for boundary in boundary_report:
                        if boundary['page'] == page:
                            boundary['status'] = 'conflicting_order'

        ownership, review = {}, []
        for chunk in chunks:
            p = chunk.get('structural_provenance')
            if isinstance(p, str):
                p = json.loads(p)
            if (str(chunk['document_id']) != str(document_id) or not p or p['version'] != 1
                    or p['document_id'] != str(document_id) or p['revision'] != structure['revision']
                    or p['chunk_sha256'] != text_digest(chunk['chunk_text'])):
                raise ValueError('invalid_index_chunk_revision')
            if chunk['id'] in ownership:
                raise ValueError('duplicate_index_chunk_id')
            page = p.get('source_page')
            if page is not None and (type(page) is not int or not 1 <= page <= last_page):
                raise ValueError('invalid_index_source_page')
            indices = pages_to_sections.get(page, [])
            cursor, interval_owners = 0, set()
            for interval in p['intervals']:
                start, end = interval['start'], interval['end']
                if (type(start) is not int or type(end) is not int or start != cursor
                        or not start < end <= len(chunk['chunk_text'])):
                    raise ValueError('invalid_index_chunk_intervals')
                cursor = end
                spans = interval['source_spans']
                for span in spans:
                    a, b = pos(span['start']), pos(span['end'])
                    if a >= b or (page is not None and (a[0] != page or b[0] != page)):
                        raise ValueError('inconsistent_index_source_page')
                if chunk['chunk_text'][start:end].isspace():
                    continue
                owner = None
                if len(indices) == 1:
                    owner = indices[0]
                elif page in cuts and interval['status'] == 'exact' and spans:
                    matched = [i for j, i in enumerate(indices) if all(
                        cuts[page][j] <= pos(s['start']) < pos(s['end']) <= cuts[page][j+1]
                        for s in spans)]
                    if len(matched) == 1:
                        owner = matched[0]
                interval_owners.add(owner)
            if cursor != len(chunk['chunk_text']):
                raise ValueError('incomplete_index_chunk_intervals')
            if len(interval_owners) == 1 and None not in interval_owners:
                ownership[chunk['id']] = chapters[next(iter(interval_owners))]['id']
            else:
                rid = f'index-review-{len(review)}'
                ownership[chunk['id']] = rid
                chapters.append({'id': rid, 'title': REVIEW_CATEGORY, 'order': len(chapters),
                                 'system_review': True})
                review.append({'chunk_id': chunk['id'], 'page': page,
                               'reason': ('missing_source_page' if page is None else
                                          'outside_index_ranges' if not indices else
                                          'ambiguous_or_cross_boundary_content')})
        if not ownership:
            raise ValueError('empty_index_source')
        report = {'assigned_chunks': len(chunks) - len(review), 'review_chunks': len(review),
                  'boundaries': boundary_report, 'review_details': review}
        return {**structure, 'status': 'index_mapped', 'chapters': chapters,
                'index_mapping': report}, ownership, ('mapped_with_review' if review else 'mapped')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as error:
        return None, None, str(error) if isinstance(error, ValueError) else 'invalid_index_source_metadata'


def _index_boundary_ownership(structure, chunks, document_id, items):
    """Request-local ownership from individually verified starts and exact spans.

    Does not rewrite stored provenance or accept page ranges as heading evidence.
    Ambiguous starts, uncovered material and cross-boundary chunks fail closed.
    """
    import copy
    import json
    import unicodedata
    def normalized(title):
        return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', title).casefold()))
    def pos(p):
        return p['unit_index'], p['block_index'], p['start_offset']
    if not isinstance(structure, dict) or not structure.get('index_boundaries'):
        return None, None, 'unverified_source_ownership'
    try:
        if (structure['document_id'] != str(document_id) or structure['version'] != 1
                or structure['revision'] != metadata_revision(structure)):
            raise ValueError('invalid_index_source_revision')
        chapters = []
        for i, item in enumerate(items):
            pages = item.get('page_range')
            if pages is None and type(item.get('page_start')) is int:
                pages = [item['page_start'], item['page_start']]
            if (not isinstance(pages, list) or len(pages) != 2
                    or any(type(p) is not int or p < 1 for p in pages) or pages[0] > pages[1]):
                raise ValueError('missing_or_invalid_index_range')
            matches = [b for b in structure['index_boundaries']
                       if b['start']['unit_index'] == pages[0]
                       and normalized(b['title']) == normalized(item['display_title'])]
            if len(matches) != 1:
                raise ValueError('unverified_or_ambiguous_index_heading')
            b = matches[0]
            chapters.append({'id': b['id'], 'title': item['display_title'], 'order': i,
                             'extent': {'start': copy.deepcopy(b['start'])}})
        if not chapters:
            raise ValueError('empty_index')
        for i, c in enumerate(chapters):
            start = c['extent']['start']
            end = copy.deepcopy(chapters[i+1]['extent']['start'] if i+1 < len(chapters)
                                else structure['source_end'])
            if pos(start) >= pos(end):
                raise ValueError('invalid_index_boundary_order')
            last_page = end['unit_index'] - int(end['block_index'] == 1 and end['start_offset'] == 0)
            if items[i].get('page_range') is not None and items[i]['page_range'][1] != last_page:
                raise ValueError('index_source_page_mismatch')
            c['extent']['end'] = end
            c['whole_page_start'] = start['unit_index'] + int(start['block_index'] != 1 or start['start_offset'] != 0)
            c['whole_page_end'] = end['unit_index'] - 1
        mapped = {**structure, 'status': 'verified', 'chapters': chapters}
        mapped.pop('reason', None)
        mapped['revision'] = metadata_revision(mapped)
        local_chunks = copy.deepcopy(chunks)
        for chunk in local_chunks:
            p = chunk['structural_provenance']
            if isinstance(p, str):
                p = json.loads(p)
            if (str(chunk['document_id']) != str(document_id) or not p or p['version'] != 1
                    or p['document_id'] != str(document_id) or p['revision'] != structure['revision']
                    or p['chunk_sha256'] != text_digest(chunk['chunk_text'])):
                raise ValueError('invalid_index_chunk_revision')
            p['revision'] = mapped['revision']
            for interval in p['intervals']:
                interval['chapter_id'] = None
                interval.pop('ownership_basis', None)
                spans = interval['source_spans']
                if interval['status'] == 'exact' and spans:
                    owners = [c['id'] for c in chapters if all(
                        pos(c['extent']['start']) <= pos(span['start']) < pos(span['end']) <= pos(c['extent']['end'])
                        for span in spans)]
                    if len(owners) == 1:
                        interval['chapter_id'] = owners[0]
                # Whole-page OCR ownership requires persisted source-page evidence.
                # No page is inferred from text or a legacy section label.
                elif not spans and type(interval.get('source_page')) is int:
                    page = interval['source_page']
                    owners = [c['id'] for c in chapters if c['whole_page_start'] <= page <= c['whole_page_end']]
                    if len(owners) == 1:
                        interval['chapter_id'] = owners[0]
                        interval['ownership_basis'] = 'chapter_page_interval'
            chunk['structural_provenance'] = p
        ownership = verified_chunk_chapters(mapped, local_chunks, document_id)
        if ownership is None:
            raise ValueError('incomplete_index_source_ownership')
        return mapped, ownership, 'verified'
    except (ValueError, TypeError, KeyError, IndexError) as error:
        return None, None, str(error) if isinstance(error, ValueError) else 'invalid_index_source_metadata'


def verified_index_chapters(structure, chunks, document_id, index_items):
    """Map index ranges with local review; retain strict legacy metadata handling.

    Returned index scopes are request-local and are not verified automatic chapters.
    Persisted source metadata and input objects are never rewritten here.
    """
    import json
    import unicodedata
    def fail(reason):
        return None, None, reason
    if isinstance(structure, str):
        try:
            structure = json.loads(structure)
        except (ValueError, TypeError):
            return fail('invalid_structure')
    if any(item.get('page_start') is not None and item.get('page_range') is None for item in index_items):
        # Starts alone are not whole-page ranges. Use verified heading positions
        # so a chapter beginning midway through a page cannot steal preceding text.
        return _index_boundary_ownership(structure, chunks, document_id, index_items)
    if isinstance(structure, dict) and structure.get('index_page_provenance_version') == 1:
        return _index_page_ownership(structure, chunks, document_id, index_items)
    independent = _index_boundary_ownership(structure, chunks, document_id, index_items)
    if independent[1] is not None:
        return independent
    ownership = verified_chunk_chapters(structure, chunks, document_id)
    if ownership is None:
        return independent
    chapters = sorted(structure['chapters'], key=lambda c: c['order'])
    if len(index_items) != len(chapters):
        return fail('index_structure_mismatch')
    def normalized(title):
        return ' '.join(re.findall(r'[^\W_]+', unicodedata.normalize('NFKC', title).casefold()))
    mapped, previous_end = [], None
    for item, chapter in zip(index_items, chapters):
        pages = item.get('page_range')
        if (not isinstance(pages, list) or len(pages) != 2
                or any(type(p) is not int or p < 1 for p in pages) or pages[0] > pages[1]):
            return fail('missing_or_invalid_index_range')
        start, end = pages
        # Adjacent chapters can share a page only with verified within-page spans.
        if previous_end is not None and start not in {previous_end, previous_end + 1}:
            return fail('noncontiguous_index_ranges')
        source_start, source_end = chapter['extent']['start'], chapter['extent']['end']
        end_at_page_start = source_end['block_index'] == 1 and source_end['start_offset'] == 0
        last_page = source_end['unit_index'] - int(end_at_page_start)
        if start != source_start['unit_index'] or end != last_page:
            return fail('index_source_page_mismatch')
        if normalized(item['display_title']) != normalized(chapter['title']):
            return fail('index_source_title_mismatch')
        mapped.append({**chapter, 'title': item['display_title']})
        previous_end = end
    if not mapped:
        return fail('empty_index')
    return {**structure, 'chapters': mapped}, ownership, 'verified'
