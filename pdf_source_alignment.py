"""Digital PDF provenance adapter using the existing extraction engine.

Verification is of pypdf extraction-event provenance, NOT semantic reading order
or PDF glyph fidelity. No substring search, fuzzy matching or page-based chapter
assignment is performed. chapter_metadata separately verifies supported simple layouts.
"""
from dataclasses import dataclass, replace
import hashlib
import io
from typing import Tuple

from pypdf import PdfReader
from canonical_document import adapt_pdf_document
from document_extractors import extract_pdf_document
from document_hierarchy_evidence_mapper import map_document_to_structural_evidence
from document_hierarchy_preserver import preserve_document_hierarchy
from source_chunk_alignment import AlignmentResult, SourceBinding, align_chunks


@dataclass(frozen=True)
class PdfBlockAlignment:
    extraction_block_index: int
    page: int
    alignment: AlignmentResult
    diagnostics: Tuple[str, ...]


@dataclass(frozen=True)
class PdfAlignmentResult:
    document_id: str
    source_sha256: str
    canonical_document: object
    structural_evidence: tuple
    structure: object
    blocks: Tuple[PdfBlockAlignment, ...]
    diagnostics: Tuple[str, ...]
    # Callback order establishes extraction offsets, not section ownership.
    section_extents_verified: bool = False


def _unverified(result, diagnostic, status='unresolved'):
    return replace(result, chunks=tuple(replace(c, intervals=tuple(
        replace(i, status=status, source_spans=()) if i.status != 'generated' else i
        for i in c.intervals)) for c in result.chunks),
        diagnostics=tuple(sorted(set(result.diagnostics) | {diagnostic})))


def align_digital_pdf(file_bytes, filename, document_id, *, clean, chunk,
                      extracted_document=None, max_chars=1000, overlap=200):
    """Align the existing page extraction using an independently verified event tape.

    Reuses pypdf (the current extractor), canonical adapter/evidence/preserver,
    and Sprint 1 contract. An optional supplied extraction must match the fresh
    digital extraction exactly. Appended OCR text stays unbound; altered, split
    or reordered digital blocks fail closed independently.
    """
    digital = extract_pdf_document(file_bytes, filename, inspect_acquisition=True)
    supplied = extracted_document or digital
    canonical = adapt_pdf_document(file_bytes, filename)
    evidence = tuple(map_document_to_structural_evidence(canonical))
    structure = preserve_document_hierarchy(evidence, document_id=document_id,
        document_title=filename, source_format='pdf')
    reader = PdfReader(io.BytesIO(file_bytes))
    if reader.is_encrypted:
        reader.decrypt('')  # Existing extractors already enforce password handling.
    results = []
    for index, extracted in enumerate(supplied.blocks):
        bindings, diagnostics = [], []
        ambiguous = False
        original = digital.blocks[index] if index < len(digital.blocks) else None
        digital_text = extracted.digital_text if extracted.digital_text is not None else extracted.text
        trim = extracted.digital_text_start
        retained_end = min(len(digital_text), trim + len(extracted.text))
        same_block = (original is not None
            and (extracted.page, extracted.block_index, extracted.source_unit) ==
                (original.page, original.block_index, original.source_unit)
            and digital_text == original.text
            and (extracted.digital_text is not None or extracted.raw_text == original.raw_text)
            and 0 <= trim <= len(digital_text)
            and not digital_text[:trim].strip()
            and not digital_text[retained_end:].strip()
            and extracted.text[:retained_end-trim] == digital_text[trim:retained_end])
        if not same_block:
            diagnostics.append('extraction_identity_mismatch')
        elif extracted.acquisition_error or original.acquisition_error:
            diagnostics.append('digital_extraction_error')
        else:
            events = []
            try:
                def visitor(text, cm, tm, font, size):
                    events.append(text or '')
                returned = reader.pages[index].extract_text(visitor_text=visitor) or ''
                if reader.pages[index].rotation:
                    diagnostics.append('rotated_page_order_unverified')
                blocks = canonical.units[index].blocks
                nonempty = [text for text in events if text.strip()]
                if not nonempty:
                    diagnostics.append('no_digital_text_provenance')
                elif (returned != digital_text
                      or ''.join(events) != returned
                      or nonempty != [b.raw_text for b in blocks]):
                    diagnostics.append('ambiguous_extraction_event_tape')
                    ambiguous = True
                else:
                    # Event ordinal + cumulative length is the identity. Repeated
                    # text is never located by searching for its first occurrence.
                    offset, block_index = 0, 0
                    for text in events:
                        if text.strip():
                            start, end = max(offset, trim), min(offset+len(text), retained_end)
                            if start < end:
                                bindings.append(SourceBinding(document_id, start-trim, end-trim,
                                    blocks[block_index], block_start=start-offset))
                            block_index += 1
                        offset += len(text)
                    # Whitespace-only events have no canonical block: deliberately
                    # leave those intervals unresolved instead of inventing one.
            except Exception as error:
                diagnostics.append('event_verification_error:' + type(error).__name__)
                bindings = []
        aligned = align_chunks(document_id, extracted.text or '', bindings,
                               clean=clean, chunk=chunk, max_chars=max_chars, overlap=overlap)
        if diagnostics:
            aligned = _unverified(aligned, diagnostics[0], 'ambiguous' if ambiguous else 'unresolved')
        results.append(PdfBlockAlignment(extracted.block_index, extracted.page,
                                        aligned, tuple(diagnostics)))
    return PdfAlignmentResult(document_id, hashlib.sha256(file_bytes).hexdigest(), canonical,
                              evidence, structure, tuple(results),
                              ('semantic_reading_order_and_section_extents_unverified',))


def _upright_rendered_text(metadata):
    """Validate the composed text-to-page transform, not its individual factors."""
    import math
    tm, cm = (metadata.get(k, []) for k in ("text_matrix", "current_transformation_matrix"))
    if any(len(m) != 6 or not all(type(v) in (int, float) and math.isfinite(v) for v in m)
           for m in (tm, cm)):
        return False
    a = tm[0] * cm[0] + tm[1] * cm[2]
    b = tm[0] * cm[1] + tm[1] * cm[3]
    c = tm[2] * cm[0] + tm[3] * cm[2]
    d = tm[2] * cm[1] + tm[3] * cm[3]
    # Only upright axis-aligned text is supported. Rotation, skew, reflection
    # and singular transforms still fail closed after composition.
    return all(math.isfinite(v) for v in (a, b, c, d)) and a > 0 and d > 0 and b == 0 and c == 0


def chapter_metadata(result):
    """Persist main-heading intervals independently of subsection hierarchy.

    Exact spans resolve within-page boundaries. Whole-page interval membership
    includes OCR without claiming OCR character coordinates.
    """
    import math
    from dataclasses import asdict
    from source_chunk_alignment import metadata_revision, text_digest
    structure = result.structure
    doc = {"version": 1, "document_id": result.document_id,
           "source_sha256": result.source_sha256, "status": "unavailable",
           "nodes": [asdict(n) for n in structure.nodes], "chapters": []}
    blocks = result.canonical_document.blocks
    extraction_diagnostics = [
        {"page": b.page, "extraction_block_index": b.extraction_block_index,
         "codes": list(b.diagnostics)}
        for b in result.blocks if b.diagnostics
    ]
    if extraction_diagnostics:
        doc["extraction_diagnostics"] = extraction_diagnostics
    # Index mode can select individually verified heading starts even when the
    # automatic prominence hierarchy is unavailable. Keep evidence in existing
    # revisioned metadata; never promote these to automatic chapters.
    boundaries = []
    by_evidence = {"evidence-" + b.block_id: b for b in blocks}
    for node in structure.nodes:
        group = sorted([by_evidence[e] for e in node.evidence_ids if e in by_evidence],
                       key=lambda b: b.source_order)
        if not group or node.provenance.conflicting_evidence_ids:
            continue
        first, last = group[0], group[-1]
        page = first.source_position.unit_index
        if (any(b.source_position.unit_index != page for b in group)
                or any(b.diagnostics for b in result.blocks if b.page == page)):
            continue
        if (any(not _upright_rendered_text(b.metadata) for b in group)
                or any(b.source_order != a.source_order + 1 for a, b in zip(group, group[1:]))):
            continue
        if any(not isinstance(b.metadata.get("y"), (int, float))
               or not math.isfinite(b.metadata["y"]) for b in blocks
               if b.source_position.unit_index == page):
            continue
        if any((b.source_order < first.source_order and b.metadata["y"] <= first.metadata["y"])
               or (b.source_order > last.source_order and b.metadata["y"] >= last.metadata["y"])
               for b in blocks if b.source_position.unit_index == page and b not in group):
            continue
        boundaries.append({"id": node.node_id, "title": node.source_title,
                           "start": asdict(first.source_position)})
    doc["index_boundaries"] = boundaries
    doc["index_page_provenance_version"] = 1
    doc["source_end"] = {"unit_index": len(result.canonical_document.units) + 1,
                         "block_index": 1, "start_offset": 0}
    try:
        if any(code != "no_digital_text_provenance" for b in result.blocks for code in b.diagnostics):
            raise ValueError("unverified_extraction")
        # Main chapters use recurring, most prominent accepted heading evidence.
        # Subsection parent repairs do not determine chapter eligibility.
        by_id = {"evidence-" + b.block_id: b for b in blocks}
        candidates = []
        for node in structure.nodes:
            if (node.status == "accepted" and node.level == 1
                    and not node.provenance.conflicting_evidence_ids):
                candidates.extend(by_id[e] for e in node.evidence_ids if e in by_id)
        if not candidates:
            raise ValueError("no_reliable_main_headings")
        size = lambda b: b.metadata.get("effective_font_size", b.metadata.get("font_size", 0))
        largest = max(size(b) for b in candidates)
        candidates = sorted({b.block_id: b for b in candidates if size(b) == largest}.values(),
                            key=lambda b: b.source_order)
        groups = []
        for block in candidates:
            m = block.metadata
            if not _upright_rendered_text(m):
                raise ValueError("transformed_heading")
            if not all(math.isfinite(m[k]) for k in ("x", "y")):
                raise ValueError("missing_heading_geometry")
            previous = groups[-1][-1] if groups else None
            # Adjacent fragments of the same visual heading, including wrapped
            # titles. Any intervening body fragment starts a separate heading.
            if (previous and block.source_position.unit_index == previous.source_position.unit_index
                    and block.source_order == previous.source_order + 1
                    and m.get("font_name") == previous.metadata.get("font_name")
                    and 0 <= previous.metadata["y"] - m["y"] <= largest * 1.5):
                if previous.metadata["y"] == m["y"] and m["x"] <= previous.metadata["x"]:
                    raise ValueError("ambiguous_heading_order")
                groups[-1].append(block)
            else:
                groups.append([block])
        if not groups:
            raise ValueError("no_reliable_main_headings")
        for group in groups:
            first = group[0]
            page = first.source_position.unit_index
            diagnostic = next((b.diagnostics for b in result.blocks if b.page == page), ())
            if diagnostic:
                raise ValueError("unverified_heading_page")
            # Validate content on both sides of a boundary using actual baselines;
            # no assumption that callback order alone is visual reading order.
            for block in blocks:
                if block.source_position.unit_index != page or block in group:
                    continue
                y = block.metadata.get("y")
                if y is None or not math.isfinite(y):
                    raise ValueError("missing_boundary_geometry")
                if ((block.source_order < first.source_order and y <= first.metadata["y"])
                    or (block.source_order > group[-1].source_order and y >= group[-1].metadata["y"])):
                    doc["boundary_diagnostics"] = [{
                        "code": "ambiguous_boundary_order", "page": page,
                        "heading_fragment": first.block_id,
                        "heading_text": " ".join(b.raw_text for b in group),
                        "heading_source_order": first.source_order,
                        "heading_y": first.metadata["y"],
                        "conflicting_fragment": block.block_id,
                        "conflicting_text": block.raw_text,
                        "conflicting_source_order": block.source_order,
                        "conflicting_y": y,
                    }]
                    raise ValueError("ambiguous_boundary_order")
        # No unowned preamble: if there is content before the first chapter, use AI.
        if groups[0][0] != blocks[0] or groups[0][0].source_position.unit_index != 1:
            raise ValueError("unowned_preamble")
        for index, group in enumerate(groups):
            title = group[0].raw_text
            for previous, current in zip(group, group[1:]):
                title += (" " if previous.metadata["y"] != current.metadata["y"] else "") + current.raw_text
            title = " ".join(title.split())
            start = asdict(group[0].source_position)
            start["start_offset"] = 0
            end = (asdict(groups[index+1][0].source_position) if index+1 < len(groups)
                   else dict(start, unit_index=len(result.canonical_document.units)+1, block_index=1))
            end["start_offset"] = 0
            doc["chapters"].append({"id": group[0].block_id, "title": title, "order": index,
                "heading_fragments": [b.block_id for b in group],
                "extent": {"start": start, "end": end},
                "whole_page_start": start["unit_index"] if start["block_index"] == 1 else start["unit_index"]+1,
                "whole_page_end": end["unit_index"]-1})
        if len({c["title"].casefold() for c in doc["chapters"]}) != len(doc["chapters"]):
            raise ValueError("repeated_chapter_titles")
        doc["status"] = "verified"
    except (ValueError, TypeError, KeyError) as error:
        doc["reason"] = str(error)
        doc["chapters"] = []
    doc["revision"] = metadata_revision(doc)

    def position(p):
        return (p["unit_index"], p["block_index"], p["start_offset"])
    def owner(span):
        start, end = position(asdict(span.start)), position(asdict(span.end))
        matches = [c["id"] for c in doc["chapters"]
                   if position(c["extent"]["start"]) <= start < end <= position(c["extent"]["end"])]
        return matches[0] if len(matches) == 1 else None

    provenance = []
    for block in result.blocks:
        block_chunks = []
        for chunk in block.alignment.chunks:
            intervals = []
            for interval in chunk.intervals:
                owners = {owner(s) for s in interval.source_spans}
                chapter = next(iter(owners)) if len(owners) == 1 and None not in owners else None
                enclosed = [c["id"] for c in doc["chapters"]
                            if c["whole_page_start"] <= block.page <= c["whole_page_end"]]
                page_owner = enclosed[0] if len(enclosed) == 1 else None
                if chapter is None and not interval.source_spans and page_owner:
                    chapter = page_owner
                item = {"start": interval.chunk_start, "end": interval.chunk_end,
                        "status": interval.status, "chapter_id": chapter,
                        "source_spans": [asdict(s) for s in interval.source_spans]}
                if (not interval.source_spans and
                        all(code == "no_digital_text_provenance" for code in block.diagnostics)):
                    item["source_page"] = block.page
                if chapter and not interval.source_spans and page_owner == chapter:
                    item["ownership_basis"] = "chapter_page_interval"
                    item["source_page"] = block.page
                # Coalesce contiguous exact coordinates without losing provenance.
                if intervals:
                    prev = intervals[-1]
                    spans = prev["source_spans"]
                    following = item["source_spans"]
                    contiguous = (len(spans) == len(following) == 1
                                  and spans[0]["end"] == following[0]["start"])
                    empty = not spans and not following
                    if (prev["end"] == item["start"] and prev["status"] == item["status"]
                            and prev["chapter_id"] == chapter
                            and prev.get("ownership_basis") == item.get("ownership_basis")
                            and (contiguous or empty)):
                        prev["end"] = item["end"]
                        if contiguous:
                            spans[0]["end"] = following[0]["end"]
                        continue
                intervals.append(item)
            block_chunks.append({"version": 1, "document_id": result.document_id,
                                 "revision": doc["revision"], "chunk_sha256": text_digest(chunk.text),
                                 "source_page": (block.page if all(code == "no_digital_text_provenance"
                                                                  for code in block.diagnostics) else None),
                                 "intervals": intervals})
        provenance.append(block_chunks)
    return doc, provenance
