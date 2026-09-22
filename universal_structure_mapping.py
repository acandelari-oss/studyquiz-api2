"""Universal document structure anchoring, segmentation, and diagnostics.

The pipeline is isolated from production upload/database behavior.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

from canonical_document import CanonicalBlock, CanonicalDocumentInput, SourcePosition
from universal_structure_interpreter import (
    DocumentSection,
    DocumentStructure,
    confidence_rank,
)


STATUS_ACCEPTED = "accepted"
STATUS_PARTIAL = "partial"
STATUS_FALLBACK = "fallback"


@dataclass(frozen=True)
class Anchor:
    source_position: SourcePosition
    method: str
    confidence: float
    fidelity: str


@dataclass(frozen=True)
class AnchoredSection(DocumentSection):
    anchor: Optional[Anchor] = None


@dataclass(frozen=True)
class AnchorSummary:
    anchored: int
    unresolved: int
    exact_count: int
    normalized_count: int
    fuzzy_count: int
    model_only_count: int
    duplicate_anchor_collisions: int
    source_order_violations: int


@dataclass(frozen=True)
class AnchoredDocumentStructure:
    document_title: str
    document_type: str
    confidence: str
    analyzer: str
    analyzer_version: str
    sections: List[AnchoredSection]
    anchor_summary: AnchorSummary


@dataclass(frozen=True)
class CanonicalSegment:
    text: str
    raw_text: str
    source_start: SourcePosition
    source_end: SourcePosition
    section_id: Optional[str]
    section_title: str
    section_path: str
    section_level: Optional[int]
    parent_id: Optional[str]
    provenance: str
    mapping_confidence: float


@dataclass(frozen=True)
class DiagnosticChunk:
    text: str
    section_path: str
    source_unit: str
    segment_index: int
    chunk_index: int
    source_start: SourcePosition
    source_end: SourcePosition


@dataclass(frozen=True)
class DocumentStructureMappingResult:
    file_type: str
    filename: str
    structure_status: str
    mapping_status: str
    structure_confidence: str
    mapping_confidence: float
    sections_total: int
    sections_anchored: int
    exact_count: int
    normalized_count: int
    fuzzy_count: int
    model_only_count: int
    unresolved_sections: int
    duplicate_anchor_collisions: int
    source_order_violations: int
    boundary_errors: int
    segments: List[CanonicalSegment]
    diagnostic_chunks: List[DiagnosticChunk]
    diagnostics: Dict[str, object]


def anchor_structure(
    document: CanonicalDocumentInput,
    structure: DocumentStructure,
) -> AnchoredDocumentStructure:
    occupied: set[Tuple[str, int, int, int, int]] = set()
    anchored_sections: List[AnchoredSection] = []
    duplicate_collisions = 0
    order_violations = 0
    previous_anchor: Optional[SourcePosition] = None

    for section in structure.sections:
        anchor = _find_anchor(document, section)
        if anchor:
            key = _anchor_key(anchor.source_position)
            if key in occupied:
                duplicate_collisions += 1
                anchor = None
            elif previous_anchor and anchor.source_position < previous_anchor:
                order_violations += 1
                anchor = None
            else:
                occupied.add(key)
                previous_anchor = anchor.source_position

        anchored_sections.append(
            AnchoredSection(
                **{
                    field: getattr(section, field)
                    for field in DocumentSection.__dataclass_fields__
                },
                anchor=anchor,
            )
        )

    exact = sum(1 for section in anchored_sections if section.anchor and section.anchor.fidelity == "exact")
    normalized = sum(1 for section in anchored_sections if section.anchor and section.anchor.fidelity == "normalized")
    fuzzy = sum(1 for section in anchored_sections if section.anchor and section.anchor.fidelity == "fuzzy")
    unresolved = sum(1 for section in anchored_sections if not section.anchor)
    summary = AnchorSummary(
        anchored=len(anchored_sections) - unresolved,
        unresolved=unresolved,
        exact_count=exact,
        normalized_count=normalized,
        fuzzy_count=fuzzy,
        model_only_count=unresolved,
        duplicate_anchor_collisions=duplicate_collisions,
        source_order_violations=order_violations,
    )
    return AnchoredDocumentStructure(
        document_title=structure.document_title,
        document_type=structure.document_type,
        confidence=structure.confidence,
        analyzer=structure.analyzer,
        analyzer_version=structure.analyzer_version,
        sections=anchored_sections,
        anchor_summary=summary,
    )


def segment_document(
    document: CanonicalDocumentInput,
    anchored_structure: AnchoredDocumentStructure,
) -> List[CanonicalSegment]:
    sections_by_id = {section.section_id: section for section in anchored_structure.sections}
    anchors_by_block: Dict[str, List[AnchoredSection]] = {}
    for section in anchored_structure.sections:
        if section.anchor:
            anchors_by_block.setdefault(_block_key(section.anchor.source_position), []).append(section)

    for sections in anchors_by_block.values():
        sections.sort(key=lambda section: (section.anchor.source_position.start_offset, section.source_order))

    active: Optional[AnchoredSection] = None
    segments: List[CanonicalSegment] = []
    fallback_title = document.title or document.filename or "GENERAL"

    for block in document.blocks:
        raw_text = block.raw_text or block.text or ""
        cursor = 0
        boundaries = anchors_by_block.get(_block_key(block.source_position), [])
        for section in boundaries:
            assert section.anchor is not None
            start = max(0, min(section.anchor.source_position.start_offset, len(raw_text)))
            if start > cursor:
                segments.append(
                    _make_segment(
                        raw_text[cursor:start],
                        block,
                        cursor,
                        start,
                        active,
                        sections_by_id,
                        fallback_title,
                    )
                )
            active = section
            cursor = start
        if cursor < len(raw_text):
            segments.append(
                _make_segment(
                    raw_text[cursor:],
                    block,
                    cursor,
                    len(raw_text),
                    active,
                    sections_by_id,
                    fallback_title,
                )
            )
    return [segment for segment in segments if segment.raw_text.strip()]


def evaluate_mapping(
    document: CanonicalDocumentInput,
    structure: Optional[DocumentStructure],
    anchored: Optional[AnchoredDocumentStructure],
    segments: List[CanonicalSegment],
    chunks: List[DiagnosticChunk],
    *,
    interpretation_failed: bool = False,
) -> DocumentStructureMappingResult:
    if interpretation_failed or structure is None or anchored is None:
        return _fallback_result(document, "model interpretation unavailable")

    summary = anchored.anchor_summary
    total = len(anchored.sections)
    boundary_errors = _count_boundary_errors(segments)
    anchor_quality = _anchor_quality(summary, total)
    completeness = summary.anchored / total if total else 0.0
    mapping_confidence = max(0.0, min(1.0, (anchor_quality * 0.7) + (completeness * 0.3)))

    if (
        confidence_rank(structure.confidence) <= 1
        or summary.source_order_violations
        or summary.duplicate_anchor_collisions
        or boundary_errors
        or total == 0
    ):
        status = STATUS_FALLBACK
    elif summary.unresolved == 0 and mapping_confidence >= 0.78:
        status = STATUS_ACCEPTED
    elif summary.anchored > 0 and mapping_confidence >= 0.45:
        status = STATUS_PARTIAL
    else:
        status = STATUS_FALLBACK

    return DocumentStructureMappingResult(
        file_type=document.file_type,
        filename=document.filename,
        structure_status=status if status != STATUS_FALLBACK else STATUS_FALLBACK,
        mapping_status=status,
        structure_confidence=structure.confidence,
        mapping_confidence=round(mapping_confidence, 3),
        sections_total=total,
        sections_anchored=summary.anchored,
        exact_count=summary.exact_count,
        normalized_count=summary.normalized_count,
        fuzzy_count=summary.fuzzy_count,
        model_only_count=summary.model_only_count,
        unresolved_sections=summary.unresolved,
        duplicate_anchor_collisions=summary.duplicate_anchor_collisions,
        source_order_violations=summary.source_order_violations,
        boundary_errors=boundary_errors,
        segments=segments,
        diagnostic_chunks=chunks,
        diagnostics={
            "fallback_behavior": _fallback_behavior(document.file_type),
            "chunks_crossing_boundaries": 0,
        },
    )


def build_mapping_result(
    document: CanonicalDocumentInput,
    structure: Optional[DocumentStructure],
    *,
    interpretation_failed: bool = False,
) -> DocumentStructureMappingResult:
    if interpretation_failed or structure is None:
        return _fallback_result(document, "model interpretation unavailable")
    anchored = anchor_structure(document, structure)
    segments = segment_document(document, anchored)
    chunks = diagnostic_chunks_from_segments(segments)
    return evaluate_mapping(document, structure, anchored, segments, chunks)


def build_section_path(
    section_id: Optional[str],
    sections_by_id: Dict[str, AnchoredSection],
) -> str:
    if not section_id or section_id not in sections_by_id:
        return ""
    path = []
    current = sections_by_id[section_id]
    guard = 0
    while current and guard < 16:
        path.append(current.source_title or current.semantic_title)
        current = sections_by_id.get(current.parent_id) if current.parent_id else None
        guard += 1
    return " > ".join(reversed([item for item in path if item]))


def diagnostic_chunks_from_segments(
    segments: List[CanonicalSegment],
) -> List[DiagnosticChunk]:
    chunks: List[DiagnosticChunk] = []
    for segment_index, segment in enumerate(segments, start=1):
        cleaned = production_compatible_clean_text(segment.text)
        if not cleaned:
            continue
        for chunk_index, chunk in enumerate(production_compatible_chunk_text(cleaned), start=1):
            chunk = production_compatible_clean_text(chunk)
            if len(chunk) <= 100:
                continue
            chunks.append(
                DiagnosticChunk(
                    text=chunk,
                    section_path=segment.section_path,
                    source_unit=f"{segment.source_start.unit_type}:{segment.source_start.unit_index}",
                    segment_index=segment_index,
                    chunk_index=chunk_index,
                    source_start=segment.source_start,
                    source_end=segment.source_end,
                )
            )
    return chunks


def production_compatible_clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\.(\w)", r". \1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def production_compatible_chunk_text(text: str, max_chars: int = 1000, overlap: int = 200) -> List[str]:
    paragraphs = re.split(r"\n\s*\n", text)
    chunks: List[str] = []
    current_chunk = ""
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(paragraph) > max_chars:
            start = 0
            while start < len(paragraph):
                chunks.append(paragraph[start:start + max_chars].strip())
                start += max_chars - overlap
            continue
        if len(current_chunk) + len(paragraph) < max_chars:
            current_chunk += "\n\n" + paragraph
        else:
            chunks.append(current_chunk.strip())
            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + paragraph
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def _find_anchor(document: CanonicalDocumentInput, section: DocumentSection) -> Optional[Anchor]:
    if not section.source_title:
        return None
    blocks = _candidate_blocks(document, section)
    for method in ("exact", "whitespace", "unicode", "punctuation"):
        for block in blocks:
            match = _match_in_block(section.source_title, block, method)
            if match:
                fidelity = "exact" if method == "exact" else "normalized"
                confidence = 1.0 if method == "exact" else 0.92
                return Anchor(
                    source_position=SourcePosition(
                        unit_type=block.source_position.unit_type,
                        unit_index=block.source_position.unit_index,
                        block_index=block.source_position.block_index,
                        start_offset=match[0],
                        end_offset=match[1],
                    ),
                    method=method,
                    confidence=confidence,
                    fidelity=fidelity,
                )
    for block in blocks:
        match = _fuzzy_line_match(section.source_title, block)
        if match:
            return Anchor(
                source_position=SourcePosition(
                    unit_type=block.source_position.unit_type,
                    unit_index=block.source_position.unit_index,
                    block_index=block.source_position.block_index,
                    start_offset=match[0],
                    end_offset=match[1],
                ),
                method="unit_constrained_fuzzy",
                confidence=match[2],
                fidelity="fuzzy",
            )
    return None


def _candidate_blocks(document: CanonicalDocumentInput, section: DocumentSection) -> List[CanonicalBlock]:
    if section.expected_start:
        exact = [
            block for block in document.blocks
            if block.source_position.unit_type == section.expected_start.unit_type
            and block.source_position.unit_index == section.expected_start.unit_index
        ]
        if exact:
            return exact
    return document.blocks


def _match_in_block(title: str, block: CanonicalBlock, method: str) -> Optional[Tuple[int, int]]:
    text = block.raw_text or block.text
    if method == "exact":
        offset = text.find(title)
        return (offset, offset + len(title)) if offset >= 0 else None
    normalized_text, text_map = _normalized_with_map(text, method)
    normalized_title, _ = _normalized_with_map(title, method)
    offset = normalized_text.find(normalized_title)
    if offset < 0 or not text_map:
        return None
    start = text_map[offset]
    end = text_map[min(offset + len(normalized_title) - 1, len(text_map) - 1)] + 1
    return start, end


def _fuzzy_line_match(title: str, block: CanonicalBlock) -> Optional[Tuple[int, int, float]]:
    best = None
    title_key = _compact_key(title)
    for start, end, line in _iter_nonempty_lines(block.raw_text or block.text):
        ratio = difflib.SequenceMatcher(None, title_key, _compact_key(line)).ratio()
        if ratio >= 0.86 and (best is None or ratio > best[2]):
            best = (start, end, ratio)
    return best


def _iter_nonempty_lines(text: str) -> Iterable[Tuple[int, int, str]]:
    offset = 0
    for raw_line in text.splitlines(keepends=True):
        line = raw_line.rstrip("\r\n")
        stripped = line.strip()
        start = offset + len(line) - len(line.lstrip())
        end = start + len(stripped)
        if stripped:
            yield start, end, stripped
        offset += len(raw_line)


def _normalized_with_map(text: str, method: str) -> Tuple[str, List[int]]:
    chars: List[str] = []
    mapping: List[int] = []
    previous_space = False
    for index, char in enumerate(text):
        transformed = char
        if method in {"unicode", "punctuation"}:
            transformed = unicodedata.normalize("NFKD", transformed)
            transformed = "".join(c for c in transformed if not unicodedata.combining(c))
        for inner in transformed:
            if method == "punctuation" and unicodedata.category(inner).startswith("P"):
                inner = " "
            if method in {"whitespace", "unicode", "punctuation"} and inner.isspace():
                if previous_space:
                    continue
                inner = " "
                previous_space = True
            else:
                previous_space = False
            chars.append(inner.casefold())
            mapping.append(index)
    normalized = "".join(chars).strip()
    leading = len("".join(chars)) - len("".join(chars).lstrip())
    if leading:
        mapping = mapping[leading:]
    return normalized, mapping[:len(normalized)]


def _compact_key(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    normalized = "".join(c for c in normalized if not unicodedata.combining(c))
    normalized = re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE)
    return re.sub(r"\s+", " ", normalized).strip().casefold()


def _make_segment(
    text: str,
    block: CanonicalBlock,
    start: int,
    end: int,
    active: Optional[AnchoredSection],
    sections_by_id: Dict[str, AnchoredSection],
    fallback_title: str,
) -> CanonicalSegment:
    source_start = SourcePosition(
        block.source_position.unit_type,
        block.source_position.unit_index,
        block.source_position.block_index,
        start,
        start,
    )
    source_end = SourcePosition(
        block.source_position.unit_type,
        block.source_position.unit_index,
        block.source_position.block_index,
        end,
        end,
    )
    path = build_section_path(active.section_id, sections_by_id) if active else fallback_title
    return CanonicalSegment(
        text=text,
        raw_text=text,
        source_start=source_start,
        source_end=source_end,
        section_id=active.section_id if active else None,
        section_title=(active.source_title or active.semantic_title) if active else fallback_title,
        section_path=path,
        section_level=active.level if active else None,
        parent_id=active.parent_id if active else None,
        provenance="anchored_section" if active else "fallback_area",
        mapping_confidence=active.anchor.confidence if active and active.anchor else 0.0,
    )


def _anchor_key(position: SourcePosition) -> Tuple[str, int, int, int, int]:
    return (
        position.unit_type,
        position.unit_index,
        position.block_index,
        position.start_offset,
        position.end_offset or position.start_offset,
    )


def _block_key(position: SourcePosition) -> str:
    return f"{position.unit_type}:{position.unit_index}:{position.block_index}"


def _count_boundary_errors(segments: List[CanonicalSegment]) -> int:
    return sum(
        1
        for segment in segments
        if segment.source_end < segment.source_start
    )


def _anchor_quality(summary: AnchorSummary, total: int) -> float:
    if total <= 0:
        return 0.0
    weighted = (
        summary.exact_count
        + (summary.normalized_count * 0.9)
        + (summary.fuzzy_count * 0.65)
    )
    return weighted / total


def _fallback_result(document: CanonicalDocumentInput, reason: str) -> DocumentStructureMappingResult:
    return DocumentStructureMappingResult(
        file_type=document.file_type,
        filename=document.filename,
        structure_status=STATUS_FALLBACK,
        mapping_status=STATUS_FALLBACK,
        structure_confidence="LOW",
        mapping_confidence=0.0,
        sections_total=0,
        sections_anchored=0,
        exact_count=0,
        normalized_count=0,
        fuzzy_count=0,
        model_only_count=0,
        unresolved_sections=0,
        duplicate_anchor_collisions=0,
        source_order_violations=0,
        boundary_errors=0,
        segments=[],
        diagnostic_chunks=[],
        diagnostics={
            "reason": reason,
            "fallback_behavior": _fallback_behavior(document.file_type),
        },
    )


def _fallback_behavior(file_type: str) -> str:
    return {
        "pdf": "current PDF page extraction and detect_section_title behavior",
        "docx": "current DOCX Heading-style section path behavior",
        "pptx": "current PPTX slide-title section behavior",
    }.get(file_type, "current production extraction behavior")
