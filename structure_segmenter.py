"""Canonical section segmentation prototype for extracted documents."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from document_extractors import ExtractedDocument
from structure_anchor import AnchoredDocumentStructure, AnchoredSection


@dataclass(frozen=True)
class CanonicalTextSegment:
    text: str
    raw_text: str
    page: Optional[int]
    block_index: int
    section_id: Optional[str]
    section_title: str
    section_level: Optional[int]
    parent_id: Optional[str]
    source_start_offset: Optional[int]
    source_end_offset: Optional[int]
    anchor_confidence: Optional[float]
    provenance: str


@dataclass(frozen=True)
class DiagnosticChunk:
    text: str
    page: Optional[int]
    block_index: int
    section_id: Optional[str]
    section_title: str
    segment_index: int
    chunk_index: int


def production_compatible_clean_text(text: str) -> str:
    """Mirror the upload clean_text helper without importing the FastAPI app."""
    if not text:
        return ""
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = re.sub(r"([a-zA-Z])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zA-Z])", r"\1 \2", text)
    text = re.sub(r"\.(\w)", r". \1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def production_compatible_chunk_text(
    text: str,
    max_chars: int = 1000,
    overlap: int = 200,
) -> List[str]:
    """Mirror the current upload chunk_text algorithm for isolated diagnostics."""
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
                end = start + max_chars
                chunks.append(paragraph[start:end].strip())
                start += max_chars - overlap
            continue
        if len(current_chunk) + len(paragraph) < max_chars:
            current_chunk += "\n\n" + paragraph
        else:
            chunks.append(current_chunk.strip())
            overlap_text = (
                current_chunk[-overlap:]
                if len(current_chunk) > overlap
                else current_chunk
            )
            current_chunk = overlap_text + "\n\n" + paragraph

    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks


def segment_extracted_document(
    extracted_document: ExtractedDocument,
    anchored_structure: AnchoredDocumentStructure,
    *,
    fallback_section_title: Optional[str] = None,
) -> List[CanonicalTextSegment]:
    fallback_title = (
        fallback_section_title
        or extracted_document.filename
        or "GENERAL"
    )
    anchors_by_block: Dict[int, List[AnchoredSection]] = {}
    sections_by_id = {section.section_id: section for section in anchored_structure.sections}

    for section in anchored_structure.sections:
        if not section.anchor:
            continue
        if section.anchor.block is None or section.anchor.start_offset is None:
            continue
        anchors_by_block.setdefault(section.anchor.block, []).append(section)

    for sections in anchors_by_block.values():
        sections.sort(key=lambda section: (section.anchor.start_offset, section.source_order))

    active_section: Optional[AnchoredSection] = None
    segments: List[CanonicalTextSegment] = []

    for block_position, block in enumerate(extracted_document.blocks, start=1):
        block_index = block.block_index or block_position
        raw_text = block.raw_text or block.text or ""
        block_anchors = anchors_by_block.get(block_index, [])
        cursor = 0

        for section in block_anchors:
            anchor = section.anchor
            assert anchor is not None
            start = max(0, min(anchor.start_offset or 0, len(raw_text)))
            if start > cursor:
                segments.append(
                    _make_segment(
                        raw_text=raw_text[cursor:start],
                        page=block.page,
                        block_index=block_index,
                        start_offset=cursor,
                        end_offset=start,
                        active_section=active_section,
                        fallback_title=fallback_title,
                        sections_by_id=sections_by_id,
                        provenance="fallback_before_first_heading"
                        if active_section is None
                        else "continued_section",
                    )
                )
            active_section = section
            cursor = start

        if cursor < len(raw_text):
            segments.append(
                _make_segment(
                    raw_text=raw_text[cursor:],
                    page=block.page,
                    block_index=block_index,
                    start_offset=cursor,
                    end_offset=len(raw_text),
                    active_section=active_section,
                    fallback_title=fallback_title,
                    sections_by_id=sections_by_id,
                    provenance="anchored_section"
                    if active_section is not None
                    else "fallback_before_first_heading",
                )
            )

    return [segment for segment in segments if segment.raw_text.strip()]


def diagnostic_chunks_from_segments(
    segments: List[CanonicalTextSegment],
    *,
    clean_text: Callable[[str], str] = production_compatible_clean_text,
    chunk_text: Callable[[str], List[str]] = production_compatible_chunk_text,
) -> List[DiagnosticChunk]:
    diagnostic_chunks: List[DiagnosticChunk] = []

    for segment_index, segment in enumerate(segments, start=1):
        cleaned = clean_text(segment.text)
        if not cleaned:
            continue
        chunks = [clean_text(chunk) for chunk in chunk_text(cleaned)]
        chunks = [chunk for chunk in chunks if len(chunk) > 100]
        for chunk_index, chunk in enumerate(chunks, start=1):
            diagnostic_chunks.append(
                DiagnosticChunk(
                    text=chunk,
                    page=segment.page,
                    block_index=segment.block_index,
                    section_id=segment.section_id,
                    section_title=segment.section_title,
                    segment_index=segment_index,
                    chunk_index=chunk_index,
                )
            )

    return diagnostic_chunks


def _make_segment(
    *,
    raw_text: str,
    page: Optional[int],
    block_index: int,
    start_offset: int,
    end_offset: int,
    active_section: Optional[AnchoredSection],
    fallback_title: str,
    sections_by_id: Dict[str, AnchoredSection],
    provenance: str,
) -> CanonicalTextSegment:
    section_title = (
        _canonical_section_path(active_section, sections_by_id)
        if active_section
        else fallback_title
    )
    return CanonicalTextSegment(
        text=raw_text,
        raw_text=raw_text,
        page=page,
        block_index=block_index,
        section_id=active_section.section_id if active_section else None,
        section_title=section_title,
        section_level=active_section.level if active_section else None,
        parent_id=active_section.parent_id if active_section else None,
        source_start_offset=start_offset,
        source_end_offset=end_offset,
        anchor_confidence=active_section.anchor.confidence
        if active_section and active_section.anchor
        else None,
        provenance=provenance,
    )


def _canonical_section_path(
    section: AnchoredSection,
    sections_by_id: Dict[str, AnchoredSection],
) -> str:
    path = []
    current: Optional[AnchoredSection] = section
    guard = 0
    while current is not None and guard < 16:
        path.append(current.source_title or current.semantic_title)
        current = sections_by_id.get(current.parent_id) if current.parent_id else None
        guard += 1
    return " > ".join(reversed([part for part in path if part]))
