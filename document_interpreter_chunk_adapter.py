"""Isolated adapter from interpreted document hierarchy to chunk-like units.

This module is a workbench prototype only.  It does not write to the database,
does not call embeddings, and is not wired into production ingestion.  Its
purpose is to prove that the clean DocumentInterpreterResult contract can be
converted into the same kind of section-aware text chunks DOUNO already uses
downstream.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from document_interpreter_contract import (
    DocumentInterpreterResult,
    InterpretedContentBlock,
    InterpretedDocumentSection,
    validate_document_interpreter_result,
)
from structure_segmenter import (
    production_compatible_chunk_text,
    production_compatible_clean_text,
)


@dataclass(frozen=True)
class InterpreterChunk:
    """Production-shaped chunk candidate derived from interpreted sections."""

    document_id: str
    document_title: str
    chunk_text: str
    page: Optional[int]
    section_id: Optional[str]
    section_title: str
    section_path: str
    segment_index: int
    chunk_index: int
    source_block_ids: Tuple[str, ...]
    source_order_start: int
    source_order_end: int
    unresolved: bool = False


def chunks_from_interpreted_document(
    result: DocumentInterpreterResult,
    *,
    clean_text: Callable[[str], str] = production_compatible_clean_text,
    chunk_text: Callable[[str], List[str]] = production_compatible_chunk_text,
    min_chars: int = 100,
) -> List[InterpreterChunk]:
    """Convert interpreted hierarchy into section-owned chunk candidates.

    Important invariants:
    - chunks never mix content from two interpreted sections;
    - unresolved/fallback content remains unowned by a section;
    - source chronology is preserved;
    - output remains close to the current ingestion chunk shape.
    """

    validation = validate_document_interpreter_result(result)
    if not validation.is_valid:
        issue_summary = ", ".join(issue.code for issue in validation.issues)
        raise ValueError(f"Invalid DocumentInterpreterResult: {issue_summary}")

    sections_by_id = result.sections_by_id
    chunks: List[InterpreterChunk] = []
    segment_index = 0

    for section in sorted(result.sections, key=lambda item: item.source_order):
        segment_index += 1
        chunks.extend(
            _chunks_from_blocks(
                result=result,
                blocks=section.owned_blocks,
                section=section,
                section_path=_section_path(section, sections_by_id),
                segment_index=segment_index,
                unresolved=False,
                clean_text=clean_text,
                chunk_text=chunk_text,
                min_chars=min_chars,
            )
        )

    if result.unresolved_blocks:
        segment_index += 1
        chunks.extend(
            _chunks_from_blocks(
                result=result,
                blocks=result.unresolved_blocks,
                section=None,
                section_path=result.document_title,
                segment_index=segment_index,
                unresolved=True,
                clean_text=clean_text,
                chunk_text=chunk_text,
                min_chars=min_chars,
            )
        )

    return sorted(
        chunks,
        key=lambda item: (
            item.source_order_start,
            item.segment_index,
            item.chunk_index,
        ),
    )


def _chunks_from_blocks(
    *,
    result: DocumentInterpreterResult,
    blocks: Sequence[InterpretedContentBlock],
    section: Optional[InterpretedDocumentSection],
    section_path: str,
    segment_index: int,
    unresolved: bool,
    clean_text: Callable[[str], str],
    chunk_text: Callable[[str], List[str]],
    min_chars: int,
) -> List[InterpreterChunk]:
    sorted_blocks = sorted(blocks, key=lambda item: item.source_order)
    cleaned_segment = clean_text(_join_block_text(sorted_blocks))
    if not cleaned_segment:
        return []

    source_block_ids = tuple(block.block_id for block in sorted_blocks)
    source_order_start = min(block.source_order for block in sorted_blocks)
    source_order_end = max(block.source_order for block in sorted_blocks)
    page = _first_page(sorted_blocks)
    section_title = section.display_title if section else result.document_title
    section_id = section.section_id if section else None

    chunk_candidates = [clean_text(chunk) for chunk in chunk_text(cleaned_segment)]
    output: List[InterpreterChunk] = []
    for chunk_index, candidate in enumerate(chunk_candidates, start=1):
        if len(candidate) <= min_chars:
            continue
        output.append(
            InterpreterChunk(
                document_id=result.document_id,
                document_title=result.document_title,
                chunk_text=candidate,
                page=page,
                section_id=section_id,
                section_title=section_title,
                section_path=section_path,
                segment_index=segment_index,
                chunk_index=chunk_index,
                source_block_ids=source_block_ids,
                source_order_start=source_order_start,
                source_order_end=source_order_end,
                unresolved=unresolved,
            )
        )
    return output


def _join_block_text(blocks: Iterable[InterpretedContentBlock]) -> str:
    return "\n\n".join(block.text for block in blocks if block.text.strip())


def _first_page(blocks: Sequence[InterpretedContentBlock]) -> Optional[int]:
    for block in blocks:
        if block.page is not None:
            return block.page
    return None


def _section_path(
    section: InterpretedDocumentSection,
    sections_by_id: Dict[str, InterpretedDocumentSection],
) -> str:
    path = []
    current: Optional[InterpretedDocumentSection] = section
    guard = 0
    while current is not None and guard < 32:
        path.append(current.display_title)
        current = sections_by_id.get(current.parent_id) if current.parent_id else None
        guard += 1
    return " > ".join(reversed([title for title in path if title]))
