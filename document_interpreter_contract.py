"""Clean document interpretation contract for future production ingestion.

This module defines the target boundary between upload/extraction and DOUNO's
existing taxonomy/chunk pipeline.  It is intentionally isolated: importing it
does not change upload behavior, database writes, taxonomy generation, or any
student-facing feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from canonical_document import SourcePosition
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ORIGIN_DETERMINISTIC_REPAIR,
    ORIGIN_EXPLICIT_SOURCE,
    ORIGIN_MODEL_REPAIR,
    ORIGIN_UNRESOLVED,
    SOURCE_FORMAT_DOCX,
    SOURCE_FORMAT_PDF,
    SOURCE_FORMAT_PPTX,
    SourceSpan,
)


INTERPRETER_STATUS_ACCEPTED = "accepted"
INTERPRETER_STATUS_PARTIAL = "partial"
INTERPRETER_STATUS_FALLBACK = "fallback"

CONTENT_ROLE_HEADING = "heading"
CONTENT_ROLE_BODY = "body"
CONTENT_ROLE_LIST = "list"
CONTENT_ROLE_TABLE = "table"
CONTENT_ROLE_LOCAL_STRUCTURE = "local_structure"
CONTENT_ROLE_UNKNOWN = "unknown"

SUPPORTED_SOURCE_FORMATS = {
    SOURCE_FORMAT_PDF,
    SOURCE_FORMAT_DOCX,
    SOURCE_FORMAT_PPTX,
}

VALID_CONFIDENCES = {
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
}

VALID_ORIGINS = {
    ORIGIN_EXPLICIT_SOURCE,
    ORIGIN_DETERMINISTIC_REPAIR,
    ORIGIN_MODEL_REPAIR,
    ORIGIN_UNRESOLVED,
}


@dataclass(frozen=True)
class InterpretedContentBlock:
    """A source-ordered text block owned by a document section or fallback area."""

    block_id: str
    text: str
    raw_text: str
    source_order: int
    source_span: SourceSpan
    role: str = CONTENT_ROLE_BODY
    page: Optional[int] = None
    section_id: Optional[str] = None
    provenance: str = ORIGIN_EXPLICIT_SOURCE
    confidence: str = CONFIDENCE_HIGH
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InterpretedDocumentSection:
    """A source hierarchy node with the content blocks it owns."""

    section_id: str
    title: str
    level: int
    source_order: int
    source_span: SourceSpan
    parent_id: Optional[str] = None
    logical_title: Optional[str] = None
    confidence: str = CONFIDENCE_HIGH
    origin: str = ORIGIN_EXPLICIT_SOURCE
    owned_blocks: List[InterpretedContentBlock] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def display_title(self) -> str:
        return self.logical_title or self.title


@dataclass(frozen=True)
class DocumentInterpreterResult:
    """The future ingestion handoff: hierarchy plus section-owned content."""

    document_id: str
    source_format: str
    document_title: str
    status: str
    confidence: str
    sections: List[InterpretedDocumentSection] = field(default_factory=list)
    unresolved_blocks: List[InterpretedContentBlock] = field(default_factory=list)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_authoritative_structure(self) -> bool:
        return (
            self.status == INTERPRETER_STATUS_ACCEPTED
            and self.confidence == CONFIDENCE_HIGH
            and bool(self.sections)
        )

    @property
    def all_content_blocks(self) -> List[InterpretedContentBlock]:
        blocks = [
            block
            for section in self.sections
            for block in section.owned_blocks
        ]
        blocks.extend(self.unresolved_blocks)
        return sorted(blocks, key=lambda block: block.source_order)

    @property
    def sections_by_id(self) -> Dict[str, InterpretedDocumentSection]:
        return {section.section_id: section for section in self.sections}


@dataclass(frozen=True)
class DocumentInterpreterValidationIssue:
    code: str
    message: str
    object_id: Optional[str] = None


@dataclass(frozen=True)
class DocumentInterpreterValidationReport:
    issues: List[DocumentInterpreterValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def validate_document_interpreter_result(
    result: DocumentInterpreterResult,
) -> DocumentInterpreterValidationReport:
    """Validate the contract without repairing or changing interpretation."""

    issues: List[DocumentInterpreterValidationIssue] = []

    if result.source_format not in SUPPORTED_SOURCE_FORMATS:
        issues.append(
            DocumentInterpreterValidationIssue(
                "unsupported_source_format",
                f"Unsupported source format: {result.source_format}",
            )
        )

    if result.status not in {
        INTERPRETER_STATUS_ACCEPTED,
        INTERPRETER_STATUS_PARTIAL,
        INTERPRETER_STATUS_FALLBACK,
    }:
        issues.append(
            DocumentInterpreterValidationIssue(
                "invalid_status",
                f"Invalid interpreter status: {result.status}",
            )
        )

    if result.confidence not in VALID_CONFIDENCES:
        issues.append(
            DocumentInterpreterValidationIssue(
                "invalid_confidence",
                f"Invalid interpreter confidence: {result.confidence}",
            )
        )

    sections_by_id = result.sections_by_id
    previous_section_order: Optional[int] = None
    seen_section_ids: set[str] = set()
    seen_block_ids: Dict[str, str] = {}

    for section in result.sections:
        if section.section_id in seen_section_ids:
            issues.append(
                DocumentInterpreterValidationIssue(
                    "duplicate_section_id",
                    "Section IDs must be unique.",
                    section.section_id,
                )
            )
        seen_section_ids.add(section.section_id)

        if not section.title.strip():
            issues.append(
                DocumentInterpreterValidationIssue(
                    "empty_section_title",
                    "Section title cannot be empty.",
                    section.section_id,
                )
            )

        if section.level < 1:
            issues.append(
                DocumentInterpreterValidationIssue(
                    "invalid_section_level",
                    "Section level must be >= 1.",
                    section.section_id,
                )
            )

        if section.confidence not in VALID_CONFIDENCES:
            issues.append(
                DocumentInterpreterValidationIssue(
                    "invalid_section_confidence",
                    f"Invalid section confidence: {section.confidence}",
                    section.section_id,
                )
            )

        if section.origin not in VALID_ORIGINS:
            issues.append(
                DocumentInterpreterValidationIssue(
                    "invalid_section_origin",
                    f"Invalid section origin: {section.origin}",
                    section.section_id,
                )
            )

        if (
            previous_section_order is not None
            and section.source_order < previous_section_order
        ):
            issues.append(
                DocumentInterpreterValidationIssue(
                    "section_source_order_decrease",
                    "Sections must be ordered by source chronology.",
                    section.section_id,
                )
            )
        previous_section_order = section.source_order

        if section.parent_id:
            parent = sections_by_id.get(section.parent_id)
            if parent is None:
                issues.append(
                    DocumentInterpreterValidationIssue(
                        "missing_parent_section",
                        "Parent section must exist.",
                        section.section_id,
                    )
                )
            elif parent.source_order >= section.source_order:
                issues.append(
                    DocumentInterpreterValidationIssue(
                        "parent_not_before_child",
                        "Parent section must precede child section.",
                        section.section_id,
                    )
                )
            elif parent.level >= section.level:
                issues.append(
                    DocumentInterpreterValidationIssue(
                        "parent_level_not_above_child",
                        "Parent level must be above child level.",
                        section.section_id,
                    )
                )

        for block in section.owned_blocks:
            _validate_block(
                block,
                issues,
                seen_block_ids,
                owner_id=section.section_id,
                require_section_id=section.section_id,
            )

    for block in result.unresolved_blocks:
        _validate_block(
            block,
            issues,
            seen_block_ids,
            owner_id="unresolved",
            require_section_id=None,
        )

    if result.status == INTERPRETER_STATUS_ACCEPTED and not result.sections:
        issues.append(
            DocumentInterpreterValidationIssue(
                "accepted_without_sections",
                "Accepted interpretation must contain at least one section.",
            )
        )

    if (
        result.status == INTERPRETER_STATUS_FALLBACK
        and result.sections
        and result.confidence == CONFIDENCE_HIGH
    ):
        issues.append(
            DocumentInterpreterValidationIssue(
                "fallback_with_high_confidence_sections",
                "Fallback interpretation should not claim high-confidence structure.",
            )
        )

    return DocumentInterpreterValidationReport(issues=issues)


def _validate_block(
    block: InterpretedContentBlock,
    issues: List[DocumentInterpreterValidationIssue],
    seen_block_ids: Dict[str, str],
    *,
    owner_id: str,
    require_section_id: Optional[str],
) -> None:
    if block.block_id in seen_block_ids:
        issues.append(
            DocumentInterpreterValidationIssue(
                "duplicate_content_block_id",
                (
                    "Content block is owned more than once "
                    f"({seen_block_ids[block.block_id]} and {owner_id})."
                ),
                block.block_id,
            )
        )
    seen_block_ids[block.block_id] = owner_id

    if not block.text.strip():
        issues.append(
            DocumentInterpreterValidationIssue(
                "empty_content_block",
                "Content block text cannot be empty.",
                block.block_id,
            )
        )

    if require_section_id is not None and block.section_id != require_section_id:
        issues.append(
            DocumentInterpreterValidationIssue(
                "owned_block_section_mismatch",
                "Owned block section_id must match its owning section.",
                block.block_id,
            )
        )

    if require_section_id is None and block.section_id is not None:
        issues.append(
            DocumentInterpreterValidationIssue(
                "unresolved_block_has_section",
                "Unresolved fallback blocks must not claim section ownership.",
                block.block_id,
            )
        )

    if block.confidence not in VALID_CONFIDENCES:
        issues.append(
            DocumentInterpreterValidationIssue(
                "invalid_block_confidence",
                f"Invalid block confidence: {block.confidence}",
                block.block_id,
            )
        )


def source_span(
    unit_type: str,
    unit_index: int,
    block_index: int,
    *,
    start_offset: int = 0,
    end_offset: Optional[int] = None,
) -> SourceSpan:
    """Convenience constructor for tests and future interpreters."""

    start = SourcePosition(
        unit_type=unit_type,
        unit_index=unit_index,
        block_index=block_index,
        start_offset=start_offset,
        end_offset=end_offset,
    )
    return SourceSpan(start=start, end=start)
