"""DOCX adapter for the clean DocumentInterpreterResult contract.

This module is intentionally isolated from production upload.  It proves the
new interpreter contract on the safest source format: Word-native heading
styles and document flow order.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from canonical_document import (
    ROLE_BODY,
    ROLE_HEADING,
    ROLE_LIST,
    ROLE_TABLE,
    CanonicalBlock,
    CanonicalDocumentInput,
    adapt_docx_document,
)
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_DOCX,
    SourceSpan,
)
from document_interpreter_contract import (
    CONTENT_ROLE_BODY,
    CONTENT_ROLE_HEADING,
    CONTENT_ROLE_LIST,
    CONTENT_ROLE_TABLE,
    CONTENT_ROLE_UNKNOWN,
    DocumentInterpreterResult,
    InterpretedContentBlock,
    InterpretedDocumentSection,
    INTERPRETER_STATUS_ACCEPTED,
    INTERPRETER_STATUS_FALLBACK,
    INTERPRETER_STATUS_PARTIAL,
    validate_document_interpreter_result,
)


def interpret_docx_document(
    file_bytes: bytes,
    filename: str,
    *,
    document_id: str,
) -> DocumentInterpreterResult:
    """Interpret a DOCX file into source sections and owned content blocks."""

    canonical_document = adapt_docx_document(file_bytes, filename)
    return interpret_canonical_docx_document(
        canonical_document,
        document_id=document_id,
    )


def interpret_canonical_docx_document(
    document: CanonicalDocumentInput,
    *,
    document_id: str,
) -> DocumentInterpreterResult:
    """Interpret an already canonicalized DOCX document."""

    if document.file_type != SOURCE_FORMAT_DOCX:
        return DocumentInterpreterResult(
            document_id=document_id,
            source_format=document.file_type,
            document_title=document.title or document.filename,
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[
                _content_block(block, section_id=None)
                for block in document.blocks
                if (block.text or "").strip()
            ],
            diagnostics={
                "fallback_reason": "not_docx",
                "source_format": document.file_type,
            },
        )

    sections: List[InterpretedDocumentSection] = []
    unresolved_blocks: List[InterpretedContentBlock] = []
    pending_owned_blocks: Dict[str, List[InterpretedContentBlock]] = {}
    section_stack: Dict[int, InterpretedDocumentSection] = {}
    diagnostics = {
        "heading_count": 0,
        "content_blocks": 0,
        "unresolved_blocks": 0,
        "heading_level_gaps": 0,
        "heading_level_jumps": [],
    }
    active_section: Optional[InterpretedDocumentSection] = None

    for block in document.blocks:
        if not (block.text or "").strip():
            continue

        if block.role_hint == ROLE_HEADING and block.native_hierarchy_hint:
            level = max(1, int(block.native_hierarchy_hint))
            parent = _nearest_parent(section_stack, level)
            expected_parent = section_stack.get(level - 1) if level > 1 else None
            confidence = CONFIDENCE_HIGH
            if level > 1 and expected_parent is None:
                diagnostics["heading_level_gaps"] += 1
                diagnostics["heading_level_jumps"].append(
                    {
                        "block_id": block.block_id,
                        "title": block.text,
                        "level": level,
                        "attached_parent_id": parent.section_id if parent else None,
                    }
                )
                confidence = CONFIDENCE_MEDIUM

            section_id = f"docx-section-{len(sections) + 1:04d}"
            heading_block = _content_block(block, section_id=section_id)
            pending_owned_blocks[section_id] = [heading_block]
            section = InterpretedDocumentSection(
                section_id=section_id,
                title=block.text,
                logical_title=block.text,
                level=level,
                parent_id=parent.section_id if parent else None,
                source_order=block.source_order,
                source_span=_span_from_block(block),
                confidence=confidence,
                origin=ORIGIN_EXPLICIT_SOURCE,
                owned_blocks=pending_owned_blocks[section_id],
                metadata={
                    "style_hint": block.style_hint,
                    "native_hierarchy_hint": block.native_hierarchy_hint,
                    "block_id": block.block_id,
                },
            )
            sections.append(section)
            active_section = section
            section_stack[level] = section
            for stale_level in list(section_stack):
                if stale_level > level:
                    del section_stack[stale_level]
            diagnostics["heading_count"] += 1
            diagnostics["content_blocks"] += 1
            continue

        content_section_id = active_section.section_id if active_section else None
        content_block = _content_block(block, section_id=content_section_id)
        diagnostics["content_blocks"] += 1
        if active_section:
            pending_owned_blocks[active_section.section_id].append(content_block)
        else:
            unresolved_blocks.append(content_block)
            diagnostics["unresolved_blocks"] += 1

    if not sections:
        result = DocumentInterpreterResult(
            document_id=document_id,
            source_format=SOURCE_FORMAT_DOCX,
            document_title=document.title or document.filename,
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=unresolved_blocks,
            diagnostics={
                **diagnostics,
                "fallback_reason": "docx_contains_no_heading_styles",
            },
        )
        return _with_validation_diagnostics(result)

    status = (
        INTERPRETER_STATUS_PARTIAL
        if unresolved_blocks or diagnostics["heading_level_gaps"]
        else INTERPRETER_STATUS_ACCEPTED
    )
    confidence = CONFIDENCE_MEDIUM if status == INTERPRETER_STATUS_PARTIAL else CONFIDENCE_HIGH
    result = DocumentInterpreterResult(
        document_id=document_id,
        source_format=SOURCE_FORMAT_DOCX,
        document_title=document.title or document.filename,
        status=status,
        confidence=confidence,
        sections=sections,
        unresolved_blocks=unresolved_blocks,
        diagnostics=diagnostics,
    )
    return _with_validation_diagnostics(result)


def _nearest_parent(
    section_stack: Dict[int, InterpretedDocumentSection],
    level: int,
) -> Optional[InterpretedDocumentSection]:
    for candidate_level in range(level - 1, 0, -1):
        parent = section_stack.get(candidate_level)
        if parent is not None:
            return parent
    return None


def _content_block(
    block: CanonicalBlock,
    *,
    section_id: Optional[str],
) -> InterpretedContentBlock:
    return InterpretedContentBlock(
        block_id=block.block_id,
        text=block.text,
        raw_text=block.raw_text,
        source_order=block.source_order,
        source_span=_span_from_block(block),
        role=_content_role(block.role_hint),
        page=None,
        section_id=section_id,
        provenance=ORIGIN_EXPLICIT_SOURCE,
        confidence=CONFIDENCE_HIGH if section_id else CONFIDENCE_LOW,
        metadata={
            "style_hint": block.style_hint,
            "native_hierarchy_hint": block.native_hierarchy_hint,
            **block.metadata,
        },
    )


def _content_role(role_hint: str) -> str:
    if role_hint == ROLE_HEADING:
        return CONTENT_ROLE_HEADING
    if role_hint == ROLE_BODY:
        return CONTENT_ROLE_BODY
    if role_hint == ROLE_LIST:
        return CONTENT_ROLE_LIST
    if role_hint == ROLE_TABLE:
        return CONTENT_ROLE_TABLE
    return CONTENT_ROLE_UNKNOWN


def _span_from_block(block: CanonicalBlock) -> SourceSpan:
    return SourceSpan(
        start=block.source_position,
        end=block.source_position,
    )


def _with_validation_diagnostics(
    result: DocumentInterpreterResult,
) -> DocumentInterpreterResult:
    report = validate_document_interpreter_result(result)
    return DocumentInterpreterResult(
        document_id=result.document_id,
        source_format=result.source_format,
        document_title=result.document_title,
        status=result.status,
        confidence=result.confidence,
        sections=result.sections,
        unresolved_blocks=result.unresolved_blocks,
        diagnostics={
            **result.diagnostics,
            "contract_validation": {
                "is_valid": report.is_valid,
                "issues": [
                    {
                        "code": issue.code,
                        "message": issue.message,
                        "object_id": issue.object_id,
                    }
                    for issue in report.issues
                ],
            },
        },
    )
