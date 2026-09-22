"""PPTX adapter for the clean DocumentInterpreterResult contract.

This isolated adapter preserves slide boundaries and title/body ownership
without changing production upload.  It intentionally does not infer a deeper
course hierarchy from slide order yet; each slide is treated as a real source
unit whose content must stay together.
"""

from __future__ import annotations

from typing import List, Optional

from canonical_document import (
    ROLE_BODY,
    ROLE_LIST,
    ROLE_TABLE,
    ROLE_TITLE,
    CanonicalBlock,
    CanonicalDocumentInput,
    CanonicalUnit,
    adapt_pptx_document,
)
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    ORIGIN_DETERMINISTIC_REPAIR,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_PPTX,
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


def interpret_pptx_document(
    file_bytes: bytes,
    filename: str,
    *,
    document_id: str,
) -> DocumentInterpreterResult:
    """Interpret a PPTX file into slide-owned document sections."""

    canonical_document = adapt_pptx_document(file_bytes, filename)
    return interpret_canonical_pptx_document(
        canonical_document,
        document_id=document_id,
    )


def interpret_canonical_pptx_document(
    document: CanonicalDocumentInput,
    *,
    document_id: str,
) -> DocumentInterpreterResult:
    """Interpret an already canonicalized PPTX document."""

    if document.file_type != SOURCE_FORMAT_PPTX:
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
                "fallback_reason": "not_pptx",
                "source_format": document.file_type,
            },
        )

    sections: List[InterpretedDocumentSection] = []
    diagnostics = {
        "slide_count": len(document.units),
        "title_slide_count": 0,
        "titleless_slide_count": 0,
        "content_blocks": 0,
        "titleless_slide_ids": [],
    }

    for unit in sorted(document.units, key=lambda item: item.unit_index):
        blocks = [block for block in unit.blocks if (block.text or "").strip()]
        if not blocks:
            continue

        title_block = _first_title_block(blocks)
        title = (
            title_block.text
            if title_block is not None
            else unit.display_label
            or f"Slide {unit.unit_index}"
        )
        has_explicit_title = title_block is not None
        confidence = CONFIDENCE_HIGH if has_explicit_title else CONFIDENCE_MEDIUM
        origin = ORIGIN_EXPLICIT_SOURCE if has_explicit_title else ORIGIN_DETERMINISTIC_REPAIR
        section_id = f"pptx-slide-{unit.unit_index:04d}"
        owned_blocks = [
            _content_block(block, section_id=section_id)
            for block in blocks
        ]
        section = InterpretedDocumentSection(
            section_id=section_id,
            title=title,
            logical_title=title,
            level=1,
            parent_id=None,
            source_order=blocks[0].source_order,
            source_span=_span_from_unit(unit, blocks),
            confidence=confidence,
            origin=origin,
            owned_blocks=owned_blocks,
            metadata={
                **unit.metadata,
                "unit_id": unit.unit_id,
                "unit_type": unit.unit_type,
                "unit_index": unit.unit_index,
                "display_label": unit.display_label,
                "has_explicit_title": has_explicit_title,
                "title_block_id": title_block.block_id if title_block else None,
            },
        )
        sections.append(section)
        diagnostics["content_blocks"] += len(owned_blocks)
        if has_explicit_title:
            diagnostics["title_slide_count"] += 1
        else:
            diagnostics["titleless_slide_count"] += 1
            diagnostics["titleless_slide_ids"].append(unit.unit_id)

    if not sections:
        result = DocumentInterpreterResult(
            document_id=document_id,
            source_format=SOURCE_FORMAT_PPTX,
            document_title=document.title or document.filename,
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            diagnostics={
                **diagnostics,
                "fallback_reason": "pptx_contains_no_extractable_slide_content",
            },
        )
        return _with_validation_diagnostics(result)

    status = (
        INTERPRETER_STATUS_PARTIAL
        if diagnostics["titleless_slide_count"]
        else INTERPRETER_STATUS_ACCEPTED
    )
    result = DocumentInterpreterResult(
        document_id=document_id,
        source_format=SOURCE_FORMAT_PPTX,
        document_title=document.title or document.filename,
        status=status,
        confidence=CONFIDENCE_MEDIUM if status == INTERPRETER_STATUS_PARTIAL else CONFIDENCE_HIGH,
        sections=sections,
        diagnostics=diagnostics,
    )
    return _with_validation_diagnostics(result)


def _first_title_block(blocks: List[CanonicalBlock]) -> Optional[CanonicalBlock]:
    return next((block for block in blocks if block.role_hint == ROLE_TITLE), None)


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
        page=block.source_position.unit_index if block.source_position.unit_type == "slide" else None,
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
    if role_hint == ROLE_TITLE:
        return CONTENT_ROLE_HEADING
    if role_hint == ROLE_BODY:
        return CONTENT_ROLE_BODY
    if role_hint == ROLE_LIST:
        return CONTENT_ROLE_LIST
    if role_hint == ROLE_TABLE:
        return CONTENT_ROLE_TABLE
    return CONTENT_ROLE_UNKNOWN


def _span_from_unit(
    unit: CanonicalUnit,
    blocks: List[CanonicalBlock],
) -> SourceSpan:
    return SourceSpan(
        start=blocks[0].source_position,
        end=blocks[-1].source_position,
    )


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
