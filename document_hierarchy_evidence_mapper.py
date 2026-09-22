"""Isolated mapping from canonical source blocks to structural evidence.

This module preserves evidence already present in CanonicalDocumentInput /
CanonicalUnit / CanonicalBlock.  It deliberately does not decide hierarchy,
detect numbering, infer visual headings, or call any model.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, List, Optional

from canonical_document import (
    ROLE_BODY,
    ROLE_HEADING,
    ROLE_LIST,
    ROLE_TABLE,
    ROLE_TITLE,
    CanonicalBlock,
    CanonicalDocumentInput,
    CanonicalUnit,
)
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONTENT_KIND_BODY,
    CONTENT_KIND_DOCUMENT_BOUNDARY,
    CONTENT_KIND_LIST_ITEM,
    CONTENT_KIND_TABLE,
    CONTENT_KIND_UNKNOWN,
    EVIDENCE_ROLE_BODY,
    EVIDENCE_ROLE_HEADING,
    EVIDENCE_ROLE_LIST,
    EVIDENCE_ROLE_TABLE,
    EVIDENCE_ROLE_TITLE,
    EVIDENCE_ROLE_UNKNOWN,
    SourceSpan,
    StructuralEvidence,
)


def map_document_to_structural_evidence(
    document: CanonicalDocumentInput,
) -> List[StructuralEvidence]:
    """Map every canonical block in source order to StructuralEvidence."""

    units_by_key = {
        (unit.unit_type, unit.unit_index): unit
        for unit in document.units
    }
    evidence: List[StructuralEvidence] = []
    for block in document.blocks:
        unit = units_by_key.get((block.source_position.unit_type, block.source_position.unit_index))
        evidence.append(map_block_to_structural_evidence(document, block, unit=unit))
    return evidence


def map_block_to_structural_evidence(
    document: CanonicalDocumentInput,
    block: CanonicalBlock,
    *,
    unit: Optional[CanonicalUnit] = None,
) -> StructuralEvidence:
    """Transfer canonical block evidence without interpreting hierarchy."""

    native_evidence = _native_evidence(block)
    visual_evidence = _visual_evidence(block)
    numbering_evidence = _numbering_evidence(block)
    structural_context = _structural_context(block, unit)
    metadata = _residual_metadata(block)
    source_span = SourceSpan(
        start=block.source_position,
        end=replace(
            block.source_position,
            start_offset=block.source_position.end_offset
            if block.source_position.end_offset is not None
            else block.source_position.start_offset + len(block.raw_text or block.text or ""),
        ),
    )
    return StructuralEvidence(
        evidence_id=f"evidence-{block.block_id}",
        source_format=document.file_type,
        source_order=block.source_order,
        source_span=source_span,
        raw_text=block.raw_text,
        normalized_text=block.text,
        evidence_role=_evidence_role(block.role_hint),
        content_kind=_content_kind(block.role_hint),
        native_evidence=native_evidence,
        visual_evidence=visual_evidence,
        numbering_evidence=numbering_evidence,
        structural_context=structural_context,
        confidence_hint=_confidence_hint(block),
        metadata=metadata,
    )


def _evidence_role(role_hint: str) -> str:
    if role_hint == ROLE_TITLE:
        return EVIDENCE_ROLE_TITLE
    if role_hint == ROLE_HEADING:
        return EVIDENCE_ROLE_HEADING
    if role_hint == ROLE_LIST:
        return EVIDENCE_ROLE_LIST
    if role_hint == ROLE_TABLE:
        return EVIDENCE_ROLE_TABLE
    if role_hint == ROLE_BODY:
        return EVIDENCE_ROLE_BODY
    return EVIDENCE_ROLE_UNKNOWN


def _content_kind(role_hint: str) -> str:
    if role_hint in {ROLE_TITLE, ROLE_HEADING}:
        return CONTENT_KIND_DOCUMENT_BOUNDARY
    if role_hint == ROLE_LIST:
        return CONTENT_KIND_LIST_ITEM
    if role_hint == ROLE_TABLE:
        return CONTENT_KIND_TABLE
    if role_hint == ROLE_BODY:
        return CONTENT_KIND_BODY
    return CONTENT_KIND_UNKNOWN


def _native_evidence(block: CanonicalBlock) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {
        "role_hint": block.role_hint,
    }
    if block.style_hint:
        evidence["style_hint"] = block.style_hint
    if block.native_hierarchy_hint is not None:
        evidence["native_hierarchy_hint"] = block.native_hierarchy_hint

    for key in (
        "paragraph_index",
        "body_item_index",
        "shape_index",
        "shape_name",
        "placeholder_type",
        "placeholder_idx",
        "bullet_level",
        "layout_name",
        "layout_index",
        "page_index",
        "pdf_text_unit",
        "font_subtype",
        "text_matrix",
        "current_transformation_matrix",
    ):
        if key in block.metadata:
            evidence[key] = block.metadata[key]
    return evidence


def _visual_evidence(block: CanonicalBlock) -> Dict[str, Any]:
    visual_keys = (
        "font_size",
        "font_weight",
        "font_name",
        "x",
        "y",
        "x0",
        "y0",
        "x1",
        "y1",
        "left",
        "top",
        "width",
        "height",
        "page_width",
        "page_height",
        "bbox",
        "prominence",
    )
    return {
        key: block.metadata[key]
        for key in visual_keys
        if key in block.metadata
    }


def _numbering_evidence(block: CanonicalBlock) -> Dict[str, Any]:
    evidence: Dict[str, Any] = {}
    numbering = block.metadata.get("numbering")
    if isinstance(numbering, dict):
        evidence.update(numbering)
    elif numbering is not None:
        evidence["numbering"] = numbering
    for key in ("numbering_pattern", "list_level"):
        if key in block.metadata:
            evidence[key] = block.metadata[key]
    if block.role_hint == ROLE_LIST and block.native_hierarchy_hint is not None:
        evidence.setdefault("list_level", block.native_hierarchy_hint)
    return evidence


def _structural_context(
    block: CanonicalBlock,
    unit: Optional[CanonicalUnit],
) -> Dict[str, Any]:
    context: Dict[str, Any] = {
        "block_id": block.block_id,
        "unit_type": block.source_position.unit_type,
        "unit_index": block.source_position.unit_index,
        "block_index": block.source_position.block_index,
    }
    if unit is not None:
        context["unit_id"] = unit.unit_id
        if unit.display_label:
            context["unit_display_label"] = unit.display_label
    return context


def _confidence_hint(block: CanonicalBlock) -> str:
    if block.role_hint in {ROLE_TITLE, ROLE_HEADING} and block.native_hierarchy_hint is not None:
        return CONFIDENCE_HIGH
    if block.role_hint in {ROLE_TITLE, ROLE_HEADING, ROLE_LIST, ROLE_TABLE}:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_LOW


def _residual_metadata(block: CanonicalBlock) -> Dict[str, Any]:
    mapped_keys = {
        "paragraph_index",
        "body_item_index",
        "shape_index",
        "shape_name",
        "placeholder_type",
        "placeholder_idx",
        "bullet_level",
        "layout_name",
        "layout_index",
        "page_index",
        "pdf_text_unit",
        "font_subtype",
        "text_matrix",
        "current_transformation_matrix",
        "font_size",
        "font_weight",
        "font_name",
        "x",
        "y",
        "x0",
        "y0",
        "x1",
        "y1",
        "left",
        "top",
        "width",
        "height",
        "page_width",
        "page_height",
        "bbox",
        "prominence",
        "numbering",
        "numbering_pattern",
        "list_level",
    }
    return {
        key: value
        for key, value in block.metadata.items()
        if key not in mapped_keys
    }
