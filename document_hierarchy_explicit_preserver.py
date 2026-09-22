"""Deterministic explicit-source hierarchy preservation.

This module starts after the evidence boundary.  It consumes StructuralEvidence
only and preserves the smallest trusted hierarchy currently supported:
explicit native DOCX Heading 1 / Heading 2 / Heading 3 evidence.
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    EVIDENCE_ROLE_HEADING,
    EvidenceProvenance,
    CanonicalDocumentNode,
    CanonicalDocumentStructure,
    NODE_KIND_DOCUMENT_SECTION,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_DOCX,
    STATUS_ACCEPTED,
    STATUS_PARTIAL,
    STATUS_UNRESOLVED,
    StructuralEvidence,
)


SUPPORTED_DOCX_HEADING_LEVELS = {1, 2, 3}


def preserve_explicit_docx_hierarchy(
    evidence: Iterable[StructuralEvidence],
    *,
    document_id: str,
    document_title: str = "",
) -> CanonicalDocumentStructure:
    """Preserve explicit DOCX Heading 1-3 evidence as canonical nodes.

    The function does not read DOCX files, inspect canonical blocks, parse
    numbering text, use Word numbering metadata, or infer semantic structure.
    """

    evidence_items = list(evidence)
    explicit_headings = [
        item
        for item in sorted(evidence_items, key=lambda item: item.source_order)
        if _is_supported_docx_heading(item)
    ]

    nodes: List[CanonicalDocumentNode] = []
    stack: Dict[int, CanonicalDocumentNode] = {}
    unresolved_parent_count = 0

    for index, item in enumerate(explicit_headings, start=1):
        level = _heading_level(item)
        parent = stack.get(level - 1) if level and level > 1 else None
        parent_id = parent.node_id if parent is not None else None
        parent_missing = bool(level and level > 1 and parent is None)
        if parent_missing:
            unresolved_parent_count += 1

        node = CanonicalDocumentNode(
            node_id=f"docx-heading-{index:04d}",
            source_title=item.normalized_text,
            logical_title=item.normalized_text,
            level=level,
            parent_id=parent_id,
            source_order=item.source_order,
            source_span=item.source_span,
            origin=ORIGIN_EXPLICIT_SOURCE,
            confidence=CONFIDENCE_HIGH,
            status=STATUS_UNRESOLVED if parent_missing else STATUS_ACCEPTED,
            evidence_ids=[item.evidence_id],
            provenance=EvidenceProvenance(
                supporting_evidence_ids=[item.evidence_id],
                decision_source=ORIGIN_EXPLICIT_SOURCE,
                repair_reason="missing_explicit_parent" if parent_missing else None,
            ),
            node_kind=NODE_KIND_DOCUMENT_SECTION,
        )
        nodes.append(node)

        if level is not None:
            stack[level] = node
            for stale_level in list(stack):
                if stale_level > level:
                    del stack[stale_level]

    ignored_non_heading_count = len(evidence_items) - len(explicit_headings)
    structure_status = _structure_status(nodes, unresolved_parent_count)
    structure_confidence = _structure_confidence(nodes, unresolved_parent_count)

    return CanonicalDocumentStructure(
        document_id=document_id,
        source_format=SOURCE_FORMAT_DOCX,
        document_title=document_title,
        structure_status=structure_status,
        structure_confidence=structure_confidence,
        nodes=nodes,
        diagnostics={
            "input_evidence_count": len(evidence_items),
            "explicit_heading_count": len(explicit_headings),
            "accepted_node_count": sum(1 for node in nodes if node.status == STATUS_ACCEPTED),
            "unresolved_parent_count": unresolved_parent_count,
            "ignored_non_heading_count": ignored_non_heading_count,
        },
    )


def render_explicit_hierarchy(structure: CanonicalDocumentStructure) -> str:
    """Render a tiny deterministic diagnostic hierarchy for tests/debugging."""

    by_id = {node.node_id: node for node in structure.nodes}
    lines: List[str] = []
    for node in sorted(structure.nodes, key=lambda item: item.source_order):
        label = "accepted" if node.status == STATUS_ACCEPTED else "unresolved-parent"
        indent = _render_indent(node, by_id)
        lines.append(f"{indent}[{label}] L{node.level} {node.source_title}")
    return "\n".join(lines)


def _is_supported_docx_heading(item: StructuralEvidence) -> bool:
    return (
        item.source_format == SOURCE_FORMAT_DOCX
        and item.evidence_role == EVIDENCE_ROLE_HEADING
        and _heading_level(item) in SUPPORTED_DOCX_HEADING_LEVELS
    )


def _heading_level(item: StructuralEvidence) -> Optional[int]:
    value = item.native_evidence.get("native_hierarchy_hint")
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _structure_status(
    nodes: List[CanonicalDocumentNode],
    unresolved_parent_count: int,
) -> str:
    if not nodes:
        return STATUS_UNRESOLVED
    if unresolved_parent_count:
        return STATUS_PARTIAL
    return STATUS_ACCEPTED


def _structure_confidence(
    nodes: List[CanonicalDocumentNode],
    unresolved_parent_count: int,
) -> str:
    if not nodes:
        return CONFIDENCE_LOW
    if unresolved_parent_count:
        return CONFIDENCE_MEDIUM
    return CONFIDENCE_HIGH


def _render_indent(
    node: CanonicalDocumentNode,
    by_id: Dict[str, CanonicalDocumentNode],
) -> str:
    depth = 0
    parent_id = node.parent_id
    while parent_id and parent_id in by_id:
        depth += 1
        parent_id = by_id[parent_id].parent_id
    return "  " * depth
