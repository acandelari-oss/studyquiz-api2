"""Deterministic cross-format document hierarchy preservation.

This isolated layer consumes StructuralEvidence and returns a
CanonicalDocumentStructure for DOCX, PPTX, and digital/text PDF.  It is not
connected to production ingestion, chunking, taxonomy, or downstream learning
features.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence

from document_hierarchy_consistency import analyze_docx_heading_numbering_consistency
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONTENT_KIND_BODY,
    CONTENT_KIND_DOCUMENT_BOUNDARY,
    CONTENT_KIND_LIST_ITEM,
    CONTENT_KIND_TABLE,
    EVIDENCE_ROLE_BODY,
    EVIDENCE_ROLE_HEADING,
    EVIDENCE_ROLE_LIST,
    EVIDENCE_ROLE_TABLE,
    EVIDENCE_ROLE_TITLE,
    EvidenceProvenance,
    CanonicalDocumentNode,
    CanonicalDocumentStructure,
    NODE_KIND_DOCUMENT_SECTION,
    ORIGIN_DETERMINISTIC_REPAIR,
    ORIGIN_EXPLICIT_SOURCE,
    ORIGIN_UNRESOLVED,
    SOURCE_FORMAT_DOCX,
    SOURCE_FORMAT_PDF,
    SOURCE_FORMAT_PPTX,
    STATUS_ACCEPTED,
    STATUS_PARTIAL,
    STATUS_UNRESOLVED,
    StructuralEvidence,
)
from document_hierarchy_explicit_preserver import preserve_explicit_docx_hierarchy


PDF_NUMBERING_RE = re.compile(r"^\s*(?P<number>\d+(?:\.\d+){0,2})(?:[.)])?\s+(?P<title>\S.+?)\s*$")


@dataclass(frozen=True)
class HierarchyValidationIssue:
    code: str
    node_id: Optional[str] = None
    message: str = ""


@dataclass(frozen=True)
class HierarchyValidationReport:
    issues: List[HierarchyValidationIssue] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return not self.issues


def preserve_document_hierarchy(
    evidence: Iterable[StructuralEvidence],
    *,
    document_id: str,
    document_title: str = "",
    source_format: Optional[str] = None,
) -> CanonicalDocumentStructure:
    """Preserve deterministic source hierarchy for supported formats."""

    evidence_items = sorted(list(evidence), key=lambda item: item.source_order)
    detected_format = source_format or _detect_source_format(evidence_items)
    if detected_format == SOURCE_FORMAT_DOCX:
        structure = _preserve_docx(evidence_items, document_id=document_id, document_title=document_title)
    elif detected_format == SOURCE_FORMAT_PPTX:
        structure = _preserve_pptx(evidence_items, document_id=document_id, document_title=document_title)
    elif detected_format == SOURCE_FORMAT_PDF:
        structure = _preserve_pdf(evidence_items, document_id=document_id, document_title=document_title)
    else:
        structure = CanonicalDocumentStructure(
            document_id=document_id,
            source_format=detected_format or "unknown",
            document_title=document_title,
            structure_status=STATUS_UNRESOLVED,
            structure_confidence=CONFIDENCE_LOW,
            diagnostics={
                "input_evidence_count": len(evidence_items),
                "unsupported_source_format": detected_format,
            },
        )
    validation = validate_canonical_document_structure(structure, evidence_items)
    return _with_validation(structure, validation)


def validate_canonical_document_structure(
    structure: CanonicalDocumentStructure,
    evidence: Iterable[StructuralEvidence],
) -> HierarchyValidationReport:
    """Validate preserved hierarchy without repairing it."""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    nodes_by_id = {node.node_id: node for node in structure.nodes}
    seen_evidence: Dict[str, str] = {}
    issues: List[HierarchyValidationIssue] = []
    previous_order: Optional[int] = None

    for node in structure.nodes:
        if previous_order is not None and node.source_order < previous_order:
            issues.append(HierarchyValidationIssue("source_order_decrease", node.node_id))
        previous_order = node.source_order
        if node.parent_id == node.node_id:
            issues.append(HierarchyValidationIssue("self_parent", node.node_id))
        if node.parent_id:
            parent = nodes_by_id.get(node.parent_id)
            if parent is None:
                issues.append(HierarchyValidationIssue("missing_parent", node.node_id))
            elif parent.source_order >= node.source_order:
                issues.append(HierarchyValidationIssue("parent_not_before_child", node.node_id))
            if parent is not None and node.level and parent.level and node.level > parent.level + 1:
                issues.append(HierarchyValidationIssue("impossible_hierarchy_jump", node.node_id))
        if not node.evidence_ids:
            issues.append(HierarchyValidationIssue("missing_evidence_provenance", node.node_id))
        for evidence_id in node.evidence_ids:
            if evidence_id not in evidence_by_id:
                issues.append(HierarchyValidationIssue("unknown_evidence_id", node.node_id))
            if evidence_id in seen_evidence:
                issues.append(HierarchyValidationIssue("duplicate_evidence_node", node.node_id))
            seen_evidence[evidence_id] = node.node_id
            item = evidence_by_id.get(evidence_id)
            if item is not None and item.content_kind in {CONTENT_KIND_LIST_ITEM, CONTENT_KIND_TABLE}:
                issues.append(HierarchyValidationIssue("local_content_accepted", node.node_id))
    return HierarchyValidationReport(issues=issues)


def render_document_hierarchy_diagnostics(structure: CanonicalDocumentStructure) -> str:
    """Render accepted hierarchy and unresolved/local diagnostics for inspection."""

    lines = [f"DOCUMENT: {structure.document_title or structure.document_id}"]
    by_id = {node.node_id: node for node in structure.nodes}
    for node in sorted(structure.nodes, key=lambda item: item.source_order):
        indent = "  " * _node_depth(node, by_id)
        lines.append(
            f"{indent}[{node.status}][{node.confidence}] L{node.level} {node.source_title}"
        )
        lines.append(f"{indent}  evidence: {', '.join(node.evidence_ids)}")
    for item in structure.diagnostics.get("unresolved_items", []):
        lines.append(
            f"[unresolved][{item.get('confidence', CONFIDENCE_LOW)}] {item.get('text')} "
            f"source_order={item.get('source_order')} reason={item.get('reason')}"
        )
    for item in structure.diagnostics.get("ignored_local_items", []):
        lines.append(
            f"[ignored-local] {item.get('text')} source_order={item.get('source_order')} "
            f"reason={item.get('reason')}"
        )
    for item in structure.diagnostics.get("conflicts", []):
        lines.append(
            f"[conflict] {item.get('evidence_id')} node={item.get('node_id')} "
            f"reason={item.get('reason')}"
        )
    return "\n".join(lines)


def _preserve_docx(
    evidence_items: List[StructuralEvidence],
    *,
    document_id: str,
    document_title: str,
) -> CanonicalDocumentStructure:
    structure = preserve_explicit_docx_hierarchy(
        evidence_items,
        document_id=document_id,
        document_title=document_title,
    )
    consistency = analyze_docx_heading_numbering_consistency(evidence_items, structure)
    conflicts = [
        {
            "evidence_id": record.evidence_id,
            "node_id": record.node_id,
            "reason": "heading_numbering_conflict",
            "heading_level": record.heading_level,
            "numbering_level": record.numbering_level,
        }
        for record in consistency.records
        if record.result == "conflicting"
    ]
    ignored_local = [
        _diagnostic_item(item, "docx_non_heading_numbering_not_promoted")
        for item in evidence_items
        if item.source_format == SOURCE_FORMAT_DOCX
        and item.numbering_evidence
        and item.evidence_role != EVIDENCE_ROLE_HEADING
    ]
    diagnostics = {
        **structure.diagnostics,
        "conflicts": conflicts,
        "ignored_local_items": ignored_local,
        "unresolved_items": [],
        "docx_numbering_consistency": consistency.diagnostics,
    }
    return _copy_structure(structure, diagnostics=diagnostics)


def _preserve_pptx(
    evidence_items: List[StructuralEvidence],
    *,
    document_id: str,
    document_title: str,
) -> CanonicalDocumentStructure:
    nodes: List[CanonicalDocumentNode] = []
    unresolved: List[Dict[str, Any]] = []
    ignored_local: List[Dict[str, Any]] = []
    current_section: Optional[CanonicalDocumentNode] = None
    title_index = 0

    for item in evidence_items:
        if item.source_format != SOURCE_FORMAT_PPTX:
            continue
        if item.evidence_role == EVIDENCE_ROLE_TITLE and _is_pptx_title_placeholder(item):
            title_index += 1
            is_section = _is_pptx_section_header(item)
            level = 1 if current_section is None or is_section else 2
            parent_id = None if level == 1 else current_section.node_id
            node = _node(
                node_id=f"pptx-title-{title_index:04d}",
                item=item,
                level=level,
                parent_id=parent_id,
                origin=ORIGIN_EXPLICIT_SOURCE,
                confidence=CONFIDENCE_HIGH if is_section else CONFIDENCE_MEDIUM,
                status=STATUS_ACCEPTED,
            )
            nodes.append(node)
            if is_section:
                current_section = node
        elif item.content_kind == CONTENT_KIND_LIST_ITEM or item.evidence_role == EVIDENCE_ROLE_LIST:
            ignored_local.append(_diagnostic_item(item, "pptx_bullet_local_content"))
        elif item.evidence_role in {EVIDENCE_ROLE_BODY, EVIDENCE_ROLE_TABLE}:
            ignored_local.append(_diagnostic_item(item, "pptx_non_title_content_not_promoted"))
        else:
            unresolved.append(_diagnostic_item(item, "pptx_insufficient_title_evidence"))

    return _structure(
        document_id=document_id,
        source_format=SOURCE_FORMAT_PPTX,
        document_title=document_title,
        nodes=nodes,
        diagnostics={
            "input_evidence_count": len(evidence_items),
            "accepted_node_count": len(nodes),
            "unresolved_items": unresolved,
            "ignored_local_items": ignored_local,
            "conflicts": [],
        },
    )


def _preserve_pdf(
    evidence_items: List[StructuralEvidence],
    *,
    document_id: str,
    document_title: str,
) -> CanonicalDocumentStructure:
    pdf_items = [item for item in evidence_items if item.source_format == SOURCE_FORMAT_PDF]
    body_font = _dominant_font_size(pdf_items)
    running_headers = _running_header_keys(pdf_items)
    candidates = [
        _pdf_candidate(item, body_font, running_headers)
        for item in pdf_items
    ]
    candidates = [candidate for candidate in candidates if candidate["candidate"]]
    recurring_patterns = {
        pattern
        for pattern, count in Counter(candidate["pattern"] for candidate in candidates).items()
        if pattern is not None and count >= 2
    }
    visual_levels = {
        pattern: index
        for index, pattern in enumerate(
            sorted(recurring_patterns, key=lambda value: value[0], reverse=True),
            start=1,
        )
    }
    accepted_candidates = [
        candidate
        for candidate in candidates
        if candidate["pattern"] in recurring_patterns
        and not candidate["running_header"]
        and not candidate["local_numbered_list"]
    ]

    nodes: List[CanonicalDocumentNode] = []
    stack: Dict[int, CanonicalDocumentNode] = {}
    for index, candidate in enumerate(sorted(accepted_candidates, key=lambda value: value["item"].source_order), start=1):
        item = candidate["item"]
        level = candidate["numbering_level"] or visual_levels.get(candidate["pattern"]) or 1
        parent = stack.get(level - 1) if level and level > 1 else None
        parent_id = parent.node_id if parent is not None else None
        parent_missing = bool(level and level > 1 and parent is None)
        node = _node(
            node_id=f"pdf-heading-{index:04d}",
            item=item,
            level=level,
            parent_id=parent_id,
            origin=ORIGIN_DETERMINISTIC_REPAIR,
            confidence=CONFIDENCE_HIGH if candidate["numbering_level"] else CONFIDENCE_MEDIUM,
            status=STATUS_UNRESOLVED if parent_missing else STATUS_ACCEPTED,
            repair_reason="missing_visual_parent" if parent_missing else "pdf_visual_numbering_pattern",
        )
        nodes.append(node)
        if level is not None:
            stack[level] = node
            for stale_level in list(stack):
                if stale_level > level:
                    del stack[stale_level]

    accepted_ids = {evidence_id for node in nodes for evidence_id in node.evidence_ids}
    unresolved = []
    ignored_local = []
    for candidate in candidates:
        item = candidate["item"]
        if item.evidence_id in accepted_ids:
            continue
        if candidate["running_header"]:
            ignored_local.append(_diagnostic_item(item, "pdf_repeated_running_header"))
        elif candidate["local_numbered_list"]:
            ignored_local.append(_diagnostic_item(item, "pdf_numbered_list_without_visual_structure"))
        else:
            unresolved.append(_diagnostic_item(item, "pdf_insufficient_recurring_visual_structure"))

    return _structure(
        document_id=document_id,
        source_format=SOURCE_FORMAT_PDF,
        document_title=document_title,
        nodes=nodes,
        diagnostics={
            "input_evidence_count": len(evidence_items),
            "accepted_node_count": len(nodes),
            "body_font_size": body_font,
            "recurring_visual_patterns": sorted(str(pattern) for pattern in recurring_patterns),
            "unresolved_items": unresolved,
            "ignored_local_items": ignored_local,
            "conflicts": [],
        },
    )


def _pdf_candidate(item: StructuralEvidence, body_font: Optional[float], running_headers: set) -> Dict[str, Any]:
    text = item.normalized_text.strip()
    font_size = _float_or_none(item.visual_evidence.get("font_size"))
    numbering_level = _pdf_numbering_level(text)
    prominent = _is_visually_prominent(font_size, body_font)
    short = _is_short_standalone(text)
    running_header = _pdf_running_header_key(item) in running_headers
    local_numbered_list = bool(_simple_integer_prefix(text)) and not prominent
    pattern = None
    visual_level = None
    if prominent and short:
        pattern = _pdf_visual_pattern(item)
    return {
        "item": item,
        "candidate": bool((prominent and short) or numbering_level is not None),
        "pattern": pattern,
        "numbering_level": numbering_level if prominent else None,
        "visual_level": visual_level,
        "running_header": running_header,
        "local_numbered_list": local_numbered_list,
    }


def _dominant_font_size(items: Sequence[StructuralEvidence]) -> Optional[float]:
    sizes = [
        _float_or_none(item.visual_evidence.get("font_size"))
        for item in items
        if _float_or_none(item.visual_evidence.get("font_size")) is not None
    ]
    if not sizes:
        return None
    return Counter(sizes).most_common(1)[0][0]


def _is_visually_prominent(font_size: Optional[float], body_font: Optional[float]) -> bool:
    if font_size is None or body_font is None:
        return False
    return font_size > body_font


def _pdf_visual_pattern(item: StructuralEvidence) -> Optional[tuple]:
    font_size = _float_or_none(item.visual_evidence.get("font_size"))
    if font_size is None:
        return None
    return (
        round(font_size, 1),
        str(item.visual_evidence.get("font_name", "")),
        str(item.visual_evidence.get("font_subtype", "")),
    )


def _pdf_numbering_level(text: str) -> Optional[int]:
    match = PDF_NUMBERING_RE.match(text)
    if not match:
        return None
    number = match.group("number")
    if "." not in number:
        return 1
    return len(number.split("."))


def _simple_integer_prefix(text: str) -> Optional[str]:
    match = re.match(r"^\s*\d+[.)]?\s+\S+", text)
    if not match:
        return None
    dotted = PDF_NUMBERING_RE.match(text)
    if dotted and "." in dotted.group("number"):
        return None
    return match.group(0)


def _is_short_standalone(text: str) -> bool:
    words = text.split()
    return 1 <= len(words) <= 12


def _running_header_keys(items: Sequence[StructuralEvidence]) -> set:
    by_key: Dict[tuple, set] = defaultdict(set)
    for item in items:
        key = _pdf_running_header_key(item)
        if key is None:
            continue
        by_key[key].add(item.source_span.start.unit_index)
    return {key for key, pages in by_key.items() if len(pages) >= 2}


def _pdf_running_header_key(item: StructuralEvidence) -> Optional[tuple]:
    text = item.normalized_text.strip()
    if not text:
        return None
    y = _float_or_none(item.visual_evidence.get("y0") or item.visual_evidence.get("y"))
    page_height = _float_or_none(item.visual_evidence.get("page_height"))
    if y is None or page_height is None or page_height <= 0:
        return None
    if y < page_height * 0.10 or y > page_height * 0.90:
        return (text.lower(), round(y / page_height, 2))
    return None


def _is_pptx_title_placeholder(item: StructuralEvidence) -> bool:
    placeholder = str(item.native_evidence.get("placeholder_type", "")).upper()
    return item.evidence_role == EVIDENCE_ROLE_TITLE and "TITLE" in placeholder


def _is_pptx_section_header(item: StructuralEvidence) -> bool:
    layout_name = str(item.native_evidence.get("layout_name", "")).lower()
    return "section" in layout_name


def _node(
    *,
    node_id: str,
    item: StructuralEvidence,
    level: Optional[int],
    parent_id: Optional[str],
    origin: str,
    confidence: str,
    status: str,
    repair_reason: Optional[str] = None,
) -> CanonicalDocumentNode:
    return CanonicalDocumentNode(
        node_id=node_id,
        source_title=item.normalized_text,
        logical_title=item.normalized_text,
        level=level,
        parent_id=parent_id,
        source_order=item.source_order,
        source_span=item.source_span,
        origin=origin,
        confidence=confidence,
        status=status,
        evidence_ids=[item.evidence_id],
        provenance=EvidenceProvenance(
            supporting_evidence_ids=[item.evidence_id],
            decision_source=origin,
            repair_reason=repair_reason,
        ),
        node_kind=NODE_KIND_DOCUMENT_SECTION,
    )


def _structure(
    *,
    document_id: str,
    source_format: str,
    document_title: str,
    nodes: List[CanonicalDocumentNode],
    diagnostics: Dict[str, Any],
) -> CanonicalDocumentStructure:
    unresolved_count = sum(1 for node in nodes if node.status == STATUS_UNRESOLVED)
    status = STATUS_UNRESOLVED if not nodes else STATUS_PARTIAL if unresolved_count else STATUS_ACCEPTED
    confidence = CONFIDENCE_LOW if not nodes else CONFIDENCE_MEDIUM if unresolved_count else CONFIDENCE_HIGH
    return CanonicalDocumentStructure(
        document_id=document_id,
        source_format=source_format,
        document_title=document_title,
        structure_status=status,
        structure_confidence=confidence,
        nodes=nodes,
        diagnostics=diagnostics,
    )


def _copy_structure(
    structure: CanonicalDocumentStructure,
    *,
    diagnostics: Dict[str, Any],
) -> CanonicalDocumentStructure:
    return CanonicalDocumentStructure(
        document_id=structure.document_id,
        source_format=structure.source_format,
        document_title=structure.document_title,
        structure_status=structure.structure_status,
        structure_confidence=structure.structure_confidence,
        nodes=list(structure.nodes),
        diagnostics=diagnostics,
    )


def _with_validation(
    structure: CanonicalDocumentStructure,
    validation: HierarchyValidationReport,
) -> CanonicalDocumentStructure:
    diagnostics = {
        **structure.diagnostics,
        "validation": {
            "is_valid": validation.is_valid,
            "issues": [
                {"code": issue.code, "node_id": issue.node_id, "message": issue.message}
                for issue in validation.issues
            ],
        },
    }
    return _copy_structure(structure, diagnostics=diagnostics)


def _diagnostic_item(item: StructuralEvidence, reason: str) -> Dict[str, Any]:
    return {
        "evidence_id": item.evidence_id,
        "source_order": item.source_order,
        "text": item.normalized_text,
        "reason": reason,
        "source_span": item.source_span,
        "confidence": CONFIDENCE_LOW,
    }


def _node_depth(node: CanonicalDocumentNode, by_id: Dict[str, CanonicalDocumentNode]) -> int:
    depth = 0
    parent_id = node.parent_id
    while parent_id and parent_id in by_id:
        depth += 1
        parent_id = by_id[parent_id].parent_id
    return depth


def _detect_source_format(evidence_items: Sequence[StructuralEvidence]) -> Optional[str]:
    if not evidence_items:
        return None
    return evidence_items[0].source_format


def _float_or_none(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
