"""Deterministic consistency diagnostics for preserved hierarchy evidence.

This module does not repair or mutate hierarchy.  It compares explicit DOCX
heading levels already preserved in CanonicalDocumentStructure with native Word
numbering evidence already present in StructuralEvidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from document_hierarchy_contracts import (
    EVIDENCE_ROLE_HEADING,
    CanonicalDocumentStructure,
    SOURCE_FORMAT_DOCX,
    StructuralEvidence,
)


CONSISTENCY_CONSISTENT = "consistent"
CONSISTENCY_CONFLICTING = "conflicting"
CONSISTENCY_NOT_NUMBERED = "not_numbered"
CONSISTENCY_INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class HeadingNumberingConsistency:
    evidence_id: str
    node_id: str
    heading_level: int
    numbering_level: Optional[int]
    result: str
    numbering_source: Optional[str] = None


@dataclass(frozen=True)
class HeadingNumberingConsistencyReport:
    records: List[HeadingNumberingConsistency] = field(default_factory=list)
    diagnostics: Dict[str, int] = field(default_factory=dict)


def analyze_docx_heading_numbering_consistency(
    evidence: Iterable[StructuralEvidence],
    structure: CanonicalDocumentStructure,
) -> HeadingNumberingConsistencyReport:
    """Compare explicit DOCX heading levels with native Word numbering depth.

    Numbering is diagnostic only.  This function never changes node levels,
    parent IDs, statuses, origin, confidence, or structure diagnostics.
    """

    evidence_by_id = {item.evidence_id: item for item in evidence}
    records: List[HeadingNumberingConsistency] = []

    for node in sorted(structure.nodes, key=lambda item: item.source_order):
        if not node.evidence_ids:
            continue
        item = evidence_by_id.get(node.evidence_ids[0])
        if item is None or not _is_explicit_docx_heading(item):
            continue
        heading_level = _heading_level(item)
        if heading_level is None:
            continue
        numbering_level = _numbering_level(item)
        result = _consistency_result(item, heading_level, numbering_level)
        records.append(
            HeadingNumberingConsistency(
                evidence_id=item.evidence_id,
                node_id=node.node_id,
                heading_level=heading_level,
                numbering_level=numbering_level,
                result=result,
                numbering_source=item.numbering_evidence.get("source"),
            )
        )

    return HeadingNumberingConsistencyReport(
        records=records,
        diagnostics={
            "explicit_heading_count": len(records),
            "numbered_heading_count": sum(
                1
                for record in records
                if record.result in {CONSISTENCY_CONSISTENT, CONSISTENCY_CONFLICTING}
            ),
            "consistent_heading_count": sum(
                1 for record in records if record.result == CONSISTENCY_CONSISTENT
            ),
            "conflicting_heading_count": sum(
                1 for record in records if record.result == CONSISTENCY_CONFLICTING
            ),
            "uncomparable_heading_count": sum(
                1
                for record in records
                if record.result in {CONSISTENCY_NOT_NUMBERED, CONSISTENCY_INSUFFICIENT}
            ),
        },
    )


def render_heading_numbering_consistency(
    structure: CanonicalDocumentStructure,
    report: HeadingNumberingConsistencyReport,
) -> str:
    """Render tiny diagnostic output for tests/debugging."""

    records_by_node = {record.node_id: record for record in report.records}
    lines: List[str] = []
    for node in sorted(structure.nodes, key=lambda item: item.source_order):
        lines.append(f"L{node.level} {node.source_title}")
        record = records_by_node.get(node.node_id)
        if record is None:
            continue
        numbering = (
            f"numbering_level={record.numbering_level}"
            if record.numbering_level is not None
            else "numbering_level=None"
        )
        lines.append(f"  numbering: {numbering} -> {record.result}")
    return "\n".join(lines)


def _is_explicit_docx_heading(item: StructuralEvidence) -> bool:
    return (
        item.source_format == SOURCE_FORMAT_DOCX
        and item.evidence_role == EVIDENCE_ROLE_HEADING
        and _heading_level(item) in {1, 2, 3}
    )


def _heading_level(item: StructuralEvidence) -> Optional[int]:
    try:
        return int(item.native_evidence.get("native_hierarchy_hint"))
    except (TypeError, ValueError):
        return None


def _numbering_level(item: StructuralEvidence) -> Optional[int]:
    if "ilvl" not in item.numbering_evidence:
        return None
    try:
        return int(item.numbering_evidence["ilvl"]) + 1
    except (TypeError, ValueError):
        return None


def _consistency_result(
    item: StructuralEvidence,
    heading_level: int,
    numbering_level: Optional[int],
) -> str:
    if not item.numbering_evidence:
        return CONSISTENCY_NOT_NUMBERED
    if numbering_level is None:
        return CONSISTENCY_INSUFFICIENT
    if numbering_level == heading_level:
        return CONSISTENCY_CONSISTENT
    return CONSISTENCY_CONFLICTING
