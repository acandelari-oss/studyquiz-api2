"""Diagnostic authority classification for DOCX numbering evidence.

This module consumes only the evidence-first interpretation outputs:

- StructuralEvidence
- CanonicalDocumentStructure from explicit DOCX heading preservation
- HeadingNumberingConsistencyReport
- DocxNumberingObservationReport

It classifies what structural authority DOCX numbering evidence currently has.
It never creates hierarchy, repairs levels, assigns parents, or mutates the
canonical document structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from document_hierarchy_consistency import (
    CONSISTENCY_CONFLICTING,
    CONSISTENCY_CONSISTENT,
    HeadingNumberingConsistencyReport,
)
from document_hierarchy_contracts import (
    CONTENT_KIND_TABLE,
    EVIDENCE_ROLE_TABLE,
    CanonicalDocumentStructure,
    StructuralEvidence,
)
from document_hierarchy_numbering_observer import DocxNumberingObservationReport


AUTHORITY_EXPLICIT_CONFIRMED = "EXPLICIT_CONFIRMED"
AUTHORITY_EXPLICIT_CONFLICT = "EXPLICIT_CONFLICT"
AUTHORITY_TRUSTED_NUMBERING_SYSTEM = "TRUSTED_NUMBERING_SYSTEM"
AUTHORITY_AMBIGUOUS_NUMBERING = "AMBIGUOUS_NUMBERING"
AUTHORITY_LOCAL_OR_UNSUPPORTED = "LOCAL_OR_UNSUPPORTED"


@dataclass(frozen=True)
class NumberingAuthorityRecord:
    evidence_id: str
    source_order: int
    authority_class: str
    basis: Dict[str, Any] = field(default_factory=dict)
    node_id: Optional[str] = None
    numbering_group_id: Optional[str] = None


@dataclass(frozen=True)
class DocxNumberingAuthorityReport:
    records: List[NumberingAuthorityRecord] = field(default_factory=list)
    diagnostics: Dict[str, int] = field(default_factory=dict)


def classify_docx_numbering_authority(
    evidence: Iterable[StructuralEvidence],
    structure: CanonicalDocumentStructure,
    consistency_report: HeadingNumberingConsistencyReport,
    numbering_report: DocxNumberingObservationReport,
) -> DocxNumberingAuthorityReport:
    """Classify DOCX numbering authority without changing hierarchy."""

    evidence_by_id = {item.evidence_id: item for item in evidence}
    nodes_by_evidence_id = {
        evidence_id: node
        for node in structure.nodes
        for evidence_id in node.evidence_ids
    }
    records: List[NumberingAuthorityRecord] = []

    for consistency in consistency_report.records:
        item = evidence_by_id.get(consistency.evidence_id)
        if item is None:
            continue
        node = nodes_by_evidence_id.get(consistency.evidence_id)
        if consistency.result == CONSISTENCY_CONSISTENT:
            records.append(
                NumberingAuthorityRecord(
                    evidence_id=consistency.evidence_id,
                    node_id=consistency.node_id,
                    source_order=item.source_order,
                    authority_class=AUTHORITY_EXPLICIT_CONFIRMED,
                    numbering_group_id=_group_id_from_evidence(item),
                    basis={
                        "step": "heading_numbering_consistency",
                        "consistency": consistency.result,
                        "heading_level": consistency.heading_level,
                        "numbering_level": consistency.numbering_level,
                        "numbering_source": consistency.numbering_source,
                        "node_id": node.node_id if node is not None else consistency.node_id,
                    },
                )
            )
        elif consistency.result == CONSISTENCY_CONFLICTING:
            records.append(
                NumberingAuthorityRecord(
                    evidence_id=consistency.evidence_id,
                    node_id=consistency.node_id,
                    source_order=item.source_order,
                    authority_class=AUTHORITY_EXPLICIT_CONFLICT,
                    numbering_group_id=_group_id_from_evidence(item),
                    basis={
                        "step": "heading_numbering_consistency",
                        "consistency": consistency.result,
                        "heading_level": consistency.heading_level,
                        "numbering_level": consistency.numbering_level,
                        "numbering_source": consistency.numbering_source,
                        "node_id": node.node_id if node is not None else consistency.node_id,
                    },
                )
            )

    recorded_evidence_ids = {record.evidence_id for record in records}
    for observation in numbering_report.observations:
        if observation.evidence_id in recorded_evidence_ids:
            continue
        item = evidence_by_id.get(observation.evidence_id)
        records.append(
            NumberingAuthorityRecord(
                evidence_id=observation.evidence_id,
                source_order=observation.source_order,
                authority_class=_authority_for_observation(item),
                numbering_group_id=observation.group_id,
                basis={
                    "step": "numbering_observation",
                    "num_id": observation.num_id,
                    "abstract_num_id": observation.abstract_num_id,
                    "ilvl": observation.ilvl,
                    "numbering_depth": observation.numbering_depth,
                    "transition_from_previous": observation.transition_from_previous,
                    "interrupted_since_previous": observation.interrupted_since_previous,
                    "context_heading_evidence_id": observation.context_heading_evidence_id,
                },
            )
        )

    ordered_records = sorted(records, key=lambda record: (record.source_order, record.evidence_id))
    return DocxNumberingAuthorityReport(
        records=ordered_records,
        diagnostics=_diagnostics(ordered_records),
    )


def render_docx_numbering_authority(report: DocxNumberingAuthorityReport) -> str:
    """Render compact diagnostics; ambiguous numbering is not rendered as a tree."""

    lines: List[str] = []
    for record in report.records:
        label = f"[{record.authority_class}] {record.evidence_id}"
        if record.node_id:
            label += f" node={record.node_id}"
        if record.numbering_group_id:
            label += f" group={record.numbering_group_id}"
        lines.append(label)
        if "heading_level" in record.basis or "numbering_level" in record.basis:
            lines.append(
                "  "
                f"heading={record.basis.get('heading_level')} "
                f"numbering={record.basis.get('numbering_level')}"
            )
        elif "numbering_depth" in record.basis:
            lines.append(
                "  "
                f"depth={record.basis.get('numbering_depth')} "
                f"transition={record.basis.get('transition_from_previous')}"
            )
    return "\n".join(lines)


def _authority_for_observation(
    item: Optional[StructuralEvidence],
) -> str:
    if item is None:
        return AUTHORITY_AMBIGUOUS_NUMBERING
    if item.evidence_role == EVIDENCE_ROLE_TABLE or item.content_kind == CONTENT_KIND_TABLE:
        return AUTHORITY_LOCAL_OR_UNSUPPORTED
    return AUTHORITY_AMBIGUOUS_NUMBERING


def _group_id_from_evidence(item: StructuralEvidence) -> Optional[str]:
    num_id = item.numbering_evidence.get("num_id")
    if num_id is not None:
        return f"num:{num_id}"
    abstract_num_id = item.numbering_evidence.get("abstract_num_id")
    if abstract_num_id is not None:
        return f"abstract:{abstract_num_id}"
    return None


def _diagnostics(records: List[NumberingAuthorityRecord]) -> Dict[str, int]:
    return {
        "explicit_confirmed_count": _count(records, AUTHORITY_EXPLICIT_CONFIRMED),
        "explicit_conflict_count": _count(records, AUTHORITY_EXPLICIT_CONFLICT),
        "trusted_numbering_system_count": _count(records, AUTHORITY_TRUSTED_NUMBERING_SYSTEM),
        "ambiguous_numbering_count": _count(records, AUTHORITY_AMBIGUOUS_NUMBERING),
        "local_or_unsupported_count": _count(records, AUTHORITY_LOCAL_OR_UNSUPPORTED),
    }


def _count(records: List[NumberingAuthorityRecord], authority_class: str) -> int:
    return sum(1 for record in records if record.authority_class == authority_class)
