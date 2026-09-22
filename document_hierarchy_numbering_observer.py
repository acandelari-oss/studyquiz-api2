"""Diagnostic-only observation of native DOCX numbering evidence.

This module observes non-heading DOCX numbering patterns from StructuralEvidence.
It does not create hierarchy nodes, assign parents, classify local/global lists,
or mutate CanonicalDocumentStructure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from document_hierarchy_contracts import (
    EVIDENCE_ROLE_HEADING,
    SOURCE_FORMAT_DOCX,
    StructuralEvidence,
)


TRANSITION_START = "start"
TRANSITION_SAME_DEPTH = "same_depth"
TRANSITION_DESCEND = "descend"
TRANSITION_ASCEND = "ascend"
TRANSITION_RESTART = "restart"
TRANSITION_INSUFFICIENT = "insufficient"


@dataclass(frozen=True)
class DocxNumberingObservation:
    evidence_id: str
    source_order: int
    group_id: str
    num_id: Optional[Any]
    abstract_num_id: Optional[Any]
    ilvl: Optional[int]
    numbering_depth: Optional[int]
    numbering_source: Optional[str]
    numbering_format: Optional[str]
    level_text: Optional[str]
    start: Optional[Any]
    start_override: Optional[Any]
    transition_from_previous: str
    interrupted_since_previous: bool = False
    previous_evidence_id: Optional[str] = None
    context_heading_evidence_id: Optional[str] = None


@dataclass(frozen=True)
class DocxNumberingObservationReport:
    observations: List[DocxNumberingObservation] = field(default_factory=list)
    diagnostics: Dict[str, int] = field(default_factory=dict)


def observe_docx_numbering_patterns(
    evidence: Iterable[StructuralEvidence],
) -> DocxNumberingObservationReport:
    """Observe native DOCX numbering patterns outside explicit headings."""

    evidence_items = sorted(list(evidence), key=lambda item: item.source_order)
    observations: List[DocxNumberingObservation] = []
    previous_by_group: Dict[str, DocxNumberingObservation] = {}
    last_explicit_heading_id: Optional[str] = None

    for item in evidence_items:
        if _is_explicit_docx_heading(item):
            last_explicit_heading_id = item.evidence_id
            continue
        if not _is_observable_numbered_docx_evidence(item):
            continue

        group_id = _group_id(item)
        previous = previous_by_group.get(group_id)
        observation = DocxNumberingObservation(
            evidence_id=item.evidence_id,
            source_order=item.source_order,
            group_id=group_id,
            num_id=item.numbering_evidence.get("num_id"),
            abstract_num_id=item.numbering_evidence.get("abstract_num_id"),
            ilvl=_int_or_none(item.numbering_evidence.get("ilvl")),
            numbering_depth=_numbering_depth(item),
            numbering_source=item.numbering_evidence.get("source"),
            numbering_format=item.numbering_evidence.get("format"),
            level_text=item.numbering_evidence.get("level_text"),
            start=item.numbering_evidence.get("start"),
            start_override=item.numbering_evidence.get("start_override"),
            transition_from_previous=_transition(previous, item),
            interrupted_since_previous=_interrupted_between(
                evidence_items,
                previous,
                item,
                group_id,
            ),
            previous_evidence_id=previous.evidence_id if previous is not None else None,
            context_heading_evidence_id=last_explicit_heading_id,
        )
        observations.append(observation)
        previous_by_group[group_id] = observation

    return DocxNumberingObservationReport(
        observations=observations,
        diagnostics=_diagnostics(observations),
    )


def render_docx_numbering_observations(
    report: DocxNumberingObservationReport,
) -> str:
    """Render diagnostic numbering observations, not a document tree."""

    lines: List[str] = []
    current_group = None
    for observation in sorted(report.observations, key=lambda item: (item.group_id, item.source_order)):
        if observation.group_id != current_group:
            current_group = observation.group_id
            lines.append(f"NUMBERING GROUP {observation.group_id}")
        interrupted = " interrupted" if observation.interrupted_since_previous else ""
        lines.append(
            f"  [depth {observation.numbering_depth}] {observation.evidence_id}"
            f" transition={observation.transition_from_previous}{interrupted}"
        )
    return "\n".join(lines)


def _is_observable_numbered_docx_evidence(item: StructuralEvidence) -> bool:
    return (
        item.source_format == SOURCE_FORMAT_DOCX
        and bool(item.numbering_evidence)
        and not _is_explicit_docx_heading(item)
    )


def _is_explicit_docx_heading(item: StructuralEvidence) -> bool:
    return (
        item.source_format == SOURCE_FORMAT_DOCX
        and item.evidence_role == EVIDENCE_ROLE_HEADING
        and _int_or_none(item.native_evidence.get("native_hierarchy_hint")) in {1, 2, 3}
    )


def _group_id(item: StructuralEvidence) -> str:
    num_id = item.numbering_evidence.get("num_id")
    if num_id is not None:
        return f"num:{num_id}"
    abstract_num_id = item.numbering_evidence.get("abstract_num_id")
    if abstract_num_id is not None:
        return f"abstract:{abstract_num_id}"
    return f"unknown:{item.evidence_id}"


def _numbering_depth(item: StructuralEvidence) -> Optional[int]:
    ilvl = _int_or_none(item.numbering_evidence.get("ilvl"))
    return ilvl + 1 if ilvl is not None else None


def _transition(
    previous: Optional[DocxNumberingObservation],
    current: StructuralEvidence,
) -> str:
    if previous is None:
        return TRANSITION_START
    if current.numbering_evidence.get("start_override") is not None:
        return TRANSITION_RESTART
    current_depth = _numbering_depth(current)
    previous_depth = previous.numbering_depth
    if current_depth is None or previous_depth is None:
        return TRANSITION_INSUFFICIENT
    if current_depth == previous_depth:
        return TRANSITION_SAME_DEPTH
    if current_depth > previous_depth:
        return TRANSITION_DESCEND
    return TRANSITION_ASCEND


def _interrupted_between(
    evidence_items: List[StructuralEvidence],
    previous: Optional[DocxNumberingObservation],
    current: StructuralEvidence,
    group_id: str,
) -> bool:
    if previous is None:
        return False
    for item in evidence_items:
        if previous.source_order >= item.source_order or item.source_order >= current.source_order:
            continue
        if not _is_observable_numbered_docx_evidence(item):
            return True
        if _group_id(item) != group_id:
            return True
    return False


def _diagnostics(observations: List[DocxNumberingObservation]) -> Dict[str, int]:
    return {
        "numbered_non_heading_count": len(observations),
        "numbering_group_count": len({observation.group_id for observation in observations}),
        "depth_1_count": sum(1 for observation in observations if observation.numbering_depth == 1),
        "depth_2_count": sum(1 for observation in observations if observation.numbering_depth == 2),
        "depth_3_plus_count": sum(
            1
            for observation in observations
            if observation.numbering_depth is not None and observation.numbering_depth >= 3
        ),
        "interrupted_group_count": len(
            {
                observation.group_id
                for observation in observations
                if observation.interrupted_since_previous
            }
        ),
        "restart_count": sum(
            1
            for observation in observations
            if observation.transition_from_previous == TRANSITION_RESTART
        ),
    }


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
