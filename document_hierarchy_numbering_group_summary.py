"""Read-only summaries of native DOCX numbering groups.

Step 14 aggregates deterministic facts about each native Word ``num_id``.  It
does not decide whether a numbering group is trusted, structural, local, or
global.  It never creates hierarchy, repairs levels, assigns parents, or mutates
the canonical document structure.
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
    EVIDENCE_ROLE_HEADING,
    CanonicalDocumentStructure,
    StructuralEvidence,
)
from document_hierarchy_numbering_authority import (
    AUTHORITY_AMBIGUOUS_NUMBERING,
    AUTHORITY_EXPLICIT_CONFLICT,
    AUTHORITY_EXPLICIT_CONFIRMED,
    AUTHORITY_LOCAL_OR_UNSUPPORTED,
    AUTHORITY_TRUSTED_NUMBERING_SYSTEM,
    DocxNumberingAuthorityReport,
)
from document_hierarchy_numbering_observer import (
    TRANSITION_RESTART,
    DocxNumberingObservationReport,
)


@dataclass(frozen=True)
class UngroupableNumberingEvidence:
    evidence_id: str
    source_order: int
    reason: str
    abstract_num_id: Optional[Any] = None
    ilvl: Optional[int] = None
    numbering_depth: Optional[int] = None


@dataclass(frozen=True)
class DocxNumberingGroupSummary:
    group_id: str
    num_id: Any
    abstract_num_ids: List[Any] = field(default_factory=list)
    numbering_sources: List[str] = field(default_factory=list)
    total_evidence_count: int = 0
    explicit_heading_count: int = 0
    non_heading_count: int = 0
    explicit_confirmed_count: int = 0
    explicit_conflict_count: int = 0
    trusted_numbering_system_count: int = 0
    ambiguous_count: int = 0
    local_or_unsupported_count: int = 0
    observed_ilvls: List[int] = field(default_factory=list)
    observed_depths: List[int] = field(default_factory=list)
    heading_levels: List[int] = field(default_factory=list)
    heading_depth_pairs: Dict[str, int] = field(default_factory=dict)
    min_source_order: Optional[int] = None
    max_source_order: Optional[int] = None
    distinct_unit_count: int = 0
    first_unit: Optional[str] = None
    last_unit: Optional[str] = None
    interruption_count: int = 0
    restart_count: int = 0
    transition_types: List[str] = field(default_factory=list)
    evidence_roles: List[str] = field(default_factory=list)
    content_kinds: List[str] = field(default_factory=list)
    evidence_ids: List[str] = field(default_factory=list)
    node_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocxNumberingGroupSummaryReport:
    groups: List[DocxNumberingGroupSummary] = field(default_factory=list)
    ungroupable_evidence: List[UngroupableNumberingEvidence] = field(default_factory=list)
    diagnostics: Dict[str, int] = field(default_factory=dict)


def summarize_docx_numbering_groups(
    evidence: Iterable[StructuralEvidence],
    structure: CanonicalDocumentStructure,
    consistency_report: HeadingNumberingConsistencyReport,
    numbering_report: DocxNumberingObservationReport,
    authority_report: DocxNumberingAuthorityReport,
) -> DocxNumberingGroupSummaryReport:
    """Aggregate factual usage of each native DOCX ``num_id``."""

    evidence_items = sorted(list(evidence), key=lambda item: item.source_order)
    nodes_by_evidence_id = {
        evidence_id: node
        for node in structure.nodes
        for evidence_id in node.evidence_ids
    }
    consistency_by_evidence_id = {
        record.evidence_id: record
        for record in consistency_report.records
    }
    observations_by_evidence_id = {
        observation.evidence_id: observation
        for observation in numbering_report.observations
    }
    authority_by_evidence_id = {
        record.evidence_id: record
        for record in authority_report.records
    }

    builders: Dict[Any, Dict[str, Any]] = {}
    ungroupable: List[UngroupableNumberingEvidence] = []

    for item in evidence_items:
        if not item.numbering_evidence:
            continue
        num_id = item.numbering_evidence.get("num_id")
        if num_id is None:
            ungroupable.append(_ungroupable(item, observations_by_evidence_id.get(item.evidence_id)))
            continue
        builder = builders.setdefault(num_id, _new_builder(num_id))
        _add_evidence(
            builder,
            item,
            node_id=nodes_by_evidence_id.get(item.evidence_id).node_id
            if item.evidence_id in nodes_by_evidence_id
            else None,
            consistency=consistency_by_evidence_id.get(item.evidence_id),
            observation=observations_by_evidence_id.get(item.evidence_id),
            authority_class=authority_by_evidence_id.get(item.evidence_id).authority_class
            if item.evidence_id in authority_by_evidence_id
            else None,
        )

    groups = [_finalize_builder(builder) for builder in builders.values()]
    groups = sorted(groups, key=lambda group: (group.min_source_order is None, group.min_source_order, str(group.num_id)))
    ungroupable = sorted(ungroupable, key=lambda item: (item.source_order, item.evidence_id))

    return DocxNumberingGroupSummaryReport(
        groups=groups,
        ungroupable_evidence=ungroupable,
        diagnostics=_report_diagnostics(groups, ungroupable),
    )


def render_docx_numbering_group_summary(report: DocxNumberingGroupSummaryReport) -> str:
    """Render factual numbering-group summaries without trust labels."""

    lines: List[str] = []
    for group in report.groups:
        lines.append(f"NUMBERING GROUP num_id={group.num_id}")
        lines.append(f"abstract_num_ids: {', '.join(str(value) for value in group.abstract_num_ids) or 'none'}")
        lines.append(
            f"evidence: total={group.total_evidence_count} "
            f"explicit_headings={group.explicit_heading_count} non_heading={group.non_heading_count}"
        )
        lines.append(
            f"authority: confirmed={group.explicit_confirmed_count} "
            f"conflicts={group.explicit_conflict_count} ambiguous={group.ambiguous_count} "
            f"local_or_unsupported={group.local_or_unsupported_count}"
        )
        lines.append(f"depths: {', '.join(str(value) for value in group.observed_depths) or 'none'}")
        lines.append(f"source_order: {group.min_source_order} -> {group.max_source_order}")
        if group.heading_depth_pairs:
            pairs = ", ".join(
                f"{pair}:{count}"
                for pair, count in sorted(group.heading_depth_pairs.items())
            )
            lines.append(f"heading_depth_pairs: {pairs}")
        lines.append(f"interruptions: {group.interruption_count}")
        lines.append(f"restarts: {group.restart_count}")
    if report.ungroupable_evidence:
        lines.append("UNGROUPABLE NUMBERING")
        for item in report.ungroupable_evidence:
            lines.append(f"  {item.evidence_id} reason={item.reason}")
    return "\n".join(lines)


def _new_builder(num_id: Any) -> Dict[str, Any]:
    return {
        "num_id": num_id,
        "abstract_num_ids": set(),
        "numbering_sources": set(),
        "total_evidence_count": 0,
        "explicit_heading_count": 0,
        "non_heading_count": 0,
        "explicit_confirmed_count": 0,
        "explicit_conflict_count": 0,
        "trusted_numbering_system_count": 0,
        "ambiguous_count": 0,
        "local_or_unsupported_count": 0,
        "observed_ilvls": set(),
        "observed_depths": set(),
        "heading_levels": set(),
        "heading_depth_pairs": {},
        "source_orders": [],
        "units": [],
        "interruption_count": 0,
        "restart_count": 0,
        "transition_types": set(),
        "evidence_roles": set(),
        "content_kinds": set(),
        "evidence_ids": [],
        "node_ids": [],
    }


def _add_evidence(
    builder: Dict[str, Any],
    item: StructuralEvidence,
    *,
    node_id: Optional[str],
    consistency,
    observation,
    authority_class: Optional[str],
) -> None:
    numbering = item.numbering_evidence
    builder["total_evidence_count"] += 1
    builder["evidence_ids"].append(item.evidence_id)
    builder["source_orders"].append(item.source_order)
    builder["evidence_roles"].add(item.evidence_role)
    builder["content_kinds"].add(item.content_kind)
    builder["units"].append(_unit_key(item))
    if node_id:
        builder["node_ids"].append(node_id)

    _add_if_present(builder["abstract_num_ids"], numbering.get("abstract_num_id"))
    _add_if_present(builder["numbering_sources"], numbering.get("source"))
    ilvl = _int_or_none(numbering.get("ilvl"))
    if ilvl is not None:
        builder["observed_ilvls"].add(ilvl)
        builder["observed_depths"].add(ilvl + 1)

    if _is_explicit_heading(item):
        builder["explicit_heading_count"] += 1
        heading_level = _heading_level(item)
        if heading_level is not None:
            builder["heading_levels"].add(heading_level)
        if consistency is not None and consistency.numbering_level is not None:
            pair = f"H{consistency.heading_level}:D{consistency.numbering_level}"
            builder["heading_depth_pairs"][pair] = builder["heading_depth_pairs"].get(pair, 0) + 1
    else:
        builder["non_heading_count"] += 1

    if observation is not None:
        if observation.interrupted_since_previous:
            builder["interruption_count"] += 1
        if observation.transition_from_previous:
            builder["transition_types"].add(observation.transition_from_previous)
        if observation.transition_from_previous == TRANSITION_RESTART:
            builder["restart_count"] += 1

    if authority_class == AUTHORITY_EXPLICIT_CONFIRMED:
        builder["explicit_confirmed_count"] += 1
    elif authority_class == AUTHORITY_EXPLICIT_CONFLICT:
        builder["explicit_conflict_count"] += 1
    elif authority_class == AUTHORITY_TRUSTED_NUMBERING_SYSTEM:
        builder["trusted_numbering_system_count"] += 1
    elif authority_class == AUTHORITY_AMBIGUOUS_NUMBERING:
        builder["ambiguous_count"] += 1
    elif authority_class == AUTHORITY_LOCAL_OR_UNSUPPORTED:
        builder["local_or_unsupported_count"] += 1


def _finalize_builder(builder: Dict[str, Any]) -> DocxNumberingGroupSummary:
    source_orders = sorted(builder["source_orders"])
    units = list(builder["units"])
    distinct_units = sorted(set(units))
    return DocxNumberingGroupSummary(
        group_id=f"num:{builder['num_id']}",
        num_id=builder["num_id"],
        abstract_num_ids=_sorted_values(builder["abstract_num_ids"]),
        numbering_sources=_sorted_values(builder["numbering_sources"]),
        total_evidence_count=builder["total_evidence_count"],
        explicit_heading_count=builder["explicit_heading_count"],
        non_heading_count=builder["non_heading_count"],
        explicit_confirmed_count=builder["explicit_confirmed_count"],
        explicit_conflict_count=builder["explicit_conflict_count"],
        trusted_numbering_system_count=builder["trusted_numbering_system_count"],
        ambiguous_count=builder["ambiguous_count"],
        local_or_unsupported_count=builder["local_or_unsupported_count"],
        observed_ilvls=_sorted_values(builder["observed_ilvls"]),
        observed_depths=_sorted_values(builder["observed_depths"]),
        heading_levels=_sorted_values(builder["heading_levels"]),
        heading_depth_pairs=dict(sorted(builder["heading_depth_pairs"].items())),
        min_source_order=source_orders[0] if source_orders else None,
        max_source_order=source_orders[-1] if source_orders else None,
        distinct_unit_count=len(distinct_units),
        first_unit=units[0] if units else None,
        last_unit=units[-1] if units else None,
        interruption_count=builder["interruption_count"],
        restart_count=builder["restart_count"],
        transition_types=_sorted_values(builder["transition_types"]),
        evidence_roles=_sorted_values(builder["evidence_roles"]),
        content_kinds=_sorted_values(builder["content_kinds"]),
        evidence_ids=list(builder["evidence_ids"]),
        node_ids=list(builder["node_ids"]),
    )


def _ungroupable(
    item: StructuralEvidence,
    observation,
) -> UngroupableNumberingEvidence:
    numbering = item.numbering_evidence
    ilvl = _int_or_none(numbering.get("ilvl"))
    return UngroupableNumberingEvidence(
        evidence_id=item.evidence_id,
        source_order=item.source_order,
        reason="missing_num_id",
        abstract_num_id=numbering.get("abstract_num_id"),
        ilvl=ilvl,
        numbering_depth=observation.numbering_depth if observation is not None else (ilvl + 1 if ilvl is not None else None),
    )


def _report_diagnostics(
    groups: List[DocxNumberingGroupSummary],
    ungroupable: List[UngroupableNumberingEvidence],
) -> Dict[str, int]:
    return {
        "numbering_group_count": len(groups),
        "ungroupable_evidence_count": len(ungroupable),
        "groups_with_explicit_headings": sum(1 for group in groups if group.explicit_heading_count > 0),
        "groups_with_non_heading_evidence": sum(1 for group in groups if group.non_heading_count > 0),
        "mixed_groups": sum(
            1
            for group in groups
            if group.explicit_heading_count > 0 and group.non_heading_count > 0
        ),
        "groups_with_conflicts": sum(1 for group in groups if group.explicit_conflict_count > 0),
        "groups_with_local_or_unsupported_evidence": sum(
            1 for group in groups if group.local_or_unsupported_count > 0
        ),
    }


def _is_explicit_heading(item: StructuralEvidence) -> bool:
    return item.evidence_role == EVIDENCE_ROLE_HEADING and _heading_level(item) in {1, 2, 3}


def _heading_level(item: StructuralEvidence) -> Optional[int]:
    return _int_or_none(item.native_evidence.get("native_hierarchy_hint"))


def _unit_key(item: StructuralEvidence) -> str:
    context = item.structural_context
    unit_type = context.get("unit_type", item.source_span.start.unit_type)
    unit_index = context.get("unit_index", item.source_span.start.unit_index)
    return f"{unit_type}:{unit_index}"


def _add_if_present(target: set, value: Any) -> None:
    if value is not None:
        target.add(value)


def _sorted_values(values: set) -> List[Any]:
    return sorted(values, key=lambda value: str(value))


def _int_or_none(value: Any) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
