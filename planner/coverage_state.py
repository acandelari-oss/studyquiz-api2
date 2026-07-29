"""Deterministic Coverage Mode state derived from Survey Study Plan progression."""

from dataclasses import dataclass, field
from typing import Mapping, Sequence, Set

from .planner_models import PlannerContext


PROFESSOR_MODE_COVERAGE = "coverage"
PROFESSOR_MODE_ADAPTIVE = "adaptive"
COVERAGE_STATUS_INCOMPLETE = "coverage_incomplete"
COVERAGE_STATUS_COMPLETE = "coverage_complete"


@dataclass(frozen=True)
class CoverageState:
    """Coverage state for category-level initial evidence collection."""

    covered_categories: Sequence[str] = field(default_factory=tuple)
    categories_requiring_coverage: Sequence[str] = field(default_factory=tuple)

    @property
    def complete(self) -> bool:
        return not self.categories_requiring_coverage


def evaluate_coverage_state(context: PlannerContext) -> CoverageState:
    """Return categories completed by previous Survey/Coverage Study Plans.

    Survey/Coverage progression deliberately ignores learning evidence such as
    quiz accuracy, quiz coverage, flashcards, mastery, and recency. A category
    is considered complete for Survey only after it belongs to a completed
    Survey/Coverage Study Plan.
    """

    covered = []
    requiring = []
    completed_identities = {
        _category_identity(category)
        for category in context.completed_survey_categories
        if _category_identity(category)
    }

    for category in context.categories:
        if _category_identity(category) in completed_identities:
            covered.append(category)
        else:
            requiring.append(category)

    return CoverageState(
        covered_categories=tuple(covered),
        categories_requiring_coverage=tuple(requiring),
    )


def ordered_coverage_categories(
    initial_ranked_categories: Sequence[str],
    categories_requiring_coverage: Sequence[str],
    previous_skipped_categories: Sequence[str] = (),
) -> Sequence[str]:
    """Return Coverage continuation order without using new performance data."""

    requiring_set = set(categories_requiring_coverage)
    ordered = []
    seen = set()

    for category in previous_skipped_categories:
        if category in requiring_set and category not in seen:
            ordered.append(category)
            seen.add(category)

    for category in initial_ranked_categories:
        if category in requiring_set and category not in seen:
            ordered.append(category)
            seen.add(category)

    return tuple(ordered)


def ordered_first_exposure_categories(
    *,
    ranked_categories: Sequence[str],
    previously_scheduled_categories: Sequence[str],
    priority_categories: Sequence[str] = (),
    enabled: bool = True,
) -> Sequence[str]:
    """Return Survey/Coverage order with priority as an initial-start signal only.

    Student priorities indicate where initial evaluation should begin. They must
    not cause already scheduled categories to be repeated before the remaining
    syllabus has received first exposure.
    """

    if not enabled:
        return tuple(ranked_categories)

    ranked = tuple(ranked_categories)
    priority_identities = {
        _category_identity(category)
        for category in priority_categories
        if _category_identity(category)
    }

    if not previously_scheduled_categories:
        return _priority_first(
            ranked_categories=ranked,
            priority_identities=priority_identities,
        )

    scheduled_identities = {
        _category_identity(category)
        for category in previously_scheduled_categories
        if _category_identity(category)
    }
    if not scheduled_identities:
        return _priority_first(
            ranked_categories=ranked,
            priority_identities=priority_identities,
        )

    if all(_category_identity(category) in scheduled_identities for category in ranked):
        return ranked

    unseen = []
    scheduled = []

    for category in ranked:
        identity = _category_identity(category)

        if identity not in scheduled_identities:
            unseen.append(category)
        else:
            scheduled.append(category)

    return tuple(
        _priority_first(
            ranked_categories=unseen,
            priority_identities=priority_identities,
        )
        + tuple(scheduled)
    )


def _priority_first(
    *,
    ranked_categories: Sequence[str],
    priority_identities: Set[str],
) -> Sequence[str]:
    """Move priority categories to the front while preserving deterministic rank."""

    if not priority_identities:
        return tuple(ranked_categories)

    priority = []
    other = []

    for category in ranked_categories:
        if _category_identity(category) in priority_identities:
            priority.append(category)
        else:
            other.append(category)

    return tuple(priority + other)


def _category_identity(category) -> str:
    return " ".join(str(category or "").strip().split()).casefold()


def coverage_metadata(
    *,
    coverage_state: CoverageState,
    accepted_categories: Sequence[str],
    skipped_categories: Sequence[str],
    continuation_order: Sequence[str],
    target_modules: int,
    buffer_modules: int,
    maximum_modules: int,
    category_required_modules: Mapping[str, int],
) -> Mapping[str, object]:
    """Build persistence metadata for the current Coverage Study Plan."""

    return {
        "professor_mode": PROFESSOR_MODE_COVERAGE,
        "coverage_status": (
            COVERAGE_STATUS_COMPLETE
            if coverage_state.complete
            else COVERAGE_STATUS_INCOMPLETE
        ),
        "coverage_complete": coverage_state.complete,
        "coverage_target_modules": target_modules,
        "coverage_buffer_modules": buffer_modules,
        "coverage_max_modules": maximum_modules,
        "coverage_covered_categories": list(coverage_state.covered_categories),
        "coverage_categories_requiring_coverage": list(
            coverage_state.categories_requiring_coverage
        ),
        "coverage_accepted_categories": list(accepted_categories),
        "coverage_skipped_categories": list(skipped_categories),
        "coverage_continuation_order": list(continuation_order),
        "coverage_category_required_modules": dict(category_required_modules),
    }
