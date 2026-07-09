"""Deterministic category selection logic for the Study Planner.

The selector ranks project categories from already-computed analytics. It does
not call AI services, access the database, or depend on FastAPI/main.py.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence

from .priority_policy import calculate_priority_score


class CategorySelectionReason(str, Enum):
    """Human-readable reasons a category was prioritized."""

    LOW_PERFORMANCE = "LOW_PERFORMANCE"
    NOT_RECENTLY_REVIEWED = "NOT_RECENTLY_REVIEWED"
    LOW_COVERAGE = "LOW_COVERAGE"
    HIGH_PRIORITY = "HIGH_PRIORITY"


@dataclass(frozen=True)
class CategoryAnalytics:
    """Analytics signals available for one category.

    Values are expected to be normalized where possible:
    - accuracy: 0.0 to 1.0, where lower means weaker performance.
    - coverage: 0.0 to 1.0, where lower means less study coverage.
    - days_since_review: higher means the category is more stale.
    - priority_weight: 1.0 is neutral; higher values make a category more urgent.
    """

    accuracy: Optional[float] = None
    coverage: Optional[float] = None
    days_since_review: Optional[int] = None
    priority_weight: float = 1.0


@dataclass(frozen=True)
class CategoryPriority:
    """Ranked category priority produced by the selector."""

    category: str
    priority_score: float
    reasons: Sequence[CategorySelectionReason] = field(default_factory=tuple)


class CategorySelector:
    """Rank categories from analytics using deterministic scoring."""

    LOW_PERFORMANCE_THRESHOLD = 0.70
    LOW_COVERAGE_THRESHOLD = 0.70
    STALE_REVIEW_DAYS = 14
    HIGH_PRIORITY_WEIGHT = 1.0

    def select_categories(
        self,
        project_categories: Sequence[str],
        category_analytics: Mapping[str, CategoryAnalytics],
        planner_preferences: Optional[Mapping[str, object]] = None,
    ) -> Sequence[CategoryPriority]:
        """Return categories sorted by descending planning priority.

        Planner preferences are accepted to keep the interface ready for future
        personalization, but they do not affect V1 scoring.
        """

        del planner_preferences

        priorities = [
            self._build_priority(category, category_analytics.get(category))
            for category in project_categories
        ]
        return self._rank_priorities(priorities)

    def _build_priority(
        self,
        category: str,
        analytics: Optional[CategoryAnalytics],
    ) -> CategoryPriority:
        """Calculate score and explanations for a single category."""

        analytics = analytics or CategoryAnalytics()
        return CategoryPriority(
            category=category,
            priority_score=calculate_priority_score(
                accuracy=analytics.accuracy,
                coverage=analytics.coverage,
                days_since_review=analytics.days_since_review,
                priority_weight=analytics.priority_weight,
            ),
            reasons=self._generate_reasons(analytics),
        )

    def _generate_reasons(
        self,
        analytics: CategoryAnalytics,
    ) -> Sequence[CategorySelectionReason]:
        """Generate deterministic explanations for a category score."""

        reasons = []

        if analytics.accuracy is not None and analytics.accuracy < self.LOW_PERFORMANCE_THRESHOLD:
            reasons.append(CategorySelectionReason.LOW_PERFORMANCE)

        if (
            analytics.days_since_review is not None
            and analytics.days_since_review >= self.STALE_REVIEW_DAYS
        ):
            reasons.append(CategorySelectionReason.NOT_RECENTLY_REVIEWED)

        if analytics.coverage is not None and analytics.coverage < self.LOW_COVERAGE_THRESHOLD:
            reasons.append(CategorySelectionReason.LOW_COVERAGE)

        if analytics.priority_weight > self.HIGH_PRIORITY_WEIGHT:
            reasons.append(CategorySelectionReason.HIGH_PRIORITY)

        return tuple(reasons)

    def _rank_priorities(
        self,
        priorities: Sequence[CategoryPriority],
    ) -> Sequence[CategoryPriority]:
        """Sort priorities by score, then category name for stable tie-breaking."""

        return tuple(
            sorted(
                priorities,
                key=lambda priority: (-priority.priority_score, priority.category.casefold()),
            )
        )
