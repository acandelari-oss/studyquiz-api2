"""Deterministic Professor weekly strategy primitives.

This module is the first non-narrative Professor layer. It analyzes an existing
PlannerContext and produces machine-readable strategy codes only. It does not
generate natural language, call AI services, mutate Week generation, or affect
Planner scheduling/activity creation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence, cast

from .category_selector import CategoryAnalytics, CategorySelector
from .planner_models import PlannerContext


class ProfessorCategoryStrategyCode(str, Enum):
    """Deterministic instructional strategy selected for one category."""

    EXPLORE = "explore"
    ASSESSMENT = "assessment"
    REINFORCE = "reinforce"
    REVIEW = "review"


class ProfessorDepthCode(str, Enum):
    """Deterministic depth level for one category."""

    LIGHT = "light"
    NORMAL = "normal"
    DEEP = "deep"


class ProfessorReasoningCode(str, Enum):
    """Machine-readable reason for a category strategy decision."""

    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"
    LOW_COVERAGE = "LOW_COVERAGE"
    LOW_PERFORMANCE = "LOW_PERFORMANCE"
    NOT_RECENTLY_REVIEWED = "NOT_RECENTLY_REVIEWED"
    HIGH_PRIORITY = "HIGH_PRIORITY"
    STABLE_PERFORMANCE = "STABLE_PERFORMANCE"


class ProfessorWeeklyGoalCode(str, Enum):
    """Machine-readable weekly goal selected from project-level evidence."""

    CALIBRATE_COVERAGE = "CALIBRATE_COVERAGE"
    IMPROVE_WEAK_AREAS = "IMPROVE_WEAK_AREAS"
    REVIEW_RETENTION = "REVIEW_RETENTION"
    ASSESS_AND_MAINTAIN = "ASSESS_AND_MAINTAIN"


@dataclass(frozen=True)
class ProfessorCategoryStrategy:
    """Deterministic Professor strategy for one category."""

    category: str
    strategy: ProfessorCategoryStrategyCode
    depth: ProfessorDepthCode
    reasoning_code: ProfessorReasoningCode
    priority_score: float = 0.0


@dataclass(frozen=True)
class ProfessorWeeklyStrategy:
    """Deterministic strategy object for a future Professor reasoning layer."""

    weekly_goal_code: ProfessorWeeklyGoalCode
    category_strategies: Sequence[ProfessorCategoryStrategy] = field(default_factory=tuple)
    priority_categories: Sequence[str] = field(default_factory=tuple)
    secondary_categories: Sequence[str] = field(default_factory=tuple)


class ProfessorWeeklyStrategyBuilder:
    """Build deterministic weekly Professor strategy from PlannerContext."""

    LOW_COVERAGE_THRESHOLD = 0.50
    LOW_PERFORMANCE_THRESHOLD = 0.60
    STALE_REVIEW_DAYS = 14
    HIGH_PRIORITY_WEIGHT = 1.0
    PRIORITY_CATEGORY_LIMIT = 3

    def __init__(
        self,
        category_selector: Optional[CategorySelector] = None,
    ) -> None:
        """Create a strategy builder with deterministic category ranking."""

        self.category_selector = category_selector or CategorySelector()

    def build_strategy(self, context: PlannerContext) -> ProfessorWeeklyStrategy:
        """Return a deterministic strategy object for the provided context."""

        analytics = cast(Mapping[str, CategoryAnalytics], context.analytics)
        ranked_categories = self.category_selector.select_categories(
            project_categories=context.categories,
            category_analytics=analytics,
            planner_preferences={},
        )

        category_strategies = tuple(
            self._build_category_strategy(
                category=priority.category,
                analytics=analytics.get(priority.category),
                topic_count=len(context.topics_by_category.get(priority.category, ())),
                priority_score=priority.priority_score,
            )
            for priority in ranked_categories
        )

        priority_categories = tuple(
            strategy.category
            for strategy in category_strategies[: self.PRIORITY_CATEGORY_LIMIT]
        )
        secondary_categories = tuple(
            strategy.category
            for strategy in category_strategies[self.PRIORITY_CATEGORY_LIMIT :]
        )

        return ProfessorWeeklyStrategy(
            weekly_goal_code=self._determine_weekly_goal(category_strategies),
            category_strategies=category_strategies,
            priority_categories=priority_categories,
            secondary_categories=secondary_categories,
        )

    def _build_category_strategy(
        self,
        category: str,
        analytics: Optional[CategoryAnalytics],
        topic_count: int,
        priority_score: float,
    ) -> ProfessorCategoryStrategy:
        """Return deterministic strategy/depth/reasoning for one category."""

        analytics = analytics or CategoryAnalytics()
        reasoning_code = self._determine_reasoning_code(analytics)
        strategy = self._determine_strategy(reasoning_code)
        depth = self._determine_depth(
            analytics=analytics,
            reasoning_code=reasoning_code,
            topic_count=topic_count,
        )

        return ProfessorCategoryStrategy(
            category=category,
            strategy=strategy,
            depth=depth,
            reasoning_code=reasoning_code,
            priority_score=priority_score,
        )

    def _determine_reasoning_code(
        self,
        analytics: CategoryAnalytics,
    ) -> ProfessorReasoningCode:
        """Return the dominant deterministic reason for a category decision."""

        if analytics.coverage is None and analytics.accuracy is None:
            return ProfessorReasoningCode.INSUFFICIENT_EVIDENCE

        if analytics.coverage is not None and analytics.coverage < self.LOW_COVERAGE_THRESHOLD:
            return ProfessorReasoningCode.LOW_COVERAGE

        if analytics.accuracy is not None and analytics.accuracy < self.LOW_PERFORMANCE_THRESHOLD:
            return ProfessorReasoningCode.LOW_PERFORMANCE

        if (
            analytics.days_since_review is not None
            and analytics.days_since_review >= self.STALE_REVIEW_DAYS
        ):
            return ProfessorReasoningCode.NOT_RECENTLY_REVIEWED

        if analytics.priority_weight > self.HIGH_PRIORITY_WEIGHT:
            return ProfessorReasoningCode.HIGH_PRIORITY

        return ProfessorReasoningCode.STABLE_PERFORMANCE

    def _determine_strategy(
        self,
        reasoning_code: ProfessorReasoningCode,
    ) -> ProfessorCategoryStrategyCode:
        """Map deterministic reasoning codes to strategy codes."""

        if reasoning_code in {
            ProfessorReasoningCode.INSUFFICIENT_EVIDENCE,
            ProfessorReasoningCode.LOW_COVERAGE,
        }:
            return ProfessorCategoryStrategyCode.EXPLORE

        if reasoning_code == ProfessorReasoningCode.LOW_PERFORMANCE:
            return ProfessorCategoryStrategyCode.REINFORCE

        if reasoning_code == ProfessorReasoningCode.NOT_RECENTLY_REVIEWED:
            return ProfessorCategoryStrategyCode.REVIEW

        return ProfessorCategoryStrategyCode.ASSESSMENT

    def _determine_depth(
        self,
        analytics: CategoryAnalytics,
        reasoning_code: ProfessorReasoningCode,
        topic_count: int,
    ) -> ProfessorDepthCode:
        """Return deterministic depth code from evidence and category size."""

        if topic_count <= 1:
            return ProfessorDepthCode.LIGHT

        if reasoning_code in {
            ProfessorReasoningCode.INSUFFICIENT_EVIDENCE,
            ProfessorReasoningCode.LOW_COVERAGE,
            ProfessorReasoningCode.LOW_PERFORMANCE,
        }:
            return ProfessorDepthCode.DEEP

        if (
            analytics.accuracy is not None
            and analytics.accuracy >= 0.85
            and analytics.coverage is not None
            and analytics.coverage >= 0.85
        ):
            return ProfessorDepthCode.LIGHT

        return ProfessorDepthCode.NORMAL

    def _determine_weekly_goal(
        self,
        category_strategies: Sequence[ProfessorCategoryStrategy],
    ) -> ProfessorWeeklyGoalCode:
        """Return the dominant deterministic weekly goal code."""

        reasoning_codes = tuple(
            strategy.reasoning_code
            for strategy in category_strategies
        )

        if any(
            code in {
                ProfessorReasoningCode.INSUFFICIENT_EVIDENCE,
                ProfessorReasoningCode.LOW_COVERAGE,
            }
            for code in reasoning_codes
        ):
            return ProfessorWeeklyGoalCode.CALIBRATE_COVERAGE

        if ProfessorReasoningCode.LOW_PERFORMANCE in reasoning_codes:
            return ProfessorWeeklyGoalCode.IMPROVE_WEAK_AREAS

        if ProfessorReasoningCode.NOT_RECENTLY_REVIEWED in reasoning_codes:
            return ProfessorWeeklyGoalCode.REVIEW_RETENTION

        return ProfessorWeeklyGoalCode.ASSESS_AND_MAINTAIN
