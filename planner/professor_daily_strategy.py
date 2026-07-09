"""Deterministic Professor daily strategy decisions.

This module decides activity type, depth, and target exercise size for each
scheduled category in a daily Planner session. It emits machine-readable codes
only: no natural language, no LLM calls, no persistence, and no frontend data.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence, cast

from .category_selector import CategoryAnalytics
from .planner_models import DailyPlan, PlannerContext
from .professor_strategy import (
    ProfessorCategoryStrategy,
    ProfessorCategoryStrategyCode,
    ProfessorDepthCode,
    ProfessorWeeklyStrategy,
)


class ProfessorDailyActivityType(str, Enum):
    """Professor-selected learning activity for one scheduled category."""

    QUIZ = "QUIZ"
    FLASHCARDS = "FLASHCARDS"
    QUIZ_PLUS_FLASHCARDS = "QUIZ_PLUS_FLASHCARDS"


class ProfessorDailyGoalCode(str, Enum):
    """Machine-readable daily session goal code."""

    EXPLORE_AND_MEASURE = "EXPLORE_AND_MEASURE"
    ASSESS_KNOWLEDGE = "ASSESS_KNOWLEDGE"
    REINFORCE_LEARNING = "REINFORCE_LEARNING"
    REVIEW_RETENTION = "REVIEW_RETENTION"
    MIXED_OBJECTIVES = "MIXED_OBJECTIVES"


class ProfessorDailyReasoningCode(str, Enum):
    """Machine-readable reason for a daily activity decision."""

    EXPLORE_NEW_AREA = "EXPLORE_NEW_AREA"
    ASSESS_KNOWLEDGE = "ASSESS_KNOWLEDGE"
    LOW_ACCURACY = "LOW_ACCURACY"
    HIGH_PRIORITY = "HIGH_PRIORITY"
    LONG_TIME_WITHOUT_REVIEW = "LONG_TIME_WITHOUT_REVIEW"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    PERFORMANCE_UNSTABLE = "PERFORMANCE_UNSTABLE"


class ProfessorDailySummaryCode(str, Enum):
    """Machine-readable daily summary code for future narrative generation."""

    EVIDENCE_GATHERING = "EVIDENCE_GATHERING"
    PRACTICE_FOCUSED = "PRACTICE_FOCUSED"
    REVIEW_FOCUSED = "REVIEW_FOCUSED"
    MULTI_CATEGORY_SESSION = "MULTI_CATEGORY_SESSION"


@dataclass(frozen=True)
class ProfessorDailyActivityStrategy:
    """Deterministic Professor strategy for one scheduled allocation."""

    category: str
    activity_type: ProfessorDailyActivityType
    depth: ProfessorDepthCode
    estimated_questions: int = 0
    estimated_flashcards: int = 0
    reasoning_codes: Sequence[ProfessorDailyReasoningCode] = field(default_factory=tuple)
    allocation_index: int = 0


@dataclass(frozen=True)
class ProfessorDailyStrategy:
    """Deterministic daily strategy for a scheduled Planner session."""

    daily_goal_code: ProfessorDailyGoalCode
    activities: Sequence[ProfessorDailyActivityStrategy] = field(default_factory=tuple)
    summary_codes: Sequence[ProfessorDailySummaryCode] = field(default_factory=tuple)


class ProfessorDailyStrategyBuilder:
    """Build deterministic daily Professor strategy from scheduled allocations."""

    UNSTABLE_REVIEW_ACCURACY_THRESHOLD = 0.75

    QUIZ_SIZE_BY_DEPTH = {
        ProfessorDepthCode.LIGHT: 6,
        ProfessorDepthCode.NORMAL: 10,
        ProfessorDepthCode.DEEP: 15,
    }
    FLASHCARD_SIZE_BY_DEPTH = {
        ProfessorDepthCode.LIGHT: 8,
        ProfessorDepthCode.NORMAL: 12,
        ProfessorDepthCode.DEEP: 18,
    }

    def build_strategy(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        daily_session: DailyPlan,
    ) -> ProfessorDailyStrategy:
        """Return deterministic strategy for one scheduled daily session."""

        weekly_by_category = {
            category_strategy.category: category_strategy
            for category_strategy in weekly_strategy.category_strategies
        }
        analytics = cast(Mapping[str, CategoryAnalytics], context.analytics)

        activities = tuple(
            self._build_activity_strategy(
                allocation=allocation,
                allocation_index=index,
                weekly_category_strategy=weekly_by_category.get(allocation.category),
                analytics=analytics.get(allocation.category),
            )
            for index, allocation in enumerate(daily_session.planned_allocations)
            if allocation.selected_topics
        )

        return ProfessorDailyStrategy(
            daily_goal_code=self._determine_daily_goal(activities),
            activities=activities,
            summary_codes=self._determine_summary_codes(activities),
        )

    def _build_activity_strategy(
        self,
        allocation,
        allocation_index: int,
        weekly_category_strategy: ProfessorCategoryStrategy,
        analytics: CategoryAnalytics = None,
    ) -> ProfessorDailyActivityStrategy:
        """Return one activity strategy for a scheduled category allocation."""

        weekly_category_strategy = weekly_category_strategy or ProfessorCategoryStrategy(
            category=allocation.category,
            strategy=ProfessorCategoryStrategyCode.EXPLORE,
            depth=ProfessorDepthCode.NORMAL,
            reasoning_code=None,
        )
        analytics = analytics or CategoryAnalytics()

        activity_type = self._select_activity_type(
            weekly_category_strategy=weekly_category_strategy,
            analytics=analytics,
        )
        reasoning_codes = self._build_reasoning_codes(
            weekly_category_strategy=weekly_category_strategy,
            analytics=analytics,
            activity_type=activity_type,
        )

        return ProfessorDailyActivityStrategy(
            category=allocation.category,
            activity_type=activity_type,
            depth=weekly_category_strategy.depth,
            estimated_questions=self._estimate_questions(
                activity_type=activity_type,
                depth=weekly_category_strategy.depth,
            ),
            estimated_flashcards=self._estimate_flashcards(
                activity_type=activity_type,
                depth=weekly_category_strategy.depth,
            ),
            reasoning_codes=reasoning_codes,
            allocation_index=allocation_index,
        )

    def _select_activity_type(
        self,
        weekly_category_strategy: ProfessorCategoryStrategy,
        analytics: CategoryAnalytics,
    ) -> ProfessorDailyActivityType:
        """Select a deterministic activity type from weekly strategy codes."""

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.EXPLORE:
            return ProfessorDailyActivityType.QUIZ

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.ASSESSMENT:
            return ProfessorDailyActivityType.QUIZ

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.REINFORCE:
            return ProfessorDailyActivityType.FLASHCARDS

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.REVIEW:
            if self._performance_unstable(analytics):
                return ProfessorDailyActivityType.QUIZ
            return ProfessorDailyActivityType.FLASHCARDS

        return ProfessorDailyActivityType.QUIZ

    def _build_reasoning_codes(
        self,
        weekly_category_strategy: ProfessorCategoryStrategy,
        analytics: CategoryAnalytics,
        activity_type: ProfessorDailyActivityType,
    ) -> Sequence[ProfessorDailyReasoningCode]:
        """Return deterministic reasoning codes for one activity decision."""

        codes = []

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.EXPLORE:
            codes.append(ProfessorDailyReasoningCode.EXPLORE_NEW_AREA)

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.ASSESSMENT:
            codes.append(ProfessorDailyReasoningCode.ASSESS_KNOWLEDGE)

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.REINFORCE:
            codes.append(ProfessorDailyReasoningCode.LOW_ACCURACY)

        if weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.REVIEW:
            codes.append(ProfessorDailyReasoningCode.LONG_TIME_WITHOUT_REVIEW)

        if analytics.accuracy is None and analytics.coverage is None:
            codes.append(ProfessorDailyReasoningCode.INSUFFICIENT_DATA)

        if analytics.priority_weight > 1.0:
            codes.append(ProfessorDailyReasoningCode.HIGH_PRIORITY)

        if (
            weekly_category_strategy.strategy == ProfessorCategoryStrategyCode.REVIEW
            and activity_type == ProfessorDailyActivityType.QUIZ
        ):
            codes.append(ProfessorDailyReasoningCode.PERFORMANCE_UNSTABLE)

        return tuple(dict.fromkeys(codes))

    def _estimate_questions(
        self,
        activity_type: ProfessorDailyActivityType,
        depth: ProfessorDepthCode,
    ) -> int:
        """Return target quiz size for a Professor-selected activity."""

        if activity_type in {
            ProfessorDailyActivityType.QUIZ,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        }:
            return self.QUIZ_SIZE_BY_DEPTH[depth]

        return 0

    def _estimate_flashcards(
        self,
        activity_type: ProfessorDailyActivityType,
        depth: ProfessorDepthCode,
    ) -> int:
        """Return target flashcard size for a Professor-selected activity."""

        if activity_type in {
            ProfessorDailyActivityType.FLASHCARDS,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        }:
            return self.FLASHCARD_SIZE_BY_DEPTH[depth]

        return 0

    def _performance_unstable(
        self,
        analytics: CategoryAnalytics,
    ) -> bool:
        """Return whether review should use quiz because performance is unstable."""

        return (
            analytics.accuracy is not None
            and analytics.accuracy < self.UNSTABLE_REVIEW_ACCURACY_THRESHOLD
        )

    def _determine_daily_goal(
        self,
        activities: Sequence[ProfessorDailyActivityStrategy],
    ) -> ProfessorDailyGoalCode:
        """Return deterministic daily goal code from activity strategies."""

        activity_types = {activity.activity_type for activity in activities}
        reasoning_codes = {
            code
            for activity in activities
            for code in activity.reasoning_codes
        }

        if len(activity_types) > 1:
            return ProfessorDailyGoalCode.MIXED_OBJECTIVES

        if ProfessorDailyReasoningCode.EXPLORE_NEW_AREA in reasoning_codes:
            return ProfessorDailyGoalCode.EXPLORE_AND_MEASURE

        if ProfessorDailyReasoningCode.LOW_ACCURACY in reasoning_codes:
            return ProfessorDailyGoalCode.REINFORCE_LEARNING

        if ProfessorDailyReasoningCode.LONG_TIME_WITHOUT_REVIEW in reasoning_codes:
            return ProfessorDailyGoalCode.REVIEW_RETENTION

        return ProfessorDailyGoalCode.ASSESS_KNOWLEDGE

    def _determine_summary_codes(
        self,
        activities: Sequence[ProfessorDailyActivityStrategy],
    ) -> Sequence[ProfessorDailySummaryCode]:
        """Return deterministic summary codes for future Professor narration."""

        codes = []

        if len(activities) > 1:
            codes.append(ProfessorDailySummaryCode.MULTI_CATEGORY_SESSION)

        if any(activity.activity_type == ProfessorDailyActivityType.QUIZ for activity in activities):
            codes.append(ProfessorDailySummaryCode.EVIDENCE_GATHERING)

        if any(
            activity.activity_type == ProfessorDailyActivityType.FLASHCARDS
            for activity in activities
        ):
            codes.append(ProfessorDailySummaryCode.PRACTICE_FOCUSED)

        if any(
            ProfessorDailyReasoningCode.LONG_TIME_WITHOUT_REVIEW in activity.reasoning_codes
            for activity in activities
        ):
            codes.append(ProfessorDailySummaryCode.REVIEW_FOCUSED)

        return tuple(dict.fromkeys(codes))
