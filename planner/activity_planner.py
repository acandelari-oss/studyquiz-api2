"""Activity configuration logic for daily Study Planner sessions."""

from dataclasses import replace
from typing import Sequence

from .planner_models import (
    Activity,
    ActivityConfiguration,
    DailyPlan,
    PlannerContext,
)
from .planner_state import ActivityType
from .professor_daily_strategy import (
    ProfessorDailyActivityType,
    ProfessorDailyActivityStrategy,
    ProfessorDailyStrategy,
)


class ActivityPlanner:
    """Configure ordered learning activities for each daily plan.

    This planner only creates activity configuration. It does not execute
    quizzes, generate flashcards, call endpoints, or persist anything.
    """

    DEFAULT_QUESTION_STYLE = "balanced"
    DEFAULT_DIFFICULTY = "medium"

    def plan_daily_plans(
        self,
        context: PlannerContext,
        daily_plans: Sequence[DailyPlan],
        daily_strategies: Sequence[ProfessorDailyStrategy],
    ) -> Sequence[DailyPlan]:
        """Return DailyPlans with configured activity objects."""

        return tuple(
            self.plan_daily_plan(
                context=context,
                daily_plan=daily_plan,
                daily_strategy=daily_strategy,
            )
            for daily_plan, daily_strategy in zip(daily_plans, daily_strategies)
        )

    def plan_daily_plan(
        self,
        context: PlannerContext,
        daily_plan: DailyPlan,
        daily_strategy: ProfessorDailyStrategy,
    ) -> DailyPlan:
        """Translate one ProfessorDailyStrategy into executable activities."""

        activities = []

        for strategy_index, activity_strategy in enumerate(daily_strategy.activities, start=1):
            allocation = daily_plan.planned_allocations[activity_strategy.allocation_index]

            if not allocation.selected_topics:
                continue

            for activity_type in self._activity_types_for_strategy(activity_strategy):
                if activity_type == ProfessorDailyActivityType.FLASHCARDS:
                    activities.append(
                        self._build_flashcards_activity(
                            daily_plan=daily_plan,
                            allocation=allocation,
                            activity_strategy=activity_strategy,
                            strategy_index=strategy_index,
                        )
                    )
                elif activity_type == ProfessorDailyActivityType.QUIZ:
                    activities.append(
                        self._build_quiz_activity(
                            context=context,
                            daily_plan=daily_plan,
                            allocation=allocation,
                            activity_strategy=activity_strategy,
                            strategy_index=strategy_index,
                        )
                    )

        return replace(daily_plan, activities=tuple(activities))

    def _activity_types_for_strategy(
        self,
        activity_strategy: ProfessorDailyActivityStrategy,
    ) -> Sequence[ProfessorDailyActivityType]:
        """Return executable activity type sequence for a Professor decision."""

        if activity_strategy.activity_type == ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS:
            return (
                ProfessorDailyActivityType.FLASHCARDS,
                ProfessorDailyActivityType.QUIZ,
            )

        return (activity_strategy.activity_type,)

    def _build_flashcards_activity(
        self,
        daily_plan: DailyPlan,
        allocation,
        activity_strategy: ProfessorDailyActivityStrategy,
        strategy_index: int,
    ) -> Activity:
        """Create a Flashcards activity for one planned allocation."""

        selected_topics = tuple(allocation.selected_topics)

        return Activity(
            id=f"{daily_plan.id}-strategy-{strategy_index}-flashcards",
            type=ActivityType.FLASHCARDS,
            configuration=ActivityConfiguration(
                category=allocation.category,
                selected_topics=selected_topics,
                estimated_duration_minutes=allocation.estimated_duration_minutes,
                num_cards=activity_strategy.estimated_flashcards,
            ),
        )

    def _build_quiz_activity(
        self,
        context: PlannerContext,
        daily_plan: DailyPlan,
        allocation,
        activity_strategy: ProfessorDailyActivityStrategy,
        strategy_index: int,
    ) -> Activity:
        """Create a Quiz activity for one planned allocation."""

        selected_topics = tuple(allocation.selected_topics)

        return Activity(
            id=f"{daily_plan.id}-strategy-{strategy_index}-quiz",
            type=ActivityType.QUIZ,
            configuration=ActivityConfiguration(
                category=allocation.category,
                selected_topics=selected_topics,
                estimated_duration_minutes=allocation.estimated_duration_minutes,
                num_questions=activity_strategy.estimated_questions,
                question_style=context.preferences.question_style or self.DEFAULT_QUESTION_STYLE,
                difficulty=self.DEFAULT_DIFFICULTY,
            ),
        )
