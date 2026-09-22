"""Activity configuration logic for daily Study Planner sessions."""

from dataclasses import dataclass, replace
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
    USEFUL_QUIZ_MIN_QUESTIONS = 10
    USEFUL_QUIZ_TARGET_MAX_QUESTIONS = 15

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
        pending_quiz_candidates = []

        for strategy_index, activity_strategy in enumerate(daily_strategy.activities, start=1):
            allocation = daily_plan.planned_allocations[activity_strategy.allocation_index]

            if not allocation.selected_topics:
                continue

            for activity_type in self._activity_types_for_strategy(activity_strategy):
                if activity_type == ProfessorDailyActivityType.FLASHCARDS:
                    if pending_quiz_candidates and activity_strategy.activity_type != ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS:
                        activities.extend(
                            self._build_grouped_quiz_activities(
                                context=context,
                                daily_plan=daily_plan,
                                candidates=pending_quiz_candidates,
                            )
                        )
                        pending_quiz_candidates = []

                    activities.append(
                        self._build_flashcards_activity(
                            daily_plan=daily_plan,
                            allocation=allocation,
                            activity_strategy=activity_strategy,
                            strategy_index=strategy_index,
                        )
                    )
                elif activity_type == ProfessorDailyActivityType.QUIZ:
                    pending_quiz_candidates.append(
                        _QuizCompositionCandidate(
                            allocation=allocation,
                            activity_strategy=activity_strategy,
                            strategy_index=strategy_index,
                        )
                    )

        if pending_quiz_candidates:
            activities.extend(
                self._build_grouped_quiz_activities(
                    context=context,
                    daily_plan=daily_plan,
                    candidates=pending_quiz_candidates,
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

    def _build_grouped_quiz_activities(
        self,
        context: PlannerContext,
        daily_plan: DailyPlan,
        candidates: Sequence["_QuizCompositionCandidate"],
    ) -> Sequence[Activity]:
        """Create substantial quiz activities from compatible module allocations.

        The grouping scope is deliberately limited to allocations already present
        in the current DailyPlan/ProfessorModule. This preserves the existing
        module boundary and category-splitting behavior.
        """

        grouped = []
        index = 0
        candidates = tuple(candidates)

        while index < len(candidates):
            candidate = candidates[index]

            if candidate.activity_strategy.estimated_questions >= self.USEFUL_QUIZ_MIN_QUESTIONS:
                grouped.append((candidate,))
                index += 1
                continue

            group = [candidate]
            index += 1

            while index < len(candidates):
                next_candidate = candidates[index]
                next_topic_count = len(next_candidate.allocation.selected_topics)
                grouped_topic_count = self._group_topic_count((*group, next_candidate))
                grouped_question_count = self._group_question_count((*group, next_candidate))

                if grouped_topic_count > self.USEFUL_QUIZ_TARGET_MAX_QUESTIONS:
                    break

                group.append(next_candidate)
                index += 1

                if (
                    grouped_question_count >= self.USEFUL_QUIZ_TARGET_MAX_QUESTIONS
                    and next_topic_count > 0
                ):
                    break

            grouped.append(tuple(group))

        return tuple(
            self._build_grouped_quiz_activity(
                context=context,
                daily_plan=daily_plan,
                candidates=group,
                group_index=group_index,
            )
            for group_index, group in enumerate(grouped, start=1)
        )

    def _build_grouped_quiz_activity(
        self,
        context: PlannerContext,
        daily_plan: DailyPlan,
        candidates: Sequence["_QuizCompositionCandidate"],
        group_index: int,
    ) -> Activity:
        """Create one quiz activity from one or more category allocations."""

        selected_topics = tuple(
            topic
            for candidate in candidates
            for topic in self._topics_with_category(
                candidate.allocation.selected_topics,
                candidate.allocation.category,
            )
        )
        categories = tuple(
            dict.fromkeys(
                candidate.allocation.category
                for candidate in candidates
                if candidate.allocation.category
            )
        )
        question_count = self._group_question_count(candidates)

        return Activity(
            id=self._grouped_quiz_activity_id(
                daily_plan=daily_plan,
                candidates=candidates,
                group_index=group_index,
            ),
            type=ActivityType.QUIZ,
            configuration=ActivityConfiguration(
                category=categories[0] if len(categories) == 1 else None,
                selected_topics=selected_topics,
                estimated_duration_minutes=self._quiz_duration_minutes(
                    context=context,
                    question_count=question_count,
                    fallback_minutes=sum(
                        candidate.allocation.estimated_duration_minutes
                        for candidate in candidates
                    ),
                ),
                num_questions=question_count,
                question_style=context.preferences.question_style or self.DEFAULT_QUESTION_STYLE,
                difficulty=self.DEFAULT_DIFFICULTY,
            ),
        )

    def _grouped_quiz_activity_id(
        self,
        daily_plan: DailyPlan,
        candidates: Sequence["_QuizCompositionCandidate"],
        group_index: int,
    ) -> str:
        if len(candidates) == 1:
            return f"{daily_plan.id}-strategy-{candidates[0].strategy_index}-quiz"

        first_index = candidates[0].strategy_index
        last_index = candidates[-1].strategy_index
        return f"{daily_plan.id}-quiz-group-{group_index}-strategies-{first_index}-{last_index}"

    def _group_topic_count(
        self,
        candidates: Sequence["_QuizCompositionCandidate"],
    ) -> int:
        return sum(
            len(candidate.allocation.selected_topics)
            for candidate in candidates
        )

    def _group_question_count(
        self,
        candidates: Sequence["_QuizCompositionCandidate"],
    ) -> int:
        estimated_questions = sum(
            max(0, int(candidate.activity_strategy.estimated_questions or 0))
            for candidate in candidates
        )
        topic_count = self._group_topic_count(candidates)

        if len(candidates) == 1:
            return max(topic_count, estimated_questions)

        return max(
            topic_count,
            min(estimated_questions, self.USEFUL_QUIZ_TARGET_MAX_QUESTIONS),
        )

    def _quiz_duration_minutes(
        self,
        context: PlannerContext,
        question_count: int,
        fallback_minutes: float,
    ) -> float:
        question_pace_seconds = context.preferences.question_pace_seconds or 0
        if question_pace_seconds <= 0:
            return fallback_minutes

        return round((question_count * question_pace_seconds) / 60, 4)

    def _topics_with_category(
        self,
        selected_topics,
        category,
    ) -> Sequence:
        return tuple(
            topic
            if getattr(topic, "category", None)
            else replace(topic, category=category)
            for topic in selected_topics
        )


@dataclass(frozen=True)
class _QuizCompositionCandidate:
    """Internal activity-composition candidate for one quiz-capable allocation."""

    allocation: object
    activity_strategy: ProfessorDailyActivityStrategy
    strategy_index: int
