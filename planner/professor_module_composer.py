"""Budget-aware Professor module composition.

This module owns the first Professor-level composition pass: it turns ordered
category allocations and deterministic Professor activity strategies into the
actual ordered Study Plan modules. It does not generate activities, call
learning endpoints, persist data, or produce natural language.
"""

from dataclasses import dataclass, field, replace
from datetime import date
from math import floor
from typing import Optional, Sequence

from .planner_models import DailyPlan, PlannerContext
from .professor_daily_strategy import (
    ProfessorDailyActivityStrategy,
    ProfessorDailyActivityType,
    ProfessorDailyStrategy,
    ProfessorDailyStrategyBuilder,
)
from .professor_strategy import ProfessorWeeklyGoalCode, ProfessorWeeklyStrategy
from .session_allocator import CategoryAllocation


@dataclass(frozen=True)
class ProfessorModule:
    """One Professor-composed Study Plan module."""

    module_index: int
    allocations: Sequence[CategoryAllocation] = field(default_factory=tuple)
    daily_strategy: ProfessorDailyStrategy = field(default_factory=ProfessorDailyStrategy)
    estimated_duration_minutes: float = 0.0


class ProfessorModuleComposer:
    """Compose budget-respecting modules from Professor strategy decisions."""

    SECONDS_PER_MINUTE = 60
    FLASHCARD_SECONDS = 30
    DEFAULT_NEGATIVE_BUFFER_MINUTES = 2
    MAX_CATEGORIES_PER_MODULE = 3

    def __init__(
        self,
        daily_strategy_builder: Optional[ProfessorDailyStrategyBuilder] = None,
        negative_buffer_minutes: int = DEFAULT_NEGATIVE_BUFFER_MINUTES,
    ) -> None:
        self.daily_strategy_builder = daily_strategy_builder or ProfessorDailyStrategyBuilder()
        self.negative_buffer_minutes = max(0, int(negative_buffer_minutes or 0))

    def compose_modules(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        allocations: Sequence[CategoryAllocation],
        max_visible_modules: Optional[int] = None,
    ) -> Sequence[ProfessorModule]:
        """Return ordered modules while respecting the preferred duration."""

        if context.planning_budget_minutes <= 0:
            return ()

        modules = []
        current_allocations = []
        current_strategies = []
        current_duration = 0.0
        pending_allocations = list(allocations)

        while pending_allocations:
            if max_visible_modules is not None and len(modules) >= max_visible_modules:
                break

            if len(current_allocations) >= self.MAX_CATEGORIES_PER_MODULE:
                modules.append(
                    self._build_module(
                        module_index=len(modules) + 1,
                        allocations=current_allocations,
                        strategies=current_strategies,
                        estimated_duration_minutes=current_duration,
                    )
                )
                current_allocations = []
                current_strategies = []
                current_duration = 0.0
                continue

            allocation = pending_allocations[0]
            if self._would_combine_student_priority_categories(
                context=context,
                current_allocations=current_allocations,
                next_allocation=allocation,
            ):
                filler = self._pop_next_compatible_filler(
                    context=context,
                    weekly_strategy=weekly_strategy,
                    pending_allocations=pending_allocations,
                    current_allocations=current_allocations,
                    available_budget_minutes=(
                        context.planning_budget_minutes - current_duration
                    ),
                )

                if filler is not None:
                    (
                        adjusted_strategy,
                        adjusted_allocation,
                        estimated_duration,
                    ) = filler
                    current_allocations.append(adjusted_allocation)
                    current_strategies.append(adjusted_strategy)
                    current_duration += estimated_duration

                    if self._coverage_module_complete(
                        context=context,
                        weekly_strategy=weekly_strategy,
                        current_allocations=current_allocations,
                        current_duration_minutes=current_duration,
                    ):
                        modules.append(
                            self._build_module(
                                module_index=len(modules) + 1,
                                allocations=current_allocations,
                                strategies=current_strategies,
                                estimated_duration_minutes=current_duration,
                            )
                        )
                        current_allocations = []
                        current_strategies = []
                        current_duration = 0.0

                    continue

                modules.append(
                    self._build_module(
                        module_index=len(modules) + 1,
                        allocations=current_allocations,
                        strategies=current_strategies,
                        estimated_duration_minutes=current_duration,
                    )
                )
                current_allocations = []
                current_strategies = []
                current_duration = 0.0
                continue

            remaining_budget = context.planning_budget_minutes - current_duration
            candidate = self._build_budgeted_activity_strategy(
                context=context,
                weekly_strategy=weekly_strategy,
                allocation=allocation,
                available_budget_minutes=remaining_budget,
                allocation_index=len(current_allocations),
            )

            if candidate is None:
                if current_allocations:
                    filler = self._pop_next_compatible_filler(
                        context=context,
                        weekly_strategy=weekly_strategy,
                        pending_allocations=pending_allocations,
                        current_allocations=current_allocations,
                        available_budget_minutes=remaining_budget,
                    )

                    if filler is not None:
                        (
                            adjusted_strategy,
                            adjusted_allocation,
                            estimated_duration,
                        ) = filler
                        current_allocations.append(adjusted_allocation)
                        current_strategies.append(adjusted_strategy)
                        current_duration += estimated_duration

                        if self._coverage_module_complete(
                            context=context,
                            weekly_strategy=weekly_strategy,
                            current_allocations=current_allocations,
                            current_duration_minutes=current_duration,
                        ):
                            modules.append(
                                self._build_module(
                                    module_index=len(modules) + 1,
                                    allocations=current_allocations,
                                    strategies=current_strategies,
                                    estimated_duration_minutes=current_duration,
                                )
                            )
                            current_allocations = []
                            current_strategies = []
                            current_duration = 0.0

                        continue

                    modules.append(
                        self._build_module(
                            module_index=len(modules) + 1,
                            allocations=current_allocations,
                            strategies=current_strategies,
                            estimated_duration_minutes=current_duration,
                        )
                    )
                    current_allocations = []
                    current_strategies = []
                    current_duration = 0.0
                    continue

                candidate = self._build_budgeted_activity_strategy(
                    context=context,
                    weekly_strategy=weekly_strategy,
                    allocation=allocation,
                    available_budget_minutes=context.planning_budget_minutes,
                    allocation_index=0,
                )

                if candidate is None:
                    pending_allocations.pop(0)
                    continue

            adjusted_strategy, adjusted_allocation, estimated_duration = candidate
            current_allocations.append(adjusted_allocation)
            current_strategies.append(adjusted_strategy)
            current_duration += estimated_duration
            pending_allocations.pop(0)

            if self._coverage_module_complete(
                context=context,
                weekly_strategy=weekly_strategy,
                current_allocations=current_allocations,
                current_duration_minutes=current_duration,
            ):
                modules.append(
                    self._build_module(
                        module_index=len(modules) + 1,
                        allocations=current_allocations,
                        strategies=current_strategies,
                        estimated_duration_minutes=current_duration,
                    )
                )
                current_allocations = []
                current_strategies = []
                current_duration = 0.0

        if current_allocations and (
            max_visible_modules is None
            or len(modules) < max_visible_modules
        ):
            modules.append(
                self._build_module(
                    module_index=len(modules) + 1,
                    allocations=current_allocations,
                    strategies=current_strategies,
                    estimated_duration_minutes=current_duration,
                )
            )

        return tuple(modules)

    def _pop_next_compatible_filler(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        pending_allocations: list[CategoryAllocation],
        current_allocations: Sequence[CategoryAllocation],
        available_budget_minutes: float,
    ):
        """Pop and return the next later allocation that can fill this module."""

        if (
            not current_allocations
            or available_budget_minutes <= 0
            or len(current_allocations) >= self.MAX_CATEGORIES_PER_MODULE
        ):
            return None

        for index, allocation in enumerate(pending_allocations[1:], start=1):
            if self._would_combine_student_priority_categories(
                context=context,
                current_allocations=current_allocations,
                next_allocation=allocation,
            ):
                continue

            candidate = self._build_budgeted_activity_strategy(
                context=context,
                weekly_strategy=weekly_strategy,
                allocation=allocation,
                available_budget_minutes=available_budget_minutes,
                allocation_index=len(current_allocations),
            )

            if candidate is None:
                continue

            pending_allocations.pop(index)
            return candidate

        return None

    def _would_combine_student_priority_categories(
        self,
        context: PlannerContext,
        current_allocations: Sequence[CategoryAllocation],
        next_allocation: CategoryAllocation,
    ) -> bool:
        """Return True when adding an allocation would put two student priorities together."""

        if not current_allocations:
            return False

        priority_categories = set(context.preferences.priority_categories or ())
        if next_allocation.category not in priority_categories:
            return False

        return any(
            allocation.category in priority_categories
            for allocation in current_allocations
        )

    def _coverage_module_complete(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        current_allocations: Sequence[CategoryAllocation],
        current_duration_minutes: float,
    ) -> bool:
        """Return whether a Coverage module should stop accepting filler categories."""

        if not self._is_coverage_strategy(weekly_strategy):
            return False

        if not current_allocations:
            return False

        if len(current_allocations) >= self.MAX_CATEGORIES_PER_MODULE:
            return True

        lower_bound = max(
            0,
            context.planning_budget_minutes - self.negative_buffer_minutes,
        )
        return lower_bound <= current_duration_minutes <= context.planning_budget_minutes

    def _is_coverage_strategy(
        self,
        weekly_strategy: ProfessorWeeklyStrategy,
    ) -> bool:
        return weekly_strategy.weekly_goal_code == ProfessorWeeklyGoalCode.CALIBRATE_COVERAGE

    def _build_budgeted_activity_strategy(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        allocation: CategoryAllocation,
        available_budget_minutes: float,
        allocation_index: int,
    ):
        """Return an adjusted strategy/allocation tuple if it fits."""

        if available_budget_minutes <= 0 or not allocation.selected_topics:
            return None

        strategy = self._build_base_activity_strategy(
            context=context,
            weekly_strategy=weekly_strategy,
            allocation=allocation,
        )
        adjusted_strategy = self._resize_strategy_to_budget(
            context=context,
            weekly_strategy=weekly_strategy,
            allocation=allocation,
            strategy=strategy,
            available_budget_minutes=available_budget_minutes,
            allocation_index=allocation_index,
        )

        if adjusted_strategy is None:
            return None

        estimated_duration = self._estimate_activity_duration_minutes(
            context=context,
            strategy=adjusted_strategy,
        )

        adjusted_allocation = replace(
            allocation,
            estimated_duration_minutes=estimated_duration,
        )

        return adjusted_strategy, adjusted_allocation, estimated_duration

    def _build_base_activity_strategy(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        allocation: CategoryAllocation,
    ) -> ProfessorDailyActivityStrategy:
        """Build the default Professor strategy for one allocation."""

        daily_strategy = self.daily_strategy_builder.build_strategy(
            context=context,
            weekly_strategy=weekly_strategy,
            daily_session=DailyPlan(
                id="module-candidate",
                date=context.week_start_date or date.today(),
                day_name="Module",
                planned_allocations=(allocation,),
            ),
        )

        return daily_strategy.activities[0]

    def _resize_strategy_to_budget(
        self,
        context: PlannerContext,
        weekly_strategy: ProfessorWeeklyStrategy,
        allocation: CategoryAllocation,
        strategy: ProfessorDailyActivityStrategy,
        available_budget_minutes: float,
        allocation_index: int,
    ) -> Optional[ProfessorDailyActivityStrategy]:
        """Shrink activity size if needed while preserving topic coverage."""

        minimum_count = len(allocation.selected_topics)
        question_pace_seconds = context.preferences.question_pace_seconds or 0

        estimated_questions = strategy.estimated_questions
        estimated_flashcards = strategy.estimated_flashcards

        if strategy.activity_type in {
            ProfessorDailyActivityType.QUIZ,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        }:
            if question_pace_seconds <= 0:
                return None

            max_questions = floor(
                available_budget_minutes
                * self.SECONDS_PER_MINUTE
                / question_pace_seconds
            )
            if self._is_coverage_strategy(weekly_strategy):
                estimated_questions = self._coverage_quiz_question_count(
                    topic_count=minimum_count,
                    max_questions=max_questions,
                    allocation_index=allocation_index,
                )
            else:
                estimated_questions = min(
                    max(estimated_questions, minimum_count),
                    max_questions,
                )

            if estimated_questions < minimum_count:
                return None

        if strategy.activity_type in {
            ProfessorDailyActivityType.FLASHCARDS,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        }:
            max_flashcards = floor(
                available_budget_minutes
                * self.SECONDS_PER_MINUTE
                / self.FLASHCARD_SECONDS
            )
            estimated_flashcards = min(
                max(estimated_flashcards, minimum_count),
                max_flashcards,
            )

            if estimated_flashcards < minimum_count:
                return None

        adjusted_strategy = replace(
            strategy,
            estimated_questions=estimated_questions,
            estimated_flashcards=estimated_flashcards,
            allocation_index=allocation_index,
        )

        if (
            self._estimate_activity_duration_minutes(
                context=context,
                strategy=adjusted_strategy,
            )
            > available_budget_minutes
        ):
            return None

        return adjusted_strategy

    def _coverage_quiz_question_count(
        self,
        topic_count: int,
        max_questions: int,
        allocation_index: int,
    ) -> int:
        """Return a complete-pass quiz size for Coverage Mode."""

        if topic_count <= 0 or max_questions < topic_count:
            return 0

        if allocation_index == 0:
            preferred_passes = 2 if topic_count <= 10 else 1
            if topic_count * preferred_passes <= max_questions:
                return topic_count * preferred_passes
            return topic_count

        if topic_count * 2 <= max_questions:
            return topic_count * 2

        return topic_count

    def _estimate_activity_duration_minutes(
        self,
        context: PlannerContext,
        strategy: ProfessorDailyActivityStrategy,
    ) -> float:
        """Estimate planned activity duration from Professor activity size."""

        question_pace_seconds = context.preferences.question_pace_seconds or 0
        seconds = 0

        if strategy.activity_type in {
            ProfessorDailyActivityType.QUIZ,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        }:
            seconds += strategy.estimated_questions * question_pace_seconds

        if strategy.activity_type in {
            ProfessorDailyActivityType.FLASHCARDS,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        }:
            seconds += strategy.estimated_flashcards * self.FLASHCARD_SECONDS

        return round(seconds / self.SECONDS_PER_MINUTE, 4)

    def _build_module(
        self,
        module_index: int,
        allocations: Sequence[CategoryAllocation],
        strategies: Sequence[ProfessorDailyActivityStrategy],
        estimated_duration_minutes: float,
    ) -> ProfessorModule:
        """Build a ProfessorModule from accumulated allocations and strategies."""

        daily_strategy = ProfessorDailyStrategy(
            daily_goal_code=self.daily_strategy_builder._determine_daily_goal(strategies),
            activities=tuple(strategies),
            summary_codes=self.daily_strategy_builder._determine_summary_codes(strategies),
        )

        return ProfessorModule(
            module_index=module_index,
            allocations=tuple(allocations),
            daily_strategy=daily_strategy,
            estimated_duration_minutes=round(estimated_duration_minutes, 4),
        )
