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
from .professor_strategy import ProfessorWeeklyStrategy
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

    def __init__(
        self,
        daily_strategy_builder: Optional[ProfessorDailyStrategyBuilder] = None,
    ) -> None:
        self.daily_strategy_builder = daily_strategy_builder or ProfessorDailyStrategyBuilder()

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
        allocation_index = 0

        while allocation_index < len(allocations):
            if max_visible_modules is not None and len(modules) >= max_visible_modules:
                break

            allocation = allocations[allocation_index]
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
                    allocation_index += 1
                    continue

            adjusted_strategy, adjusted_allocation, estimated_duration = candidate
            current_allocations.append(adjusted_allocation)
            current_strategies.append(adjusted_strategy)
            current_duration += estimated_duration
            allocation_index += 1

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
            estimated_questions = min(estimated_questions, max_questions)

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
            estimated_flashcards = min(estimated_flashcards, max_flashcards)

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
