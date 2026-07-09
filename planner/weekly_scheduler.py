"""Deterministic weekly scheduling for pre-allocated Planner segments.

The scheduler receives category allocations that were already produced by the
Session Allocator. It preserves their order and distributes complete
allocations across the student's available weekly sessions.
"""

from dataclasses import dataclass, field
from typing import Sequence

from .session_allocator import CategoryAllocation


@dataclass(frozen=True)
class WeeklySession:
    """One scheduled weekly study session."""

    session_index: int
    allocations: Sequence[CategoryAllocation] = field(default_factory=tuple)
    estimated_duration_minutes: float = 0.0


class WeeklyScheduler:
    """Schedule ordered category allocations into weekly study sessions."""

    FIRST_ADDITIONAL_CATEGORY_REMAINING_RATIO = 0.40
    ADDITIONAL_CATEGORY_REMAINING_RATIO_STEP = 0.30

    def schedule_week(
        self,
        allocations: Sequence[CategoryAllocation],
        number_of_sessions: int,
        planning_budget_minutes: float,
    ) -> Sequence[WeeklySession]:
        """Return ordered weekly sessions without exceeding session budgets.

        Allocations are consumed in their existing order. The scheduler never
        skips a higher-priority allocation to fit a lower-priority one.
        """

        if number_of_sessions <= 0:
            return ()

        sessions = []
        next_allocation_index = 0

        for session_index in range(1, number_of_sessions + 1):
            session_allocations = []
            session_duration = 0.0
            session_categories = set()

            while next_allocation_index < len(allocations):
                allocation = allocations[next_allocation_index]

                if not self._can_fit(
                    allocation=allocation,
                    current_duration=session_duration,
                    planning_budget_minutes=planning_budget_minutes,
                    current_categories=session_categories,
                ):
                    break

                session_allocations.append(allocation)
                session_duration += allocation.estimated_duration_minutes
                session_categories.add(allocation.category)
                next_allocation_index += 1

            sessions.append(
                WeeklySession(
                    session_index=session_index,
                    allocations=tuple(session_allocations),
                    estimated_duration_minutes=round(session_duration, 4),
                )
            )

        return tuple(sessions)

    def _can_fit(
        self,
        allocation: CategoryAllocation,
        current_duration: float,
        planning_budget_minutes: float,
        current_categories: set,
    ) -> bool:
        """Return whether an allocation fits entirely in the remaining budget."""

        if planning_budget_minutes <= 0:
            return False

        if allocation.estimated_duration_minutes < 0:
            return False

        if not current_duration + allocation.estimated_duration_minutes <= planning_budget_minutes:
            return False

        if allocation.category in current_categories or not current_categories:
            return True

        return self._has_enough_remaining_budget_for_new_category(
            current_duration=current_duration,
            planning_budget_minutes=planning_budget_minutes,
            current_category_count=len(current_categories),
        )

    def _has_enough_remaining_budget_for_new_category(
        self,
        current_duration: float,
        planning_budget_minutes: float,
        current_category_count: int,
    ) -> bool:
        """Return whether adding another category preserves depth over breadth.

        The more categories already present in a session, the more remaining
        time is required before a new category can be introduced. This keeps
        short sessions focused while still allowing longer sessions to include
        more categories when the earlier allocations are genuinely small.
        """

        remaining_budget = planning_budget_minutes - current_duration
        required_ratio = (
            self.FIRST_ADDITIONAL_CATEGORY_REMAINING_RATIO
            + max(current_category_count - 1, 0)
            * self.ADDITIONAL_CATEGORY_REMAINING_RATIO_STEP
        )

        return remaining_budget >= planning_budget_minutes * required_ratio
