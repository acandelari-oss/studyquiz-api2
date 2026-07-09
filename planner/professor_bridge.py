"""Deterministic Professor narrative bridge for Planner weeks.

The bridge prepares empty narrative containers only. Professor Voice owns
generated text; unavailable Professor content must remain empty so clients can
hide those sections instead of displaying placeholders.
"""

from dataclasses import replace

from .planner_models import (
    DailyPlan,
    DailySummary,
    Week,
)


class ProfessorBridge:
    """Attach empty Professor narrative containers to planned weeks."""

    def enrich_week(self, week: Week) -> Week:
        """Return a week with DailySummary containers and no placeholder text."""

        return replace(
            week,
            daily_plans=tuple(
                self.enrich_daily_plan(daily_plan)
                for daily_plan in week.daily_plans
            ),
        )

    def enrich_daily_plan(self, daily_plan: DailyPlan) -> DailyPlan:
        """Return a daily plan with an empty summary container."""

        return replace(
            daily_plan,
            summary=self._build_summary(daily_plan.summary),
        )

    def _build_summary(self, summary: DailySummary = None) -> DailySummary:
        """Create or preserve a DailySummary without injecting placeholder text."""

        return summary or DailySummary()
