import unittest
from datetime import date

from planner.planner_models import DailyPlan, Week
from planner.planner_state import WeekStatus
from planner.professor_bridge import ProfessorBridge


class PlannerProfessorBridgeTests(unittest.TestCase):
    def test_enrich_week_adds_empty_narrative_containers(self):
        week = Week(
            id="week-1",
            start_date=date(2026, 6, 29),
            end_date=date(2026, 7, 5),
            status=WeekStatus.PLANNED,
            daily_plans=(
                DailyPlan(
                    id="day-1",
                    date=date(2026, 6, 29),
                    day_name="Monday",
                ),
            ),
        )

        enriched_week = ProfessorBridge().enrich_week(week)
        daily_plan = enriched_week.daily_plans[0]

        self.assertEqual(
            enriched_week.weekly_briefing,
            "",
        )
        self.assertEqual(
            daily_plan.objective,
            "",
        )
        self.assertEqual(
            daily_plan.briefing,
            "",
        )
        self.assertIsNotNone(daily_plan.summary)
        self.assertEqual(
            daily_plan.summary.professor_debrief,
            "",
        )
        self.assertEqual(
            daily_plan.summary.homework_recommendations,
            (),
        )
        self.assertIsNone(daily_plan.summary.active_recall_offer)
        self.assertIsNone(daily_plan.summary.office_hours_offer)


if __name__ == "__main__":
    unittest.main()
