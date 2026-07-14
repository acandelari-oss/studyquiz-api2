import unittest
from datetime import date

from planner.category_selector import CategoryAnalytics
from planner.planner_engine import PlannerEngine, generate_week
from planner.planner_models import PlannerContext, PlannerPreferences, SelectedTopic
from planner.planner_state import ActivityType, ExecutionStatus, WeekStatus


class PlannerEngineTests(unittest.TestCase):
    def _topics(self, prefix, count):
        return tuple(
            SelectedTopic(id=f"{prefix}-{index}", title=f"{prefix} Topic {index}", order=index)
            for index in range(1, count + 1)
        )

    def test_generate_week_builds_planning_only_week(self):
        context = PlannerContext(
            categories=("Contracts", "Property"),
            topics_by_category={
                "Contracts": self._topics("contract", 2),
                "Property": self._topics("property", 3),
            },
            analytics={
                "Contracts": CategoryAnalytics(accuracy=0.90),
                "Property": CategoryAnalytics(accuracy=0.50),
            },
            preferences=PlannerPreferences(question_pace_seconds=60),
            number_of_sessions=3,
            planning_budget_minutes=2,
            week_start_date=date(2026, 6, 29),
            week_id="week-test",
        )

        week = PlannerEngine().generate_week(context)

        self.assertEqual(week.id, "week-test")
        self.assertEqual(week.status, WeekStatus.PLANNED)
        self.assertEqual(week.start_date, date(2026, 6, 29))
        self.assertEqual(week.end_date, date(2026, 7, 5))
        self.assertEqual(len(week.daily_plans), 3)

        self.assertEqual(week.daily_plans[0].id, "week-test-session-1")
        self.assertEqual(week.daily_plans[0].day_name, "Monday")
        self.assertIn("foundation", week.weekly_briefing)
        self.assertTrue(week.weekly_briefing)
        self.assertTrue(week.daily_plans[0].objective)
        self.assertTrue(week.daily_plans[0].briefing)
        self.assertIsNotNone(week.daily_plans[0].summary)
        self.assertEqual(
            [activity.type for activity in week.daily_plans[0].activities],
            [ActivityType.FLASHCARDS],
        )
        self.assertTrue(
            all(
                activity.execution.status == ExecutionStatus.NOT_STARTED
                for activity in week.daily_plans[0].activities
            )
        )

        scheduled_categories = [
            allocation.category
            for daily_plan in week.daily_plans
            for allocation in daily_plan.planned_allocations
        ]

        self.assertEqual(scheduled_categories, [
            "Property",
            "Property",
            "Contracts",
        ])

    def test_daily_plans_preserve_scheduler_allocation_order(self):
        context = PlannerContext(
            categories=("A", "B", "C"),
            topics_by_category={
                "A": self._topics("a", 2),
                "B": self._topics("b", 1),
                "C": self._topics("c", 1),
            },
            analytics={
                "A": CategoryAnalytics(accuracy=0.40),
                "B": CategoryAnalytics(accuracy=0.60),
                "C": CategoryAnalytics(accuracy=0.80),
            },
            preferences=PlannerPreferences(question_pace_seconds=60),
            number_of_sessions=2,
            planning_budget_minutes=2,
            week_start_date=date(2026, 6, 29),
        )

        week = PlannerEngine().generate_week(context)

        self.assertEqual(
            [
                [allocation.category for allocation in daily_plan.planned_allocations]
                for daily_plan in week.daily_plans
            ],
            [["A"], ["B"]],
        )

    def test_empty_context_produces_valid_empty_week(self):
        week = generate_week(
            PlannerContext(
                number_of_sessions=2,
                planning_budget_minutes=30,
                week_start_date=date(2026, 6, 29),
            )
        )

        self.assertEqual(len(week.daily_plans), 0)

    def test_activity_planner_populates_daily_plan_activities(self):
        week = PlannerEngine().generate_week(
            PlannerContext(
                categories=("Genetics",),
                topics_by_category={
                    "Genetics": self._topics("genetics", 3),
                },
                analytics={
                    "Genetics": CategoryAnalytics(accuracy=0.80, coverage=0.80),
                },
                preferences=PlannerPreferences(
                    question_pace_seconds=60,
                    question_style="exam",
                ),
                number_of_sessions=2,
                planning_budget_minutes=2,
                week_start_date=date(2026, 6, 29),
            )
        )

        first_day_activities = week.daily_plans[0].activities

        self.assertEqual(
            [activity.type for activity in first_day_activities],
            [ActivityType.QUIZ],
        )
        self.assertEqual(first_day_activities[0].configuration.num_questions, 2)
        self.assertEqual(first_day_activities[0].configuration.question_style, "exam")
        self.assertEqual(first_day_activities[0].configuration.difficulty, "medium")
        self.assertIsNone(first_day_activities[0].result)

    def test_assessment_uses_context_order_and_quiz_only_full_topic_coverage(self):
        week = PlannerEngine().generate_assessment_week(
            PlannerContext(
                categories=("B", "A"),
                topics_by_category={
                    "B": self._topics("b", 3),
                    "A": self._topics("a", 2),
                },
                analytics={
                    "A": CategoryAnalytics(accuracy=0.10, coverage=0.10),
                    "B": CategoryAnalytics(accuracy=0.90, coverage=0.90),
                },
                preferences=PlannerPreferences(
                    question_pace_seconds=60,
                    question_style="reasoning",
                ),
                number_of_sessions=1,
                planning_budget_minutes=2,
                week_start_date=date(2026, 6, 29),
                week_id="assessment-test",
            )
        )

        self.assertEqual(week.plan_type, "assessment")
        self.assertGreater(len(week.daily_plans), 1)

        allocations = [
            allocation
            for daily_plan in week.daily_plans
            for allocation in daily_plan.planned_allocations
        ]
        self.assertEqual(
            [allocation.category for allocation in allocations],
            ["B", "B", "A"],
        )
        self.assertEqual(
            [
                topic.title
                for allocation in allocations
                for topic in allocation.selected_topics
            ],
            [
                "b Topic 1",
                "b Topic 2",
                "b Topic 3",
                "a Topic 1",
                "a Topic 2",
            ],
        )

        activities = [
            activity
            for daily_plan in week.daily_plans
            for activity in daily_plan.activities
        ]
        self.assertTrue(activities)
        self.assertTrue(all(activity.type == ActivityType.QUIZ for activity in activities))
        self.assertTrue(
            all(
                activity.configuration.num_questions
                >= len(activity.configuration.selected_topics)
                for activity in activities
            )
        )
        self.assertTrue(
            all(
                activity.configuration.question_style == "reasoning"
                for activity in activities
            )
        )

    def test_assessment_covers_programme_beyond_visible_study_plan_limit(self):
        topics = self._topics("large", 13)

        week = PlannerEngine().generate_assessment_week(
            PlannerContext(
                categories=("Large",),
                topics_by_category={"Large": topics},
                analytics={
                    "Large": CategoryAnalytics(accuracy=0.95, coverage=0.95),
                },
                preferences=PlannerPreferences(
                    question_pace_seconds=60,
                    question_style="exam",
                ),
                number_of_sessions=1,
                planning_budget_minutes=1,
                week_start_date=date(2026, 6, 29),
                week_id="assessment-large-test",
            )
        )

        self.assertEqual(week.plan_type, "assessment")
        self.assertEqual(len(week.daily_plans), 13)

        selected_topic_ids = [
            topic.id
            for daily_plan in week.daily_plans
            for allocation in daily_plan.planned_allocations
            for topic in allocation.selected_topics
        ]
        self.assertEqual(selected_topic_ids, [topic.id for topic in topics])
        self.assertEqual(len(selected_topic_ids), len(set(selected_topic_ids)))

        activities = [
            activity
            for daily_plan in week.daily_plans
            for activity in daily_plan.activities
        ]
        self.assertEqual(len(activities), 13)
        self.assertTrue(all(activity.type == ActivityType.QUIZ for activity in activities))
        self.assertTrue(
            all(activity.configuration.num_questions >= 1 for activity in activities)
        )

    def test_assessment_quiz_size_uses_budget_instead_of_topic_count_only(self):
        week = PlannerEngine().generate_assessment_week(
            PlannerContext(
                categories=("A",),
                topics_by_category={"A": self._topics("a", 2)},
                analytics={
                    "A": CategoryAnalytics(accuracy=0.95, coverage=0.95),
                },
                preferences=PlannerPreferences(
                    question_pace_seconds=60,
                    question_style="balanced",
                ),
                number_of_sessions=1,
                planning_budget_minutes=10,
                week_start_date=date(2026, 6, 29),
                week_id="assessment-budget-test",
            )
        )

        activity = week.daily_plans[0].activities[0]

        self.assertEqual(len(activity.configuration.selected_topics), 2)
        self.assertGreater(activity.configuration.num_questions, 2)
        self.assertLessEqual(activity.configuration.estimated_duration_minutes, 10)


if __name__ == "__main__":
    unittest.main()
