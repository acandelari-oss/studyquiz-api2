import unittest
from datetime import date

from planner.activity_planner import ActivityPlanner
from planner.planner_models import (
    DailyPlan,
    PlannerContext,
    PlannerPreferences,
    SelectedTopic,
)
from planner.planner_state import ActivityType, ExecutionStatus
from planner.professor_daily_strategy import (
    ProfessorDailyActivityStrategy,
    ProfessorDailyActivityType,
    ProfessorDailyStrategy,
)
from planner.professor_strategy import ProfessorDepthCode
from planner.session_allocator import CategoryAllocation


class PlannerActivityPlannerTests(unittest.TestCase):
    def setUp(self):
        self.planner = ActivityPlanner()

    def _allocation(self, category, topic_count, duration=3.0):
        return CategoryAllocation(
            category=category,
            selected_topics=tuple(
                SelectedTopic(id=f"{category}-{index}", title=f"{category} Topic {index}")
                for index in range(1, topic_count + 1)
            ),
            estimated_duration_minutes=duration,
        )

    def _daily_plan(self, *allocations):
        return DailyPlan(
            id="day-1",
            date=date(2026, 6, 30),
            day_name="Tuesday",
            planned_allocations=allocations,
        )

    def _daily_strategy(self, *activity_strategies):
        return ProfessorDailyStrategy(
            daily_goal_code="TEST",
            activities=activity_strategies,
        )

    def _activity_strategy(
        self,
        category,
        activity_type,
        allocation_index=0,
        depth=ProfessorDepthCode.NORMAL,
        questions=10,
        flashcards=12,
    ):
        return ProfessorDailyActivityStrategy(
            category=category,
            activity_type=activity_type,
            depth=depth,
            estimated_questions=questions,
            estimated_flashcards=flashcards,
            allocation_index=allocation_index,
        )

    def test_translates_professor_quiz_strategies_to_quiz_activities(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(
                self._allocation("Genetics", 2),
                self._allocation("Chemistry", 3),
            ),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.QUIZ,
                    allocation_index=0,
                    questions=10,
                ),
                self._activity_strategy(
                    "Chemistry",
                    ProfessorDailyActivityType.QUIZ,
                    allocation_index=1,
                    questions=15,
                ),
            ),
        )

        self.assertEqual(
            [activity.type for activity in daily_plan.activities],
            [
                ActivityType.QUIZ,
                ActivityType.QUIZ,
            ],
        )
        self.assertEqual(
            [activity.configuration.category for activity in daily_plan.activities],
            ["Genetics", "Chemistry"],
        )
        self.assertEqual(
            [activity.configuration.num_questions for activity in daily_plan.activities],
            [10, 15],
        )

    def test_translates_professor_flashcard_strategy_to_flashcards(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Genetics", 1)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.FLASHCARDS,
                    flashcards=8,
                ),
            ),
        )

        self.assertEqual(
            [activity.type for activity in daily_plan.activities],
            [ActivityType.FLASHCARDS],
        )
        self.assertEqual(daily_plan.activities[0].configuration.num_cards, 8)

    def test_mandatory_topic_coverage_is_reflected_in_counts(self):
        flashcard_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Chemistry", 1)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Chemistry",
                    ProfessorDailyActivityType.FLASHCARDS,
                    flashcards=8,
                ),
            ),
        )
        quiz_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Genetics", 3)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.QUIZ,
                    questions=10,
                ),
            ),
        )

        flashcards = flashcard_plan.activities[0]
        quiz = quiz_plan.activities[0]

        self.assertEqual(flashcards.configuration.num_cards, 8)
        self.assertEqual(quiz.configuration.num_questions, 10)
        self.assertGreaterEqual(
            flashcards.configuration.num_cards,
            len(flashcards.configuration.selected_topics),
        )
        self.assertGreaterEqual(
            quiz.configuration.num_questions,
            len(quiz.configuration.selected_topics),
        )

    def test_selected_topics_and_estimated_duration_are_preserved(self):
        allocation = self._allocation("Genetics", 2, duration=4.5)
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(allocation),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.QUIZ,
                ),
            ),
        )

        for activity in daily_plan.activities:
            self.assertEqual(activity.configuration.selected_topics, allocation.selected_topics)
            self.assertEqual(activity.configuration.estimated_duration_minutes, 4.5)

    def test_activities_have_empty_execution_and_result(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Genetics", 1)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.FLASHCARDS,
                ),
            ),
        )

        for activity in daily_plan.activities:
            self.assertEqual(activity.execution.status, ExecutionStatus.NOT_STARTED)
            self.assertIsNone(activity.execution.started_at)
            self.assertIsNone(activity.execution.completed_at)
            self.assertIsNone(activity.execution.actual_duration)
            self.assertIsNone(activity.result)

    def test_quiz_uses_preference_question_style_when_available(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(
                preferences=PlannerPreferences(question_style="exam")
            ),
            daily_plan=self._daily_plan(self._allocation("Genetics", 2)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.QUIZ,
                    questions=10,
                ),
            ),
        )

        quiz = daily_plan.activities[0]

        self.assertEqual(quiz.configuration.question_style, "exam")
        self.assertEqual(quiz.configuration.difficulty, "medium")

    def test_quiz_defaults_question_style_to_balanced(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Genetics", 2)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.QUIZ,
                    questions=10,
                ),
            ),
        )

        quiz = daily_plan.activities[0]

        self.assertEqual(quiz.configuration.question_style, "balanced")
        self.assertEqual(quiz.configuration.difficulty, "medium")

    def test_empty_allocation_does_not_create_activity(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Genetics", 0)),
            daily_strategy=self._daily_strategy(),
        )

        self.assertEqual(daily_plan.activities, ())

    def test_translates_quiz_plus_flashcards_to_two_activities(self):
        daily_plan = self.planner.plan_daily_plan(
            context=PlannerContext(),
            daily_plan=self._daily_plan(self._allocation("Genetics", 2)),
            daily_strategy=self._daily_strategy(
                self._activity_strategy(
                    "Genetics",
                    ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
                    questions=10,
                    flashcards=12,
                ),
            ),
        )

        self.assertEqual(
            [activity.type for activity in daily_plan.activities],
            [ActivityType.FLASHCARDS, ActivityType.QUIZ],
        )
        self.assertEqual(daily_plan.activities[0].configuration.num_cards, 12)
        self.assertEqual(daily_plan.activities[1].configuration.num_questions, 10)


if __name__ == "__main__":
    unittest.main()
