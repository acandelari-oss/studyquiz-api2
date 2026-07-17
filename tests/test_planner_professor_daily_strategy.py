import unittest
from datetime import date

from planner.category_selector import CategoryAnalytics
from planner.planner_models import DailyPlan, PlannerContext, SelectedTopic
from planner.professor_daily_strategy import (
    ProfessorDailyActivityType,
    ProfessorDailyStrategyBuilder,
)
from planner.professor_strategy import (
    ProfessorCategoryStrategy,
    ProfessorCategoryStrategyCode,
    ProfessorDepthCode,
    ProfessorReasoningCode,
    ProfessorWeeklyGoalCode,
    ProfessorWeeklyStrategy,
)
from planner.session_allocator import CategoryAllocation


class ProfessorDailyStrategyBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = ProfessorDailyStrategyBuilder()

    def _allocation(self, category, topic_count=2):
        return CategoryAllocation(
            category=category,
            selected_topics=tuple(
                SelectedTopic(id=f"{category}-{index}", title=f"{category} Topic {index}")
                for index in range(1, topic_count + 1)
            ),
            estimated_duration_minutes=10,
        )

    def _daily_plan(self, *allocations):
        return DailyPlan(
            id="day-1",
            date=date(2026, 7, 4),
            day_name="Saturday",
            planned_allocations=allocations,
        )

    def _weekly_strategy(self, category, strategy, depth=ProfessorDepthCode.NORMAL):
        return ProfessorWeeklyStrategy(
            weekly_goal_code=ProfessorWeeklyGoalCode.ASSESS_AND_MAINTAIN,
            category_strategies=(
                ProfessorCategoryStrategy(
                    category=category,
                    strategy=strategy,
                    depth=depth,
                    reasoning_code=ProfessorReasoningCode.STABLE_PERFORMANCE,
                ),
            ),
            priority_categories=(category,),
        )

    def _activity_for(self, strategy, analytics=None):
        daily_strategy = self.builder.build_strategy(
            context=PlannerContext(
                categories=("Category",),
                analytics={"Category": analytics} if analytics else {},
            ),
            weekly_strategy=strategy,
            daily_session=self._daily_plan(self._allocation("Category")),
        )

        return daily_strategy.activities[0]

    def test_explore_maps_to_quiz(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.EXPLORE,
            )
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.QUIZ)

    def test_reinforce_maps_to_flashcards(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.REINFORCE,
            )
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.FLASHCARDS)

    def test_reinforce_with_quiz_weakness_maps_to_quiz(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.REINFORCE,
            ),
            analytics=CategoryAnalytics(
                accuracy=0.40,
                quiz_accuracy=0.40,
                flashcard_accuracy=0.90,
                coverage=0.90,
            ),
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.QUIZ)

    def test_reinforce_with_flashcard_weakness_maps_to_flashcards(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.REINFORCE,
            ),
            analytics=CategoryAnalytics(
                accuracy=0.50,
                quiz_accuracy=0.90,
                flashcard_accuracy=0.50,
                coverage=0.90,
            ),
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.FLASHCARDS)

    def test_reinforce_with_mixed_weakness_maps_to_mixed_activity(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.REINFORCE,
            ),
            analytics=CategoryAnalytics(
                accuracy=0.40,
                quiz_accuracy=0.40,
                flashcard_accuracy=0.50,
                coverage=0.90,
            ),
        )

        self.assertEqual(
            activity.activity_type,
            ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS,
        )

    def test_assessment_maps_to_quiz(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.ASSESSMENT,
            )
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.QUIZ)

    def test_review_maps_to_flashcards(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.REVIEW,
            ),
            analytics=CategoryAnalytics(accuracy=0.90, coverage=0.90),
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.FLASHCARDS)

    def test_review_with_unstable_performance_maps_to_quiz(self):
        activity = self._activity_for(
            self._weekly_strategy(
                "Category",
                ProfessorCategoryStrategyCode.REVIEW,
            ),
            analytics=CategoryAnalytics(accuracy=0.60, coverage=0.90),
        )

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.QUIZ)

    def test_quiz_size_mapping_by_depth(self):
        expected = {
            ProfessorDepthCode.LIGHT: 6,
            ProfessorDepthCode.NORMAL: 10,
            ProfessorDepthCode.DEEP: 15,
        }

        for depth, question_count in expected.items():
            activity = self._activity_for(
                self._weekly_strategy(
                    "Category",
                    ProfessorCategoryStrategyCode.ASSESSMENT,
                    depth=depth,
                )
            )
            self.assertEqual(activity.estimated_questions, question_count)

    def test_flashcard_size_mapping_by_depth(self):
        expected = {
            ProfessorDepthCode.LIGHT: 8,
            ProfessorDepthCode.NORMAL: 12,
            ProfessorDepthCode.DEEP: 18,
        }

        for depth, flashcard_count in expected.items():
            activity = self._activity_for(
                self._weekly_strategy(
                    "Category",
                    ProfessorCategoryStrategyCode.REINFORCE,
                    depth=depth,
                )
            )
            self.assertEqual(activity.estimated_flashcards, flashcard_count)

    def test_multiple_categories_produce_independent_activity_strategies(self):
        daily_strategy = self.builder.build_strategy(
            context=PlannerContext(
                categories=("A", "B", "C"),
                analytics={
                    "A": CategoryAnalytics(accuracy=0.90, coverage=0.90),
                    "B": CategoryAnalytics(accuracy=0.40, coverage=0.90),
                    "C": CategoryAnalytics(accuracy=0.80, coverage=0.90),
                },
            ),
            weekly_strategy=ProfessorWeeklyStrategy(
                weekly_goal_code=ProfessorWeeklyGoalCode.ASSESS_AND_MAINTAIN,
                category_strategies=(
                    ProfessorCategoryStrategy(
                        category="A",
                        strategy=ProfessorCategoryStrategyCode.ASSESSMENT,
                        depth=ProfessorDepthCode.LIGHT,
                        reasoning_code=ProfessorReasoningCode.STABLE_PERFORMANCE,
                    ),
                    ProfessorCategoryStrategy(
                        category="B",
                        strategy=ProfessorCategoryStrategyCode.REINFORCE,
                        depth=ProfessorDepthCode.DEEP,
                        reasoning_code=ProfessorReasoningCode.LOW_PERFORMANCE,
                    ),
                    ProfessorCategoryStrategy(
                        category="C",
                        strategy=ProfessorCategoryStrategyCode.REVIEW,
                        depth=ProfessorDepthCode.NORMAL,
                        reasoning_code=ProfessorReasoningCode.NOT_RECENTLY_REVIEWED,
                    ),
                ),
            ),
            daily_session=self._daily_plan(
                self._allocation("A"),
                self._allocation("B"),
                self._allocation("C"),
            ),
        )

        self.assertEqual(
            [activity.category for activity in daily_strategy.activities],
            ["A", "B", "C"],
        )
        self.assertEqual(
            [activity.activity_type for activity in daily_strategy.activities],
            [
                ProfessorDailyActivityType.QUIZ,
                ProfessorDailyActivityType.FLASHCARDS,
                ProfessorDailyActivityType.FLASHCARDS,
            ],
        )


if __name__ == "__main__":
    unittest.main()
