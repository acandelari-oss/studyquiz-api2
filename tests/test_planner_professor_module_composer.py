import unittest
from datetime import date

from planner.category_selector import CategoryAnalytics
from planner.planner_models import PlannerContext, PlannerPreferences, SelectedTopic
from planner.professor_daily_strategy import ProfessorDailyActivityType
from planner.professor_module_composer import ProfessorModuleComposer
from planner.professor_strategy import (
    ProfessorCategoryStrategy,
    ProfessorCategoryStrategyCode,
    ProfessorDepthCode,
    ProfessorReasoningCode,
    ProfessorWeeklyGoalCode,
    ProfessorWeeklyStrategy,
)
from planner.session_allocator import CategoryAllocation


class ProfessorModuleComposerTests(unittest.TestCase):
    def setUp(self):
        self.composer = ProfessorModuleComposer()

    def _topics(self, category, count):
        return tuple(
            SelectedTopic(
                id=f"{category}-{index}",
                title=f"{category} Topic {index}",
                order=index,
            )
            for index in range(1, count + 1)
        )

    def _allocation(self, category, topic_count=2):
        return CategoryAllocation(
            category=category,
            selected_topics=self._topics(category, topic_count),
            estimated_duration_minutes=0,
        )

    def _context(self, categories, budget=30, pace=90):
        return PlannerContext(
            categories=categories,
            analytics={
                category: CategoryAnalytics(accuracy=0.80, coverage=0.80)
                for category in categories
            },
            preferences=PlannerPreferences(question_pace_seconds=pace),
            planning_budget_minutes=budget,
            week_start_date=date(2026, 7, 6),
        )

    def _weekly_strategy(self, categories, depth=ProfessorDepthCode.NORMAL):
        return ProfessorWeeklyStrategy(
            weekly_goal_code=ProfessorWeeklyGoalCode.ASSESS_AND_MAINTAIN,
            category_strategies=tuple(
                ProfessorCategoryStrategy(
                    category=category,
                    strategy=ProfessorCategoryStrategyCode.ASSESSMENT,
                    depth=depth,
                    reasoning_code=ProfessorReasoningCode.STABLE_PERFORMANCE,
                )
                for category in categories
            ),
            priority_categories=tuple(categories[:3]),
            secondary_categories=tuple(categories[3:]),
        )

    def test_composes_only_required_modules_without_empty_placeholders(self):
        categories = ("A", "B")

        modules = self.composer.compose_modules(
            context=self._context(categories, budget=45, pace=90),
            weekly_strategy=self._weekly_strategy(categories),
            allocations=(
                self._allocation("A", 2),
                self._allocation("B", 2),
            ),
            max_visible_modules=12,
        )

        self.assertEqual(len(modules), 1)
        self.assertEqual(
            [allocation.category for allocation in modules[0].allocations],
            ["A", "B"],
        )
        self.assertLessEqual(modules[0].estimated_duration_minutes, 45)

    def test_starts_new_module_when_next_activity_exceeds_budget(self):
        categories = ("A", "B", "C")

        modules = self.composer.compose_modules(
            context=self._context(categories, budget=25, pace=90),
            weekly_strategy=self._weekly_strategy(categories, depth=ProfessorDepthCode.DEEP),
            allocations=(
                self._allocation("A", 2),
                self._allocation("B", 2),
                self._allocation("C", 2),
            ),
            max_visible_modules=12,
        )

        self.assertEqual(len(modules), 3)
        self.assertTrue(
            all(module.estimated_duration_minutes <= 25 for module in modules)
        )
        self.assertEqual(
            [
                allocation.category
                for module in modules
                for allocation in module.allocations
            ],
            ["A", "B", "C"],
        )

    def test_quiz_size_shrinks_to_remaining_budget_without_losing_topics(self):
        categories = ("A",)

        modules = self.composer.compose_modules(
            context=self._context(categories, budget=9, pace=90),
            weekly_strategy=self._weekly_strategy(categories, depth=ProfessorDepthCode.DEEP),
            allocations=(self._allocation("A", 3),),
            max_visible_modules=12,
        )

        activity = modules[0].daily_strategy.activities[0]

        self.assertEqual(activity.activity_type, ProfessorDailyActivityType.QUIZ)
        self.assertEqual(activity.estimated_questions, 6)
        self.assertGreaterEqual(activity.estimated_questions, 3)
        self.assertLessEqual(modules[0].estimated_duration_minutes, 9)

    def test_visible_limit_caps_output_without_forcing_module_count(self):
        categories = ("A", "B", "C")

        modules = self.composer.compose_modules(
            context=self._context(categories, budget=10, pace=90),
            weekly_strategy=self._weekly_strategy(categories, depth=ProfessorDepthCode.DEEP),
            allocations=(
                self._allocation("A", 2),
                self._allocation("B", 2),
                self._allocation("C", 2),
            ),
            max_visible_modules=2,
        )

        self.assertEqual(len(modules), 2)
        self.assertEqual(
            [
                allocation.category
                for module in modules
                for allocation in module.allocations
            ],
            ["A", "B"],
        )


if __name__ == "__main__":
    unittest.main()
