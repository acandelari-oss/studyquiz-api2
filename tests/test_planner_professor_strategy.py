import unittest

from planner.category_selector import CategoryAnalytics
from planner.planner_models import PlannerContext, SelectedTopic
from planner.professor_strategy import (
    ProfessorCategoryStrategyCode,
    ProfessorDepthCode,
    ProfessorReasoningCode,
    ProfessorWeeklyGoalCode,
    ProfessorWeeklyStrategyBuilder,
)


class ProfessorWeeklyStrategyBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = ProfessorWeeklyStrategyBuilder()

    def _topics(self, prefix, count):
        return tuple(
            SelectedTopic(id=f"{prefix}-{index}", title=f"{prefix} Topic {index}", order=index)
            for index in range(1, count + 1)
        )

    def test_builds_category_strategy_codes_from_planner_context(self):
        strategy = self.builder.build_strategy(
            PlannerContext(
                categories=("Coverage", "Performance", "Review", "Stable"),
                topics_by_category={
                    "Coverage": self._topics("coverage", 3),
                    "Performance": self._topics("performance", 3),
                    "Review": self._topics("review", 2),
                    "Stable": self._topics("stable", 2),
                },
                analytics={
                    "Coverage": CategoryAnalytics(accuracy=0.80, coverage=0.20),
                    "Performance": CategoryAnalytics(accuracy=0.40, coverage=0.90),
                    "Review": CategoryAnalytics(
                        accuracy=0.80,
                        coverage=0.90,
                        days_since_review=20,
                    ),
                    "Stable": CategoryAnalytics(accuracy=0.90, coverage=0.90),
                },
            )
        )

        by_category = {
            category_strategy.category: category_strategy
            for category_strategy in strategy.category_strategies
        }

        self.assertEqual(
            by_category["Coverage"].strategy,
            ProfessorCategoryStrategyCode.EXPLORE,
        )
        self.assertEqual(
            by_category["Coverage"].reasoning_code,
            ProfessorReasoningCode.LOW_COVERAGE,
        )
        self.assertEqual(
            by_category["Coverage"].depth,
            ProfessorDepthCode.DEEP,
        )
        self.assertEqual(
            by_category["Performance"].strategy,
            ProfessorCategoryStrategyCode.REINFORCE,
        )
        self.assertEqual(
            by_category["Review"].strategy,
            ProfessorCategoryStrategyCode.REVIEW,
        )
        self.assertEqual(
            by_category["Stable"].strategy,
            ProfessorCategoryStrategyCode.ASSESSMENT,
        )

    def test_missing_evidence_uses_explore_strategy_and_calibration_goal(self):
        strategy = self.builder.build_strategy(
            PlannerContext(
                categories=("New Category",),
                topics_by_category={
                    "New Category": self._topics("new", 2),
                },
                analytics={},
            )
        )

        category_strategy = strategy.category_strategies[0]

        self.assertEqual(
            category_strategy.strategy,
            ProfessorCategoryStrategyCode.EXPLORE,
        )
        self.assertEqual(
            category_strategy.reasoning_code,
            ProfessorReasoningCode.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(
            strategy.weekly_goal_code,
            ProfessorWeeklyGoalCode.CALIBRATE_COVERAGE,
        )

    def test_stable_single_topic_category_uses_light_depth(self):
        strategy = self.builder.build_strategy(
            PlannerContext(
                categories=("Stable",),
                topics_by_category={
                    "Stable": self._topics("stable", 1),
                },
                analytics={
                    "Stable": CategoryAnalytics(accuracy=0.90, coverage=0.90),
                },
            )
        )

        self.assertEqual(
            strategy.category_strategies[0].depth,
            ProfessorDepthCode.LIGHT,
        )

    def test_priority_and_secondary_categories_are_deterministic(self):
        strategy = self.builder.build_strategy(
            PlannerContext(
                categories=("A", "B", "C", "D"),
                topics_by_category={
                    "A": self._topics("a", 2),
                    "B": self._topics("b", 2),
                    "C": self._topics("c", 2),
                    "D": self._topics("d", 2),
                },
                analytics={
                    "A": CategoryAnalytics(accuracy=0.40, coverage=0.40),
                    "B": CategoryAnalytics(accuracy=0.50, coverage=0.80),
                    "C": CategoryAnalytics(accuracy=0.80, coverage=0.80),
                    "D": CategoryAnalytics(accuracy=0.90, coverage=0.90),
                },
            )
        )

        self.assertEqual(len(strategy.priority_categories), 3)
        self.assertEqual(len(strategy.secondary_categories), 1)
        self.assertEqual(
            tuple(
                category_strategy.category
                for category_strategy in strategy.category_strategies
            ),
            strategy.priority_categories + strategy.secondary_categories,
        )


if __name__ == "__main__":
    unittest.main()
