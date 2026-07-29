import unittest

from planner.category_selector import CategoryAnalytics, CategorySelector
from planner.student_preferences import (
    PRIORITY_CATEGORY_SCORE_MULTIPLIER,
    apply_student_preference_score_bonus,
)


class PlannerStudentPreferenceTests(unittest.TestCase):
    def test_priority_category_receives_exact_bonus(self):
        adjusted = apply_student_preference_score_bonus(
            category="Contracts",
            planner_score=100.0,
            planner_preferences={"priorityCategories": ["Contracts"]},
        )

        self.assertEqual(adjusted, 100.0 * PRIORITY_CATEGORY_SCORE_MULTIPLIER)

    def test_non_priority_category_remains_unchanged(self):
        adjusted = apply_student_preference_score_bonus(
            category="Property",
            planner_score=100.0,
            planner_preferences={"priorityCategories": ["Contracts"]},
        )

        self.assertEqual(adjusted, 100.0)

    def test_ordering_changes_only_from_deterministic_score_bonus(self):
        priorities = CategorySelector().select_categories(
            project_categories=("A", "B"),
            category_analytics={
                "A": CategoryAnalytics(accuracy=0.50),
                "B": CategoryAnalytics(accuracy=0.52),
            },
            planner_preferences={"priorityCategories": ["B"]},
        )

        self.assertEqual([priority.category for priority in priorities], ["B", "A"])

    def test_learning_evidence_remains_primary_over_priority_bonus(self):
        priorities = CategorySelector().select_categories(
            project_categories=("A", "B"),
            category_analytics={
                "A": CategoryAnalytics(accuracy=0.20),
                "B": CategoryAnalytics(accuracy=0.95),
            },
            planner_preferences={"priorityCategories": ["B"]},
        )

        self.assertEqual([priority.category for priority in priorities], ["A", "B"])

    def test_identical_inputs_produce_identical_outputs(self):
        selector = CategorySelector()
        kwargs = {
            "project_categories": ("A", "B", "C"),
            "category_analytics": {
                "A": CategoryAnalytics(accuracy=0.50),
                "B": CategoryAnalytics(accuracy=0.52),
                "C": CategoryAnalytics(accuracy=0.90),
            },
            "planner_preferences": {"priorityCategories": ["B"]},
        }

        first = selector.select_categories(**kwargs)
        second = selector.select_categories(**kwargs)

        self.assertEqual(first, second)

    def test_changing_multiplier_changes_ordering_deterministically(self):
        kwargs = {
            "project_categories": ("A", "B"),
            "category_analytics": {
                "A": CategoryAnalytics(accuracy=0.50),
                "B": CategoryAnalytics(accuracy=0.52),
            },
            "planner_preferences": {"priorityCategories": ["B"]},
        }

        neutral = CategorySelector(
            priority_category_score_multiplier=1.0,
        ).select_categories(**kwargs)
        boosted = CategorySelector(
            priority_category_score_multiplier=PRIORITY_CATEGORY_SCORE_MULTIPLIER,
        ).select_categories(**kwargs)

        self.assertEqual([priority.category for priority in neutral], ["A", "B"])
        self.assertEqual([priority.category for priority in boosted], ["B", "A"])


if __name__ == "__main__":
    unittest.main()
