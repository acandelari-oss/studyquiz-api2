import unittest

from planner.coverage_state import (
    evaluate_coverage_state,
    ordered_first_exposure_categories,
    ordered_coverage_categories,
)
from planner.planner_models import PlannerContext


class CoverageStateTests(unittest.TestCase):
    def test_coverage_uses_completed_survey_plans_not_learning_evidence(self):
        context = PlannerContext(
            categories=("CompletedSurvey", "QuizCoveredOnly", "FlashcardOnly"),
            analytics={
                "QuizCoveredOnly": object(),
                "FlashcardOnly": object(),
            },
            completed_survey_categories=("CompletedSurvey",),
        )

        state = evaluate_coverage_state(context)

        self.assertEqual(state.covered_categories, ("CompletedSurvey",))
        self.assertEqual(
            state.categories_requiring_coverage,
            ("QuizCoveredOnly", "FlashcardOnly"),
        )
        self.assertFalse(state.complete)

    def test_continuation_order_preserves_previous_order_and_excludes_covered(self):
        ordered = ordered_coverage_categories(
            initial_ranked_categories=("B", "A", "C", "D"),
            categories_requiring_coverage=("A", "C"),
        )

        self.assertEqual(ordered, ("A", "C"))

    def test_first_exposure_order_without_previous_plans_starts_with_priorities(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("A", "B", "C"),
            previously_scheduled_categories=(),
            priority_categories=("C",),
        )

        self.assertEqual(ordered, ("C", "A", "B"))

    def test_first_exposure_order_moves_unscheduled_categories_first(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("A", "B", "C", "D", "E"),
            previously_scheduled_categories=("A", "C", "E"),
            priority_categories=(),
        )

        self.assertEqual(ordered, ("B", "D", "A", "C", "E"))

    def test_first_exposure_order_does_not_promote_scheduled_priorities_for_reuse(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("A", "B", "C", "D", "E"),
            previously_scheduled_categories=("A", "C", "E"),
            priority_categories=("E",),
        )

        self.assertEqual(ordered, ("B", "D", "A", "C", "E"))

    def test_first_exposure_order_starts_with_unscheduled_priorities(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("A", "B", "C", "D", "E"),
            previously_scheduled_categories=("A", "C"),
            priority_categories=("E",),
        )

        self.assertEqual(ordered, ("E", "B", "D", "A", "C"))

    def test_first_exposure_order_preserves_rank_inside_unseen_group(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("C", "A", "D", "B"),
            previously_scheduled_categories=("A",),
            priority_categories=(),
        )

        self.assertEqual(ordered, ("C", "D", "B", "A"))

    def test_first_exposure_order_restores_normal_rank_after_all_scheduled(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("B", "A", "C"),
            previously_scheduled_categories=("A", "B", "C"),
            priority_categories=("C",),
        )

        self.assertEqual(ordered, ("B", "A", "C"))

    def test_first_exposure_order_can_be_disabled_for_non_survey_exploration_modes(self):
        ordered = ordered_first_exposure_categories(
            ranked_categories=("A", "B", "C"),
            previously_scheduled_categories=("A",),
            priority_categories=("A",),
            enabled=False,
        )

        self.assertEqual(ordered, ("A", "B", "C"))


if __name__ == "__main__":
    unittest.main()
