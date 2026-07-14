import unittest

from planner.category_selector import CategoryAnalytics
from planner.survey_bootstrap import (
    apply_survey_bootstrap_bias,
    should_apply_survey_bootstrap,
)


class PlannerSurveyBootstrapTests(unittest.TestCase):
    def test_survey_bootstrap_applies_only_without_history_or_evidence(self):
        self.assertTrue(
            should_apply_survey_bootstrap(
                survey={"A": "practice"},
                is_first_study_plan=True,
                has_objective_learning_evidence=False,
            )
        )
        self.assertFalse(
            should_apply_survey_bootstrap(
                survey={"A": "practice"},
                is_first_study_plan=False,
                has_objective_learning_evidence=False,
            )
        )
        self.assertFalse(
            should_apply_survey_bootstrap(
                survey={"A": "practice"},
                is_first_study_plan=True,
                has_objective_learning_evidence=True,
            )
        )

    def test_survey_answers_create_moderate_priority_bias(self):
        analytics = {
            "Confident": CategoryAnalytics(),
            "Unsure": CategoryAnalytics(),
            "Practice": CategoryAnalytics(),
        }

        adjusted = apply_survey_bootstrap_bias(
            analytics=analytics,
            survey={
                "Confident": "confident",
                "Unsure": "unsure",
                "Practice": "practice",
            },
        )

        self.assertLess(adjusted["Confident"].priority_weight, 1.0)
        self.assertGreater(adjusted["Unsure"].priority_weight, 1.0)
        self.assertGreater(
            adjusted["Practice"].priority_weight,
            adjusted["Unsure"].priority_weight,
        )
        self.assertEqual(analytics["Confident"].priority_weight, 1.0)


if __name__ == "__main__":
    unittest.main()
