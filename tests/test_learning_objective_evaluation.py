import unittest

from planner.learning_objective_evaluation import (
    LearningObjectiveConfidence,
    LearningObjectiveEvaluator,
    LearningObjectiveEvidenceCode,
    LearningObjectiveStatus,
)


class LearningObjectiveEvaluationTests(unittest.TestCase):
    def test_high_quiz_score_demonstrates_objective(self):
        evaluation = LearningObjectiveEvaluator().evaluate(
            module_objective="By the end, you should distinguish the main concepts.",
            activity_type="quiz",
            runtime_result={
                "activity_type": "quiz",
                "completed": True,
                "accuracy": 0.85,
            },
        )

        self.assertEqual(evaluation.status, LearningObjectiveStatus.DEMONSTRATED)
        self.assertEqual(evaluation.confidence, LearningObjectiveConfidence.HIGH)
        self.assertIn(
            LearningObjectiveEvidenceCode.QUIZ_ACCURACY_AT_OR_ABOVE_THRESHOLD,
            evaluation.evidence,
        )
        self.assertEqual(evaluation.metadata["accuracy"], 0.85)

    def test_low_quiz_score_does_not_yet_demonstrate_objective(self):
        evaluation = LearningObjectiveEvaluator().evaluate(
            module_objective="By the end, you should distinguish the main concepts.",
            activity_type="quiz",
            runtime_result={
                "activity_type": "quiz",
                "completed": True,
                "correct": 5,
                "total": 10,
            },
        )

        self.assertEqual(
            evaluation.status,
            LearningObjectiveStatus.NOT_YET_DEMONSTRATED,
        )
        self.assertEqual(evaluation.confidence, LearningObjectiveConfidence.MEDIUM)
        self.assertIn(
            LearningObjectiveEvidenceCode.QUIZ_ACCURACY_BELOW_THRESHOLD,
            evaluation.evidence,
        )
        self.assertEqual(evaluation.metadata["accuracy"], 0.5)

    def test_no_quiz_evidence_is_insufficient_evidence(self):
        evaluation = LearningObjectiveEvaluator().evaluate(
            module_objective="By the end, you should distinguish the main concepts.",
            activity_type="quiz",
            runtime_result={
                "activity_type": "quiz",
                "completed": True,
            },
        )

        self.assertEqual(
            evaluation.status,
            LearningObjectiveStatus.INSUFFICIENT_EVIDENCE,
        )
        self.assertEqual(evaluation.confidence, LearningObjectiveConfidence.LOW)
        self.assertIn(
            LearningObjectiveEvidenceCode.QUIZ_EVIDENCE_MISSING,
            evaluation.evidence,
        )

    def test_missing_objective_is_insufficient_evidence(self):
        evaluation = LearningObjectiveEvaluator().evaluate(
            module_objective="",
            activity_type="quiz",
            runtime_result={
                "activity_type": "quiz",
                "accuracy": 0.95,
            },
        )

        self.assertEqual(
            evaluation.status,
            LearningObjectiveStatus.INSUFFICIENT_EVIDENCE,
        )
        self.assertIn(
            LearningObjectiveEvidenceCode.MODULE_OBJECTIVE_MISSING,
            evaluation.evidence,
        )


if __name__ == "__main__":
    unittest.main()
