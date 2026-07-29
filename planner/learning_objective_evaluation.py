"""Deterministic learning objective evaluation.

This module evaluates whether an activity has demonstrated the educational
objective using only existing runtime evidence. It does not call AI services,
generate text, mutate Planner state, or introduce new analytics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Optional, Sequence


class LearningObjectiveStatus(str, Enum):
    DEMONSTRATED = "DEMONSTRATED"
    NOT_YET_DEMONSTRATED = "NOT_YET_DEMONSTRATED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class LearningObjectiveConfidence(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class LearningObjectiveEvidenceCode(str, Enum):
    MODULE_OBJECTIVE_PRESENT = "ModuleObjectivePresent"
    MODULE_OBJECTIVE_MISSING = "ModuleObjectiveMissing"
    QUIZ_ACTIVITY_PRESENT = "QuizActivityPresent"
    QUIZ_COMPLETED = "QuizCompleted"
    QUIZ_EVIDENCE_MISSING = "QuizEvidenceMissing"
    QUIZ_ACCURACY_AT_OR_ABOVE_THRESHOLD = "QuizAccuracyAtOrAboveThreshold"
    QUIZ_ACCURACY_BELOW_THRESHOLD = "QuizAccuracyBelowThreshold"
    ACTIVITY_TYPE_UNSUPPORTED = "ActivityTypeUnsupported"


@dataclass(frozen=True)
class LearningObjectiveEvaluation:
    """Structured result of deterministic objective evaluation."""

    status: LearningObjectiveStatus
    confidence: LearningObjectiveConfidence
    evidence: Sequence[LearningObjectiveEvidenceCode] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


class LearningObjectiveEvaluator:
    """Evaluate module objective demonstration from existing runtime evidence."""

    DEMONSTRATED_ACCURACY_THRESHOLD = 0.8

    def evaluate(
        self,
        *,
        module_objective: Optional[str],
        activity_type: Optional[str] = None,
        runtime_result: Optional[Any] = None,
    ) -> LearningObjectiveEvaluation:
        evidence = []

        if not str(module_objective or "").strip():
            return LearningObjectiveEvaluation(
                status=LearningObjectiveStatus.INSUFFICIENT_EVIDENCE,
                confidence=LearningObjectiveConfidence.LOW,
                evidence=(LearningObjectiveEvidenceCode.MODULE_OBJECTIVE_MISSING,),
                metadata={},
            )

        evidence.append(LearningObjectiveEvidenceCode.MODULE_OBJECTIVE_PRESENT)

        quiz_result = self._quiz_result(activity_type, runtime_result)

        if quiz_result is None:
            if activity_type and not self._is_quiz_type(activity_type):
                evidence.append(LearningObjectiveEvidenceCode.ACTIVITY_TYPE_UNSUPPORTED)
            else:
                evidence.append(LearningObjectiveEvidenceCode.QUIZ_EVIDENCE_MISSING)
            return LearningObjectiveEvaluation(
                status=LearningObjectiveStatus.INSUFFICIENT_EVIDENCE,
                confidence=LearningObjectiveConfidence.LOW,
                evidence=tuple(evidence),
                metadata={},
            )

        evidence.append(LearningObjectiveEvidenceCode.QUIZ_ACTIVITY_PRESENT)

        if self._is_completed(quiz_result):
            evidence.append(LearningObjectiveEvidenceCode.QUIZ_COMPLETED)

        accuracy = self._accuracy(quiz_result)

        if accuracy is None:
            evidence.append(LearningObjectiveEvidenceCode.QUIZ_EVIDENCE_MISSING)
            return LearningObjectiveEvaluation(
                status=LearningObjectiveStatus.INSUFFICIENT_EVIDENCE,
                confidence=LearningObjectiveConfidence.LOW,
                evidence=tuple(evidence),
                metadata={},
            )

        if accuracy >= self.DEMONSTRATED_ACCURACY_THRESHOLD:
            evidence.append(
                LearningObjectiveEvidenceCode.QUIZ_ACCURACY_AT_OR_ABOVE_THRESHOLD
            )
            return LearningObjectiveEvaluation(
                status=LearningObjectiveStatus.DEMONSTRATED,
                confidence=LearningObjectiveConfidence.HIGH,
                evidence=tuple(evidence),
                metadata={
                    "activity_type": "quiz",
                    "accuracy": accuracy,
                    "threshold": self.DEMONSTRATED_ACCURACY_THRESHOLD,
                },
            )

        evidence.append(LearningObjectiveEvidenceCode.QUIZ_ACCURACY_BELOW_THRESHOLD)
        return LearningObjectiveEvaluation(
            status=LearningObjectiveStatus.NOT_YET_DEMONSTRATED,
            confidence=LearningObjectiveConfidence.MEDIUM,
            evidence=tuple(evidence),
            metadata={
                "activity_type": "quiz",
                "accuracy": accuracy,
                "threshold": self.DEMONSTRATED_ACCURACY_THRESHOLD,
            },
        )

    def _quiz_result(
        self,
        activity_type: Optional[str],
        runtime_result: Optional[Any],
    ) -> Optional[Any]:
        if runtime_result is None:
            return None

        activity_results = self._value(runtime_result, "activity_results") or self._value(
            runtime_result, "activityResults"
        ) or ()

        for result in activity_results:
            if self._is_quiz_type(
                self._value(result, "activity_type") or self._value(result, "activityType")
            ):
                return result

        if self._is_quiz_type(activity_type) or self._is_quiz_type(
            self._value(runtime_result, "activity_type")
            or self._value(runtime_result, "activityType")
        ):
            return runtime_result

        return None

    def _is_completed(self, runtime_result: Any) -> bool:
        completed = self._value(runtime_result, "completed")

        if completed is not None:
            return bool(completed)

        return self._accuracy(runtime_result) is not None

    def _accuracy(self, runtime_result: Any) -> Optional[float]:
        accuracy = self._value(runtime_result, "accuracy")

        if accuracy is not None:
            try:
                value = float(accuracy)
            except (TypeError, ValueError):
                return None

            if value > 1:
                value = value / 100

            return max(0.0, min(1.0, value))

        correct = self._value(runtime_result, "correct")
        total = self._value(runtime_result, "total")

        if correct is None or total in (None, 0):
            return None

        try:
            return max(0.0, min(1.0, float(correct) / float(total)))
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _is_quiz_type(self, activity_type: Optional[Any]) -> bool:
        return "QUIZ" in str(activity_type or "").upper()

    def _value(self, value: Any, key: str) -> Any:
        if isinstance(value, Mapping):
            return value.get(key)

        return getattr(value, key, None)
