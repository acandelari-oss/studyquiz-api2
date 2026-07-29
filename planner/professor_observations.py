"""Deterministic Professor observation layer.

ProfessorObservations translates existing Planner and runtime data into typed
educational observations. It does not call AI services, build prompts, generate
text, mutate Planner state, or make new planning decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .learning_objective_evaluation import (
    LearningObjectiveEvaluator,
    LearningObjectiveStatus,
)
from .professor_knowledge import ProfessorKnowledge


@dataclass(frozen=True)
class ProfessorObservation:
    """Base observation fact produced deterministically from existing data."""

    confidence: float
    reason: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GoalNotYetDemonstrated(ProfessorObservation):
    """The module objective exists, but runtime quiz evidence is insufficient."""

    module_index: int = 0


@dataclass(frozen=True)
class FirstExposure(ProfessorObservation):
    """The available planning evidence marks the module/category as first exposure."""

    module_index: int = 0
    category: Optional[str] = None


@dataclass(frozen=True)
class CoverageIncomplete(ProfessorObservation):
    """The current Study Plan intentionally covers only part of the material."""

    remaining_category_count: int = 0
    remaining_topic_count: int = 0


@dataclass(frozen=True)
class PriorityCategoryIncluded(ProfessorObservation):
    """A selected category carries an existing deterministic priority signal."""

    category: str = ""
    priority_score: float = 0.0


class ProfessorObservationBuilder:
    """Build V0 Professor observations from existing deterministic inputs."""

    INSUFFICIENT_EVIDENCE_CODES = {
        "INSUFFICIENT_EVIDENCE",
        "LOW_COVERAGE",
        "CALIBRATE_COVERAGE",
        "EXPLORE",
    }

    def __init__(
        self,
        learning_objective_evaluator: Optional[LearningObjectiveEvaluator] = None,
    ):
        self.learning_objective_evaluator = (
            learning_objective_evaluator or LearningObjectiveEvaluator()
        )

    def build(
        self,
        *,
        knowledge: ProfessorKnowledge,
        module_results_by_index: Optional[Mapping[int, Mapping[str, Any]]] = None,
    ) -> Sequence[ProfessorObservation]:
        """Return deterministic observations for the supplied ProfessorKnowledge."""

        module_results_by_index = module_results_by_index or {}
        observations = []

        observations.extend(
            self._goal_observations(
                knowledge=knowledge,
                module_results_by_index=module_results_by_index,
            )
        )
        observations.extend(self._first_exposure_observations(knowledge))
        observations.extend(self._coverage_observations(knowledge))
        observations.extend(self._priority_category_observations(knowledge))

        return tuple(observations)

    def _goal_observations(
        self,
        *,
        knowledge: ProfessorKnowledge,
        module_results_by_index: Mapping[int, Mapping[str, Any]],
    ) -> Sequence[GoalNotYetDemonstrated]:
        observations = []

        for teaching_context in knowledge.teaching_contexts:
            module_index = int(getattr(teaching_context, "module_index", 0) or 0)
            expected_mastery = str(
                getattr(teaching_context, "expected_mastery", "") or ""
            ).strip()

            if not module_index or not expected_mastery:
                continue

            module_activity_type = self._module_activity_type(knowledge, module_index)
            module_has_quiz = "QUIZ" in module_activity_type

            if not module_has_quiz:
                continue

            module_results = module_results_by_index.get(module_index, {})
            evaluation = self.learning_objective_evaluator.evaluate(
                module_objective=expected_mastery,
                activity_type=module_activity_type,
                runtime_result=module_results,
            )

            if evaluation.status != LearningObjectiveStatus.NOT_YET_DEMONSTRATED:
                continue

            observations.append(
                GoalNotYetDemonstrated(
                    module_index=module_index,
                    confidence=0.8,
                    reason="Learning objective evaluation is not yet demonstrated.",
                    metadata={
                        "expected_mastery": expected_mastery,
                        "learning_objective_evaluation": {
                            "status": evaluation.status.value,
                            "confidence": evaluation.confidence.value,
                            "evidence": tuple(
                                evidence.value for evidence in evaluation.evidence
                            ),
                            "metadata": dict(evaluation.metadata),
                        },
                    },
                )
            )

        return tuple(observations)

    def _first_exposure_observations(
        self,
        knowledge: ProfessorKnowledge,
    ) -> Sequence[FirstExposure]:
        observations = []

        for module_strategy in knowledge.module_strategies:
            module_index = int(getattr(module_strategy, "module_index", 0) or 0)

            for activity in getattr(module_strategy, "activities", ()) or ():
                reasoning_codes = {
                    str(code or "").upper()
                    for code in getattr(activity, "reasoning_codes", ()) or ()
                }

                if not reasoning_codes.intersection(self.INSUFFICIENT_EVIDENCE_CODES):
                    continue

                category = getattr(activity, "category", None)
                observations.append(
                    FirstExposure(
                        module_index=module_index,
                        category=category,
                        confidence=0.8,
                        reason=(
                            "The module activity carries an existing insufficient "
                            "evidence signal."
                        ),
                        metadata={
                            "reasoning_codes": tuple(sorted(reasoning_codes)),
                        },
                    )
                )

        return tuple(observations)

    def _coverage_observations(
        self,
        knowledge: ProfessorKnowledge,
    ) -> Sequence[CoverageIncomplete]:
        remaining_topic_count = sum(
            len(topics)
            for topics in (knowledge.remaining_topics_by_category or {}).values()
        )
        remaining_category_count = len(knowledge.remaining_categories or ())

        if not knowledge.additional_modules_remain and remaining_topic_count <= 0:
            return ()

        return (
            CoverageIncomplete(
                confidence=1.0,
                reason=(
                    "The current Study Plan does not include every topic from "
                    "the available project material."
                ),
                metadata={
                    "additional_modules_remain": knowledge.additional_modules_remain,
                    "remaining_categories": tuple(knowledge.remaining_categories or ()),
                },
                remaining_category_count=remaining_category_count,
                remaining_topic_count=remaining_topic_count,
            ),
        )

    def _priority_category_observations(
        self,
        knowledge: ProfessorKnowledge,
    ) -> Sequence[PriorityCategoryIncluded]:
        selected_categories = {
            str(category)
            for category in knowledge.selected_categories
            if str(category).strip()
        }
        observations = []

        for strategy in knowledge.category_strategies:
            category = str(getattr(strategy, "category", "") or "")

            if category not in selected_categories:
                continue

            priority_score = float(getattr(strategy, "priority_score", 0.0) or 0.0)
            reasoning_code = getattr(strategy, "reasoning_code", None)

            if priority_score <= 0 and not reasoning_code:
                continue

            observations.append(
                PriorityCategoryIncluded(
                    category=category,
                    priority_score=priority_score,
                    confidence=1.0,
                    reason=(
                        "A selected category has an existing deterministic "
                        "priority signal."
                    ),
                    metadata={
                        "reasoning_code": reasoning_code,
                        "strategy_code": getattr(strategy, "strategy_code", None),
                        "depth_code": getattr(strategy, "depth_code", None),
                    },
                )
            )

        return tuple(observations)

    def _module_activity_type(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        activity_types = []

        for activity in knowledge.activity_sizes:
            if int(getattr(activity, "module_index", 0) or 0) != module_index:
                continue

            activity_type = str(getattr(activity, "activity_type", "") or "").upper()

            if activity_type:
                activity_types.append(activity_type)

        for module_strategy in knowledge.module_strategies:
            if int(getattr(module_strategy, "module_index", 0) or 0) != module_index:
                continue

            for activity in getattr(module_strategy, "activities", ()) or ():
                activity_type = str(getattr(activity, "activity_type", "") or "").upper()

                if activity_type:
                    activity_types.append(activity_type)

        if any("QUIZ" in activity_type for activity_type in activity_types):
            return "QUIZ"

        if any("FLASHCARD" in activity_type for activity_type in activity_types):
            return "FLASHCARDS"

        return activity_types[0] if activity_types else ""
