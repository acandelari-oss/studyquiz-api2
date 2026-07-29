import unittest

from planner.professor_knowledge import (
    ProfessorKnowledge,
    ProfessorKnowledgeActivitySize,
    ProfessorKnowledgeCategoryStrategy,
    ProfessorKnowledgeModuleActivityStrategy,
    ProfessorKnowledgeModuleStrategy,
    ProfessorKnowledgeTopic,
    ProfessorTeachingContext,
)
from planner.professor_observations import (
    CoverageIncomplete,
    FirstExposure,
    GoalNotYetDemonstrated,
    PriorityCategoryIncluded,
    ProfessorObservationBuilder,
)


class PlannerProfessorObservationTests(unittest.TestCase):
    def _knowledge(
        self,
        *,
        additional_modules_remain=False,
        include_remaining_topics=False,
        include_first_exposure=False,
        include_priority_strategy=False,
        activity_type="QUIZ",
    ):
        reasoning_codes = (
            ("INSUFFICIENT_EVIDENCE",)
            if include_first_exposure
            else ("ASSESS_KNOWLEDGE",)
        )

        remaining_topics_by_category = {}
        remaining_categories = ()

        if include_remaining_topics:
            remaining_categories = ("Future Category",)
            remaining_topics_by_category = {
                "Future Category": (
                    ProfessorKnowledgeTopic(
                        id="future-topic-1",
                        title="Future Topic",
                        order=1,
                    ),
                ),
            }

        category_strategies = ()

        if include_priority_strategy:
            category_strategies = (
                ProfessorKnowledgeCategoryStrategy(
                    category="Property",
                    strategy_code="explore",
                    depth_code="deep",
                    reasoning_code="LOW_COVERAGE",
                    priority_score=25.0,
                ),
            )

        return ProfessorKnowledge(
            project_id="project-1",
            project_name="Private Law",
            module_count=1,
            visible_module_count=1,
            additional_modules_remain=additional_modules_remain,
            selected_categories=("Property",),
            selected_topics_by_category={
                "Property": (
                    ProfessorKnowledgeTopic(
                        id="topic-1",
                        title="Ownership",
                        order=1,
                    ),
                ),
            },
            activity_sizes=(
                ProfessorKnowledgeActivitySize(
                    activity_id="activity-1",
                    module_index=1,
                    activity_type=activity_type,
                    category="Property",
                    num_questions=10 if "QUIZ" in activity_type else None,
                    num_cards=12 if "FLASHCARD" in activity_type else None,
                    estimated_duration_minutes=15,
                ),
            ),
            category_strategies=category_strategies,
            module_strategies=(
                ProfessorKnowledgeModuleStrategy(
                    module_index=1,
                    daily_goal_code="ASSESS_CURRENT_LEVEL",
                    activities=(
                        ProfessorKnowledgeModuleActivityStrategy(
                            category="Property",
                            activity_type=activity_type,
                            depth_code="NORMAL",
                            estimated_questions=10 if "QUIZ" in activity_type else 0,
                            estimated_flashcards=12 if "FLASHCARD" in activity_type else 0,
                            reasoning_codes=reasoning_codes,
                        ),
                    ),
                ),
            ),
            teaching_contexts=(
                ProfessorTeachingContext(
                    module_index=1,
                    conceptual_summary="This module establishes the current concept.",
                    prerequisite_level="foundations",
                    learning_progression="This supports the next teaching decision.",
                    expected_mastery="By the end, you should explain the core relation.",
                    activity_rationale="The objective is to verify conceptual stability.",
                ),
            ),
            remaining_categories=remaining_categories,
            remaining_topics_by_category=remaining_topics_by_category,
        )

    def test_goal_not_yet_demonstrated_from_low_learning_objective_evaluation(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(),
            module_results_by_index={
                1: {
                    "activity_results": [
                        {
                            "activity_type": "quiz",
                            "accuracy": 0.55,
                        },
                    ],
                },
            },
        )

        goal_observations = [
            observation
            for observation in observations
            if isinstance(observation, GoalNotYetDemonstrated)
        ]

        self.assertEqual(len(goal_observations), 1)
        self.assertEqual(goal_observations[0].module_index, 1)
        self.assertEqual(goal_observations[0].confidence, 0.8)
        evaluation = goal_observations[0].metadata["learning_objective_evaluation"]
        self.assertEqual(evaluation["status"], "NOT_YET_DEMONSTRATED")
        self.assertIn("QuizAccuracyBelowThreshold", evaluation["evidence"])

    def test_goal_not_yet_demonstrated_is_not_emitted_without_quiz_evidence(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(),
            module_results_by_index={},
        )

        self.assertFalse(
            any(
                isinstance(observation, GoalNotYetDemonstrated)
                for observation in observations
            )
        )

    def test_goal_not_yet_demonstrated_is_not_emitted_when_quiz_evidence_exists(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(),
            module_results_by_index={
                1: {
                    "activity_results": [
                        {
                            "activity_type": "quiz",
                            "accuracy": 0.8,
                        },
                    ],
                },
            },
        )

        self.assertFalse(
            any(
                isinstance(observation, GoalNotYetDemonstrated)
                for observation in observations
            )
        )

    def test_goal_not_yet_demonstrated_requires_quiz_activity(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(activity_type="FLASHCARDS"),
            module_results_by_index={},
        )

        self.assertFalse(
            any(
                isinstance(observation, GoalNotYetDemonstrated)
                for observation in observations
            )
        )

    def test_first_exposure_from_existing_insufficient_evidence_code(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(include_first_exposure=True),
        )

        first_exposure = [
            observation
            for observation in observations
            if isinstance(observation, FirstExposure)
        ]

        self.assertEqual(len(first_exposure), 1)
        self.assertEqual(first_exposure[0].module_index, 1)
        self.assertEqual(first_exposure[0].category, "Property")
        self.assertIn("INSUFFICIENT_EVIDENCE", first_exposure[0].metadata["reasoning_codes"])

    def test_coverage_incomplete_from_remaining_topics(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(
                additional_modules_remain=True,
                include_remaining_topics=True,
            ),
        )

        coverage = [
            observation
            for observation in observations
            if isinstance(observation, CoverageIncomplete)
        ]

        self.assertEqual(len(coverage), 1)
        self.assertEqual(coverage[0].remaining_category_count, 1)
        self.assertEqual(coverage[0].remaining_topic_count, 1)
        self.assertTrue(coverage[0].metadata["additional_modules_remain"])

    def test_priority_category_included_from_existing_strategy(self):
        observations = ProfessorObservationBuilder().build(
            knowledge=self._knowledge(include_priority_strategy=True),
        )

        priority = [
            observation
            for observation in observations
            if isinstance(observation, PriorityCategoryIncluded)
        ]

        self.assertEqual(len(priority), 1)
        self.assertEqual(priority[0].category, "Property")
        self.assertEqual(priority[0].priority_score, 25.0)
        self.assertEqual(priority[0].metadata["reasoning_code"], "LOW_COVERAGE")

    def test_priority_category_ignores_unselected_categories(self):
        knowledge = self._knowledge(include_priority_strategy=True)
        knowledge = ProfessorKnowledge(
            **{
                **knowledge.__dict__,
                "selected_categories": ("Contracts",),
            }
        )

        observations = ProfessorObservationBuilder().build(knowledge=knowledge)

        self.assertFalse(
            any(
                isinstance(observation, PriorityCategoryIncluded)
                for observation in observations
            )
        )


if __name__ == "__main__":
    unittest.main()
