"""Backend Planner State Evaluation for project entry flow.

This module decides the current Planner entry state from persisted Planner
weeks and real learning evidence. It does not generate weeks, execute
activities, or contain Professor logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Set

from sqlalchemy import text

from .planner_models import Week
from .planner_repository import PlannerRepository
from .planner_state import PlannerState


LEARNING_COVERAGE_READY_THRESHOLD = 0.50


@dataclass(frozen=True)
class LearningCoverage:
    """Coverage of topics with at least one real learning interaction."""

    covered_topics: int
    total_topics: int
    ratio: float


@dataclass(frozen=True)
class PlannerStateEvaluation:
    """Authoritative backend decision for a project's Planner entry state."""

    state: PlannerState
    learning_coverage: LearningCoverage
    active_week: Optional[Week] = None


class PlannerStateEvaluator:
    """Evaluate Planner entry state for one explicit project."""

    def __init__(
        self,
        db: Any,
        repository: Optional[PlannerRepository] = None,
    ) -> None:
        self.db = db
        self.repository = repository or PlannerRepository(db)

    def evaluate(self, project_id: str) -> PlannerStateEvaluation:
        """Return the Planner state for one project without mutating data."""

        active_week = self.repository.load_active_week(project_id=project_id)
        learning_coverage = self.calculate_learning_coverage(project_id=project_id)

        if active_week:
            return PlannerStateEvaluation(
                state=PlannerState.ACTIVE_WEEK,
                learning_coverage=learning_coverage,
                active_week=active_week,
            )

        if learning_coverage.ratio < LEARNING_COVERAGE_READY_THRESHOLD:
            return PlannerStateEvaluation(
                state=PlannerState.NEW_PROJECT,
                learning_coverage=learning_coverage,
            )

        return PlannerStateEvaluation(
            state=PlannerState.READY_FOR_FIRST_PLAN,
            learning_coverage=learning_coverage,
        )

    def calculate_learning_coverage(self, project_id: str) -> LearningCoverage:
        """Calculate covered topics / total topics using real interactions."""

        topic_rows = self.db.execute(
            text("""
                select topic
                from topics
                where project_id = :project_id
                and topic is not null
                and is_display_topic = true
            """),
            {"project_id": project_id},
        ).fetchall()
        topic_keys = {
            self._normalize_topic(row[0])
            for row in topic_rows
            if self._normalize_topic(row[0])
        }

        if not topic_keys:
            return LearningCoverage(
                covered_topics=0,
                total_topics=0,
                ratio=0.0,
            )

        evidence_topic_keys = (
            self._load_completed_quiz_topic_keys(project_id=project_id)
            | self._load_reviewed_flashcard_topic_keys(project_id=project_id)
        )
        covered_topic_count = len(topic_keys & evidence_topic_keys)

        return LearningCoverage(
            covered_topics=covered_topic_count,
            total_topics=len(topic_keys),
            ratio=covered_topic_count / len(topic_keys),
        )

    def _load_completed_quiz_topic_keys(self, project_id: str) -> Set[str]:
        rows = self.db.execute(
            text("""
                select coalesce(qa.topic, qq.topic) as topic
                from quiz_answers qa
                join quiz_questions qq
                    on qq.id = qa.question_id
                join quizzes q
                    on q.id = qq.quiz_id
                where q.project_id = :project_id
                and coalesce(qa.topic, qq.topic) is not null
            """),
            {"project_id": project_id},
        ).fetchall()

        return {
            self._normalize_topic(row[0])
            for row in rows
            if self._normalize_topic(row[0])
        }

    def _load_reviewed_flashcard_topic_keys(self, project_id: str) -> Set[str]:
        rows = self.db.execute(
            text("""
                select topic
                from flashcard_reviews
                where project_id = :project_id
                and topic is not null
            """),
            {"project_id": project_id},
        ).fetchall()

        return {
            self._normalize_topic(row[0])
            for row in rows
            if self._normalize_topic(row[0])
        }

    def _normalize_topic(self, value: Any) -> str:
        return " ".join(str(value or "").strip().lower().split())


def serialize_planner_state_evaluation(
    evaluation: PlannerStateEvaluation,
) -> dict:
    """Serialize a Planner State Evaluation for API responses."""

    return {
        "state": evaluation.state.value,
        "learning_coverage": {
            "covered_topics": evaluation.learning_coverage.covered_topics,
            "total_topics": evaluation.learning_coverage.total_topics,
            "ratio": evaluation.learning_coverage.ratio,
        },
        "week": None,
    }
