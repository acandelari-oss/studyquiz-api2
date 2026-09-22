import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from learning_intelligence_service import get_learning_intelligence


class LearningIntelligenceServiceTests(unittest.TestCase):
    fixed_now = datetime(2026, 7, 30, 12, 0, tzinfo=timezone.utc)

    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.db = self.engine.connect()
        self.db.execute(text("""
            create table learning_sessions (
                id text primary key,
                user_id text,
                project_id text,
                session_type text,
                started_at text,
                completed_at text,
                status text
            )
        """))
        self.db.commit()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _insert_session(
        self,
        session_id,
        session_type,
        status,
        started_at="2026-07-30T10:00:00+00:00",
        completed_at="2026-07-30T10:30:00+00:00",
        user_id="user-1",
        project_id="project-1",
    ):
        self.db.execute(
            text("""
                insert into learning_sessions
                (id, user_id, project_id, session_type, started_at, completed_at, status)
                values
                (:id, :user_id, :project_id, :session_type, :started_at, :completed_at, :status)
            """),
            {
                "id": session_id,
                "user_id": user_id,
                "project_id": project_id,
                "session_type": session_type,
                "started_at": started_at,
                "completed_at": completed_at,
                "status": status,
            },
        )
        self.db.commit()

    def _insights_by_type(self):
        return {
            insight["type"]: insight
            for insight in get_learning_intelligence(
                self.db,
                "user-1",
                now=self.fixed_now,
            )
        }

    def test_empty_user_returns_recent_inactivity_only(self):
        insights = self._insights_by_type()

        self.assertEqual(list(insights.keys()), ["recent_activity"])
        self.assertEqual(insights["recent_activity"]["level"], "warning")

    def test_high_completion_rate_is_positive(self):
        for index in range(9):
            self._insert_session("completed-{}".format(index), "quiz", "completed")
        self._insert_session("abandoned-1", "quiz", "abandoned")

        insight = self._insights_by_type()["completion_rate"]

        self.assertEqual(insight["level"], "positive")
        self.assertIn("90.0%", insight["message"])

    def test_low_completion_rate_is_warning(self):
        self._insert_session("completed-1", "quiz", "completed")
        self._insert_session("abandoned-1", "quiz", "abandoned")
        self._insert_session("abandoned-2", "flashcards", "abandoned")

        insight = self._insights_by_type()["completion_rate"]

        self.assertEqual(insight["level"], "warning")
        self.assertIn("33.3%", insight["message"])

    def test_dominant_completed_activity_generates_info_insight(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("quiz-2", "quiz", "completed")
        self._insert_session("flashcards-1", "flashcards", "completed")

        insight = self._insights_by_type()["favorite_activity"]

        self.assertEqual(insight["level"], "info")
        self.assertIn("Quizzes", insight["message"])

    def test_no_favorite_activity_when_no_completed_majority_exists(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("flashcards-1", "flashcards", "completed")

        insights = self._insights_by_type()

        self.assertNotIn("favorite_activity", insights)

    def test_recent_inactivity_warning_when_no_completed_session_in_last_week(self):
        self._insert_session(
            "old-completed",
            "quiz",
            "completed",
            started_at="2026-07-01T10:00:00+00:00",
            completed_at="2026-07-01T10:30:00+00:00",
        )

        insights = get_learning_intelligence(
            self.db,
            "user-1",
            now=self.fixed_now,
        )
        recent = [
            insight for insight in insights
            if insight["type"] == "recent_activity"
        ][0]

        self.assertEqual(recent["level"], "warning")

    def test_duration_classification_short_balanced_and_long(self):
        scenarios = [
            (
                "short",
                "2026-07-30T10:00:00+00:00",
                "2026-07-30T10:05:00+00:00",
                "Short study sessions",
            ),
            (
                "balanced",
                "2026-07-30T10:00:00+00:00",
                "2026-07-30T10:30:00+00:00",
                "Balanced study sessions",
            ),
            (
                "long",
                "2026-07-30T10:00:00+00:00",
                "2026-07-30T11:00:01+00:00",
                "Long study sessions",
            ),
        ]

        for label, started_at, completed_at, expected_title in scenarios:
            with self.subTest(label=label):
                self.db.execute(text("delete from learning_sessions"))
                self.db.commit()
                self._insert_session(
                    "session-{}".format(label),
                    "planner",
                    "completed",
                    started_at=started_at,
                    completed_at=completed_at,
                )

                insight = self._insights_by_type()["average_session_duration"]

                self.assertEqual(insight["title"], expected_title)

    def test_invalid_timestamps_do_not_fail(self):
        self._insert_session(
            "invalid",
            "quiz",
            "completed",
            started_at="not-a-date",
            completed_at=None,
        )

        insights = self._insights_by_type()

        self.assertIn("completion_rate", insights)
        self.assertIn("average_session_duration", insights)

    def test_only_authenticated_user_sessions_are_analyzed(self):
        self._insert_session("user-1-session", "quiz", "completed")
        self._insert_session(
            "user-2-session",
            "flashcards",
            "abandoned",
            user_id="user-2",
        )

        insight = self._insights_by_type()["completion_rate"]

        self.assertEqual(insight["level"], "positive")
        self.assertIn("100.0%", insight["message"])

    def test_intelligence_can_be_scoped_to_one_project(self):
        self._insert_session("project-1-completed", "quiz", "completed")
        self._insert_session(
            "project-2-abandoned",
            "quiz",
            "abandoned",
            project_id="project-2",
        )

        insights = {
            insight["type"]: insight
            for insight in get_learning_intelligence(
                self.db,
                "user-1",
                now=self.fixed_now,
                project_id="project-1",
            )
        }

        self.assertEqual(insights["completion_rate"]["level"], "positive")
        self.assertIn("100.0%", insights["completion_rate"]["message"])


if __name__ == "__main__":
    unittest.main()
