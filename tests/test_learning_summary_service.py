import unittest
from datetime import datetime, timezone

from sqlalchemy import create_engine, text

from learning_summary_service import get_learning_summary, summarize_learning_sessions


class LearningSummaryServiceTests(unittest.TestCase):
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
        self.db.execute(text("""
            create table quiz_attempts (
                id text primary key,
                user_id text,
                project_id text,
                score integer,
                total_questions integer,
                created_at text
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

    def _insert_quiz_attempt(
        self,
        attempt_id,
        score,
        total_questions,
        created_at="2026-07-30T10:30:00+00:00",
        user_id="user-1",
        project_id="project-1",
    ):
        self.db.execute(
            text("""
                insert into quiz_attempts
                (id, user_id, project_id, score, total_questions, created_at)
                values
                (:id, :user_id, :project_id, :score, :total_questions, :created_at)
            """),
            {
                "id": attempt_id,
                "user_id": user_id,
                "project_id": project_id,
                "score": score,
                "total_questions": total_questions,
                "created_at": created_at,
            },
        )
        self.db.commit()

    def test_zero_sessions(self):
        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(summary["total_sessions"], 0)
        self.assertEqual(summary["completed_sessions"], 0)
        self.assertEqual(summary["abandoned_sessions"], 0)
        self.assertEqual(summary["completion_rate"], 0)
        self.assertEqual(summary["total_study_seconds"], 0)
        self.assertEqual(summary["current_streak"], 0)
        self.assertIsNone(summary["favorite_activity"])
        self.assertEqual(summary["activities"]["quiz"], 0)
        self.assertEqual(summary["activities"]["flashcards"], 0)
        self.assertEqual(summary["activities"]["ask"], 0)
        self.assertEqual(summary["activities"]["active_recall"], 0)
        self.assertEqual(summary["activities"]["planner"], 0)
        self.assertEqual(summary["quiz_accuracy_history"], [])

    def test_only_completed_sessions(self):
        self._insert_session("session-1", "quiz", "completed")
        self._insert_session(
            "session-2",
            "flashcards",
            "completed",
            completed_at="2026-07-30T10:15:00+00:00",
        )

        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(summary["total_sessions"], 2)
        self.assertEqual(summary["completed_sessions"], 2)
        self.assertEqual(summary["abandoned_sessions"], 0)
        self.assertEqual(summary["completion_rate"], 100.0)
        self.assertEqual(summary["total_study_seconds"], 2700)
        self.assertIsInstance(summary["current_streak"], int)
        self.assertEqual(summary["activities"]["quiz"], 1)
        self.assertEqual(summary["activities"]["flashcards"], 1)
        self.assertEqual(summary["quiz_accuracy_history"], [])

    def test_only_abandoned_sessions(self):
        self._insert_session("session-1", "ask", "abandoned")
        self._insert_session("session-2", "active_recall", "abandoned")

        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(summary["total_sessions"], 2)
        self.assertEqual(summary["completed_sessions"], 0)
        self.assertEqual(summary["abandoned_sessions"], 2)
        self.assertEqual(summary["completion_rate"], 0.0)
        self.assertEqual(summary["total_study_seconds"], 0)
        self.assertEqual(summary["current_streak"], 0)
        self.assertEqual(summary["activities"]["ask"], 1)
        self.assertEqual(summary["activities"]["active_recall"], 1)

    def test_mixed_sessions_are_user_scoped(self):
        self._insert_session("session-1", "quiz", "completed")
        self._insert_session("session-2", "quiz", "abandoned")
        self._insert_session("session-3", "planner", "completed")
        self._insert_session("session-other-user", "quiz", "completed", user_id="user-2")

        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(summary["total_sessions"], 3)
        self.assertEqual(summary["completed_sessions"], 2)
        self.assertEqual(summary["abandoned_sessions"], 1)
        self.assertEqual(summary["completion_rate"], 66.7)
        self.assertEqual(summary["total_study_seconds"], 3600)
        self.assertIsInstance(summary["current_streak"], int)
        self.assertEqual(summary["favorite_activity"], "quiz")
        self.assertEqual(summary["activities"]["quiz"], 2)
        self.assertEqual(summary["activities"]["planner"], 1)

    def test_summary_can_be_scoped_to_one_project(self):
        self._insert_session("project-1-session", "quiz", "completed")
        self._insert_session(
            "project-2-session",
            "flashcards",
            "completed",
            project_id="project-2",
        )
        self._insert_quiz_attempt("project-1-attempt", score=8, total_questions=10)
        self._insert_quiz_attempt(
            "project-2-attempt",
            score=2,
            total_questions=10,
            project_id="project-2",
        )

        summary = get_learning_summary(self.db, "user-1", project_id="project-1")

        self.assertEqual(summary["total_sessions"], 1)
        self.assertEqual(summary["activities"]["quiz"], 1)
        self.assertEqual(summary["activities"]["flashcards"], 0)
        self.assertEqual(len(summary["quiz_accuracy_history"]), 1)
        self.assertEqual(summary["quiz_accuracy_history"][0]["accuracy"], 80.0)

    def test_invalid_timestamps_do_not_fail_or_add_minutes(self):
        self._insert_session(
            "session-1",
            "quiz",
            "completed",
            started_at="not-a-date",
            completed_at="2026-07-30T10:30:00+00:00",
        )
        self._insert_session(
            "session-2",
            "flashcards",
            "completed",
            started_at=None,
            completed_at=None,
        )

        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(summary["total_sessions"], 2)
        self.assertEqual(summary["completed_sessions"], 2)
        self.assertEqual(summary["completion_rate"], 100.0)
        self.assertEqual(summary["total_study_seconds"], 0)
        self.assertEqual(summary["current_streak"], 0)

    def test_current_streak_counts_consecutive_completed_days_ending_today(self):
        now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
        rows = [
            ("quiz", "completed", "2026-07-31T09:00:00+00:00", "2026-07-31T09:20:00+00:00"),
            ("flashcards", "completed", "2026-07-30T09:00:00+00:00", "2026-07-30T09:20:00+00:00"),
            ("ask", "completed", "2026-07-29T09:00:00+00:00", "2026-07-29T09:20:00+00:00"),
        ]

        summary = summarize_learning_sessions(rows, now=now)

        self.assertEqual(summary["current_streak"], 3)

    def test_current_streak_may_end_yesterday(self):
        now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
        rows = [
            ("quiz", "completed", "2026-07-30T09:00:00+00:00", "2026-07-30T09:20:00+00:00"),
            ("flashcards", "completed", "2026-07-29T09:00:00+00:00", "2026-07-29T09:20:00+00:00"),
        ]

        summary = summarize_learning_sessions(rows, now=now)

        self.assertEqual(summary["current_streak"], 2)

    def test_current_streak_ignores_abandoned_sessions_and_duplicate_days(self):
        now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
        rows = [
            ("quiz", "completed", "2026-07-31T09:00:00+00:00", "2026-07-31T09:20:00+00:00"),
            ("flashcards", "completed", "2026-07-31T10:00:00+00:00", "2026-07-31T10:20:00+00:00"),
            ("ask", "abandoned", "2026-07-30T09:00:00+00:00", "2026-07-30T09:20:00+00:00"),
        ]

        summary = summarize_learning_sessions(rows, now=now)

        self.assertEqual(summary["current_streak"], 1)

    def test_current_streak_is_zero_when_today_and_yesterday_are_missing(self):
        now = datetime(2026, 7, 31, 12, 0, tzinfo=timezone.utc)
        rows = [
            ("quiz", "completed", "2026-07-29T09:00:00+00:00", "2026-07-29T09:20:00+00:00"),
            ("flashcards", "completed", "2026-07-28T09:00:00+00:00", "2026-07-28T09:20:00+00:00"),
        ]

        summary = summarize_learning_sessions(rows, now=now)

        self.assertEqual(summary["current_streak"], 0)

    def test_quiz_accuracy_history_uses_quiz_attempts_chronologically(self):
        self._insert_quiz_attempt(
            "attempt-2",
            score=8,
            total_questions=10,
            created_at="2026-07-31T09:00:00+00:00",
        )
        self._insert_quiz_attempt(
            "attempt-1",
            score=6,
            total_questions=10,
            created_at="2026-07-30T09:00:00+00:00",
        )

        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(
            summary["quiz_accuracy_history"],
            [
                {
                    "completed_at": "2026-07-30T09:00:00+00:00",
                    "accuracy": 60.0,
                },
                {
                    "completed_at": "2026-07-31T09:00:00+00:00",
                    "accuracy": 80.0,
                },
            ],
        )

    def test_quiz_accuracy_history_ignores_other_users_and_invalid_totals(self):
        self._insert_quiz_attempt("attempt-1", score=7, total_questions=10)
        self._insert_quiz_attempt("attempt-invalid", score=4, total_questions=0)
        self._insert_quiz_attempt(
            "attempt-other-user",
            score=10,
            total_questions=10,
            user_id="user-2",
        )

        summary = get_learning_summary(self.db, "user-1")

        self.assertEqual(len(summary["quiz_accuracy_history"]), 1)
        self.assertEqual(summary["quiz_accuracy_history"][0]["accuracy"], 70.0)


if __name__ == "__main__":
    unittest.main()
