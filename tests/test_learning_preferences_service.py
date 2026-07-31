import unittest

from sqlalchemy import create_engine, text

from learning_preferences_service import get_learning_preferences


class LearningPreferencesServiceTests(unittest.TestCase):
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
        completed_at="2026-07-30T10:20:00+00:00",
        user_id="user-1",
    ):
        self.db.execute(
            text("""
                insert into learning_sessions
                (id, user_id, project_id, session_type, started_at, completed_at, status)
                values
                (:id, :user_id, 'project-1', :session_type, :started_at, :completed_at, :status)
            """),
            {
                "id": session_id,
                "user_id": user_id,
                "session_type": session_type,
                "started_at": started_at,
                "completed_at": completed_at,
                "status": status,
            },
        )
        self.db.commit()

    def _preferences(self):
        return get_learning_preferences(self.db, "user-1")

    def test_user_with_no_sessions_returns_empty_state(self):
        preferences = self._preferences()

        self.assertEqual(preferences["total_sessions_observed"], 0)
        self.assertEqual(preferences["completed_sessions_observed"], 0)
        self.assertIsNone(preferences["preferred_activity"])
        self.assertIsNone(preferences["typical_session_seconds"])
        self.assertIsNone(preferences["session_duration_profile"])
        self.assertIsNone(preferences["most_reliable_activity"])
        self.assertEqual(preferences["completion_rate_by_activity"], {
            "quiz": None,
            "flashcards": None,
            "ask": None,
            "active_recall": None,
            "planner": None,
        })

    def test_authenticated_user_scoping(self):
        self._insert_session("user-1-session", "quiz", "completed")
        self._insert_session(
            "user-2-session",
            "flashcards",
            "completed",
            user_id="user-2",
        )

        preferences = self._preferences()

        self.assertEqual(preferences["total_sessions_observed"], 1)
        self.assertEqual(preferences["completed_sessions_observed"], 1)
        self.assertEqual(preferences["completion_rate_by_activity"]["quiz"], 100.0)
        self.assertIsNone(preferences["completion_rate_by_activity"]["flashcards"])

    def test_preferred_activity_with_sufficient_evidence(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("flashcards-1", "flashcards", "completed")
        self._insert_session("flashcards-2", "flashcards", "completed")
        self._insert_session("flashcards-3", "flashcards", "completed")

        preferences = self._preferences()

        self.assertEqual(preferences["preferred_activity"], "flashcards")

    def test_preferred_activity_tie_returns_null(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("quiz-2", "quiz", "completed")
        self._insert_session("flashcards-1", "flashcards", "completed")
        self._insert_session("flashcards-2", "flashcards", "completed")

        preferences = self._preferences()

        self.assertIsNone(preferences["preferred_activity"])

    def test_preferred_activity_requires_three_completed_sessions_overall(self):
        self._insert_session("flashcards-1", "flashcards", "completed")
        self._insert_session("flashcards-2", "flashcards", "completed")

        preferences = self._preferences()

        self.assertIsNone(preferences["preferred_activity"])

    def test_median_duration_with_odd_sample_count(self):
        samples = [
            ("2026-07-30T10:00:00+00:00", "2026-07-30T10:05:00+00:00"),
            ("2026-07-30T10:00:00+00:00", "2026-07-30T10:20:00+00:00"),
            ("2026-07-30T10:00:00+00:00", "2026-07-30T11:00:00+00:00"),
        ]
        for index, (started_at, completed_at) in enumerate(samples):
            self._insert_session(
                "session-{}".format(index),
                "quiz",
                "completed",
                started_at=started_at,
                completed_at=completed_at,
            )

        preferences = self._preferences()

        self.assertEqual(preferences["typical_session_seconds"], 1200)

    def test_median_duration_with_even_sample_count(self):
        self._insert_session(
            "session-1",
            "quiz",
            "completed",
            started_at="2026-07-30T10:00:00+00:00",
            completed_at="2026-07-30T10:10:00+00:00",
        )
        self._insert_session(
            "session-2",
            "quiz",
            "completed",
            started_at="2026-07-30T10:00:00+00:00",
            completed_at="2026-07-30T10:20:00+00:00",
        )
        self._insert_session(
            "session-3",
            "quiz",
            "completed",
            started_at="2026-07-30T10:00:00+00:00",
            completed_at="2026-07-30T10:30:00+00:00",
        )
        self._insert_session(
            "session-4",
            "quiz",
            "completed",
            started_at="2026-07-30T10:00:00+00:00",
            completed_at="2026-07-30T10:40:00+00:00",
        )

        preferences = self._preferences()

        self.assertEqual(preferences["typical_session_seconds"], 1500)

    def test_invalid_and_reversed_timestamps_are_ignored_for_duration(self):
        self._insert_session(
            "valid-1",
            "quiz",
            "completed",
            started_at="2026-07-30T10:00:00+00:00",
            completed_at="2026-07-30T10:10:00+00:00",
        )
        self._insert_session(
            "invalid",
            "quiz",
            "completed",
            started_at="not-a-date",
            completed_at="2026-07-30T10:20:00+00:00",
        )
        self._insert_session(
            "reversed",
            "quiz",
            "completed",
            started_at="2026-07-30T11:00:00+00:00",
            completed_at="2026-07-30T10:00:00+00:00",
        )

        preferences = self._preferences()

        self.assertIsNone(preferences["typical_session_seconds"])

    def test_short_balanced_and_long_duration_profiles(self):
        scenarios = [
            (
                "short",
                [
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T10:05:00+00:00"),
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T10:06:00+00:00"),
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T10:07:00+00:00"),
                ],
                "short",
            ),
            (
                "balanced",
                [
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T10:20:00+00:00"),
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T10:25:00+00:00"),
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T10:30:00+00:00"),
                ],
                "balanced",
            ),
            (
                "long",
                [
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T11:00:00+00:00"),
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T11:05:00+00:00"),
                    ("2026-07-30T10:00:00+00:00", "2026-07-30T11:10:00+00:00"),
                ],
                "long",
            ),
        ]

        for label, samples, expected_profile in scenarios:
            with self.subTest(label=label):
                self.db.execute(text("delete from learning_sessions"))
                self.db.commit()
                for index, (started_at, completed_at) in enumerate(samples):
                    self._insert_session(
                        "{}-{}".format(label, index),
                        "quiz",
                        "completed",
                        started_at=started_at,
                        completed_at=completed_at,
                    )

                preferences = self._preferences()

                self.assertEqual(
                    preferences["session_duration_profile"],
                    expected_profile,
                )

    def test_completion_rate_for_each_activity_and_no_attempts(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("quiz-2", "quiz", "completed")
        self._insert_session("quiz-3", "quiz", "abandoned")
        self._insert_session("flashcards-1", "flashcards", "abandoned")
        self._insert_session("ask-1", "ask", "completed")
        self._insert_session("active-recall-1", "active_recall", "completed")
        self._insert_session("planner-1", "planner", "completed")
        self._insert_session("planner-2", "planner", "abandoned")

        rates = self._preferences()["completion_rate_by_activity"]

        self.assertEqual(rates["quiz"], 66.7)
        self.assertEqual(rates["flashcards"], 0.0)
        self.assertEqual(rates["ask"], 100.0)
        self.assertEqual(rates["active_recall"], 100.0)
        self.assertEqual(rates["planner"], 50.0)

    def test_most_reliable_activity(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("quiz-2", "quiz", "completed")
        self._insert_session("quiz-3", "quiz", "abandoned")
        self._insert_session("flashcards-1", "flashcards", "completed")
        self._insert_session("flashcards-2", "flashcards", "completed")
        self._insert_session("flashcards-3", "flashcards", "completed")

        preferences = self._preferences()

        self.assertEqual(preferences["most_reliable_activity"], "flashcards")

    def test_most_reliable_activity_tie_returns_null(self):
        for activity in ("quiz", "flashcards"):
            for index in range(3):
                self._insert_session(
                    "{}-{}".format(activity, index),
                    activity,
                    "completed",
                )

        preferences = self._preferences()

        self.assertIsNone(preferences["most_reliable_activity"])

    def test_most_reliable_activity_requires_three_attempts(self):
        self._insert_session("quiz-1", "quiz", "completed")
        self._insert_session("quiz-2", "quiz", "completed")
        self._insert_session("flashcards-1", "flashcards", "completed")
        self._insert_session("flashcards-2", "flashcards", "abandoned")

        preferences = self._preferences()

        self.assertIsNone(preferences["most_reliable_activity"])

    def test_unsupported_session_types_are_ignored(self):
        self._insert_session("unknown-1", "video", "completed")
        self._insert_session("quiz-1", "quiz", "completed")

        preferences = self._preferences()

        self.assertEqual(preferences["total_sessions_observed"], 1)
        self.assertEqual(preferences["completed_sessions_observed"], 1)
        self.assertNotIn("video", preferences["completion_rate_by_activity"])


if __name__ == "__main__":
    unittest.main()
