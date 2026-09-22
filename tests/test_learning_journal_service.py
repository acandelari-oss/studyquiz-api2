import unittest

from sqlalchemy import create_engine, text

from learning_journal_service import get_learning_journal


class LearningJournalServiceTests(unittest.TestCase):
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
        started_at,
        completed_at,
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

    def test_zero_sessions_returns_empty_list(self):
        self.assertEqual(get_learning_journal(self.db, "user-1"), [])

    def test_completed_session_returns_duration_seconds(self):
        self._insert_session(
            "session-1",
            "quiz",
            "completed",
            "2026-07-30T10:00:00+00:00",
            "2026-07-30T10:25:24+00:00",
        )

        journal = get_learning_journal(self.db, "user-1")

        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["id"], "session-1")
        self.assertEqual(journal[0]["session_type"], "quiz")
        self.assertEqual(journal[0]["status"], "completed")
        self.assertEqual(journal[0]["duration_seconds"], 1524)

    def test_abandoned_session_has_null_duration(self):
        self._insert_session(
            "session-1",
            "flashcards",
            "abandoned",
            "2026-07-30T10:00:00+00:00",
            "2026-07-30T10:25:24+00:00",
        )

        journal = get_learning_journal(self.db, "user-1")

        self.assertIsNone(journal[0]["duration_seconds"])

    def test_invalid_timestamps_do_not_fail_and_return_null_duration(self):
        self._insert_session(
            "session-1",
            "ask",
            "completed",
            "not-a-date",
            "2026-07-30T10:25:24+00:00",
        )
        self._insert_session(
            "session-2",
            "active_recall",
            "completed",
            "2026-07-30T11:00:00+00:00",
            None,
        )
        self._insert_session(
            "session-3",
            "planner",
            "completed",
            "2026-07-30T12:00:00+00:00",
            "2026-07-30T11:00:00+00:00",
        )

        journal = get_learning_journal(self.db, "user-1")

        self.assertEqual(len(journal), 3)
        self.assertTrue(all(entry["duration_seconds"] is None for entry in journal))

    def test_sessions_are_ordered_by_started_at_desc_and_user_scoped(self):
        self._insert_session(
            "old-session",
            "quiz",
            "completed",
            "2026-07-29T10:00:00+00:00",
            "2026-07-29T10:10:00+00:00",
        )
        self._insert_session(
            "new-session",
            "planner",
            "completed",
            "2026-07-30T10:00:00+00:00",
            "2026-07-30T10:10:00+00:00",
        )
        self._insert_session(
            "other-user-session",
            "quiz",
            "completed",
            "2026-07-31T10:00:00+00:00",
            "2026-07-31T10:10:00+00:00",
            user_id="user-2",
        )

        journal = get_learning_journal(self.db, "user-1")

        self.assertEqual([entry["id"] for entry in journal], [
            "new-session",
            "old-session",
        ])

    def test_journal_can_be_scoped_to_one_project(self):
        self._insert_session(
            "project-1-session",
            "quiz",
            "completed",
            "2026-07-30T10:00:00+00:00",
            "2026-07-30T10:10:00+00:00",
        )
        self._insert_session(
            "project-2-session",
            "flashcards",
            "completed",
            "2026-07-30T11:00:00+00:00",
            "2026-07-30T11:10:00+00:00",
            project_id="project-2",
        )

        journal = get_learning_journal(self.db, "user-1", project_id="project-1")

        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["id"], "project-1-session")

    def test_limit_and_offset_are_applied(self):
        self._insert_session(
            "session-1",
            "quiz",
            "completed",
            "2026-07-30T10:00:00+00:00",
            "2026-07-30T10:10:00+00:00",
        )
        self._insert_session(
            "session-2",
            "flashcards",
            "completed",
            "2026-07-30T11:00:00+00:00",
            "2026-07-30T11:10:00+00:00",
        )
        self._insert_session(
            "session-3",
            "ask",
            "completed",
            "2026-07-30T12:00:00+00:00",
            "2026-07-30T12:10:00+00:00",
        )

        journal = get_learning_journal(self.db, "user-1", limit=1, offset=1)

        self.assertEqual(len(journal), 1)
        self.assertEqual(journal[0]["id"], "session-2")


if __name__ == "__main__":
    unittest.main()
