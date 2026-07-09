import unittest
from datetime import date

from sqlalchemy import create_engine, text

from planner.context_builder import build_real_planner_context


class PlannerContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.db = self.engine.connect()
        self._create_schema()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _create_schema(self):
        self.db.execute(text("""
            create table projects (
                id text primary key,
                name text not null,
                created_at text,
                user_id text,
                topic_status text,
                taxonomy_language text
            )
        """))
        self.db.execute(text("""
            create table topics (
                id text primary key,
                project_id text,
                category text,
                topic text,
                is_display_topic boolean
            )
        """))
        self.db.execute(text("""
            create table quizzes (
                id text primary key,
                project_id text,
                user_id text
            )
        """))
        self.db.execute(text("""
            create table quiz_questions (
                id text primary key,
                quiz_id text,
                topic text
            )
        """))
        self.db.execute(text("""
            create table quiz_answers (
                id text primary key,
                quiz_id text,
                question_id text,
                user_id text,
                is_correct boolean,
                created_at text,
                topic text
            )
        """))
        self.db.execute(text("""
            create table flashcard_reviews (
                id text primary key,
                project_id text,
                user_id text,
                is_correct boolean,
                reviewed_at text,
                topic text
            )
        """))
        self.db.commit()

    def test_builds_context_from_project_topics_and_learning_evidence(self):
        self.db.execute(text("""
            insert into projects
            (id, name, created_at, user_id, topic_status, taxonomy_language)
            values
            ('project-1', 'Private Law', '2026-06-01', 'user-1', 'completed', 'it')
        """))
        self.db.execute(text("""
            insert into topics
            (id, project_id, category, topic, is_display_topic)
            values
            ('topic-2', 'project-1', 'BENI', 'Beni immobili', true),
            ('topic-1', 'project-1', 'BENI', 'Beni mobili', true),
            ('topic-3', 'project-1', 'CONTRATTI', 'Appalto', true)
        """))
        self.db.execute(text("""
            insert into quizzes (id, project_id, user_id)
            values ('quiz-1', 'project-1', 'user-1')
        """))
        self.db.execute(text("""
            insert into quiz_questions (id, quiz_id, topic)
            values ('question-1', 'quiz-1', 'Beni immobili')
        """))
        self.db.execute(text("""
            insert into quiz_answers
            (id, quiz_id, question_id, user_id, is_correct, created_at, topic)
            values
            ('answer-1', 'quiz-1', 'question-1', 'user-1', true, '2026-06-28', 'beni immobili')
        """))
        self.db.execute(text("""
            insert into flashcard_reviews
            (id, project_id, user_id, is_correct, reviewed_at, topic)
            values
            ('review-1', 'project-1', 'user-1', false, '2026-06-29', 'Beni mobili')
        """))
        self.db.commit()

        context = build_real_planner_context(
            self.db,
            project_id="project-1",
            user_id="user-1",
            today=date(2026, 7, 1),
        )

        self.assertEqual(context.project["id"], "project-1")
        self.assertEqual(context.categories, ("BENI", "CONTRATTI"))
        self.assertEqual(
            [topic.title for topic in context.topics_by_category["BENI"]],
            ["Beni immobili", "Beni mobili"],
        )
        self.assertEqual(
            [topic.order for topic in context.topics_by_category["BENI"]],
            [1, 2],
        )
        self.assertEqual(context.analytics["BENI"].accuracy, 0.5)
        self.assertEqual(context.analytics["BENI"].coverage, 1.0)
        self.assertEqual(context.analytics["BENI"].days_since_review, 2)
        self.assertIsNone(context.analytics["CONTRATTI"].accuracy)
        self.assertEqual(context.analytics["CONTRATTI"].coverage, 0.0)
        self.assertEqual(context.preferences.question_pace_seconds, 60)
        self.assertEqual(context.preferences.question_style, "balanced")
        self.assertEqual(context.number_of_sessions, 4)
        self.assertEqual(context.planning_budget_minutes, 3)
        self.assertEqual(context.week_start_date, date(2026, 6, 29))

    def test_empty_database_returns_valid_empty_context_with_defaults(self):
        context = build_real_planner_context(
            self.db,
            today=date(2026, 7, 1),
        )

        self.assertIsNone(context.project)
        self.assertEqual(context.categories, ())
        self.assertEqual(context.topics_by_category, {})
        self.assertEqual(context.analytics, {})
        self.assertEqual(context.preferences.question_pace_seconds, 60)
        self.assertEqual(context.preferences.question_style, "balanced")
        self.assertEqual(context.number_of_sessions, 4)
        self.assertEqual(context.planning_budget_minutes, 3)
        self.assertEqual(context.week_start_date, date(2026, 6, 29))

    def test_without_project_id_does_not_select_latest_project(self):
        self.db.execute(text("""
            insert into projects
            (id, name, created_at, user_id, topic_status, taxonomy_language)
            values
            ('project-1', 'Genetics', '2026-06-01', 'user-1', 'completed', 'en'),
            ('project-2', 'Private Law', '2026-07-01', 'user-1', 'completed', 'it')
        """))
        self.db.execute(text("""
            insert into topics
            (id, project_id, category, topic, is_display_topic)
            values
            ('topic-1', 'project-2', 'DIRITTO PRIVATO', 'Beni mobili', true)
        """))
        self.db.commit()

        context = build_real_planner_context(
            self.db,
            today=date(2026, 7, 1),
        )

        self.assertIsNone(context.project)
        self.assertEqual(context.categories, ())


if __name__ == "__main__":
    unittest.main()
