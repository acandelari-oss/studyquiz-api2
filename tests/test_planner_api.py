import json
import unittest

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

import main
from main import app
from planner.demo_context import build_demo_planner_context
from planner.planner_engine import PlannerEngine
from planner.planner_repository import PlannerRepository
from planner.planner_serializers import serialize_planner_domain


class PlannerApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self._create_schema()
        self._seed_project()

    def tearDown(self):
        self.engine.dispose()

    def _session(self):
        return self.engine.connect()

    def _create_schema(self):
        with self.engine.begin() as db:
            db.execute(text("""
                create table projects (
                    id text primary key,
                    name text not null,
                    created_at text,
                    user_id text,
                    topic_status text,
                    taxonomy_language text
                )
            """))
            db.execute(text("""
                create table topics (
                    id text primary key,
                    project_id text,
                    category text,
                    topic text,
                    is_display_topic boolean
                )
            """))
            db.execute(text("""
                create table quizzes (
                    id text primary key,
                    project_id text,
                    user_id text
                )
            """))
            db.execute(text("""
                create table quiz_questions (
                    id text primary key,
                    quiz_id text,
                    topic text
                )
            """))
            db.execute(text("""
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
            db.execute(text("""
                create table flashcard_reviews (
                    id text primary key,
                    project_id text,
                    user_id text,
                    is_correct boolean,
                    reviewed_at text,
                    topic text
                )
            """))
            db.execute(text("""
                create table flashcards (
                    id text primary key,
                    project_id text,
                    user_id text,
                    topic text
                )
            """))
            db.execute(text("""
                create table planner_weeks (
                    id text primary key,
                    project_id text not null,
                    start_date text not null,
                    end_date text not null,
                    status text not null,
                    planning_parameters text not null,
                    weekly_briefing text,
                    weekly_statistics text not null,
                    weekly_review text,
                    next_week_options text,
                    created_at text default CURRENT_TIMESTAMP,
                    updated_at text default CURRENT_TIMESTAMP
                )
            """))
            db.execute(text("""
                create unique index planner_weeks_one_active_per_project_idx
                on planner_weeks(project_id)
                where status = 'ACTIVE'
            """))
            db.execute(text("""
                create table planner_daily_plans (
                    id text primary key,
                    week_id text not null,
                    session_index integer not null,
                    plan_date text not null,
                    day_name text not null,
                    status text not null,
                    objective text,
                    briefing text,
                    planned_allocations text not null,
                    summary text,
                    created_at text default CURRENT_TIMESTAMP,
                    updated_at text default CURRENT_TIMESTAMP
                )
            """))
            db.execute(text("""
                create table planner_activities (
                    id text primary key,
                    daily_plan_id text not null,
                    activity_index integer not null,
                    activity_type text not null,
                    configuration text not null,
                    created_at text default CURRENT_TIMESTAMP,
                    updated_at text default CURRENT_TIMESTAMP
                )
            """))

    def _seed_project(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into projects
                (id, name, created_at, user_id, topic_status, taxonomy_language)
                values
                ('project-low', 'Low Coverage', '2026-06-28', 'user-1', 'completed', 'en'),
                ('project-ready', 'Ready Coverage', '2026-06-29', 'user-1', 'completed', 'en'),
                ('project-active', 'Active Week', '2026-06-30', 'user-1', 'completed', 'en'),
                ('project-generated-only', 'Generated Only', '2026-07-01', 'user-1', 'completed', 'en'),
                ('project-survey-only', 'Survey Only', '2026-07-02', 'user-1', 'completed', 'en'),
                ('project-assessment', 'Assessment Project', '2026-07-03', 'user-1', 'completed', 'en')
            """))
            db.execute(text("""
                insert into topics
                (id, project_id, category, topic, is_display_topic)
                values
                ('low-topic-1', 'project-low', 'LOW', 'Topic 1', true),
                ('low-topic-2', 'project-low', 'LOW', 'Topic 2', true),
                ('low-topic-3', 'project-low', 'LOW', 'Topic 3', true),
                ('low-topic-4', 'project-low', 'LOW', 'Topic 4', true),
                ('low-topic-5', 'project-low', 'LOW', 'Topic 5', true),
                ('ready-topic-1', 'project-ready', 'READY', 'Ready 1', true),
                ('ready-topic-2', 'project-ready', 'READY', 'Ready 2', true),
                ('ready-topic-3', 'project-ready', 'READY', 'Ready 3', true),
                ('ready-topic-4', 'project-ready', 'READY', 'Ready 4', true),
                ('active-topic-1', 'project-active', 'ACTIVE', 'Active 1', true),
                ('active-topic-2', 'project-active', 'ACTIVE', 'Active 2', true),
                ('generated-topic-1', 'project-generated-only', 'GENERATED', 'Generated 1', true),
                ('generated-topic-2', 'project-generated-only', 'GENERATED', 'Generated 2', true),
                ('survey-topic-1', 'project-survey-only', 'A_CONFIDENT', 'A Topic', true),
                ('survey-topic-2', 'project-survey-only', 'B_PRACTICE', 'B Topic', true),
                ('survey-topic-3', 'project-survey-only', 'C_UNSURE', 'C Topic', true),
                ('assessment-topic-1', 'project-assessment', 'A', 'A Topic 1', true),
                ('assessment-topic-2', 'project-assessment', 'A', 'A Topic 2', true),
                ('assessment-topic-3', 'project-assessment', 'B', 'B Topic 1', true)
            """))
            db.execute(text("""
                insert into quizzes (id, project_id, user_id)
                values
                ('low-quiz', 'project-low', 'user-1'),
                ('ready-quiz', 'project-ready', 'user-1'),
                ('generated-quiz', 'project-generated-only', 'user-1')
            """))
            db.execute(text("""
                insert into quiz_questions (id, quiz_id, topic)
                values
                ('low-question-1', 'low-quiz', 'Topic 1'),
                ('ready-question-1', 'ready-quiz', 'Ready 1'),
                ('ready-question-2', 'ready-quiz', 'Ready 2'),
                ('ready-question-3', 'ready-quiz', 'Ready 3'),
                ('generated-question-1', 'generated-quiz', 'Generated 1'),
                ('generated-question-2', 'generated-quiz', 'Generated 2')
            """))
            db.execute(text("""
                insert into quiz_answers
                (id, quiz_id, question_id, user_id, is_correct, created_at, topic)
                values
                ('low-answer-1', 'low-quiz', 'low-question-1', 'user-1', true, '2026-07-01', 'Topic 1'),
                ('ready-answer-1', 'ready-quiz', 'ready-question-1', 'user-1', true, '2026-07-01', 'Ready 1'),
                ('ready-answer-2', 'ready-quiz', 'ready-question-2', 'user-1', true, '2026-07-01', 'Ready 2')
            """))
            db.execute(text("""
                insert into flashcard_reviews
                (id, project_id, user_id, is_correct, reviewed_at, topic)
                values
                ('ready-review-1', 'project-ready', 'user-1', true, '2026-07-01', 'Ready 3')
            """))
            db.execute(text("""
                insert into flashcards
                (id, project_id, user_id, topic)
                values
                ('generated-flashcard-1', 'project-generated-only', 'user-1', 'Generated 2')
            """))

    def _create_active_week(self):
        week = PlannerEngine().generate_week(build_demo_planner_context())

        with self.engine.connect() as db:
            PlannerRepository(db).save_active_week(
                project_id="project-active",
                week=week,
            )

    def test_unscoped_planner_week_endpoint_rejects_missing_project(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/planner/week")

        self.assertEqual(response.status_code, 400)
        self.assertEqual(
            response.json()["detail"],
            "project_id is required for Study Planner.",
        )

    def test_project_planner_week_endpoint_returns_new_project_for_low_coverage(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-low/planner/week")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "NEW_PROJECT")
        self.assertEqual(data["week"], None)
        self.assertEqual(
            data["learning_coverage"],
            {
                "covered_topics": 1,
                "total_topics": 5,
                "ratio": 0.2,
            },
        )

        with self.engine.connect() as db:
            self.assertEqual(
                db.execute(text("select count(*) from planner_weeks")).scalar(),
                0,
            )

    def test_project_planner_week_endpoint_returns_ready_for_first_plan(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-ready/planner/week")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "READY_FOR_FIRST_PLAN")
        self.assertEqual(data["week"], None)
        self.assertEqual(data["learning_coverage"]["covered_topics"], 3)
        self.assertEqual(data["learning_coverage"]["total_topics"], 4)
        self.assertEqual(data["learning_coverage"]["ratio"], 0.75)

        with self.engine.connect() as db:
            self.assertEqual(
                db.execute(text("select count(*) from planner_weeks")).scalar(),
                0,
            )

    def test_project_planner_week_endpoint_returns_active_week_when_present(self):
        self._create_active_week()
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-active/planner/week")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "ACTIVE_WEEK")
        self.assertIsNotNone(data["week"])
        self.assertEqual(data["week"]["status"], "ACTIVE")

    def test_generated_quiz_and_flashcards_without_interactions_do_not_count(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-generated-only/planner/week")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "NEW_PROJECT")
        self.assertEqual(data["learning_coverage"]["covered_topics"], 0)
        self.assertEqual(data["learning_coverage"]["total_topics"], 2)
        self.assertEqual(data["learning_coverage"]["ratio"], 0.0)

    def test_project_planner_week_endpoint_returns_404_for_unknown_project(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/missing-project/planner/week")

        self.assertEqual(response.status_code, 404)

    def test_generate_week_from_new_project_configuration(self):
        client = TestClient(app)

        payload = {
            "survey": {
                "LOW": "practice",
            },
            "study_language": "Italian",
            "preferences": {
                "studyDurationMinutes": 45,
                "questionPaceSeconds": 90,
                "questionStyle": "reasoning",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-low/planner/week/generate",
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "ACTIVE_WEEK")
        self.assertIsNotNone(data["week"])
        self.assertEqual(data["week"]["status"], "ACTIVE")
        self.assertEqual(data["week"]["study_language"], "Italian")
        self.assertEqual(len(data["week"]["daily_plans"]), 1)
        self.assertEqual(
            data["week"]["weekly_statistics"]["metadata"]["max_visible_modules"],
            main.MAX_VISIBLE_PLANNER_MODULES,
        )
        self.assertFalse(
            data["week"]["weekly_statistics"]["metadata"][
                "additional_modules_remain"
            ]
        )
        quiz_activities = [
            activity
            for daily_plan in data["week"]["daily_plans"]
            for activity in daily_plan["activities"]
            if activity["type"] == "QUIZ"
        ]
        self.assertTrue(quiz_activities)
        self.assertTrue(
            all(
                activity["configuration"]["question_style"] == "reasoning"
                for activity in quiz_activities
            )
        )

        with self.engine.connect() as db:
            row = db.execute(text("""
                select planning_parameters
                from planner_weeks
                where project_id = 'project-low'
            """)).fetchone()
            planning_parameters = json.loads(row[0])

        self.assertEqual(planning_parameters["studyDurationMinutes"], 45)
        self.assertEqual(planning_parameters["questionPaceSeconds"], 90)
        self.assertEqual(
            planning_parameters["maxVisibleModules"],
            main.MAX_VISIBLE_PLANNER_MODULES,
        )
        self.assertFalse(planning_parameters["additionalModulesRemain"])
        self.assertEqual(planning_parameters["questionStyle"], "reasoning")
        self.assertEqual(planning_parameters["studyLanguage"], "Italian")
        self.assertEqual(planning_parameters["study_language"], "Italian")
        self.assertEqual(planning_parameters["survey"], {"LOW": "practice"})

    def test_generate_week_from_ready_configuration_without_survey(self):
        client = TestClient(app)

        payload = {
            "survey": None,
            "preferences": {
                "studyDurationMinutes": 60,
                "questionPaceSeconds": 120,
                "questionStyle": "exam",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-ready/planner/week/generate",
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "ACTIVE_WEEK")
        self.assertIsNotNone(data["week"])
        self.assertEqual(len(data["week"]["daily_plans"]), 1)
        quiz_activities = [
            activity
            for daily_plan in data["week"]["daily_plans"]
            for activity in daily_plan["activities"]
            if activity["type"] == "QUIZ"
        ]
        self.assertTrue(quiz_activities)
        self.assertTrue(
            all(
                activity["configuration"]["question_style"] == "exam"
                for activity in quiz_activities
            )
        )

    def test_generate_week_uses_survey_as_first_plan_bootstrap_bias(self):
        client = TestClient(app)

        payload = {
            "survey": {
                "A_CONFIDENT": "confident",
                "B_PRACTICE": "practice",
                "C_UNSURE": "unsure",
            },
            "preferences": {
                "studyDurationMinutes": 45,
                "questionPaceSeconds": 90,
                "questionStyle": "balanced",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-survey-only/planner/week/generate",
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        scheduled_categories = [
            allocation["category"]
            for daily_plan in data["week"]["daily_plans"]
            for allocation in daily_plan["planned_allocations"]
        ]

        self.assertEqual(
            scheduled_categories,
            ["B_PRACTICE", "C_UNSURE", "A_CONFIDENT"],
        )

    def test_generate_week_rejects_invalid_survey_category(self):
        client = TestClient(app)

        payload = {
            "survey": {
                "NOT A PROJECT CATEGORY": "practice",
            },
            "preferences": {
                "studyDurationMinutes": 45,
                "questionPaceSeconds": 90,
                "sessionsPerWeek": 3,
                "questionStyle": "balanced",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-low/planner/week/generate",
                json=payload,
            )

        self.assertEqual(response.status_code, 400)

        with self.engine.connect() as db:
            self.assertEqual(
                db.execute(text("""
                    select count(*)
                    from planner_weeks
                    where project_id = 'project-low'
                """)).scalar(),
                0,
            )

    def test_generate_assessment_ignores_survey_and_persists_plan_type(self):
        client = TestClient(app)

        payload = {
            "survey": {
                "NOT USED": "practice",
            },
            "study_language": "Italian",
            "preferences": {
                "studyDurationMinutes": 30,
                "questionPaceSeconds": 60,
                "questionStyle": "exam",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-assessment/planner/assessment/generate",
                json=payload,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "ACTIVE_WEEK")
        self.assertEqual(data["week"]["plan_type"], "assessment")

        activities = [
            activity
            for daily_plan in data["week"]["daily_plans"]
            for activity in daily_plan["activities"]
        ]
        self.assertTrue(activities)
        self.assertTrue(all(activity["type"] == "QUIZ" for activity in activities))
        self.assertTrue(
            all(
                activity["configuration"]["num_questions"]
                >= len(activity["configuration"]["selected_topics"])
                for activity in activities
            )
        )

        selected_topic_titles = [
            topic["title"]
            for daily_plan in data["week"]["daily_plans"]
            for allocation in daily_plan["planned_allocations"]
            for topic in allocation["selected_topics"]
        ]
        self.assertEqual(
            selected_topic_titles,
            ["A Topic 1", "A Topic 2", "B Topic 1"],
        )

        with self.engine.connect() as db:
            row = db.execute(text("""
                select planning_parameters
                from planner_weeks
                where project_id = 'project-assessment'
            """)).fetchone()
            planning_parameters = json.loads(row[0])

        self.assertEqual(planning_parameters["plan_type"], "assessment")
        self.assertEqual(
            planning_parameters["onboarding_mode"],
            "professor_assessment",
        )
        self.assertIsNone(planning_parameters["survey"])

        with patch.object(main, "SessionLocal", self._session):
            reload_response = client.get(
                "/projects/project-assessment/planner/week",
            )

        self.assertEqual(reload_response.status_code, 200)
        self.assertEqual(reload_response.json()["week"]["plan_type"], "assessment")

    def test_completed_assessment_no_longer_blocks_normal_generation(self):
        client = TestClient(app)
        assessment_payload = {
            "survey": None,
            "preferences": {
                "studyDurationMinutes": 30,
                "questionPaceSeconds": 60,
                "questionStyle": "balanced",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            create_response = client.post(
                "/projects/project-assessment/planner/assessment/generate",
                json=assessment_payload,
            )
            complete_response = client.post(
                "/projects/project-assessment/planner/assessment/complete",
            )

        self.assertEqual(create_response.status_code, 200)
        self.assertEqual(complete_response.status_code, 200)

        study_plan_payload = {
            "survey": {
                "A": "practice",
                "B": "unsure",
            },
            "preferences": {
                "studyDurationMinutes": 30,
                "questionPaceSeconds": 60,
                "questionStyle": "balanced",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-assessment/planner/week/generate",
                json=study_plan_payload,
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["state"], "ACTIVE_WEEK")
        self.assertEqual(data["week"]["plan_type"], "study_plan")

    def test_generate_week_rejects_invalid_quiz_style(self):
        client = TestClient(app)

        payload = {
            "survey": {
                "LOW": "practice",
            },
            "preferences": {
                "studyDurationMinutes": 45,
                "questionPaceSeconds": 90,
                "sessionsPerWeek": 3,
                "questionStyle": "reference_exam",
            },
        }

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-low/planner/week/generate",
                json=payload,
            )

        self.assertEqual(response.status_code, 400)

        with self.engine.connect() as db:
            self.assertEqual(
                db.execute(text("""
                    select count(*)
                    from planner_weeks
                    where project_id = 'project-low'
                """)).scalar(),
                0,
            )

    def test_planner_serializer_matches_engine_output_shape(self):
        week = PlannerEngine().generate_week(build_demo_planner_context())
        data = serialize_planner_domain(week)

        self.assertEqual(data["id"], "demo-week")
        self.assertEqual(data["status"], "PLANNED")
        self.assertIsInstance(data["daily_plans"], list)
        self.assertIsInstance(data["daily_plans"][0]["planned_allocations"], list)
        self.assertIsInstance(data["daily_plans"][0]["activities"], list)


if __name__ == "__main__":
    unittest.main()
