import unittest
import json
from datetime import date

from sqlalchemy import create_engine, text

from planner.category_selector import CategoryAnalytics
from planner.planner_engine import PlannerEngine
from planner.planner_models import PlannerContext, PlannerPreferences, SelectedTopic
from planner.planner_repository import PlannerRepository, build_planning_parameters
from planner.planner_state import WeekStatus


class PlannerRepositoryTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine("sqlite:///:memory:")
        self.db = self.engine.connect()
        self._create_schema()

    def tearDown(self):
        self.db.close()
        self.engine.dispose()

    def _create_schema(self):
        self.db.execute(text("""
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
        self.db.execute(text("""
            create unique index planner_weeks_one_active_per_project_idx
            on planner_weeks(project_id)
            where status = 'ACTIVE'
        """))
        self.db.execute(text("""
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
        self.db.execute(text("""
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
        self.db.commit()

    def _context(self, week_id="week-1"):
        topics = (
            SelectedTopic(id="topic-1", title="Topic 1", order=1),
            SelectedTopic(id="topic-2", title="Topic 2", order=2),
        )
        return PlannerContext(
            categories=("Category A",),
            topics_by_category={"Category A": topics},
            analytics={"Category A": CategoryAnalytics(accuracy=0.8, coverage=0.8)},
            preferences=PlannerPreferences(
                question_pace_seconds=60,
                question_style="exam",
            ),
            number_of_sessions=2,
            planning_budget_minutes=2,
            week_start_date=date(2026, 6, 29),
            week_id=week_id,
        )

    def test_saves_and_reloads_generated_week(self):
        context = self._context()
        week = PlannerEngine().generate_week(context)
        repository = PlannerRepository(self.db)

        saved_week = repository.save_active_week(
            project_id="project-1",
            week=week,
            planning_parameters=build_planning_parameters(context),
        )
        loaded_week = repository.load_active_week(project_id="project-1")

        self.assertIsNotNone(loaded_week)
        self.assertEqual(saved_week.id, "week-1")
        self.assertEqual(loaded_week.id, "week-1")
        self.assertEqual(loaded_week.status, WeekStatus.ACTIVE)
        self.assertEqual(len(loaded_week.daily_plans), 1)
        self.assertEqual(
            [activity.type.value for activity in loaded_week.daily_plans[0].activities],
            ["QUIZ"],
        )
        self.assertEqual(
            loaded_week.daily_plans[0].planned_allocations[0].selected_topics[0].id,
            "topic-1",
        )
        self.assertEqual(
            loaded_week.daily_plans[0].activities[0].configuration.question_style,
            "exam",
        )
        self.assertTrue(loaded_week.weekly_briefing)

    def test_duplicate_active_week_is_not_created(self):
        repository = PlannerRepository(self.db)
        first_week = PlannerEngine().generate_week(self._context(week_id="week-1"))
        second_week = PlannerEngine().generate_week(self._context(week_id="week-2"))

        repository.save_active_week(project_id="project-1", week=first_week)
        returned_week = repository.save_active_week(
            project_id="project-1",
            week=second_week,
        )

        self.assertEqual(returned_week.id, "week-1")
        self.assertEqual(
            self.db.execute(text("""
                select count(*)
                from planner_weeks
                where project_id = 'project-1'
                and status = 'ACTIVE'
            """)).scalar(),
            1,
        )

    def test_completes_daily_plan_and_reloads_runtime_summary(self):
        repository = PlannerRepository(self.db)
        week = PlannerEngine().generate_week(self._context(week_id="week-1"))

        repository.save_active_week(project_id="project-1", week=week)
        repository.complete_daily_plan(
            project_id="project-1",
            session_index=1,
            session_results={
                "flashcardsReviewed": 0,
                "quizzesCompleted": 1,
                "quizQuestions": 5,
                "quizCorrect": 4,
                "startedAtMs": 1000,
                "completedAtMs": 61000,
                "activityResults": [{"type": "quiz"}],
            },
            professor_debrief="The module debrief.",
            homework_recommendation="Write one concise explanation.",
            study_plan_debrief="The Study Plan debrief.",
        )

        loaded_week = repository.load_active_week(project_id="project-1")

        self.assertIsNotNone(loaded_week)
        self.assertEqual(loaded_week.daily_plans[0].status.value, "COMPLETED")
        self.assertEqual(
            loaded_week.daily_plans[0].summary.session_data["quiz_questions"],
            5,
        )
        self.assertEqual(
            loaded_week.daily_plans[0].summary.professor_debrief,
            "The module debrief.",
        )
        self.assertEqual(
            loaded_week.daily_plans[0].summary.homework_recommendations[0].text,
            "Write one concise explanation.",
        )
        self.assertEqual(loaded_week.weekly_statistics.sessions_completed, 1)
        self.assertEqual(loaded_week.weekly_statistics.quiz_accuracy, 0.8)
        self.assertEqual(loaded_week.weekly_statistics.study_time, 1)
        self.assertEqual(loaded_week.weekly_review, "The Study Plan debrief.")

    def test_load_previously_scheduled_categories_deduplicates_and_scopes_project(self):
        self.db.execute(
            text("""
                insert into planner_weeks
                (
                    id,
                    project_id,
                    start_date,
                    end_date,
                    status,
                    planning_parameters,
                    weekly_statistics
                )
                values
                (
                    'week-1',
                    'project-1',
                    '2026-06-29',
                    '2026-07-05',
                    'COMPLETED',
                    :study_plan_parameters,
                    '{}'
                ),
                (
                    'week-2',
                    'project-1',
                    '2026-07-06',
                    '2026-07-12',
                    'COMPLETED',
                    :assessment_parameters,
                    '{}'
                ),
                (
                    'week-other',
                    'project-2',
                    '2026-06-29',
                    '2026-07-05',
                    'COMPLETED',
                    :study_plan_parameters,
                    '{}'
                )
            """),
            {
                "study_plan_parameters": json.dumps({"plan_type": "study_plan"}),
                "assessment_parameters": json.dumps({"plan_type": "assessment"}),
            },
        )
        self.db.execute(
            text("""
                insert into planner_daily_plans
                (
                    id,
                    week_id,
                    session_index,
                    plan_date,
                    day_name,
                    status,
                    planned_allocations
                )
                values
                (
                    'day-1',
                    'week-1',
                    1,
                    '2026-06-29',
                    'Monday',
                    'PLANNED',
                    :first_allocations
                ),
                (
                    'day-2',
                    'week-1',
                    2,
                    '2026-06-30',
                    'Tuesday',
                    'PLANNED',
                    :duplicate_allocations
                ),
                (
                    'assessment-day',
                    'week-2',
                    1,
                    '2026-07-06',
                    'Monday',
                    'PLANNED',
                    :assessment_allocations
                ),
                (
                    'other-day',
                    'week-other',
                    1,
                    '2026-06-29',
                    'Monday',
                    'PLANNED',
                    :other_project_allocations
                )
            """),
            {
                "first_allocations": json.dumps(
                    [
                        {"category": "Category A", "selected_topics": []},
                        {"category": "Category B", "selected_topics": []},
                    ]
                ),
                "duplicate_allocations": json.dumps(
                    [{"category": " category a ", "selected_topics": []}]
                ),
                "assessment_allocations": json.dumps(
                    [{"category": "Assessment Only", "selected_topics": []}]
                ),
                "other_project_allocations": json.dumps(
                    [{"category": "Other Project", "selected_topics": []}]
                ),
            },
        )
        self.db.commit()

        scheduled = PlannerRepository(self.db).load_previously_scheduled_categories(
            project_id="project-1"
        )

        self.assertEqual(scheduled, ("category a", "category b"))

    def test_load_completed_survey_categories_uses_completed_coverage_plans_only(self):
        self.db.execute(
            text("""
                insert into planner_weeks
                (
                    id,
                    project_id,
                    start_date,
                    end_date,
                    status,
                    planning_parameters,
                    weekly_statistics
                )
                values
                (
                    'completed-coverage',
                    'project-1',
                    '2026-06-29',
                    '2026-07-05',
                    'COMPLETED',
                    :coverage_parameters,
                    :coverage_statistics
                ),
                (
                    'active-coverage',
                    'project-1',
                    '2026-07-06',
                    '2026-07-12',
                    'ACTIVE',
                    :coverage_parameters,
                    :active_statistics
                ),
                (
                    'completed-adaptive',
                    'project-1',
                    '2026-07-13',
                    '2026-07-19',
                    'COMPLETED',
                    :adaptive_parameters,
                    :adaptive_statistics
                )
            """),
            {
                "coverage_parameters": json.dumps(
                    {"plan_type": "study_plan", "professor_mode": "coverage"}
                ),
                "coverage_statistics": json.dumps(
                    {
                        "metadata": {
                            "professor_mode": "coverage",
                            "coverage_accepted_categories": [
                                "Category A",
                                " Category B ",
                            ],
                        }
                    }
                ),
                "active_statistics": json.dumps(
                    {
                        "metadata": {
                            "professor_mode": "coverage",
                            "coverage_accepted_categories": ["Active Only"],
                        }
                    }
                ),
                "adaptive_parameters": json.dumps(
                    {"plan_type": "study_plan", "professor_mode": "adaptive"}
                ),
                "adaptive_statistics": json.dumps(
                    {
                        "metadata": {
                            "professor_mode": "adaptive",
                            "coverage_accepted_categories": ["Adaptive Only"],
                        }
                    }
                ),
            },
        )
        self.db.execute(
            text("""
                insert into planner_daily_plans
                (
                    id,
                    week_id,
                    session_index,
                    plan_date,
                    day_name,
                    status,
                    planned_allocations
                )
                values
                (
                    'completed-day',
                    'completed-coverage',
                    1,
                    '2026-06-29',
                    'Monday',
                    'COMPLETED',
                    :allocations
                )
            """),
            {
                "allocations": json.dumps(
                    [{"category": "Category C", "selected_topics": []}]
                )
            },
        )
        self.db.commit()

        completed = PlannerRepository(self.db).load_completed_survey_categories(
            project_id="project-1"
        )

        self.assertEqual(completed, ("category a", "category b"))


if __name__ == "__main__":
    unittest.main()
