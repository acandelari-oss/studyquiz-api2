"""Persistence boundary for generated Study Planner weeks.

The repository stores and reloads the planning snapshot produced by the
Planner Engine. It does not generate plans, evaluate planner state, execute
activities, or contain Professor logic.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import date, datetime
from typing import Any, Mapping, Optional, Sequence

from sqlalchemy import JSON, bindparam, text
from sqlalchemy.exc import IntegrityError

from .planner_models import (
    Activity,
    ActivityConfiguration,
    DailyPlan,
    DailySummary,
    HomeworkRecommendation,
    NextWeekOptions,
    PlannerPreferences,
    SelectedTopic,
    Week,
    WeeklyStatistics,
)
from .planner_serializers import serialize_planner_domain
from .planner_state import ActivityType, DailyStatus, WeekStatus
from .session_allocator import CategoryAllocation


class PlannerRepository:
    """Store and reconstruct generated Planner Week snapshots."""

    ACTIVE_STATUS = WeekStatus.ACTIVE.value

    def __init__(self, db: Any) -> None:
        self.db = db

    def load_active_week(self, project_id: str) -> Optional[Week]:
        """Return the active Week for a project, if one exists."""

        row = self.db.execute(
            text("""
                select id
                from planner_weeks
                where project_id = :project_id
                and status = :status
                order by created_at desc
                limit 1
            """),
            {
                "project_id": project_id,
                "status": self.ACTIVE_STATUS,
            },
        ).fetchone()

        if not row:
            return None

        return self.load_week(str(row[0]))

    def load_week(self, week_id: str) -> Optional[Week]:
        """Reconstruct one persisted Week domain object."""

        week_row = self.db.execute(
            text("""
                select
                    id,
                    start_date,
                    end_date,
                    status,
                    planning_parameters,
                    weekly_briefing,
                    weekly_statistics,
                    weekly_review,
                    next_week_options
                from planner_weeks
                where id = :week_id
            """),
            {"week_id": week_id},
        ).fetchone()

        if not week_row:
            return None

        daily_plans = self._load_daily_plans(week_id=week_id)

        return Week(
            id=str(week_row[0]),
            start_date=self._as_date(week_row[1]),
            end_date=self._as_date(week_row[2]),
            status=WeekStatus(str(week_row[3])),
            study_language=self._json_value(
                week_row[4],
                default={},
            ).get("study_language"),
            weekly_briefing=week_row[5] or "",
            weekly_statistics=self._build_weekly_statistics(
                self._json_value(week_row[6], default={})
            ),
            daily_plans=daily_plans,
            weekly_review=week_row[7],
            next_week_options=self._build_next_week_options(
                self._json_value(week_row[8], default=None)
            ),
        )

    def save_active_week(
        self,
        project_id: str,
        week: Week,
        planning_parameters: Optional[Mapping[str, Any]] = None,
    ) -> Week:
        """Persist a generated Week as the project's active Week.

        If an active Week already exists for the project, it is returned
        unchanged. This preserves the one-active-week invariant without
        introducing Planner State Evaluation.
        """

        existing_week = self.load_active_week(project_id=project_id)

        if existing_week:
            return existing_week

        active_week = replace(week, status=WeekStatus.ACTIVE)

        try:
            self._insert_week(
                project_id=project_id,
                week=active_week,
                planning_parameters=planning_parameters or {},
            )
            self._insert_daily_plans(week=active_week)
            self._commit_if_available()
        except IntegrityError:
            self._rollback_if_available()
            existing_week = self.load_active_week(project_id=project_id)

            if existing_week:
                return existing_week

            raise

        loaded_week = self.load_week(active_week.id)
        return loaded_week or active_week

    def _insert_week(
        self,
        project_id: str,
        week: Week,
        planning_parameters: Mapping[str, Any],
    ) -> None:
        statement = text("""
                insert into planner_weeks (
                    id,
                    project_id,
                    start_date,
                    end_date,
                    status,
                    planning_parameters,
                    weekly_briefing,
                    weekly_statistics,
                    weekly_review,
                    next_week_options
                )
                values (
                    :id,
                    :project_id,
                    :start_date,
                    :end_date,
                    :status,
                    :planning_parameters,
                    :weekly_briefing,
                    :weekly_statistics,
                    :weekly_review,
                    :next_week_options
                )
            """).bindparams(
                bindparam("planning_parameters", type_=JSON),
                bindparam("weekly_statistics", type_=JSON),
                bindparam("next_week_options", type_=JSON),
            )
        self.db.execute(
            statement,
            {
                "id": week.id,
                "project_id": project_id,
                "start_date": week.start_date,
                "end_date": week.end_date,
                "status": week.status.value,
                "planning_parameters": self._json_data(planning_parameters),
                "weekly_briefing": week.weekly_briefing,
                "weekly_statistics": self._json_data(week.weekly_statistics),
                "weekly_review": week.weekly_review,
                "next_week_options": self._json_data(week.next_week_options),
            },
        )

    def _insert_daily_plans(self, week: Week) -> None:
        for daily_index, daily_plan in enumerate(week.daily_plans, start=1):
            statement = text("""
                    insert into planner_daily_plans (
                        id,
                        week_id,
                        session_index,
                        plan_date,
                        day_name,
                        status,
                        objective,
                        briefing,
                        planned_allocations,
                        summary
                    )
                    values (
                        :id,
                        :week_id,
                        :session_index,
                        :plan_date,
                        :day_name,
                        :status,
                        :objective,
                        :briefing,
                        :planned_allocations,
                        :summary
                    )
                """).bindparams(
                    bindparam("planned_allocations", type_=JSON),
                    bindparam("summary", type_=JSON),
                )
            self.db.execute(
                statement,
                {
                    "id": daily_plan.id,
                    "week_id": week.id,
                    "session_index": daily_index,
                    "plan_date": daily_plan.date,
                    "day_name": daily_plan.day_name,
                    "status": daily_plan.status.value,
                    "objective": daily_plan.objective,
                    "briefing": daily_plan.briefing,
                    "planned_allocations": self._json_data(
                        daily_plan.planned_allocations
                    ),
                    "summary": self._json_data(daily_plan.summary),
                },
            )
            self._insert_activities(daily_plan=daily_plan)

    def _insert_activities(self, daily_plan: DailyPlan) -> None:
        for activity_index, activity in enumerate(daily_plan.activities, start=1):
            statement = text("""
                    insert into planner_activities (
                        id,
                        daily_plan_id,
                        activity_index,
                        activity_type,
                        configuration
                    )
                    values (
                        :id,
                        :daily_plan_id,
                        :activity_index,
                        :activity_type,
                        :configuration
                    )
                """).bindparams(
                    bindparam("configuration", type_=JSON),
                )
            self.db.execute(
                statement,
                {
                    "id": activity.id,
                    "daily_plan_id": daily_plan.id,
                    "activity_index": activity_index,
                    "activity_type": activity.type.value,
                    "configuration": self._json_data(activity.configuration),
                },
            )

    def _load_daily_plans(self, week_id: str) -> Sequence[DailyPlan]:
        rows = self.db.execute(
            text("""
                select
                    id,
                    plan_date,
                    day_name,
                    status,
                    objective,
                    briefing,
                    planned_allocations,
                    summary
                from planner_daily_plans
                where week_id = :week_id
                order by session_index asc
            """),
            {"week_id": week_id},
        ).fetchall()

        return tuple(
            DailyPlan(
                id=str(row[0]),
                date=self._as_date(row[1]),
                day_name=row[2],
                status=DailyStatus(str(row[3])),
                objective=row[4] or "",
                briefing=row[5] or "",
                planned_allocations=self._build_allocations(
                    self._json_value(row[6], default=[])
                ),
                activities=self._load_activities(daily_plan_id=str(row[0])),
                summary=self._build_daily_summary(
                    self._json_value(row[7], default=None)
                ),
            )
            for row in rows
        )

    def _load_activities(self, daily_plan_id: str) -> Sequence[Activity]:
        rows = self.db.execute(
            text("""
                select id, activity_type, configuration
                from planner_activities
                where daily_plan_id = :daily_plan_id
                order by activity_index asc
            """),
            {"daily_plan_id": daily_plan_id},
        ).fetchall()

        return tuple(
            Activity(
                id=str(row[0]),
                type=ActivityType(str(row[1])),
                configuration=self._build_activity_configuration(
                    self._json_value(row[2], default={})
                ),
            )
            for row in rows
        )

    def _build_allocations(self, values: Any) -> Sequence[CategoryAllocation]:
        return tuple(
            CategoryAllocation(
                category=value.get("category"),
                selected_topics=self._build_selected_topics(
                    value.get("selected_topics", [])
                ),
                estimated_duration_minutes=value.get(
                    "estimated_duration_minutes",
                    0.0,
                ) or 0.0,
            )
            for value in values or []
        )

    def _build_activity_configuration(
        self,
        value: Mapping[str, Any],
    ) -> ActivityConfiguration:
        return ActivityConfiguration(
            category=value.get("category"),
            selected_topics=self._build_selected_topics(
                value.get("selected_topics", [])
            ),
            estimated_duration_minutes=value.get("estimated_duration_minutes"),
            difficulty=value.get("difficulty"),
            question_style=value.get("question_style"),
            num_questions=value.get("num_questions"),
            num_cards=value.get("num_cards"),
        )

    def _build_selected_topics(self, values: Any) -> Sequence[SelectedTopic]:
        return tuple(
            SelectedTopic(
                id=str(value.get("id")),
                title=value.get("title") or "",
                order=value.get("order"),
            )
            for value in values or []
        )

    def _build_weekly_statistics(self, value: Mapping[str, Any]) -> WeeklyStatistics:
        return WeeklyStatistics(
            sessions_completed=value.get("sessions_completed", 0) or 0,
            quiz_accuracy=value.get("quiz_accuracy"),
            flashcard_completion=value.get("flashcard_completion"),
            study_time=value.get("study_time"),
            metadata=value.get("metadata", {}) or {},
        )

    def _build_next_week_options(
        self,
        value: Optional[Mapping[str, Any]],
    ) -> Optional[NextWeekOptions]:
        if not value:
            return None

        return NextWeekOptions(
            recommendations=tuple(value.get("recommendations", []) or []),
            metadata=value.get("metadata", {}) or {},
        )

    def _build_daily_summary(
        self,
        value: Optional[Mapping[str, Any]],
    ) -> Optional[DailySummary]:
        if not value:
            return None

        return DailySummary(
            session_data=value.get("session_data", {}) or {},
            professor_debrief=value.get("professor_debrief", "") or "",
            homework_recommendations=tuple(
                HomeworkRecommendation(
                    text=item.get("text", "") or "",
                    rationale=item.get("rationale", "") or "",
                    related_categories=tuple(
                        item.get("related_categories", []) or []
                    ),
                    estimated_effort=item.get("estimated_effort"),
                )
                for item in value.get("homework_recommendations", []) or []
            ),
            active_recall_offer=None,
            office_hours_offer=None,
            extra_session_offer=None,
        )

    def _json_data(self, value: Any) -> Any:
        return serialize_planner_domain(value)

    def _json_value(self, value: Any, default: Any) -> Any:
        if value is None:
            return default

        if isinstance(value, str):
            return json.loads(value)

        return value

    def _as_date(self, value: Any) -> date:
        if isinstance(value, datetime):
            return value.date()

        if isinstance(value, date):
            return value

        return date.fromisoformat(str(value))

    def _commit_if_available(self) -> None:
        commit = getattr(self.db, "commit", None)

        if callable(commit):
            commit()

    def _rollback_if_available(self) -> None:
        rollback = getattr(self.db, "rollback", None)

        if callable(rollback):
            rollback()


def build_planning_parameters(context: Any) -> Mapping[str, Any]:
    """Capture the generation parameters that produced a Week snapshot."""

    preferences = getattr(context, "preferences", PlannerPreferences())

    return {
        "question_pace_seconds": preferences.question_pace_seconds,
        "question_style": preferences.question_style,
        "study_language": getattr(context, "study_language", None),
        "number_of_sessions": getattr(context, "number_of_sessions", None),
        "planning_budget_minutes": getattr(
            context,
            "planning_budget_minutes",
            None,
        ),
        "week_start_date": serialize_planner_domain(
            getattr(context, "week_start_date", None)
        ),
    }
