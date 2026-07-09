"""Real PlannerContext builder backed by existing project data.

This module adapts current DOUNO database rows into the frozen PlannerContext
domain object. It does not contain Planner Engine logic, persistence logic, or
state evaluation.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Mapping, Optional

from sqlalchemy import text

from .category_selector import CategoryAnalytics
from .planner_models import PlannerContext, PlannerPreferences, SelectedTopic


DEFAULT_QUESTION_PACE_SECONDS = 60
DEFAULT_QUESTION_STYLE = "balanced"
DEFAULT_STUDY_LANGUAGE = "English"
DEFAULT_NUMBER_OF_SESSIONS = 4
DEFAULT_PLANNING_BUDGET_MINUTES = 3


def build_real_planner_context(
    db: Any,
    project_id: Optional[str] = None,
    user_id: Optional[str] = None,
    today: Optional[date] = None,
) -> PlannerContext:
    """Build a PlannerContext from existing persisted project data.

    Sprint 1 deliberately keeps the existing PlannerContext shape unchanged.
    If no explicit project is available, an empty real context with
    deterministic defaults is returned so callers never silently plan against a
    different project.
    """

    today = today or date.today()
    project = _load_project(db, project_id=project_id, user_id=user_id)

    if not project:
        return _empty_context(today=today)

    topic_rows = _load_display_topics(db, project_id=str(project["id"]))
    categories, topics_by_category, topic_lookup = _build_topic_context(
        topic_rows
    )
    analytics = _build_category_analytics(
        db=db,
        project_id=str(project["id"]),
        user_id=user_id,
        categories=categories,
        topics_by_category=topics_by_category,
        topic_lookup=topic_lookup,
        today=today,
    )

    return PlannerContext(
        project=project,
        categories=categories,
        topics_by_category=topics_by_category,
        analytics=analytics,
        preferences=PlannerPreferences(
            question_pace_seconds=DEFAULT_QUESTION_PACE_SECONDS,
            question_style=DEFAULT_QUESTION_STYLE,
        ),
        study_language=DEFAULT_STUDY_LANGUAGE,
        number_of_sessions=DEFAULT_NUMBER_OF_SESSIONS,
        planning_budget_minutes=DEFAULT_PLANNING_BUDGET_MINUTES,
        week_start_date=_week_start(today),
        week_id=f"{project['id']}-week-{_week_start(today).isoformat()}",
    )


def _load_project(
    db: Any,
    project_id: Optional[str],
    user_id: Optional[str],
) -> Optional[Mapping[str, Any]]:
    if not project_id:
        return None

    filters = []
    params = {}

    filters.append("p.id = :project_id")
    params["project_id"] = project_id

    if user_id:
        filters.append("p.user_id = :user_id")
        params["user_id"] = user_id

    where_clause = f"where {' and '.join(filters)}" if filters else ""

    row = db.execute(
        text(f"""
            select
                p.id,
                p.name,
                p.created_at,
                p.user_id,
                p.topic_status,
                p.taxonomy_language
            from projects p
            {where_clause}
            order by
                case
                    when exists (
                        select 1
                        from topics t
                        where t.project_id = p.id
                        and t.topic is not null
                        and t.is_display_topic = true
                    )
                    then 0
                    else 1
                end,
                p.created_at desc
            limit 1
        """),
        params,
    ).fetchone()

    if not row:
        return None

    return {
        "id": str(row[0]),
        "name": row[1],
        "created_at": _as_iso(row[2]),
        "user_id": str(row[3]) if row[3] else None,
        "topic_status": row[4],
        "taxonomy_language": row[5],
    }


def _load_display_topics(db: Any, project_id: str):
    return db.execute(
        text("""
            select id, category, topic
            from topics
            where project_id = :project_id
            and topic is not null
            and is_display_topic = true
            order by category asc, topic asc
        """),
        {"project_id": project_id},
    ).fetchall()


def _build_topic_context(topic_rows):
    categories = []
    category_seen = set()
    topics_by_category = {}
    topic_lookup = {}

    for row in topic_rows:
        topic_id = str(row[0])
        category = row[1] or "General"
        topic_title = row[2]

        if category not in category_seen:
            category_seen.add(category)
            categories.append(category)
            topics_by_category[category] = []

        order = len(topics_by_category[category]) + 1
        selected_topic = SelectedTopic(
            id=topic_id,
            title=topic_title,
            order=order,
        )
        topics_by_category[category].append(selected_topic)
        topic_lookup.setdefault(_normalize_topic_key(topic_title), category)

    return (
        tuple(categories),
        {
            category: tuple(topics)
            for category, topics in topics_by_category.items()
        },
        topic_lookup,
    )


def _build_category_analytics(
    db: Any,
    project_id: str,
    user_id: Optional[str],
    categories,
    topics_by_category,
    topic_lookup,
    today: date,
):
    stats = {
        category: {
            "correct": 0,
            "total": 0,
            "studied_topics": set(),
            "last_reviewed_at": None,
        }
        for category in categories
    }

    for event in _load_quiz_events(db, project_id=project_id, user_id=user_id):
        _apply_learning_event(
            stats=stats,
            topic_lookup=topic_lookup,
            topic=event["topic"],
            is_correct=event["is_correct"],
            occurred_at=event["occurred_at"],
        )

    for event in _load_flashcard_events(
        db,
        project_id=project_id,
        user_id=user_id,
    ):
        _apply_learning_event(
            stats=stats,
            topic_lookup=topic_lookup,
            topic=event["topic"],
            is_correct=event["is_correct"],
            occurred_at=event["occurred_at"],
        )

    analytics = {}

    for category in categories:
        category_stats = stats[category]
        topic_count = len(topics_by_category.get(category, ()))
        total = category_stats["total"]
        last_reviewed_at = category_stats["last_reviewed_at"]

        analytics[category] = CategoryAnalytics(
            accuracy=(
                category_stats["correct"] / total
                if total > 0
                else None
            ),
            coverage=(
                len(category_stats["studied_topics"]) / topic_count
                if topic_count > 0
                else None
            ),
            days_since_review=(
                (today - _as_date(last_reviewed_at)).days
                if last_reviewed_at
                else None
            ),
            priority_weight=1.0,
        )

    return analytics


def _load_quiz_events(
    db: Any,
    project_id: str,
    user_id: Optional[str],
):
    filters = ["q.project_id = :project_id"]
    params = {"project_id": project_id}

    if user_id:
        filters.append("qa.user_id = :user_id")
        params["user_id"] = user_id

    return [
        {
            "topic": row[0],
            "is_correct": bool(row[1]),
            "occurred_at": row[2],
        }
        for row in db.execute(
            text(f"""
                select
                    coalesce(qa.topic, qq.topic) as topic,
                    qa.is_correct,
                    qa.created_at
                from quiz_answers qa
                join quiz_questions qq
                    on qq.id = qa.question_id
                join quizzes q
                    on q.id = qq.quiz_id
                where {' and '.join(filters)}
                and coalesce(qa.topic, qq.topic) is not null
            """),
            params,
        ).fetchall()
    ]


def _load_flashcard_events(
    db: Any,
    project_id: str,
    user_id: Optional[str],
):
    filters = ["project_id = :project_id"]
    params = {"project_id": project_id}

    if user_id:
        filters.append("user_id = :user_id")
        params["user_id"] = user_id

    return [
        {
            "topic": row[0],
            "is_correct": bool(row[1]),
            "occurred_at": row[2],
        }
        for row in db.execute(
            text(f"""
                select topic, is_correct, reviewed_at
                from flashcard_reviews
                where {' and '.join(filters)}
                and topic is not null
            """),
            params,
        ).fetchall()
    ]


def _apply_learning_event(
    stats,
    topic_lookup,
    topic,
    is_correct,
    occurred_at,
):
    topic_key = _normalize_topic_key(topic)
    category = topic_lookup.get(topic_key)

    if not category:
        return

    category_stats = stats[category]
    category_stats["total"] += 1

    if is_correct:
        category_stats["correct"] += 1

    category_stats["studied_topics"].add(topic_key)

    if (
        occurred_at
        and (
            category_stats["last_reviewed_at"] is None
            or occurred_at > category_stats["last_reviewed_at"]
        )
    ):
        category_stats["last_reviewed_at"] = occurred_at


def _empty_context(today: date) -> PlannerContext:
    return PlannerContext(
        preferences=PlannerPreferences(
            question_pace_seconds=DEFAULT_QUESTION_PACE_SECONDS,
            question_style=DEFAULT_QUESTION_STYLE,
        ),
        study_language=DEFAULT_STUDY_LANGUAGE,
        number_of_sessions=DEFAULT_NUMBER_OF_SESSIONS,
        planning_budget_minutes=DEFAULT_PLANNING_BUDGET_MINUTES,
        week_start_date=_week_start(today),
        week_id=f"week-{_week_start(today).isoformat()}",
    )


def _week_start(today: date) -> date:
    return today - timedelta(days=today.weekday())


def _normalize_topic_key(topic: Any) -> str:
    return " ".join(str(topic or "").strip().lower().split())


def _as_date(value: Any) -> date:
    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    return date.fromisoformat(str(value)[:10])


def _as_iso(value: Any) -> Optional[str]:
    if value is None:
        return None

    if hasattr(value, "isoformat"):
        return value.isoformat()

    return str(value)
