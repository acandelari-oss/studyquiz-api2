from datetime import datetime, timedelta, timezone

from sqlalchemy import text


SUPPORTED_ACTIVITY_TYPES = (
    "quiz",
    "flashcards",
    "ask",
    "active_recall",
    "planner",
)


def _parse_datetime(value):
    if value is None:
        return None

    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except (TypeError, ValueError):
            return None

    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def _completed_duration_seconds(started_at, completed_at):
    started = _parse_datetime(started_at)
    completed = _parse_datetime(completed_at)

    if not started or not completed:
        return 0

    duration = (completed - started).total_seconds()
    if duration < 0:
        return 0

    return duration


def _row_value(row, key, index):
    if hasattr(row, "_mapping"):
        return row._mapping[key]
    return row[index]


def _current_streak_from_completed_days(completed_days, now=None):
    if not completed_days:
        return 0

    today = (now or datetime.now(timezone.utc)).astimezone(timezone.utc).date()
    yesterday = today - timedelta(days=1)

    if today in completed_days:
        cursor = today
    elif yesterday in completed_days:
        cursor = yesterday
    else:
        return 0

    streak = 0
    while cursor in completed_days:
        streak += 1
        cursor -= timedelta(days=1)

    return streak


def _quiz_accuracy_history(quiz_rows):
    history = []

    for row in quiz_rows:
        completed_at = _row_value(row, "completed_at", 0)
        score = _row_value(row, "score", 1)
        total_questions = _row_value(row, "total_questions", 2)

        try:
            score_value = float(score)
            total_value = float(total_questions)
        except (TypeError, ValueError):
            continue

        if total_value <= 0:
            continue

        completed = _parse_datetime(completed_at)
        if not completed:
            continue

        history.append({
            "completed_at": completed.isoformat(),
            "accuracy": round((score_value / total_value) * 100, 1),
        })

    return history


def summarize_learning_sessions(rows, now=None, quiz_rows=None):
    total_sessions = len(rows)
    completed_sessions = 0
    abandoned_sessions = 0
    total_study_seconds = 0
    activities = {
        session_type: 0
        for session_type in SUPPORTED_ACTIVITY_TYPES
    }
    completed_days = set()

    for row in rows:
        session_type = _row_value(row, "session_type", 0)
        status = _row_value(row, "status", 1)
        started_at = _row_value(row, "started_at", 2)
        completed_at = _row_value(row, "completed_at", 3)

        if session_type in activities:
            activities[session_type] += 1

        if status == "completed":
            completed_sessions += 1
            total_study_seconds += _completed_duration_seconds(
                started_at,
                completed_at,
            )
            completed = _parse_datetime(completed_at)
            if completed:
                completed_days.add(completed.date())
        elif status == "abandoned":
            abandoned_sessions += 1

    completion_rate = (
        round((completed_sessions / total_sessions) * 100, 1)
        if total_sessions > 0
        else 0
    )

    favorite_activity = None
    if total_sessions > 0:
        favorite_activity = max(
            SUPPORTED_ACTIVITY_TYPES,
            key=lambda session_type: activities[session_type],
        )
        if activities[favorite_activity] == 0:
            favorite_activity = None

    return {
        "total_sessions": total_sessions,
        "completed_sessions": completed_sessions,
        "abandoned_sessions": abandoned_sessions,
        "completion_rate": completion_rate,
        "total_study_seconds": int(total_study_seconds),
        "current_streak": _current_streak_from_completed_days(completed_days, now),
        "quiz_accuracy_history": _quiz_accuracy_history(quiz_rows or []),
        "favorite_activity": favorite_activity,
        "activities": activities,
    }


def get_learning_summary(db, user_id):
    rows = db.execute(
        text("""
            select
                session_type,
                status,
                started_at,
                completed_at
            from learning_sessions
            where user_id = :user_id
        """),
        {"user_id": user_id},
    ).fetchall()

    quiz_rows = db.execute(
        text("""
            select
                created_at as completed_at,
                score,
                total_questions
            from quiz_attempts
            where user_id = :user_id
            and total_questions > 0
            order by created_at asc
        """),
        {"user_id": user_id},
    ).fetchall()

    return summarize_learning_sessions(rows, quiz_rows=quiz_rows)
