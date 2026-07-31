from collections import Counter
from datetime import datetime, timedelta, timezone

from sqlalchemy import text

from learning_summary_service import (
    SUPPORTED_ACTIVITY_TYPES,
    _completed_duration_seconds,
    _parse_datetime,
    summarize_learning_sessions,
)


ACTIVITY_LABELS = {
    "quiz": "Quizzes",
    "flashcards": "Flashcards",
    "ask": "Ask",
    "active_recall": "Active Recall",
    "planner": "Study Planner",
}


def _row_value(row, key, index):
    if hasattr(row, "_mapping"):
        return row._mapping[key]
    return row[index]


def _insight(insight_type, level, title, message):
    return {
        "type": insight_type,
        "level": level,
        "title": title,
        "message": message,
    }


def _completion_rate_insight(summary):
    if summary["total_sessions"] <= 0:
        return None

    completion_rate = summary["completion_rate"]

    if completion_rate >= 90:
        return _insight(
            "completion_rate",
            "positive",
            "Excellent consistency",
            "You complete {:.1f}% of your learning sessions.".format(
                completion_rate,
            ),
        )

    if completion_rate >= 70:
        return _insight(
            "completion_rate",
            "neutral",
            "Steady learning rhythm",
            "You complete {:.1f}% of your learning sessions.".format(
                completion_rate,
            ),
        )

    return _insight(
        "completion_rate",
        "warning",
        "Interrupted sessions are frequent",
        "You complete {:.1f}% of your learning sessions.".format(
            completion_rate,
        ),
    )


def _completed_activity_counts(rows):
    counts = Counter()

    for row in rows:
        session_type = _row_value(row, "session_type", 0)
        status = _row_value(row, "status", 1)

        if status == "completed" and session_type in SUPPORTED_ACTIVITY_TYPES:
            counts[session_type] += 1

    return counts


def _favorite_activity_insight(rows, completed_sessions):
    if completed_sessions <= 0:
        return None

    counts = _completed_activity_counts(rows)

    if not counts:
        return None

    session_type, count = counts.most_common(1)[0]

    if count <= completed_sessions / 2:
        return None

    activity_label = ACTIVITY_LABELS.get(session_type, session_type)

    return _insight(
        "favorite_activity",
        "info",
        "Preferred learning style",
        "{} are currently your preferred study activity.".format(
            activity_label,
        ),
    )


def _average_duration_insight(summary):
    completed_sessions = summary["completed_sessions"]

    if completed_sessions <= 0:
        return None

    average_seconds = summary["total_study_seconds"] / completed_sessions

    if average_seconds < 10 * 60:
        return _insight(
            "average_session_duration",
            "info",
            "Short study sessions",
            "Your completed learning sessions are usually shorter than 10 minutes.",
        )

    if average_seconds <= 45 * 60:
        return _insight(
            "average_session_duration",
            "info",
            "Balanced study sessions",
            "Your completed learning sessions usually last between 10 and 45 minutes.",
        )

    return _insight(
        "average_session_duration",
        "info",
        "Long study sessions",
        "Your completed learning sessions usually last more than 45 minutes.",
    )


def _recent_activity_insight(rows, now=None):
    reference_now = now or datetime.now(timezone.utc)
    if reference_now.tzinfo is None:
        reference_now = reference_now.replace(tzinfo=timezone.utc)

    cutoff = reference_now.astimezone(timezone.utc) - timedelta(days=7)

    for row in rows:
        status = _row_value(row, "status", 1)
        completed_at = _row_value(row, "completed_at", 3)

        if status != "completed":
            continue

        completed = _parse_datetime(completed_at)

        if completed and completed >= cutoff:
            return None

    return _insight(
        "recent_activity",
        "warning",
        "No recent completed sessions",
        "No completed learning sessions were recorded during the last week.",
    )


def build_learning_intelligence(rows, now=None):
    summary = summarize_learning_sessions(rows)
    insights = []

    completion_rate = _completion_rate_insight(summary)
    if completion_rate:
        insights.append(completion_rate)

    favorite_activity = _favorite_activity_insight(
        rows,
        summary["completed_sessions"],
    )
    if favorite_activity:
        insights.append(favorite_activity)

    average_duration = _average_duration_insight(summary)
    if average_duration:
        insights.append(average_duration)

    recent_activity = _recent_activity_insight(rows, now=now)
    if recent_activity:
        insights.append(recent_activity)

    return insights


def get_learning_intelligence(db, user_id, now=None):
    rows = db.execute(
        text("""
            select
                session_type,
                status,
                started_at,
                completed_at
            from learning_sessions
            where user_id = :user_id
            order by started_at desc
        """),
        {"user_id": user_id},
    ).fetchall()

    return build_learning_intelligence(rows, now=now)
