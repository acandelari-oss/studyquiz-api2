from statistics import median

from sqlalchemy import text

from learning_summary_service import SUPPORTED_ACTIVITY_TYPES, _parse_datetime


MIN_COMPLETED_SESSIONS_FOR_PREFERENCE = 3
MIN_DURATIONS_FOR_TYPICAL_SESSION = 3
MIN_ATTEMPTS_FOR_RELIABILITY = 3


def _row_value(row, key, index):
    if hasattr(row, "_mapping"):
        return row._mapping[key]
    return row[index]


def _valid_completed_duration_seconds(status, started_at, completed_at):
    if status != "completed":
        return None

    started = _parse_datetime(started_at)
    completed = _parse_datetime(completed_at)

    if not started or not completed:
        return None

    duration = (completed - started).total_seconds()
    if duration < 0:
        return None

    return int(duration)


def _duration_profile(typical_session_seconds):
    if typical_session_seconds is None:
        return None
    if typical_session_seconds < 10 * 60:
        return "short"
    if typical_session_seconds <= 45 * 60:
        return "balanced"
    return "long"


def _strict_winner(values_by_activity, eligible_activity):
    candidates = [
        activity
        for activity in SUPPORTED_ACTIVITY_TYPES
        if eligible_activity(activity)
    ]

    if not candidates:
        return None

    max_value = max(values_by_activity[activity] for activity in candidates)
    winners = [
        activity
        for activity in candidates
        if values_by_activity[activity] == max_value
    ]

    if len(winners) != 1:
        return None

    return winners[0]


def build_learning_preferences(rows):
    total_by_activity = {
        activity: 0
        for activity in SUPPORTED_ACTIVITY_TYPES
    }
    completed_by_activity = {
        activity: 0
        for activity in SUPPORTED_ACTIVITY_TYPES
    }
    valid_completed_durations = []

    for row in rows:
        session_type = _row_value(row, "session_type", 0)
        status = _row_value(row, "status", 1)
        started_at = _row_value(row, "started_at", 2)
        completed_at = _row_value(row, "completed_at", 3)

        if session_type not in SUPPORTED_ACTIVITY_TYPES:
            continue

        total_by_activity[session_type] += 1

        if status == "completed":
            completed_by_activity[session_type] += 1

        duration = _valid_completed_duration_seconds(
            status,
            started_at,
            completed_at,
        )
        if duration is not None:
            valid_completed_durations.append(duration)

    total_sessions_observed = sum(total_by_activity.values())
    completed_sessions_observed = sum(completed_by_activity.values())

    preferred_activity = None
    if completed_sessions_observed >= MIN_COMPLETED_SESSIONS_FOR_PREFERENCE:
        preferred_activity = _strict_winner(
            completed_by_activity,
            lambda activity: completed_by_activity[activity] > 0,
        )

    typical_session_seconds = None
    if len(valid_completed_durations) >= MIN_DURATIONS_FOR_TYPICAL_SESSION:
        typical_session_seconds = int(median(valid_completed_durations))

    completion_rate_by_activity = {}
    for activity in SUPPORTED_ACTIVITY_TYPES:
        total = total_by_activity[activity]
        if total <= 0:
            completion_rate_by_activity[activity] = None
        else:
            completion_rate_by_activity[activity] = round(
                (completed_by_activity[activity] / total) * 100,
                1,
            )

    most_reliable_activity = _strict_winner(
        completion_rate_by_activity,
        lambda activity: total_by_activity[activity] >= MIN_ATTEMPTS_FOR_RELIABILITY,
    )

    return {
        "total_sessions_observed": total_sessions_observed,
        "completed_sessions_observed": completed_sessions_observed,
        "preferred_activity": preferred_activity,
        "typical_session_seconds": typical_session_seconds,
        "session_duration_profile": _duration_profile(typical_session_seconds),
        "completion_rate_by_activity": completion_rate_by_activity,
        "most_reliable_activity": most_reliable_activity,
    }


def get_learning_preferences(db, user_id):
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

    return build_learning_preferences(rows)
