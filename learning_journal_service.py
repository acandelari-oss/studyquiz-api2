from sqlalchemy import text

from learning_summary_service import _parse_datetime


DEFAULT_JOURNAL_LIMIT = 50
MAX_JOURNAL_LIMIT = 200


def _row_value(row, key, index):
    if hasattr(row, "_mapping"):
        return row._mapping[key]
    return row[index]


def _format_datetime(value):
    parsed = _parse_datetime(value)
    if parsed is None:
        return str(value) if value is not None else None
    return parsed.isoformat()


def _session_duration_seconds(status, started_at, completed_at):
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


def serialize_learning_journal_rows(rows):
    journal = []

    for row in rows:
        started_at = _row_value(row, "started_at", 3)
        completed_at = _row_value(row, "completed_at", 4)
        status = _row_value(row, "status", 5)

        journal.append({
            "id": str(_row_value(row, "id", 0)),
            "project_id": str(_row_value(row, "project_id", 1)),
            "session_type": _row_value(row, "session_type", 2),
            "started_at": _format_datetime(started_at),
            "completed_at": _format_datetime(completed_at),
            "status": status,
            "duration_seconds": _session_duration_seconds(
                status,
                started_at,
                completed_at,
            ),
        })

    return journal


def normalize_journal_pagination(limit=None, offset=None):
    try:
        resolved_limit = int(limit)
    except (TypeError, ValueError):
        resolved_limit = DEFAULT_JOURNAL_LIMIT

    try:
        resolved_offset = int(offset)
    except (TypeError, ValueError):
        resolved_offset = 0

    if resolved_limit < 1:
        resolved_limit = DEFAULT_JOURNAL_LIMIT
    if resolved_limit > MAX_JOURNAL_LIMIT:
        resolved_limit = MAX_JOURNAL_LIMIT
    if resolved_offset < 0:
        resolved_offset = 0

    return resolved_limit, resolved_offset


def get_learning_journal(db, user_id, limit=None, offset=None):
    resolved_limit, resolved_offset = normalize_journal_pagination(
        limit,
        offset,
    )

    rows = db.execute(
        text("""
            select
                id,
                project_id,
                session_type,
                started_at,
                completed_at,
                status
            from learning_sessions
            where user_id = :user_id
            order by started_at desc
            limit :limit
            offset :offset
        """),
        {
            "user_id": user_id,
            "limit": resolved_limit,
            "offset": resolved_offset,
        },
    ).fetchall()

    return serialize_learning_journal_rows(rows)
