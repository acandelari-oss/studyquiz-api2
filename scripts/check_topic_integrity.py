#!/usr/bin/env python3
"""Read-only topic/chunk integrity diagnostics for local development.

Reports:
- topic_chunks rows whose chunk_id no longer exists;
- topics whose project_id no longer exists;
- existing-project topics that have zero valid chunk links.

This script does not modify data.
"""

import os
from pathlib import Path

from sqlalchemy import create_engine, text


ROOT = Path(__file__).resolve().parents[1]


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def scalar(conn, sql: str):
    return conn.execute(text(sql)).scalar() or 0


def print_rows(title: str, rows) -> None:
    print()
    print(title)
    print("-" * len(title))
    if not rows:
        print("None")
        return

    for row in rows:
        print(dict(row._mapping))


def main() -> int:
    load_env_file(ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    engine = create_engine(database_url)

    with engine.connect() as conn:
        orphan_topic_chunks = scalar(
            conn,
            """
            SELECT count(*)
            FROM public.topic_chunks tc
            WHERE NOT EXISTS (
                SELECT 1
                FROM public.chunks c
                WHERE c.id = tc.chunk_id
            )
            """,
        )

        topics_missing_projects = scalar(
            conn,
            """
            SELECT count(*)
            FROM public.topics t
            WHERE t.project_id IS NOT NULL
            AND NOT EXISTS (
                SELECT 1
                FROM public.projects p
                WHERE p.id = t.project_id
            )
            """,
        )

        active_zero_evidence_topics = scalar(
            conn,
            """
            SELECT count(*)
            FROM public.topics t
            JOIN public.projects p ON p.id = t.project_id
            WHERE NOT EXISTS (
                SELECT 1
                FROM public.topic_chunks tc
                JOIN public.chunks c ON c.id = tc.chunk_id
                WHERE tc.topic_id = t.id
            )
            """,
        )

        print("DOUNO topic integrity check")
        print("===========================")
        print(f"orphan_topic_chunks: {orphan_topic_chunks}")
        print(f"topics_missing_projects: {topics_missing_projects}")
        print(f"active_project_topics_with_zero_valid_chunks: {active_zero_evidence_topics}")

        project_rows = conn.execute(
            text(
                """
                WITH topic_state AS (
                    SELECT
                        p.id AS project_id,
                        p.name AS project_name,
                        t.id AS topic_id,
                        t.topic,
                        count(c.id) AS valid_links
                    FROM public.topics t
                    JOIN public.projects p ON p.id = t.project_id
                    LEFT JOIN public.topic_chunks tc ON tc.topic_id = t.id
                    LEFT JOIN public.chunks c ON c.id = tc.chunk_id
                    GROUP BY p.id, p.name, t.id, t.topic
                )
                SELECT
                    project_id,
                    project_name,
                    count(*) AS topic_count,
                    count(*) FILTER (WHERE valid_links = 0) AS zero_evidence_topics
                FROM topic_state
                GROUP BY project_id, project_name
                HAVING count(*) FILTER (WHERE valid_links = 0) > 0
                ORDER BY zero_evidence_topics DESC, project_name ASC
                LIMIT 25
                """
            )
        ).fetchall()

        print_rows(
            "Existing projects with zero-evidence topics",
            project_rows,
        )

        topic_rows = conn.execute(
            text(
                """
                SELECT
                    p.id AS project_id,
                    p.name AS project_name,
                    t.id AS topic_id,
                    t.topic,
                    t.category
                FROM public.topics t
                JOIN public.projects p ON p.id = t.project_id
                WHERE NOT EXISTS (
                    SELECT 1
                    FROM public.topic_chunks tc
                    JOIN public.chunks c ON c.id = tc.chunk_id
                    WHERE tc.topic_id = t.id
                )
                ORDER BY p.name ASC, t.topic ASC, t.id ASC
                LIMIT 100
                """
            )
        ).fetchall()

        print_rows(
            "Sample active-project topics with zero valid chunk links",
            topic_rows,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

