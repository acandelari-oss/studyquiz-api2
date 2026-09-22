#!/usr/bin/env python3
"""Rollback-based validation for permanent project deletion.

The script creates disposable project-owned rows, calls the same deletion
service used by the API endpoint, verifies that project-owned data is gone,
verifies that a sibling project is untouched, and rolls the transaction back.

It writes only transient rows inside a transaction and leaves no test data
behind when it exits normally.
"""

import json
import os
import sys
import uuid
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from project_deletion_service import delete_project_owned_data  # noqa: E402


def load_env_file(path: Path) -> None:
    if not path.exists():
        return

    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def scalar(conn, sql: str, params: dict) -> int:
    return int(conn.execute(text(sql), params).scalar() or 0)


def insert_fixture(conn, project_id: str, sibling_project_id: str, user_id: str) -> None:
    quiz_id = str(uuid.uuid4())
    question_id = str(uuid.uuid4())
    flashcard_id = str(uuid.uuid4())
    document_id = str(uuid.uuid4())
    topic_id = str(uuid.uuid4())
    week_id = f"week-{uuid.uuid4()}"
    day_id = f"day-{uuid.uuid4()}"
    activity_id = f"activity-{uuid.uuid4()}"
    hard_run_id = str(uuid.uuid4())

    conn.execute(
        text("insert into projects (id, user_id, name) values (:id, :user_id, :name)"),
        {"id": project_id, "user_id": user_id, "name": "DELETE VALIDATION PROJECT"},
    )
    conn.execute(
        text("insert into projects (id, user_id, name) values (:id, :user_id, :name)"),
        {"id": sibling_project_id, "user_id": user_id, "name": "UNTOUCHED VALIDATION PROJECT"},
    )

    conn.execute(
        text("""
            insert into documents (id, project_id, title, text)
            values (:id, :project_id, :title, :text)
        """),
        {"id": document_id, "project_id": project_id, "title": "delete-test.pdf", "text": "text"},
    )

    chunk_id = conn.execute(
        text("""
            insert into chunks (project_id, document_id, doc_title, chunk_text, page)
            values (:project_id, :document_id, :doc_title, :chunk_text, 1)
            returning id
        """),
        {
            "project_id": project_id,
            "document_id": document_id,
            "doc_title": "delete-test.pdf",
            "chunk_text": "This is a disposable validation chunk with enough text.",
        },
    ).scalar()

    conn.execute(
        text("""
            insert into topics (id, project_id, document_id, topic, category)
            values (:id, :project_id, :document_id, :topic, :category)
        """),
        {
            "id": topic_id,
            "project_id": project_id,
            "document_id": document_id,
            "topic": "Disposable Topic",
            "category": "VALIDATION",
        },
    )
    conn.execute(
        text("insert into topic_chunks (topic_id, chunk_id) values (:topic_id, :chunk_id)"),
        {"topic_id": topic_id, "chunk_id": chunk_id},
    )

    conn.execute(
        text("""
            insert into quizzes (id, project_id, user_id, num_questions, difficulty)
            values (:id, :project_id, :user_id, 1, 'medium')
        """),
        {"id": quiz_id, "project_id": project_id, "user_id": user_id},
    )
    conn.execute(
        text("""
            insert into quiz_questions
            (id, quiz_id, question_order, question, options)
            values (:id, :quiz_id, 1, 'Question?', cast(:options as jsonb))
        """),
        {"id": question_id, "quiz_id": quiz_id, "options": json.dumps(["A", "B"])},
    )
    conn.execute(
        text("""
            insert into quiz_attempts
            (id, project_id, quiz_id, user_id, score, total_questions, topic, question_index)
            values (:id, :project_id, :quiz_id, :user_id, 1, 1, 'Disposable Topic', 0)
        """),
        {
            "id": str(uuid.uuid4()),
            "project_id": project_id,
            "quiz_id": quiz_id,
            "user_id": user_id,
        },
    )
    conn.execute(
        text("""
            insert into quiz_answers
            (id, quiz_id, question_id, user_id, is_correct, topic)
            values (:id, :quiz_id, :question_id, :user_id, true, 'Disposable Topic')
        """),
        {
            "id": str(uuid.uuid4()),
            "quiz_id": quiz_id,
            "question_id": question_id,
            "user_id": user_id,
        },
    )

    conn.execute(
        text("""
            insert into hard_quiz_generation_runs
            (
                id, project_id, project_name, quiz_id, difficulty, question_style,
                requested_questions, generated_questions, accepted_questions,
                rejected_questions, acceptance_rate, rejection_reasons_breakdown
            )
            values
            (
                :id, :project_id, 'DELETE VALIDATION PROJECT', :quiz_id, 'medium', 'standard',
                1, 1, 1, 0, 1.0, cast('{}' as jsonb)
            )
        """),
        {"id": hard_run_id, "project_id": project_id, "quiz_id": quiz_id},
    )
    conn.execute(
        text("""
            insert into hard_quiz_generation_samples
            (id, run_id, outcome, question_stem, rejection_reasons)
            values (:id, :run_id, 'accepted', 'Question?', cast('[]' as jsonb))
        """),
        {"id": str(uuid.uuid4()), "run_id": hard_run_id},
    )

    conn.execute(
        text("""
            insert into flashcards (id, project_id, user_id, question, answer, topic)
            values (:id, :project_id, :user_id, 'Q', 'A', 'Disposable Topic')
        """),
        {"id": flashcard_id, "project_id": project_id, "user_id": user_id},
    )
    conn.execute(
        text("""
            insert into flashcard_reviews
            (id, flashcard_id, project_id, user_id, is_correct, difficulty)
            values (:id, :flashcard_id, :project_id, :user_id, true, 3)
        """),
        {
            "id": str(uuid.uuid4()),
            "flashcard_id": flashcard_id,
            "project_id": project_id,
            "user_id": user_id,
        },
    )

    conn.execute(
        text("insert into recall_answers (id, project_id, topic, correct) values (:id, :project_id, 'T', true)"),
        {"id": str(uuid.uuid4()), "project_id": project_id},
    )
    conn.execute(
        text("insert into learning_events (id, project_id, user_id, topic, type, correct) values (:id, :project_id, :user_id, 'T', 'quiz', true)"),
        {"id": str(uuid.uuid4()), "project_id": project_id, "user_id": user_id},
    )
    conn.execute(
        text("""
            insert into learning_sessions
            (id, project_id, user_id, session_type, started_at, completed_at, status)
            values (:id, :project_id, :user_id, 'quiz', now(), now(), 'completed')
        """),
        {"id": str(uuid.uuid4()), "project_id": project_id, "user_id": user_id},
    )

    conn.execute(
        text("""
            insert into planner_weeks
            (id, project_id, start_date, end_date, status, planning_parameters, weekly_statistics)
            values
            (:id, :project_id, current_date, current_date, 'active', cast('{}' as jsonb), cast('{}' as jsonb))
        """),
        {"id": week_id, "project_id": project_id},
    )
    conn.execute(
        text("""
            insert into planner_daily_plans
            (id, week_id, session_index, plan_date, day_name, status, planned_allocations)
            values
            (:id, :week_id, 1, current_date, 'Today', 'planned', cast('[]' as jsonb))
        """),
        {"id": day_id, "week_id": week_id},
    )
    conn.execute(
        text("""
            insert into planner_activities
            (id, daily_plan_id, activity_index, activity_type, configuration)
            values (:id, :daily_plan_id, 1, 'quiz', cast('{}' as jsonb))
        """),
        {"id": activity_id, "daily_plan_id": day_id},
    )

    conn.execute(
        text("""
            insert into chat_messages (id, project_id, quiz_id, question_index, role, message)
            values (:id, :project_id, :quiz_id, 0, 'user', 'hello')
        """),
        {"id": str(uuid.uuid4()), "project_id": project_id, "quiz_id": quiz_id},
    )


def main() -> int:
    load_env_file(ROOT / ".env")
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("DATABASE_URL is required")

    project_id = str(uuid.uuid4())
    sibling_project_id = str(uuid.uuid4())

    engine = create_engine(database_url)
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            user_id = conn.execute(
                text("select id from auth.users order by created_at asc limit 1")
            ).scalar()
            if not user_id:
                raise RuntimeError("At least one auth user is required for validation")
            user_id = str(user_id)

            insert_fixture(conn, project_id, sibling_project_id, user_id)
            delete_project_owned_data(conn, project_id)

            checks = {
                "projects": "select count(*) from projects where id = :project_id",
                "documents": "select count(*) from documents where project_id = :project_id",
                "chunks": "select count(*) from chunks where project_id = :project_id",
                "topics": "select count(*) from topics where project_id = :project_id",
                "topic_chunks": """
                    select count(*)
                    from topic_chunks tc
                    left join topics t on t.id = tc.topic_id
                    left join chunks c on c.id = tc.chunk_id
                    where t.project_id = :project_id or c.project_id = :project_id
                """,
                "quizzes": "select count(*) from quizzes where project_id = :project_id",
                "quiz_questions": """
                    select count(*)
                    from quiz_questions qq
                    join quizzes q on q.id = qq.quiz_id
                    where q.project_id = :project_id
                """,
                "quiz_attempts": "select count(*) from quiz_attempts where project_id = :project_id",
                "quiz_answers": """
                    select count(*)
                    from quiz_answers qa
                    left join quizzes q on q.id = qa.quiz_id
                    where q.project_id = :project_id
                """,
                "flashcards": "select count(*) from flashcards where project_id = :project_id",
                "flashcard_reviews": "select count(*) from flashcard_reviews where project_id = :project_id",
                "recall_answers": "select count(*) from recall_answers where project_id = :project_id",
                "learning_events": "select count(*) from learning_events where project_id = :project_id",
                "learning_sessions": "select count(*) from learning_sessions where project_id = :project_id",
                "planner_weeks": "select count(*) from planner_weeks where project_id = :project_id",
                "planner_daily_plans": """
                    select count(*)
                    from planner_daily_plans dp
                    join planner_weeks w on w.id = dp.week_id
                    where w.project_id = :project_id
                """,
                "planner_activities": """
                    select count(*)
                    from planner_activities a
                    join planner_daily_plans dp on dp.id = a.daily_plan_id
                    join planner_weeks w on w.id = dp.week_id
                    where w.project_id = :project_id
                """,
                "chat_messages": "select count(*) from chat_messages where project_id = :project_id",
                "hard_quiz_generation_runs": "select count(*) from hard_quiz_generation_runs where project_id = :project_id",
                "hard_quiz_generation_samples": """
                    select count(*)
                    from hard_quiz_generation_samples s
                    join hard_quiz_generation_runs r on r.id = s.run_id
                    where r.project_id = :project_id
                """,
            }

            failures = {}
            for name, sql in checks.items():
                count = scalar(conn, sql, {"project_id": project_id})
                if count != 0:
                    failures[name] = count

            sibling_count = scalar(
                conn,
                "select count(*) from projects where id = :project_id",
                {"project_id": sibling_project_id},
            )
            if sibling_count != 1:
                failures["sibling_project_untouched"] = sibling_count

            if failures:
                raise AssertionError(f"Project deletion validation failed: {failures}")

            print("Project deletion lifecycle validation passed.")
            print("Disposable rows were created, deleted, verified, and rolled back.")
            return 0
        finally:
            transaction.rollback()


if __name__ == "__main__":
    raise SystemExit(main())
