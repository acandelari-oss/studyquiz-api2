"""Project deletion lifecycle helpers.

Permanent project deletion removes only data that is exclusively owned by the
project. User/account data is intentionally outside this lifecycle.
"""

from sqlalchemy import text


def delete_project_owned_data(db, project_id: str) -> None:
    """Delete all known project-exclusive data for one project.

    The caller owns the transaction boundary.
    """

    params = {"project_id": project_id}

    # Chat rows may be project-scoped directly or quiz-scoped.
    db.execute(
        text("""
            delete from chat_messages
            where project_id = :project_id
            or quiz_id in (
                select id
                from quizzes
                where project_id = :project_id
            )
        """),
        params,
    )

    # Quiz children first because quizzes currently has no project FK.
    db.execute(
        text("""
            delete from quiz_answers
            where quiz_id in (
                select id
                from quizzes
                where project_id = :project_id
            )
            or question_id in (
                select qq.id
                from quiz_questions qq
                join quizzes q on q.id = qq.quiz_id
                where q.project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("""
            delete from quiz_questions
            where quiz_id in (
                select id
                from quizzes
                where project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("delete from quiz_attempts where project_id = :project_id"),
        params,
    )

    db.execute(
        text("""
            delete from hard_quiz_generation_samples
            where run_id in (
                select id
                from hard_quiz_generation_runs
                where project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("delete from hard_quiz_generation_runs where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from quizzes where project_id = :project_id"),
        params,
    )

    # Flashcard reviews have no reliable FK today, so delete by both project
    # scope and flashcard scope.
    db.execute(
        text("""
            delete from flashcard_reviews
            where project_id = :project_id
            or flashcard_id in (
                select id
                from flashcards
                where project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("delete from flashcards where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from recall_answers where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from learning_events where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from learning_sessions where project_id = :project_id"),
        params,
    )

    db.execute(
        text("""
            delete from planner_activities
            where daily_plan_id in (
                select dp.id
                from planner_daily_plans dp
                join planner_weeks w on w.id = dp.week_id
                where w.project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("""
            delete from planner_daily_plans
            where week_id in (
                select id
                from planner_weeks
                where project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("delete from planner_weeks where project_id = :project_id"),
        params,
    )

    db.execute(
        text("""
            delete from topic_chunks
            where topic_id in (
                select id
                from topics
                where project_id = :project_id
            )
            or chunk_id in (
                select id
                from chunks
                where project_id = :project_id
            )
        """),
        params,
    )

    db.execute(
        text("delete from topics where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from chunks where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from documents where project_id = :project_id"),
        params,
    )

    db.execute(
        text("delete from projects where id = :project_id"),
        params,
    )

