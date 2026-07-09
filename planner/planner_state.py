"""Planner lifecycle and execution states used by the Study Planner domain."""

from enum import Enum


class PlannerState(str, Enum):
    """High-level lifecycle states for a project's Study Planner."""

    NEW_PROJECT = "NEW_PROJECT"
    READY_FOR_FIRST_PLAN = "READY_FOR_FIRST_PLAN"
    ACTIVE_WEEK = "ACTIVE_WEEK"
    WEEK_COMPLETED = "WEEK_COMPLETED"


class WeekStatus(str, Enum):
    """Lifecycle states for a generated planner week."""

    PLANNED = "PLANNED"
    ACTIVE = "ACTIVE"
    COMPLETED = "COMPLETED"


class DailyStatus(str, Enum):
    """Lifecycle states for a daily planner session."""

    PLANNED = "PLANNED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    SKIPPED = "SKIPPED"


class ActivityType(str, Enum):
    """Supported planner activity types.

    The enum can grow as new learning activities become available.
    """

    FLASHCARDS = "FLASHCARDS"
    QUIZ = "QUIZ"


class ExecutionStatus(str, Enum):
    """Runtime execution states shared by sessions and activities."""

    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
