"""Planner domain models.

These are pure application-domain objects only. They are not database models,
ORM entities, FastAPI request models, or response schemas.
"""

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Mapping, Optional, Sequence

from .planner_state import (
    ActivityType,
    DailyStatus,
    ExecutionStatus,
    PlannerState,
    WeekStatus,
)


@dataclass
class SelectedTopic:
    """A topic selected inside a category for a planned learning activity."""

    id: str
    title: str
    order: Optional[int] = None


@dataclass
class ActivityConfiguration:
    """Configuration needed to launch a planned learning activity."""

    category: Optional[str] = None
    selected_topics: Sequence[SelectedTopic] = field(default_factory=tuple)
    estimated_duration_minutes: Optional[float] = None
    difficulty: Optional[str] = None
    question_style: Optional[str] = None
    num_questions: Optional[int] = None
    num_cards: Optional[int] = None


@dataclass
class ActivityExecution:
    """Runtime execution data for a single activity."""

    status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[int] = None


@dataclass
class ActivityResult:
    """Educational result data produced by a completed activity."""

    accuracy: Optional[float] = None
    score: Optional[float] = None
    correct: Optional[int] = None
    wrong: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Activity:
    """A planned learning activity with configuration, execution, and result."""

    id: str
    type: ActivityType
    configuration: ActivityConfiguration = field(default_factory=ActivityConfiguration)
    execution: ActivityExecution = field(default_factory=ActivityExecution)
    result: Optional[ActivityResult] = None


@dataclass
class DailyExecution:
    """Runtime execution data for a daily planner session."""

    status: ExecutionStatus = ExecutionStatus.NOT_STARTED
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    actual_duration: Optional[int] = None


@dataclass(frozen=True)
class HomeworkRecommendation:
    """Immutable Professor homework recommendation."""

    text: str
    rationale: str = ""
    related_categories: Sequence[str] = field(default_factory=tuple)
    estimated_effort: Optional[int] = None


@dataclass(frozen=True)
class ConversationReference:
    """Reference to a shared conversation used by planner-supported dialogue."""

    conversation_id: str
    context: str = ""


@dataclass(frozen=True)
class ExtraSessionOffer:
    """Offer for an optional extra session without representing the session."""

    available: bool = False
    reason: str = ""


@dataclass
class DailySummary:
    """Completed daily session summary and Professor follow-up material."""

    session_data: Mapping[str, Any] = field(default_factory=dict)
    professor_debrief: str = ""
    homework_recommendations: Sequence[HomeworkRecommendation] = field(default_factory=tuple)
    active_recall_offer: Optional[ConversationReference] = None
    office_hours_offer: Optional[ConversationReference] = None
    extra_session_offer: Optional[ExtraSessionOffer] = None


@dataclass
class DailyPlan:
    """One planned study day with execution state, activities, and summary."""

    id: str
    date: date
    day_name: str
    status: DailyStatus = DailyStatus.PLANNED
    objective: str = ""
    briefing: str = ""
    execution: DailyExecution = field(default_factory=DailyExecution)
    planned_allocations: Sequence[Any] = field(default_factory=tuple)
    activities: Sequence[Activity] = field(default_factory=tuple)
    summary: Optional[DailySummary] = None


@dataclass
class WeeklyStatistics:
    """Aggregated learning data for a planner week."""

    sessions_completed: int = 0
    quiz_accuracy: Optional[float] = None
    flashcard_completion: Optional[float] = None
    study_time: Optional[int] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class NextWeekOptions:
    """Professor or planner options offered after a week is completed."""

    recommendations: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass
class Week:
    """A generated planner week containing daily plans and weekly outcomes."""

    id: str
    start_date: date
    end_date: date
    plan_type: str = "study_plan"
    study_language: Optional[str] = None
    status: WeekStatus = WeekStatus.PLANNED
    weekly_briefing: str = ""
    weekly_statistics: WeeklyStatistics = field(default_factory=WeeklyStatistics)
    daily_plans: Sequence[DailyPlan] = field(default_factory=tuple)
    weekly_review: Optional[str] = None
    next_week_options: Optional[NextWeekOptions] = None


@dataclass
class Planner:
    """Top-level planner state for a project."""

    state: PlannerState = PlannerState.NEW_PROJECT
    current_week: Optional[Week] = None


@dataclass(frozen=True)
class PlannerPreferences:
    """Student preferences that influence future planner generation."""

    question_pace_seconds: Optional[int] = None
    question_style: Optional[str] = None


@dataclass(frozen=True)
class PlannerContext:
    """Pure domain input required to generate a planner week.

    The fields stay intentionally generic so callers can adapt project data,
    analytics, and preferences without introducing database, FastAPI, or ORM
    dependencies into the Planner Engine.
    """

    project: Any = None
    categories: Sequence[str] = field(default_factory=tuple)
    topics_by_category: Mapping[str, Sequence[SelectedTopic]] = field(default_factory=dict)
    analytics: Mapping[str, Any] = field(default_factory=dict)
    preferences: PlannerPreferences = field(default_factory=PlannerPreferences)
    study_language: Optional[str] = None
    number_of_sessions: int = 0
    planning_budget_minutes: float = 0.0
    week_start_date: Optional[date] = None
    week_id: Optional[str] = None
