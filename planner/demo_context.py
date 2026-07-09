"""Temporary demo context builder for the first Planner API endpoint.

This module exists only to exercise the Planner pipeline before real frontend
and persistence integration are added.
"""

from datetime import date

from .category_selector import CategoryAnalytics
from .planner_models import PlannerContext, PlannerPreferences, SelectedTopic


def build_demo_planner_context() -> PlannerContext:
    """Build a deterministic mock PlannerContext for API integration testing."""

    return PlannerContext(
        categories=("Genetics", "Cell Biology", "Chemistry"),
        topics_by_category={
            "Genetics": (
                SelectedTopic(id="genetics-1", title="DNA structure", order=1),
                SelectedTopic(id="genetics-2", title="Gene expression", order=2),
                SelectedTopic(id="genetics-3", title="Genetic mutations", order=3),
                SelectedTopic(id="genetics-4", title="Chromosomal abnormalities", order=4),
            ),
            "Cell Biology": (
                SelectedTopic(id="cell-1", title="Cell membrane", order=1),
                SelectedTopic(id="cell-2", title="Organelles", order=2),
                SelectedTopic(id="cell-3", title="Cell cycle", order=3),
            ),
            "Chemistry": (
                SelectedTopic(id="chem-1", title="Chemical bonds", order=1),
                SelectedTopic(id="chem-2", title="Reaction rates", order=2),
            ),
        },
        analytics={
            "Genetics": CategoryAnalytics(accuracy=0.62, coverage=0.80, days_since_review=12),
            "Cell Biology": CategoryAnalytics(accuracy=0.78, coverage=0.55, days_since_review=20),
            "Chemistry": CategoryAnalytics(accuracy=0.85, coverage=0.90, days_since_review=4),
        },
        preferences=PlannerPreferences(question_pace_seconds=60),
        study_language="English",
        number_of_sessions=4,
        planning_budget_minutes=3,
        week_start_date=date(2026, 6, 29),
        week_id="demo-week",
    )
