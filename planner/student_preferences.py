"""Deterministic student preference adjustments for Planner ranking.

This stage runs after the base Planner score has been computed. It does not
compute analytics, call AI services, or introduce probabilistic behaviour.
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence


PRIORITY_CATEGORY_SCORE_MULTIPLIER = 1.15
MAX_STUDY_PRIORITY_CATEGORIES = 3


def apply_student_preference_score_bonus(
    *,
    category: str,
    planner_score: float,
    planner_preferences: Any = None,
    multiplier: float = PRIORITY_CATEGORY_SCORE_MULTIPLIER,
) -> float:
    """Return the final Planner score after deterministic student preferences."""

    if category in selected_priority_categories(planner_preferences):
        return planner_score * multiplier

    return planner_score


def selected_priority_categories(planner_preferences: Any = None) -> Sequence[str]:
    """Extract selected priority categories from supported preference shapes."""

    if planner_preferences is None:
        return ()

    value = None

    if isinstance(planner_preferences, dict):
        value = (
            planner_preferences.get("priority_categories")
            or planner_preferences.get("priorityCategories")
        )
    else:
        value = (
            getattr(planner_preferences, "priority_categories", None)
            or getattr(planner_preferences, "priorityCategories", None)
        )

    if not value:
        return ()

    return tuple(_unique_non_empty_strings(value)[:MAX_STUDY_PRIORITY_CATEGORIES])


def _unique_non_empty_strings(values: Iterable[Any]) -> Sequence[str]:
    seen = set()
    result = []

    for value in values:
        category = str(value or "").strip()

        if not category or category in seen:
            continue

        seen.add(category)
        result.append(category)

    return tuple(result)
