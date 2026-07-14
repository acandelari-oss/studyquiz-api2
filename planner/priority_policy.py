"""Priority scoring policy for Planner category selection.

This module owns the deterministic priority score calculation. It is pure
Python and intentionally independent from selector models, FastAPI, database
access, and AI services.
"""

from typing import Optional


PERFORMANCE_WEIGHT = 40.0
RECENCY_WEIGHT = 25.0
COVERAGE_WEIGHT = 25.0
MANUAL_PRIORITY_WEIGHT = 10.0

MAX_REVIEW_DAYS_FOR_SCORING = 30


def calculate_priority_score(
    *,
    accuracy: Optional[float] = None,
    coverage: Optional[float] = None,
    days_since_review: Optional[int] = None,
    priority_weight: float = 1.0,
) -> float:
    """Calculate a deterministic category priority score.

    The score increases when performance is weaker, review recency is stale,
    coverage is lower, or an explicit priority weight is higher than neutral.
    """

    performance_score = 0.0
    if accuracy is not None:
        performance_score = (1.0 - _clamp(accuracy)) * PERFORMANCE_WEIGHT

    recency_score = 0.0
    if days_since_review is not None:
        normalized_days = min(
            max(days_since_review, 0),
            MAX_REVIEW_DAYS_FOR_SCORING,
        ) / MAX_REVIEW_DAYS_FOR_SCORING
        recency_score = normalized_days * RECENCY_WEIGHT

    coverage_score = 0.0
    if coverage is not None:
        coverage_score = (1.0 - _clamp(coverage)) * COVERAGE_WEIGHT

    manual_priority_score = (priority_weight - 1.0) * MANUAL_PRIORITY_WEIGHT

    return round(
        performance_score + recency_score + coverage_score + manual_priority_score,
        4,
    )


def _clamp(value: float) -> float:
    """Clamp a normalized analytics value to the 0.0-1.0 range."""

    return min(max(value, 0.0), 1.0)
