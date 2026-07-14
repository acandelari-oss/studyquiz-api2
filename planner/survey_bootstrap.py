"""Survey bootstrap bias for first Study Plan generation.

The learning survey is a temporary self-assessment signal. It is not objective
analytics and it is not mastery. This module only converts valid survey answers
into a moderate priority-weight bias that can be consumed by the existing
category priority policy.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping, Optional

from .category_selector import CategoryAnalytics


SURVEY_PRIORITY_WEIGHTS = {
    "confident": 0.95,
    "unsure": 1.10,
    "practice": 1.20,
}


def should_apply_survey_bootstrap(
    *,
    survey: Optional[Mapping[str, object]],
    is_first_study_plan: bool,
    has_objective_learning_evidence: bool,
) -> bool:
    """Return whether survey self-assessment should bias initial priorities."""

    return (
        bool(survey)
        and is_first_study_plan
        and not has_objective_learning_evidence
    )


def apply_survey_bootstrap_bias(
    *,
    analytics: Mapping[str, CategoryAnalytics],
    survey: Optional[Mapping[str, object]],
) -> Mapping[str, CategoryAnalytics]:
    """Return analytics with moderate survey-derived priority weights.

    Unknown survey values are ignored here because API validation owns request
    validation. Categories without a matching survey answer remain unchanged.
    """

    if not survey:
        return analytics

    adjusted = dict(analytics)

    for category, answer in survey.items():
        if category not in adjusted:
            continue

        priority_weight = SURVEY_PRIORITY_WEIGHTS.get(str(answer))

        if priority_weight is None:
            continue

        adjusted[category] = replace(
            adjusted[category],
            priority_weight=priority_weight,
        )

    return adjusted
