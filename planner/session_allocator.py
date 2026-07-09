"""Deterministic session allocation primitives for the Study Planner."""

from dataclasses import dataclass, field
from math import ceil
from typing import Any, Sequence

from .category_selector import CategoryPriority
from .planner_models import PlannerContext, SelectedTopic


@dataclass(frozen=True)
class TopicSliceAllocation:
    """A consecutive topic slice that fits inside an available time budget."""

    category: str
    selected_topics: Sequence[SelectedTopic] = field(default_factory=tuple)
    estimated_duration_minutes: float = 0.0
    remaining_topics: Sequence[SelectedTopic] = field(default_factory=tuple)


@dataclass(frozen=True)
class CategoryAllocation:
    """One consecutive allocation segment for a category."""

    category: str
    selected_topics: Sequence[SelectedTopic] = field(default_factory=tuple)
    estimated_duration_minutes: float = 0.0


class SessionAllocator:
    """Distribute selected categories and topics across study sessions."""

    SECONDS_PER_MINUTE = 60

    def allocate_topic_slice(
        self,
        category: str,
        ordered_topics: Sequence[SelectedTopic],
        available_budget_minutes: float,
        question_pace_seconds: int,
    ) -> TopicSliceAllocation:
        """Select the consecutive topic slice that fits inside the budget.

        V1 assumes each selected topic requires one question or flashcard. A
        topic is selected only if adding it keeps the planned duration inside
        the available budget.
        """

        topics = self._normalize_topics(ordered_topics)

        if not topics or available_budget_minutes <= 0 or question_pace_seconds <= 0:
            return TopicSliceAllocation(category=category, remaining_topics=topics)

        available_budget_seconds = available_budget_minutes * self.SECONDS_PER_MINUTE
        max_topic_count = int(available_budget_seconds // question_pace_seconds)

        if max_topic_count <= 0:
            return TopicSliceAllocation(category=category, remaining_topics=topics)

        selected_topics = topics[:max_topic_count]
        remaining_topics = topics[max_topic_count:]
        estimated_duration_minutes = self._calculate_estimated_duration_minutes(
            selected_topic_count=len(selected_topics),
            question_pace_seconds=question_pace_seconds,
        )

        return TopicSliceAllocation(
            category=category,
            selected_topics=selected_topics,
            estimated_duration_minutes=estimated_duration_minutes,
            remaining_topics=remaining_topics,
        )

    def allocate_category_segments(
        self,
        category: str,
        ordered_topics: Sequence[SelectedTopic],
        available_budget_minutes: float,
        question_pace_seconds: int,
    ) -> Sequence[CategoryAllocation]:
        """Allocate an entire category into consecutive topic segments.

        This repeatedly applies the topic-slice primitive until all topics have
        been allocated, or until the provided budget/pace cannot fit even one
        topic. It does not schedule a week or create DailyPlan/Activity objects.
        """

        allocations = []
        current_topics = self._normalize_topics(ordered_topics)

        while current_topics:
            slice_allocation = self.allocate_topic_slice(
                category=category,
                ordered_topics=current_topics,
                available_budget_minutes=available_budget_minutes,
                question_pace_seconds=question_pace_seconds,
            )

            if not slice_allocation.selected_topics:
                break

            allocations.append(
                CategoryAllocation(
                    category=slice_allocation.category,
                    selected_topics=slice_allocation.selected_topics,
                    estimated_duration_minutes=slice_allocation.estimated_duration_minutes,
                )
            )
            current_topics = slice_allocation.remaining_topics

        return tuple(allocations)

    def allocate_sessions(
        self,
        context: PlannerContext,
        category_priorities: Sequence[CategoryPriority],
    ) -> Sequence[Any]:
        """Return session allocations for a future weekly plan.

        TODO: Implement session allocation in a future step.
        """

        del context, category_priorities
        raise NotImplementedError("Session allocation is not implemented yet.")

    def _normalize_topics(
        self,
        ordered_topics: Sequence[SelectedTopic],
    ) -> Sequence[SelectedTopic]:
        """Return deterministic, non-repeated topics in learning order.

        If all topics have an explicit order value, that order is used. If any
        topic lacks an order value, the caller-provided order is preserved.
        Duplicate topic IDs are ignored after their first occurrence.
        """

        unique_topics = []
        seen_topic_ids = set()

        for topic in ordered_topics:
            if topic.id in seen_topic_ids:
                continue
            seen_topic_ids.add(topic.id)
            unique_topics.append(topic)

        if unique_topics and all(topic.order is not None for topic in unique_topics):
            return tuple(sorted(unique_topics, key=lambda topic: topic.order))

        return tuple(unique_topics)

    def _calculate_estimated_duration_minutes(
        self,
        selected_topic_count: int,
        question_pace_seconds: int,
    ) -> float:
        """Convert selected topic count and question pace into minutes."""

        duration_seconds = selected_topic_count * question_pace_seconds
        return ceil((duration_seconds / self.SECONDS_PER_MINUTE) * 100) / 100
