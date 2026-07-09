import unittest

from planner.planner_models import SelectedTopic
from planner.session_allocator import SessionAllocator


class PlannerSessionAllocatorTests(unittest.TestCase):
    def setUp(self):
        self.allocator = SessionAllocator()

    def _topics(self, count):
        return tuple(
            SelectedTopic(id=f"topic-{index}", title=f"Topic {index}", order=index)
            for index in range(1, count + 1)
        )

    def test_full_category_fits(self):
        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=self._topics(3),
            available_budget_minutes=10,
            question_pace_seconds=60,
        )

        self.assertEqual([topic.id for topic in allocation.selected_topics], [
            "topic-1",
            "topic-2",
            "topic-3",
        ])
        self.assertEqual(allocation.remaining_topics, ())
        self.assertEqual(allocation.estimated_duration_minutes, 3.0)

    def test_partial_category_fits(self):
        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=self._topics(5),
            available_budget_minutes=3,
            question_pace_seconds=60,
        )

        self.assertEqual([topic.id for topic in allocation.selected_topics], [
            "topic-1",
            "topic-2",
            "topic-3",
        ])
        self.assertEqual([topic.id for topic in allocation.remaining_topics], [
            "topic-4",
            "topic-5",
        ])
        self.assertEqual(allocation.estimated_duration_minutes, 3.0)

    def test_no_topic_fits_when_budget_too_small(self):
        topics = self._topics(2)
        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=topics,
            available_budget_minutes=1,
            question_pace_seconds=90,
        )

        self.assertEqual(allocation.selected_topics, ())
        self.assertEqual(allocation.remaining_topics, topics)
        self.assertEqual(allocation.estimated_duration_minutes, 0.0)

    def test_empty_topic_list(self):
        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=(),
            available_budget_minutes=10,
            question_pace_seconds=60,
        )

        self.assertEqual(allocation.selected_topics, ())
        self.assertEqual(allocation.remaining_topics, ())

    def test_invalid_budget_or_pace_selects_no_topics(self):
        topics = self._topics(2)

        zero_budget = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=topics,
            available_budget_minutes=0,
            question_pace_seconds=60,
        )
        invalid_pace = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=topics,
            available_budget_minutes=10,
            question_pace_seconds=0,
        )

        self.assertEqual(zero_budget.selected_topics, ())
        self.assertEqual(zero_budget.remaining_topics, topics)
        self.assertEqual(invalid_pace.selected_topics, ())
        self.assertEqual(invalid_pace.remaining_topics, topics)

    def test_explicit_topic_order_is_used_when_available(self):
        unordered_topics = (
            SelectedTopic(id="topic-3", title="Topic 3", order=3),
            SelectedTopic(id="topic-1", title="Topic 1", order=1),
            SelectedTopic(id="topic-2", title="Topic 2", order=2),
        )

        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=unordered_topics,
            available_budget_minutes=2,
            question_pace_seconds=60,
        )

        self.assertEqual([topic.id for topic in allocation.selected_topics], [
            "topic-1",
            "topic-2",
        ])
        self.assertEqual([topic.id for topic in allocation.remaining_topics], [
            "topic-3",
        ])

    def test_input_order_is_preserved_when_order_is_missing(self):
        topics = (
            SelectedTopic(id="topic-3", title="Topic 3"),
            SelectedTopic(id="topic-1", title="Topic 1"),
            SelectedTopic(id="topic-2", title="Topic 2"),
        )

        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=topics,
            available_budget_minutes=2,
            question_pace_seconds=60,
        )

        self.assertEqual([topic.id for topic in allocation.selected_topics], [
            "topic-3",
            "topic-1",
        ])
        self.assertEqual([topic.id for topic in allocation.remaining_topics], [
            "topic-2",
        ])

    def test_duplicate_topics_are_not_repeated(self):
        topics = (
            SelectedTopic(id="topic-1", title="Topic 1", order=1),
            SelectedTopic(id="topic-1", title="Topic 1 duplicate", order=1),
            SelectedTopic(id="topic-2", title="Topic 2", order=2),
        )

        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=topics,
            available_budget_minutes=10,
            question_pace_seconds=60,
        )

        self.assertEqual([topic.id for topic in allocation.selected_topics], [
            "topic-1",
            "topic-2",
        ])

    def test_estimated_duration_can_be_fractional_minutes(self):
        allocation = self.allocator.allocate_topic_slice(
            category="Genetics",
            ordered_topics=self._topics(3),
            available_budget_minutes=5,
            question_pace_seconds=90,
        )

        self.assertEqual(allocation.estimated_duration_minutes, 4.5)

    def test_small_category_produces_one_segment(self):
        allocations = self.allocator.allocate_category_segments(
            category="Genetics",
            ordered_topics=self._topics(3),
            available_budget_minutes=10,
            question_pace_seconds=60,
        )

        self.assertEqual(len(allocations), 1)
        self.assertEqual([topic.id for topic in allocations[0].selected_topics], [
            "topic-1",
            "topic-2",
            "topic-3",
        ])
        self.assertEqual(allocations[0].estimated_duration_minutes, 3.0)

    def test_large_category_produces_multiple_segments(self):
        allocations = self.allocator.allocate_category_segments(
            category="Genetics",
            ordered_topics=self._topics(7),
            available_budget_minutes=3,
            question_pace_seconds=60,
        )

        self.assertEqual(len(allocations), 3)
        self.assertEqual(
            [[topic.id for topic in allocation.selected_topics] for allocation in allocations],
            [
                ["topic-1", "topic-2", "topic-3"],
                ["topic-4", "topic-5", "topic-6"],
                ["topic-7"],
            ],
        )
        self.assertEqual(
            [allocation.estimated_duration_minutes for allocation in allocations],
            [3.0, 3.0, 1.0],
        )

    def test_category_segments_preserve_order_without_missing_or_duplicate_topics(self):
        topics = self._topics(8)
        allocations = self.allocator.allocate_category_segments(
            category="Genetics",
            ordered_topics=topics,
            available_budget_minutes=2,
            question_pace_seconds=60,
        )

        allocated_topic_ids = [
            topic.id
            for allocation in allocations
            for topic in allocation.selected_topics
        ]

        self.assertEqual(allocated_topic_ids, [topic.id for topic in topics])
        self.assertEqual(len(allocated_topic_ids), len(set(allocated_topic_ids)))

    def test_category_segments_use_explicit_order(self):
        unordered_topics = (
            SelectedTopic(id="topic-3", title="Topic 3", order=3),
            SelectedTopic(id="topic-1", title="Topic 1", order=1),
            SelectedTopic(id="topic-2", title="Topic 2", order=2),
            SelectedTopic(id="topic-4", title="Topic 4", order=4),
        )

        allocations = self.allocator.allocate_category_segments(
            category="Genetics",
            ordered_topics=unordered_topics,
            available_budget_minutes=2,
            question_pace_seconds=60,
        )

        self.assertEqual(
            [[topic.id for topic in allocation.selected_topics] for allocation in allocations],
            [
                ["topic-1", "topic-2"],
                ["topic-3", "topic-4"],
            ],
        )

    def test_category_segments_return_empty_when_no_topic_can_fit(self):
        allocations = self.allocator.allocate_category_segments(
            category="Genetics",
            ordered_topics=self._topics(3),
            available_budget_minutes=1,
            question_pace_seconds=90,
        )

        self.assertEqual(allocations, ())


if __name__ == "__main__":
    unittest.main()
