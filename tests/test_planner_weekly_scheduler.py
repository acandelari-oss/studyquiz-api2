import unittest

from planner.planner_models import SelectedTopic
from planner.session_allocator import CategoryAllocation
from planner.weekly_scheduler import WeeklyScheduler


class PlannerWeeklySchedulerTests(unittest.TestCase):
    def setUp(self):
        self.scheduler = WeeklyScheduler()

    def _allocation(self, category, duration, topic_id):
        return CategoryAllocation(
            category=category,
            selected_topics=(SelectedTopic(id=topic_id, title=topic_id),),
            estimated_duration_minutes=duration,
        )

    def test_no_allocations_returns_empty_sessions(self):
        sessions = self.scheduler.schedule_week(
            allocations=(),
            number_of_sessions=3,
            planning_budget_minutes=30,
        )

        self.assertEqual(len(sessions), 3)
        self.assertEqual([session.allocations for session in sessions], [(), (), ()])
        self.assertEqual([session.estimated_duration_minutes for session in sessions], [0.0, 0.0, 0.0])

    def test_one_allocation_is_scheduled_once(self):
        allocation = self._allocation("Genetics", 15, "topic-1")

        sessions = self.scheduler.schedule_week(
            allocations=(allocation,),
            number_of_sessions=2,
            planning_budget_minutes=30,
        )

        self.assertEqual(sessions[0].allocations, (allocation,))
        self.assertEqual(sessions[0].estimated_duration_minutes, 15.0)
        self.assertEqual(sessions[1].allocations, ())

    def test_allocations_preserve_order_across_sessions(self):
        allocations = (
            self._allocation("A", 20, "a1"),
            self._allocation("A", 20, "a2"),
            self._allocation("B", 20, "b1"),
            self._allocation("C", 10, "c1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=3,
            planning_budget_minutes=40,
        )

        scheduled = [
            allocation
            for session in sessions
            for allocation in session.allocations
        ]

        self.assertEqual(scheduled, list(allocations))
        self.assertEqual(
            [[allocation.category for allocation in session.allocations] for session in sessions],
            [["A", "A"], ["B", "C"], []],
        )

    def test_session_budgets_are_respected(self):
        allocations = (
            self._allocation("A", 18, "a1"),
            self._allocation("B", 20, "b1"),
            self._allocation("C", 18, "c1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=3,
            planning_budget_minutes=35,
        )

        self.assertTrue(
            all(session.estimated_duration_minutes <= 35 for session in sessions)
        )
        self.assertEqual(
            [session.estimated_duration_minutes for session in sessions],
            [18.0, 20.0, 18.0],
        )

    def test_lower_priority_allocation_is_not_used_to_fill_remaining_budget(self):
        allocations = (
            self._allocation("A", 18, "a1"),
            self._allocation("B", 20, "b1"),
            self._allocation("C", 10, "c1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=3,
            planning_budget_minutes=28,
        )

        self.assertEqual(
            [[allocation.category for allocation in session.allocations] for session in sessions],
            [["A"], ["B"], ["C"]],
        )

    def test_small_budget_uses_fewer_categories_for_depth(self):
        allocations = (
            self._allocation("A", 5, "a1"),
            self._allocation("B", 5, "b1"),
            self._allocation("C", 5, "c1"),
            self._allocation("D", 5, "d1"),
            self._allocation("E", 5, "e1"),
            self._allocation("F", 5, "f1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=3,
            planning_budget_minutes=30,
        )

        self.assertEqual(
            [[allocation.category for allocation in session.allocations] for session in sessions],
            [["A", "B"], ["C", "D"], ["E", "F"]],
        )

    def test_large_budget_may_add_more_categories_when_allocations_are_small(self):
        allocations = (
            self._allocation("A", 5, "a1"),
            self._allocation("B", 5, "b1"),
            self._allocation("C", 5, "c1"),
            self._allocation("D", 5, "d1"),
            self._allocation("E", 5, "e1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=2,
            planning_budget_minutes=60,
        )

        self.assertEqual(
            [[allocation.category for allocation in session.allocations] for session in sessions],
            [["A", "B", "C"], ["D", "E"]],
        )

    def test_more_sessions_than_allocations_produces_empty_sessions_at_end(self):
        allocations = (
            self._allocation("A", 10, "a1"),
            self._allocation("B", 10, "b1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=4,
            planning_budget_minutes=10,
        )

        self.assertEqual(
            [len(session.allocations) for session in sessions],
            [1, 1, 0, 0],
        )

    def test_fewer_sessions_than_allocations_schedules_prefix_without_reordering(self):
        allocations = (
            self._allocation("A", 10, "a1"),
            self._allocation("B", 10, "b1"),
            self._allocation("C", 10, "c1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=2,
            planning_budget_minutes=10,
        )

        scheduled_categories = [
            allocation.category
            for session in sessions
            for allocation in session.allocations
        ]

        self.assertEqual(scheduled_categories, ["A", "B"])

    def test_oversized_allocation_fails_gracefully_without_budget_violation_or_skip(self):
        allocations = (
            self._allocation("A", 45, "a1"),
            self._allocation("B", 10, "b1"),
        )

        sessions = self.scheduler.schedule_week(
            allocations=allocations,
            number_of_sessions=2,
            planning_budget_minutes=30,
        )

        self.assertEqual([session.allocations for session in sessions], [(), ()])
        self.assertTrue(
            all(session.estimated_duration_minutes <= 30 for session in sessions)
        )

    def test_deterministic_output(self):
        allocations = (
            self._allocation("A", 10, "a1"),
            self._allocation("B", 10, "b1"),
            self._allocation("C", 10, "c1"),
        )

        first = self.scheduler.schedule_week(allocations, 2, 20)
        second = self.scheduler.schedule_week(allocations, 2, 20)

        self.assertEqual(first, second)

    def test_zero_or_negative_session_count_returns_no_sessions(self):
        allocation = self._allocation("A", 10, "a1")

        self.assertEqual(self.scheduler.schedule_week((allocation,), 0, 30), ())
        self.assertEqual(self.scheduler.schedule_week((allocation,), -1, 30), ())


if __name__ == "__main__":
    unittest.main()
