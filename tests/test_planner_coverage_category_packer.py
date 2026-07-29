import unittest

from planner.coverage_category_packer import CoverageCategoryPacker
from planner.session_allocator import CategoryAllocation
from planner.planner_models import SelectedTopic


class CoverageCategoryPackerTests(unittest.TestCase):
    def _allocations(self, category, count):
        return tuple(
            CategoryAllocation(
                category=category,
                selected_topics=(
                    SelectedTopic(
                        id=f"{category}-{index}",
                        title=f"{category} Topic {index}",
                        order=index,
                    ),
                ),
                estimated_duration_minutes=1,
            )
            for index in range(1, count + 1)
        )

    def test_ranking_aware_best_fit_skips_category_that_exceeds_buffer(self):
        packer = CoverageCategoryPacker(target_modules=12, buffer_modules=3)

        result = packer.pack(
            ranked_categories=("A", "B", "C"),
            allocations_by_category={
                "A": self._allocations("A", 8),
                "B": self._allocations("B", 9),
                "C": self._allocations("C", 7),
            },
            target_modules=12,
        )

        self.assertEqual(result.accepted_categories, ("A", "C"))
        self.assertEqual(result.skipped_categories, ("B",))
        self.assertEqual(
            [allocation.category for allocation in result.allocations],
            ["A"] * 8 + ["C"] * 7,
        )

    def test_large_first_category_is_never_truncated_by_nominal_target(self):
        packer = CoverageCategoryPacker(target_modules=12, buffer_modules=3)

        result = packer.pack(
            ranked_categories=("Large", "Small"),
            allocations_by_category={
                "Large": self._allocations("Large", 18),
                "Small": self._allocations("Small", 1),
            },
            target_modules=12,
        )

        self.assertEqual(result.accepted_categories, ("Large",))
        self.assertEqual(result.skipped_categories, ("Small",))
        self.assertEqual(len(result.allocations), 18)

    def test_identical_inputs_are_deterministic(self):
        packer = CoverageCategoryPacker(target_modules=3, buffer_modules=1)
        kwargs = {
            "ranked_categories": ("A", "B", "C"),
            "allocations_by_category": {
                "A": self._allocations("A", 2),
                "B": self._allocations("B", 3),
                "C": self._allocations("C", 1),
            },
            "target_modules": 3,
        }

        first = packer.pack(**kwargs)
        second = packer.pack(**kwargs)

        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
