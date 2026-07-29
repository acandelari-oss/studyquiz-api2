"""Deterministic category packing for Coverage Mode Study Plans.

Coverage Mode is not adaptive planning. It receives an already-determined
category order, keeps categories as complete planning units, and packs only
whole categories into the current Study Plan.
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .session_allocator import CategoryAllocation


DEFAULT_COVERAGE_TARGET_MODULES = 12
DEFAULT_COVERAGE_BUFFER_MODULES = 3


@dataclass(frozen=True)
class CoveragePackingResult:
    """Result of packing complete categories for one Coverage Study Plan."""

    allocations: Sequence[CategoryAllocation] = field(default_factory=tuple)
    accepted_categories: Sequence[str] = field(default_factory=tuple)
    skipped_categories: Sequence[str] = field(default_factory=tuple)
    remaining_categories: Sequence[str] = field(default_factory=tuple)
    category_required_modules: Mapping[str, int] = field(default_factory=dict)
    target_modules: int = DEFAULT_COVERAGE_TARGET_MODULES
    buffer_modules: int = DEFAULT_COVERAGE_BUFFER_MODULES
    maximum_modules: int = DEFAULT_COVERAGE_TARGET_MODULES + DEFAULT_COVERAGE_BUFFER_MODULES


class CoverageCategoryPacker:
    """Pack complete categories while respecting a soft Study Plan target."""

    def __init__(
        self,
        target_modules: int = DEFAULT_COVERAGE_TARGET_MODULES,
        buffer_modules: int = DEFAULT_COVERAGE_BUFFER_MODULES,
    ) -> None:
        self.target_modules = max(1, int(target_modules or DEFAULT_COVERAGE_TARGET_MODULES))
        self.buffer_modules = max(0, int(buffer_modules or 0))

    def pack(
        self,
        ranked_categories: Sequence[str],
        allocations_by_category: Mapping[str, Sequence[CategoryAllocation]],
        target_modules: int = 0,
    ) -> CoveragePackingResult:
        """Return allocations for complete categories only.

        The first category is always accepted, even when it exceeds the nominal
        maximum. This preserves the rule that a large category must not be
        truncated simply because of plan size.
        """

        effective_target = max(1, int(target_modules or self.target_modules))
        maximum_modules = effective_target + self.buffer_modules
        accepted_categories = []
        skipped_categories = []
        required_modules = {}
        current_module_estimate = 0

        for category in ranked_categories:
            category_allocations = tuple(allocations_by_category.get(category, ()))
            if not category_allocations:
                continue

            category_module_count = len(category_allocations)
            required_modules[category] = category_module_count

            if not accepted_categories:
                accepted_categories.append(category)
                current_module_estimate += category_module_count
                continue

            if current_module_estimate + category_module_count <= maximum_modules:
                accepted_categories.append(category)
                current_module_estimate += category_module_count
                continue

            skipped_categories.append(category)

        accepted_set = set(accepted_categories)
        skipped_set = set(skipped_categories)
        remaining_categories = tuple(
            category
            for category in ranked_categories
            if category not in accepted_set and category not in skipped_set
        )
        allocations = tuple(
            allocation
            for category in accepted_categories
            for allocation in allocations_by_category.get(category, ())
        )

        return CoveragePackingResult(
            allocations=allocations,
            accepted_categories=tuple(accepted_categories),
            skipped_categories=tuple(skipped_categories),
            remaining_categories=remaining_categories,
            category_required_modules=dict(required_modules),
            target_modules=effective_target,
            buffer_modules=self.buffer_modules,
            maximum_modules=maximum_modules,
        )
