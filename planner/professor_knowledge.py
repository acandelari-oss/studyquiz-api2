"""Professor planning knowledge snapshots.

ProfessorKnowledge is a backend-internal, deterministic record of what the
Professor is allowed to know before explaining a Study Plan. V1 is planning-only:
it records facts and reasoning produced elsewhere, but never creates decisions,
text, prompts, or natural language.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from .planner_models import PlannerContext, SelectedTopic, Week
from .planner_state import ActivityType
from .professor_daily_strategy import ProfessorDailyActivityType
from .professor_module_composer import ProfessorModule
from .professor_strategy import ProfessorWeeklyStrategy


@dataclass(frozen=True)
class ProfessorKnowledgeTopic:
    """Topic known to the Professor as part of planning knowledge."""

    id: str
    title: str
    order: Optional[int] = None


@dataclass(frozen=True)
class ProfessorKnowledgeActivityMix:
    """Aggregate activity mix for a Study Plan."""

    quiz_count: int = 0
    flashcard_count: int = 0
    mixed_count: int = 0


@dataclass(frozen=True)
class ProfessorKnowledgeActivitySize:
    """Planned size of one executable learning activity."""

    activity_id: str
    module_index: int
    activity_type: str
    category: Optional[str] = None
    num_questions: Optional[int] = None
    num_cards: Optional[int] = None
    estimated_duration_minutes: Optional[float] = None


@dataclass(frozen=True)
class ProfessorKnowledgePlanningConstraints:
    """Student-selected constraints used to generate the Study Plan."""

    module_duration_minutes: float = 0.0
    quiz_pace_seconds: Optional[int] = None
    question_style: Optional[str] = None
    max_visible_modules: Optional[int] = None


@dataclass(frozen=True)
class ProfessorKnowledgeCategoryStrategy:
    """Professor weekly strategy facts for one category."""

    category: str
    strategy_code: Optional[str] = None
    depth_code: Optional[str] = None
    reasoning_code: Optional[str] = None
    priority_score: float = 0.0


@dataclass(frozen=True)
class ProfessorKnowledgeModuleActivityStrategy:
    """Professor daily strategy facts for one scheduled category/activity."""

    category: str
    activity_type: str
    depth_code: Optional[str] = None
    estimated_questions: int = 0
    estimated_flashcards: int = 0
    reasoning_codes: Sequence[str] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProfessorKnowledgeModuleStrategy:
    """Professor daily/module strategy facts for one Study Plan module."""

    module_index: int
    daily_goal_code: Optional[str] = None
    summary_codes: Sequence[str] = field(default_factory=tuple)
    activities: Sequence[ProfessorKnowledgeModuleActivityStrategy] = field(default_factory=tuple)


@dataclass(frozen=True)
class ProfessorTeachingContext:
    """Deterministic educational context for one Study Plan module.

    TeachingContext contains no generated prose and makes no new decisions. It
    translates existing planning and Professor strategy facts into stable
    educational context that Professor Voice can explain.
    """

    module_index: int
    conceptual_summary: str
    prerequisite_level: str
    learning_progression: str
    expected_mastery: str
    activity_rationale: str


@dataclass(frozen=True)
class ProfessorKnowledge:
    """Planning truth available to future Professor Voice.

    V1 is deliberately backend-internal and planning-only. It contains no
    Professor Identity, no generated text, no GPT prompts, no homework/debrief
    execution data, and no conversational state.
    """

    study_language: Optional[str] = None
    project_id: Optional[str] = None
    project_name: Optional[str] = None
    taxonomy_language: Optional[str] = None
    module_count: int = 0
    visible_module_count: int = 0
    additional_modules_remain: bool = False
    selected_categories: Sequence[str] = field(default_factory=tuple)
    selected_topics_by_category: Mapping[str, Sequence[ProfessorKnowledgeTopic]] = field(default_factory=dict)
    activity_mix: ProfessorKnowledgeActivityMix = field(default_factory=ProfessorKnowledgeActivityMix)
    activity_sizes: Sequence[ProfessorKnowledgeActivitySize] = field(default_factory=tuple)
    planning_constraints: ProfessorKnowledgePlanningConstraints = field(default_factory=ProfessorKnowledgePlanningConstraints)
    weekly_goal_code: Optional[str] = None
    category_strategies: Sequence[ProfessorKnowledgeCategoryStrategy] = field(default_factory=tuple)
    module_strategies: Sequence[ProfessorKnowledgeModuleStrategy] = field(default_factory=tuple)
    teaching_contexts: Sequence[ProfessorTeachingContext] = field(default_factory=tuple)
    remaining_categories: Sequence[str] = field(default_factory=tuple)
    remaining_topics_by_category: Mapping[str, Sequence[ProfessorKnowledgeTopic]] = field(default_factory=dict)
    warnings: Mapping[str, bool] = field(default_factory=dict)


class ProfessorKnowledgeBuilder:
    """Build ProfessorKnowledge from existing deterministic Planner objects."""

    def build(
        self,
        *,
        context: PlannerContext,
        week: Week,
        weekly_strategy: Optional[ProfessorWeeklyStrategy] = None,
        modules: Sequence[ProfessorModule] = (),
        max_visible_modules: Optional[int] = None,
        additional_modules_remain: Optional[bool] = None,
    ) -> ProfessorKnowledge:
        """Return a planning-only ProfessorKnowledge snapshot."""

        selected_topics_by_category = self._selected_topics_by_category(week)
        remaining_topics_by_category = self._remaining_topics_by_category(
            context=context,
            selected_topics_by_category=selected_topics_by_category,
        )
        computed_additional_modules_remain = bool(remaining_topics_by_category)
        additional_modules_remain = (
            computed_additional_modules_remain
            if additional_modules_remain is None
            else bool(additional_modules_remain)
        )
        module_count = len(week.daily_plans)
        visible_module_count = (
            min(module_count, max_visible_modules)
            if max_visible_modules is not None
            else module_count
        )
        activity_mix = self._activity_mix(week=week, modules=modules)
        activity_sizes = self._activity_sizes(week)
        module_strategies = self._module_strategies(modules)

        return ProfessorKnowledge(
            study_language=week.study_language or context.study_language,
            project_id=self._project_value(context.project, "id"),
            project_name=self._project_value(context.project, "name"),
            taxonomy_language=self._project_value(context.project, "taxonomy_language"),
            module_count=module_count,
            visible_module_count=visible_module_count,
            additional_modules_remain=additional_modules_remain,
            selected_categories=tuple(selected_topics_by_category.keys()),
            selected_topics_by_category=selected_topics_by_category,
            activity_mix=activity_mix,
            activity_sizes=activity_sizes,
            planning_constraints=ProfessorKnowledgePlanningConstraints(
                module_duration_minutes=context.planning_budget_minutes,
                quiz_pace_seconds=context.preferences.question_pace_seconds,
                question_style=context.preferences.question_style,
                max_visible_modules=max_visible_modules,
            ),
            weekly_goal_code=self._enum_value(getattr(weekly_strategy, "weekly_goal_code", None)),
            category_strategies=self._category_strategies(weekly_strategy),
            module_strategies=module_strategies,
            teaching_contexts=self._teaching_contexts(
                module_count=module_count,
                activity_sizes=activity_sizes,
                module_strategies=module_strategies,
            ),
            remaining_categories=tuple(remaining_topics_by_category.keys()),
            remaining_topics_by_category=remaining_topics_by_category,
            warnings={
                "missing_study_language": not bool(week.study_language or context.study_language),
                "additional_modules_remain": additional_modules_remain,
                "remaining_topics_exist": bool(remaining_topics_by_category),
                "missing_weekly_strategy": weekly_strategy is None,
                "missing_module_strategy": not bool(modules),
            },
        )

    def _selected_topics_by_category(
        self,
        week: Week,
    ) -> Mapping[str, Sequence[ProfessorKnowledgeTopic]]:
        grouped = {}
        seen_by_category = {}

        for daily_plan in week.daily_plans:
            for allocation in daily_plan.planned_allocations:
                category = getattr(allocation, "category", None)
                if not category:
                    continue

                grouped.setdefault(category, [])
                seen_by_category.setdefault(category, set())

                for topic in getattr(allocation, "selected_topics", ()) or ():
                    topic_id = str(getattr(topic, "id", "") or "")
                    topic_key = topic_id or str(getattr(topic, "title", "") or "")

                    if not topic_key or topic_key in seen_by_category[category]:
                        continue

                    seen_by_category[category].add(topic_key)
                    grouped[category].append(self._knowledge_topic(topic))

        return {
            category: tuple(topics)
            for category, topics in grouped.items()
        }

    def _remaining_topics_by_category(
        self,
        *,
        context: PlannerContext,
        selected_topics_by_category: Mapping[str, Sequence[ProfessorKnowledgeTopic]],
    ) -> Mapping[str, Sequence[ProfessorKnowledgeTopic]]:
        remaining = {}

        for category, topics in context.topics_by_category.items():
            selected_ids = {
                topic.id
                for topic in selected_topics_by_category.get(category, ())
                if topic.id
            }
            selected_titles = {
                topic.title
                for topic in selected_topics_by_category.get(category, ())
                if topic.title
            }
            remaining_topics = []

            for topic in topics:
                topic_id = str(getattr(topic, "id", "") or "")
                topic_title = str(getattr(topic, "title", "") or "")

                if topic_id and topic_id in selected_ids:
                    continue

                if not topic_id and topic_title in selected_titles:
                    continue

                remaining_topics.append(self._knowledge_topic(topic))

            if remaining_topics:
                remaining[category] = tuple(remaining_topics)

        return remaining

    def _activity_mix(
        self,
        *,
        week: Week,
        modules: Sequence[ProfessorModule],
    ) -> ProfessorKnowledgeActivityMix:
        quiz_count = 0
        flashcard_count = 0
        mixed_count = 0

        if modules:
            for module in modules:
                for strategy in module.daily_strategy.activities:
                    activity_type = strategy.activity_type

                    if activity_type == ProfessorDailyActivityType.QUIZ_PLUS_FLASHCARDS:
                        mixed_count += 1
                    elif activity_type == ProfessorDailyActivityType.QUIZ:
                        quiz_count += 1
                    elif activity_type == ProfessorDailyActivityType.FLASHCARDS:
                        flashcard_count += 1
        else:
            for daily_plan in week.daily_plans:
                for activity in daily_plan.activities:
                    if activity.type == ActivityType.QUIZ:
                        quiz_count += 1
                    elif activity.type == ActivityType.FLASHCARDS:
                        flashcard_count += 1

        return ProfessorKnowledgeActivityMix(
            quiz_count=quiz_count,
            flashcard_count=flashcard_count,
            mixed_count=mixed_count,
        )

    def _activity_sizes(self, week: Week) -> Sequence[ProfessorKnowledgeActivitySize]:
        sizes = []

        for module_index, daily_plan in enumerate(week.daily_plans, start=1):
            for activity in daily_plan.activities:
                configuration = activity.configuration
                sizes.append(
                    ProfessorKnowledgeActivitySize(
                        activity_id=activity.id,
                        module_index=module_index,
                        activity_type=self._enum_value(activity.type) or str(activity.type),
                        category=configuration.category,
                        num_questions=configuration.num_questions,
                        num_cards=configuration.num_cards,
                        estimated_duration_minutes=configuration.estimated_duration_minutes,
                    )
                )

        return tuple(sizes)

    def _category_strategies(
        self,
        weekly_strategy: Optional[ProfessorWeeklyStrategy],
    ) -> Sequence[ProfessorKnowledgeCategoryStrategy]:
        if not weekly_strategy:
            return ()

        return tuple(
            ProfessorKnowledgeCategoryStrategy(
                category=strategy.category,
                strategy_code=self._enum_value(strategy.strategy),
                depth_code=self._enum_value(strategy.depth),
                reasoning_code=self._enum_value(strategy.reasoning_code),
                priority_score=strategy.priority_score,
            )
            for strategy in weekly_strategy.category_strategies
        )

    def _module_strategies(
        self,
        modules: Sequence[ProfessorModule],
    ) -> Sequence[ProfessorKnowledgeModuleStrategy]:
        return tuple(
            ProfessorKnowledgeModuleStrategy(
                module_index=module.module_index,
                daily_goal_code=self._enum_value(module.daily_strategy.daily_goal_code),
                summary_codes=tuple(
                    self._enum_value(code) or str(code)
                    for code in module.daily_strategy.summary_codes
                ),
                activities=tuple(
                    ProfessorKnowledgeModuleActivityStrategy(
                        category=strategy.category,
                        activity_type=self._enum_value(strategy.activity_type) or str(strategy.activity_type),
                        depth_code=self._enum_value(strategy.depth),
                        estimated_questions=strategy.estimated_questions,
                        estimated_flashcards=strategy.estimated_flashcards,
                        reasoning_codes=tuple(
                            self._enum_value(code) or str(code)
                            for code in strategy.reasoning_codes
                        ),
                    )
                    for strategy in module.daily_strategy.activities
                ),
            )
            for module in modules
        )

    def _teaching_contexts(
        self,
        *,
        module_count: int,
        activity_sizes: Sequence[ProfessorKnowledgeActivitySize],
        module_strategies: Sequence[ProfessorKnowledgeModuleStrategy],
    ) -> Sequence[ProfessorTeachingContext]:
        return tuple(
            ProfessorTeachingContext(
                module_index=module_index,
                conceptual_summary=self._conceptual_summary(module_index),
                prerequisite_level=self._prerequisite_level(
                    module_index=module_index,
                    module_count=module_count,
                    module_strategy=self._module_strategy(module_strategies, module_index),
                ),
                learning_progression=self._learning_progression(
                    module_index=module_index,
                    module_count=module_count,
                ),
                expected_mastery=self._expected_mastery(
                    module_index=module_index,
                    module_count=module_count,
                ),
                activity_rationale=self._activity_rationale(
                    activity_kinds=self._module_activity_kinds(
                        module_index=module_index,
                        activity_sizes=activity_sizes,
                        module_strategy=self._module_strategy(module_strategies, module_index),
                    ),
                ),
            )
            for module_index in range(1, module_count + 1)
        )

    def _module_strategy(
        self,
        module_strategies: Sequence[ProfessorKnowledgeModuleStrategy],
        module_index: int,
    ) -> Optional[ProfessorKnowledgeModuleStrategy]:
        return next(
            (
                strategy
                for strategy in module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )

    def _module_activity_kinds(
        self,
        *,
        module_index: int,
        activity_sizes: Sequence[ProfessorKnowledgeActivitySize],
        module_strategy: Optional[ProfessorKnowledgeModuleStrategy],
    ) -> frozenset[str]:
        kinds = set()

        if module_strategy:
            for activity in module_strategy.activities:
                activity_type = str(activity.activity_type or "").upper()
                if "QUIZ" in activity_type:
                    kinds.add("quiz")
                if "FLASHCARD" in activity_type:
                    kinds.add("flashcards")

        for activity in activity_sizes:
            if activity.module_index != module_index:
                continue

            activity_type = str(activity.activity_type or "").upper()
            if "QUIZ" in activity_type:
                kinds.add("quiz")
            if "FLASHCARD" in activity_type:
                kinds.add("flashcards")

        return frozenset(kinds)

    def _conceptual_summary(self, module_index: int) -> str:
        if module_index == 1:
            return (
                "This module establishes the conceptual frame needed to orient the first part of the study path. "
                "Its purpose is to clarify the central distinctions before more demanding connections are introduced."
            )

        return (
            "This module develops the conceptual relationships introduced earlier and places them in a more precise sequence. "
            "Its purpose is to make the student reason across ideas rather than treat them as isolated facts."
        )

    def _prerequisite_level(
        self,
        *,
        module_index: int,
        module_count: int,
        module_strategy: Optional[ProfessorKnowledgeModuleStrategy],
    ) -> str:
        activity_types = {
            str(activity.activity_type or "").upper()
            for activity in (module_strategy.activities if module_strategy else ())
        }

        if activity_types and all("FLASHCARD" in activity_type for activity_type in activity_types):
            return "consolidation"

        if module_count <= 1 or module_index == 1:
            return "foundations"

        if module_index >= module_count:
            return "advanced"

        return "intermediate"

    def _learning_progression(
        self,
        *,
        module_index: int,
        module_count: int,
    ) -> str:
        if module_count <= 1:
            return "This module creates the first reliable basis for the next teaching decision."

        if module_index == 1:
            return "This establishes the initial foundation required before the following modules can introduce more demanding relationships."

        if module_index >= module_count:
            return "This completes the current study path by connecting the earlier foundations with the later material."

        return "This expands the foundations from the previous modules and prepares the student for more complex reasoning."

    def _expected_mastery(
        self,
        *,
        module_index: int,
        module_count: int,
    ) -> str:
        if module_count > 1 and module_index >= module_count:
            return (
                "By the end, the student should connect the central ideas from this part of the path and use them in a coherent line of reasoning."
            )

        return (
            "By the end, the student should distinguish the central concepts, explain their relationships, and recognise which parts are stable enough to support later work."
        )

    def _activity_rationale(self, *, activity_kinds: frozenset[str]) -> str:
        has_quiz = "quiz" in activity_kinds
        has_flashcards = "flashcards" in activity_kinds

        if has_quiz and has_flashcards:
            return "The objective combines retrieval practice with reinforcement."

        if has_flashcards:
            return "The objective is long-term consolidation of terminology and relationships."

        if has_quiz:
            return "The objective is to verify conceptual stability before progressing."

        return "The objective is to clarify the conceptual structure before continuing."

    def _knowledge_topic(self, topic: SelectedTopic) -> ProfessorKnowledgeTopic:
        return ProfessorKnowledgeTopic(
            id=str(getattr(topic, "id", "") or ""),
            title=str(getattr(topic, "title", "") or ""),
            order=getattr(topic, "order", None),
        )

    def _project_value(self, project: Any, key: str) -> Optional[str]:
        if not project:
            return None

        if isinstance(project, Mapping):
            value = project.get(key)
        else:
            value = getattr(project, key, None)

        return str(value) if value is not None else None

    def _enum_value(self, value: Any) -> Optional[str]:
        if value is None:
            return None

        return str(getattr(value, "value", value))
