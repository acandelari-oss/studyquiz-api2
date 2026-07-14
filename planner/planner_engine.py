"""Public orchestration interface for future Study Planner week generation."""

from datetime import date, timedelta
from typing import Mapping, Optional, Sequence, cast

from .activity_planner import ActivityPlanner
from .category_selector import CategoryAnalytics, CategorySelector
from .planner_models import DailyPlan, PlannerContext, Week
from .planner_state import WeekStatus
from .professor_bridge import ProfessorBridge
from .professor_daily_strategy import ProfessorDailyStrategyBuilder
from .professor_knowledge import ProfessorKnowledge, ProfessorKnowledgeBuilder
from .professor_module_composer import ProfessorModule, ProfessorModuleComposer
from .professor_strategy import (
    ProfessorCategoryStrategy,
    ProfessorCategoryStrategyCode,
    ProfessorDepthCode,
    ProfessorReasoningCode,
    ProfessorWeeklyGoalCode,
    ProfessorWeeklyStrategy,
    ProfessorWeeklyStrategyBuilder,
)
from .professor_voice import ProfessorVoiceService
from .session_allocator import CategoryAllocation, SessionAllocator
from .weekly_scheduler import WeeklyScheduler


class PlannerEngine:
    """Coordinate Planner components without owning their internal algorithms."""

    def __init__(
        self,
        category_selector: Optional[CategorySelector] = None,
        session_allocator: Optional[SessionAllocator] = None,
        weekly_scheduler: Optional[WeeklyScheduler] = None,
        activity_planner: Optional[ActivityPlanner] = None,
        professor_weekly_strategy_builder: Optional[ProfessorWeeklyStrategyBuilder] = None,
        professor_daily_strategy_builder: Optional[ProfessorDailyStrategyBuilder] = None,
        professor_module_composer: Optional[ProfessorModuleComposer] = None,
        professor_knowledge_builder: Optional[ProfessorKnowledgeBuilder] = None,
        professor_voice_service: Optional[ProfessorVoiceService] = None,
        professor_bridge: Optional[ProfessorBridge] = None,
    ) -> None:
        """Create an engine from pure Planner collaborators."""

        self.category_selector = category_selector or CategorySelector()
        self.session_allocator = session_allocator or SessionAllocator()
        self.weekly_scheduler = weekly_scheduler or WeeklyScheduler()
        self.activity_planner = activity_planner or ActivityPlanner()
        self.professor_weekly_strategy_builder = (
            professor_weekly_strategy_builder
            or ProfessorWeeklyStrategyBuilder()
        )
        self.professor_daily_strategy_builder = (
            professor_daily_strategy_builder
            or ProfessorDailyStrategyBuilder()
        )
        self.professor_module_composer = (
            professor_module_composer
            or ProfessorModuleComposer(self.professor_daily_strategy_builder)
        )
        self.professor_knowledge_builder = (
            professor_knowledge_builder
            or ProfessorKnowledgeBuilder()
        )
        self.professor_voice_service = professor_voice_service or ProfessorVoiceService()
        self.professor_bridge = professor_bridge or ProfessorBridge()
        self.last_professor_knowledge: Optional[ProfessorKnowledge] = None

    def generate_week(self, context: PlannerContext) -> Week:
        """Generate a weekly study plan from a PlannerContext.

        The algorithmic steps are intentionally delegated to dedicated modules:
        category selection, category segmentation, and weekly scheduling.
        """

        category_priorities = self.category_selector.select_categories(
            project_categories=context.categories,
            category_analytics=cast(Mapping[str, CategoryAnalytics], context.analytics),
            planner_preferences={},
        )

        category_allocations = self._allocate_categories(
            context=context,
            ordered_categories=tuple(priority.category for priority in category_priorities),
        )

        weekly_strategy = self.professor_weekly_strategy_builder.build_strategy(context)
        modules = self.professor_module_composer.compose_modules(
            context=context,
            weekly_strategy=weekly_strategy,
            allocations=category_allocations,
            max_visible_modules=context.number_of_sessions or None,
        )

        week = self._build_week(context=context, modules=modules)
        self.last_professor_knowledge = self.professor_knowledge_builder.build(
            context=context,
            week=week,
            weekly_strategy=weekly_strategy,
            modules=modules,
            max_visible_modules=context.number_of_sessions or None,
        )
        return self._add_professor_voice(
            week=week,
            knowledge=self.last_professor_knowledge,
        )

    def generate_assessment_week(self, context: PlannerContext) -> Week:
        """Generate a quiz-only assessment plan in Topic View order.

        Assessment is evidence collection. It deliberately bypasses category
        priority scoring and Professor strategy decisions while reusing the
        existing allocator, activity planner, domain shape, and persistence.
        """

        category_allocations = self._allocate_categories(
            context=context,
            ordered_categories=context.categories,
        )
        modules = self.professor_module_composer.compose_modules(
            context=context,
            weekly_strategy=self._assessment_weekly_strategy(context),
            allocations=category_allocations,
            max_visible_modules=None,
        )
        daily_plans = tuple(
            self._build_daily_plan(
                week_id=context.week_id or self._assessment_week_id(context),
                week_start_date=context.week_start_date or date.today(),
                module=module,
            )
            for module in modules
        )
        daily_plans = tuple(
            self.activity_planner.plan_daily_plans(
                context=context,
                daily_plans=daily_plans,
                daily_strategies=tuple(module.daily_strategy for module in modules),
            )
        )
        start_date = context.week_start_date or date.today()
        week = Week(
            id=context.week_id or self._assessment_week_id(context),
            start_date=start_date,
            end_date=start_date + timedelta(days=6),
            plan_type="assessment",
            study_language=context.study_language,
            status=WeekStatus.PLANNED,
            daily_plans=daily_plans,
            weekly_briefing="",
        )
        self.last_professor_knowledge = None
        return week

    def _add_professor_voice(
        self,
        week: Week,
        knowledge: ProfessorKnowledge,
    ) -> Week:
        """Attach Professor Voice text generated from ProfessorKnowledge."""

        from dataclasses import replace

        return replace(
            week,
            weekly_briefing=(
                self.professor_voice_service
                .generate_study_plan_briefing(knowledge)
            ),
            daily_plans=tuple(
                replace(
                    daily_plan,
                    objective=(
                        self.professor_voice_service
                        .generate_module_objective(knowledge, index)
                    ),
                    briefing=(
                        self.professor_voice_service
                        .generate_daily_briefing(knowledge, index)
                    ),
                )
                for index, daily_plan in enumerate(week.daily_plans, start=1)
            ),
        )

    def _allocate_categories(
        self,
        context: PlannerContext,
        ordered_categories: Sequence[str],
    ) -> Sequence[CategoryAllocation]:
        """Allocate category topic segments using the Session Allocator."""

        allocations = []
        question_pace_seconds = context.preferences.question_pace_seconds or 0

        for category in ordered_categories:
            allocations.extend(
                self.session_allocator.allocate_category_segments(
                    category=category,
                    ordered_topics=context.topics_by_category.get(category, ()),
                    available_budget_minutes=context.planning_budget_minutes,
                    question_pace_seconds=question_pace_seconds,
                )
            )

        return tuple(allocations)

    def _assessment_weekly_strategy(
        self,
        context: PlannerContext,
    ) -> ProfessorWeeklyStrategy:
        """Return a neutral quiz-only strategy for assessment evidence collection."""

        category_strategies = tuple(
            ProfessorCategoryStrategy(
                category=category,
                strategy=ProfessorCategoryStrategyCode.EXPLORE,
                depth=ProfessorDepthCode.DEEP,
                reasoning_code=ProfessorReasoningCode.INSUFFICIENT_EVIDENCE,
            )
            for category in context.categories
        )
        return ProfessorWeeklyStrategy(
            weekly_goal_code=ProfessorWeeklyGoalCode.CALIBRATE_COVERAGE,
            category_strategies=category_strategies,
            priority_categories=(),
            secondary_categories=tuple(context.categories),
        )

    def _assessment_week_id(self, context: PlannerContext) -> str:
        start_date = context.week_start_date or date.today()
        project_id = None
        if isinstance(context.project, Mapping):
            project_id = context.project.get("id")
        else:
            project_id = getattr(context.project, "id", None)
        return f"{project_id or 'project'}-assessment-{start_date.isoformat()}"

    def _build_week(
        self,
        context: PlannerContext,
        modules: Sequence[ProfessorModule],
    ) -> Week:
        """Build a planning-only Week from scheduled weekly sessions."""

        start_date = context.week_start_date or date.today()
        week_id = context.week_id or f"week-{start_date.isoformat()}"
        daily_plans = tuple(
            self._build_daily_plan(
                week_id=week_id,
                week_start_date=start_date,
                module=module,
            )
            for module in modules
        )
        daily_plans = tuple(
            self.activity_planner.plan_daily_plans(
                context=context,
                daily_plans=daily_plans,
                daily_strategies=tuple(module.daily_strategy for module in modules),
            )
        )

        week = Week(
            id=week_id,
            start_date=start_date,
            end_date=start_date + timedelta(days=6),
            study_language=context.study_language,
            status=WeekStatus.PLANNED,
            daily_plans=daily_plans,
        )
        return self.professor_bridge.enrich_week(week)

    def _build_daily_plan(
        self,
        week_id: str,
        week_start_date: date,
        module: ProfessorModule,
    ) -> DailyPlan:
        """Convert a ProfessorModule into a planning-only DailyPlan."""

        session_date = week_start_date + timedelta(days=module.module_index - 1)
        return DailyPlan(
            id=f"{week_id}-session-{module.module_index}",
            date=session_date,
            day_name=session_date.strftime("%A"),
            planned_allocations=module.allocations,
        )


def generate_week(context: PlannerContext) -> Week:
    """Public Planner Engine entrypoint."""

    return PlannerEngine().generate_week(context)
