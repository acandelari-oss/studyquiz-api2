import unittest
from datetime import date

from planner.category_selector import CategoryAnalytics
from planner.planner_engine import PlannerEngine
from planner.planner_models import PlannerContext, PlannerPreferences, SelectedTopic
from planner.planner_serializers import serialize_planner_domain


class StaticProfessorVoiceService:
    def generate_study_plan_briefing(self, _knowledge):
        return "Static Study Plan briefing."

    def generate_daily_briefing(self, _knowledge, module_index):
        return f"Static module {module_index} briefing."

    def generate_module_objective(self, _knowledge, module_index):
        return f"Static module {module_index} objective."


class PlannerProfessorKnowledgeTests(unittest.TestCase):
    def _topics(self, category, count):
        return tuple(
            SelectedTopic(
                id=f"{category.lower()}-{index}",
                title=f"{category} Topic {index}",
                order=index,
            )
            for index in range(1, count + 1)
        )

    def _context(self):
        return PlannerContext(
            project={
                "id": "project-1",
                "name": "Private Law",
                "taxonomy_language": "it",
            },
            categories=("Property", "Contracts"),
            topics_by_category={
                "Property": self._topics("Property", 2),
                "Contracts": self._topics("Contracts", 2),
            },
            analytics={
                "Property": CategoryAnalytics(),
                "Contracts": CategoryAnalytics(),
            },
            preferences=PlannerPreferences(
                question_pace_seconds=60,
                question_style="exam",
            ),
            study_language="Italian",
            number_of_sessions=3,
            planning_budget_minutes=30,
            week_start_date=date(2026, 7, 6),
            week_id="knowledge-week",
        )

    def test_professor_knowledge_contains_study_language(self):
        engine = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        )
        engine.generate_week(self._context())

        knowledge = engine.last_professor_knowledge

        self.assertIsNotNone(knowledge)
        self.assertEqual(knowledge.study_language, "Italian")
        self.assertEqual(knowledge.project_id, "project-1")
        self.assertEqual(knowledge.project_name, "Private Law")
        self.assertEqual(knowledge.taxonomy_language, "it")

    def test_professor_knowledge_contains_module_count_and_activity_mix(self):
        engine = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        )
        week = engine.generate_week(self._context())
        knowledge = engine.last_professor_knowledge

        self.assertEqual(knowledge.module_count, len(week.daily_plans))
        self.assertEqual(knowledge.visible_module_count, len(week.daily_plans))
        self.assertGreater(knowledge.activity_mix.quiz_count, 0)
        self.assertEqual(knowledge.activity_mix.flashcard_count, 0)
        self.assertEqual(knowledge.activity_mix.mixed_count, 0)
        self.assertTrue(knowledge.activity_sizes)

    def test_professor_knowledge_contains_selected_categories_and_topics(self):
        engine = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        )
        engine.generate_week(self._context())
        knowledge = engine.last_professor_knowledge

        self.assertEqual(
            set(knowledge.selected_categories),
            {"Contracts", "Property"},
        )
        self.assertEqual(
            {
                category: [topic.title for topic in topics]
                for category, topics in knowledge.selected_topics_by_category.items()
            },
            {
                "Contracts": ["Contracts Topic 1", "Contracts Topic 2"],
                "Property": ["Property Topic 1", "Property Topic 2"],
            },
        )

    def test_professor_knowledge_contains_teaching_context_for_every_module(self):
        engine = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        )
        week = engine.generate_week(self._context())
        knowledge = engine.last_professor_knowledge

        self.assertEqual(
            [context.module_index for context in knowledge.teaching_contexts],
            list(range(1, len(week.daily_plans) + 1)),
        )

        for context in knowledge.teaching_contexts:
            self.assertTrue(context.conceptual_summary)
            self.assertTrue(context.prerequisite_level)
            self.assertTrue(context.learning_progression)
            self.assertTrue(context.expected_mastery)
            self.assertTrue(context.activity_rationale)

    def test_quiz_only_modules_receive_quiz_teaching_rationale(self):
        engine = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        )
        engine.generate_week(self._context())
        knowledge = engine.last_professor_knowledge

        self.assertTrue(knowledge.teaching_contexts)
        self.assertIn(
            "verify conceptual stability",
            knowledge.teaching_contexts[0].activity_rationale,
        )

    def test_professor_knowledge_does_not_change_generated_week_output(self):
        context = self._context()
        week_with_knowledge = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        ).generate_week(context)
        week_without_knowledge = PlannerEngine(
            professor_voice_service=StaticProfessorVoiceService()
        ).generate_week(context)

        self.assertEqual(
            serialize_planner_domain(week_with_knowledge),
            serialize_planner_domain(week_without_knowledge),
        )


if __name__ == "__main__":
    unittest.main()
