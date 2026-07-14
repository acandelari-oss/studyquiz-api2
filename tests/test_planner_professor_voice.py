import unittest

from planner.professor_knowledge import (
    ProfessorKnowledge,
    ProfessorKnowledgeActivityMix,
    ProfessorKnowledgeActivitySize,
    ProfessorKnowledgeModuleActivityStrategy,
    ProfessorKnowledgeModuleStrategy,
    ProfessorKnowledgePlanningConstraints,
    ProfessorTeachingContext,
    ProfessorKnowledgeTopic,
)
from planner.professor_voice import ProfessorVoiceService, ProfessorVoiceValidator


class PlannerProfessorVoiceTests(unittest.TestCase):
    def _knowledge(
        self,
        *,
        study_language="English",
        additional_modules_remain=False,
        quiz_count=2,
        flashcard_count=0,
        module_activity_type="QUIZ",
        include_second_module_category=False,
    ):
        activity_sizes = [
            ProfessorKnowledgeActivitySize(
                activity_id="activity-1",
                module_index=1,
                activity_type=module_activity_type,
                category="Property",
                num_questions=10 if "QUIZ" in module_activity_type else None,
                num_cards=12 if "FLASHCARD" in module_activity_type else None,
                estimated_duration_minutes=15,
            ),
        ]
        module_activities = [
            ProfessorKnowledgeModuleActivityStrategy(
                category="Property",
                activity_type=module_activity_type,
                depth_code="NORMAL",
                estimated_questions=10 if "QUIZ" in module_activity_type else 0,
                estimated_flashcards=12 if "FLASHCARD" in module_activity_type else 0,
                reasoning_codes=("ASSESS_KNOWLEDGE",),
            ),
        ]

        if include_second_module_category:
            activity_sizes.append(
                ProfessorKnowledgeActivitySize(
                    activity_id="activity-2",
                    module_index=1,
                    activity_type=module_activity_type,
                    category="Contracts",
                    num_questions=10 if "QUIZ" in module_activity_type else None,
                    num_cards=12 if "FLASHCARD" in module_activity_type else None,
                    estimated_duration_minutes=15,
                ),
            )
            module_activities.append(
                ProfessorKnowledgeModuleActivityStrategy(
                    category="Contracts",
                    activity_type=module_activity_type,
                    depth_code="NORMAL",
                    estimated_questions=10 if "QUIZ" in module_activity_type else 0,
                    estimated_flashcards=12 if "FLASHCARD" in module_activity_type else 0,
                    reasoning_codes=("ASSESS_KNOWLEDGE",),
                ),
            )

        if "QUIZ" in module_activity_type and "FLASHCARD" in module_activity_type:
            activity_rationale = "The objective combines retrieval practice with reinforcement."
        elif "FLASHCARD" in module_activity_type:
            activity_rationale = "The objective is long-term consolidation of terminology and relationships."
        elif "QUIZ" in module_activity_type:
            activity_rationale = "The objective is to verify conceptual stability before progressing."
        else:
            activity_rationale = "The objective is to clarify the conceptual structure before continuing."

        return ProfessorKnowledge(
            study_language=study_language,
            project_id="project-1",
            project_name="Private Law",
            taxonomy_language="it",
            module_count=2,
            visible_module_count=2,
            additional_modules_remain=additional_modules_remain,
            selected_categories=("Property", "Contracts"),
            selected_topics_by_category={
                "Property": (
                    ProfessorKnowledgeTopic(id="p1", title="Ownership", order=1),
                ),
                "Contracts": (
                    ProfessorKnowledgeTopic(id="c1", title="Agreement", order=1),
                ),
            },
            activity_mix=ProfessorKnowledgeActivityMix(
                quiz_count=quiz_count,
                flashcard_count=flashcard_count,
                mixed_count=0,
            ),
            activity_sizes=tuple(activity_sizes),
            planning_constraints=ProfessorKnowledgePlanningConstraints(
                module_duration_minutes=30,
                quiz_pace_seconds=90,
                question_style="exam",
                max_visible_modules=12,
            ),
            module_strategies=(
                ProfessorKnowledgeModuleStrategy(
                    module_index=1,
                    daily_goal_code="ASSESS_CURRENT_LEVEL",
                    activities=tuple(module_activities),
                ),
            ),
            teaching_contexts=(
                ProfessorTeachingContext(
                    module_index=1,
                    conceptual_summary=(
                        "This module develops TEST TEACHING CONTEXT for the current learning step."
                    ),
                    prerequisite_level="foundations",
                    learning_progression=(
                        "This establishes a foundation for TEST PROGRESSION before the next teaching decision."
                    ),
                    expected_mastery=(
                        "By the end, you should explain TEST MASTERY in a coherent line of reasoning."
                    ),
                    activity_rationale=activity_rationale,
                ),
            ),
        )

    def test_quiz_only_first_study_plan_briefing(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"briefing":"We will mainly use quizzes to collect evidence about your current level."}'
        )

        briefing = service.generate_study_plan_briefing(self._knowledge())

        self.assertIn("quizzes", briefing)
        self.assertIn("evidence", briefing)

    def test_remaining_categories_and_additional_modules_are_explained_by_fallback(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: ""
        )

        briefing = service.generate_study_plan_briefing(
            self._knowledge(additional_modules_remain=True)
        )

        self.assertIn("continuity", briefing)
        self.assertIn("priority", briefing)

    def test_study_language_controls_fallback_language(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: ""
        )

        briefing = service.generate_study_plan_briefing(
            self._knowledge(study_language="Italian")
        )

        self.assertIn("percorso", briefing)
        self.assertIn("quiz", briefing.lower())

    def test_fallback_generation_when_llm_fails(self):
        def fail(_prompt):
            raise RuntimeError("LLM unavailable")

        service = ProfessorVoiceService(llm_generate=fail)

        briefing = service.generate_study_plan_briefing(self._knowledge())

        self.assertTrue(briefing)
        self.assertIn("foundation", briefing)

    def test_validation_rejects_forbidden_words(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "The Planner algorithm selected these categories.",
                self._knowledge(),
            )
        )

    def test_validation_rejects_contradictory_activity_mentions(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "This plan will use flashcards for practice.",
                self._knowledge(quiz_count=2, flashcard_count=0),
            )
        )

    def test_validation_rejects_activity_mix_claim_for_quiz_only_plan(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "The choice of a mix of quizzes helps us vary the work.",
                self._knowledge(quiz_count=2, flashcard_count=0),
            )
        )

    def test_validation_allows_mix_language_when_both_activity_types_exist(self):
        validator = ProfessorVoiceValidator()

        self.assertTrue(
            validator.validate_study_plan_briefing(
                "Combining assessment and consolidation helps separate what is stable from what needs reinforcement.",
                self._knowledge(quiz_count=2, flashcard_count=2),
            )
        )

    def test_prompt_requests_interpretation_not_data_summary(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return (
                '{"briefing":"We begin with the areas that best reveal your current preparation, '
                'so the first plan can turn uncertainty into evidence for the next stage."}'
            )

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_study_plan_briefing(self._knowledge())

        self.assertIn("Interpret the educational reasoning", captured["prompt"])
        self.assertIn("Do not describe", captured["prompt"])
        self.assertIn("what was generated", captured["prompt"])
        self.assertIn("Avoid repeating information already visible", captured["prompt"])
        self.assertIn("Every sentence must explain a deterministic educational decision", captured["prompt"])
        self.assertIn("If ProfessorKnowledge.activity_mix contains only quizzes", captured["prompt"])
        self.assertIn("WHY WE START HERE", captured["prompt"])
        self.assertIn("STUDY PLAN OBJECTIVE", captured["prompt"])
        self.assertIn("WHAT COMES NEXT", captured["prompt"])
        self.assertIn("ACTIVITY REASONING", captured["prompt"])
        self.assertIn("Never sound like documentation", captured["prompt"])
        self.assertIn("80-140 words", captured["prompt"])

    def test_validation_rejects_category_list_style_output(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "The selected categories are Property and Contracts.",
                self._knowledge(),
            )
        )

    def test_validation_rejects_visible_module_statistics(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "This Study Plan contains 2 modules and 2 quizzes.",
                self._knowledge(),
            )
        )

    def test_validation_rejects_topic_list_style_output(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "The selected topics include Ownership and Agreement.",
                self._knowledge(),
            )
        )

    def test_validation_rejects_descriptive_study_plan_summary(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "The Study Plan contains 2 modules and is divided into quiz activities.",
                self._knowledge(),
            )
        )

    def test_validation_rejects_documentation_tone(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "This Study Plan shows the categories and activities selected for your work.",
                self._knowledge(),
            )
        )

    def test_validation_rejects_generated_plan_language(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_briefing(
                "The generated plan begins with quizzes to assess your preparation.",
                self._knowledge(),
            )
        )

    def test_fallback_explains_pedagogical_continuity(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        briefing = service.generate_study_plan_briefing(
            self._knowledge(additional_modules_remain=True)
        )

        self.assertIn("foundation", briefing)
        self.assertIn("objective picture", briefing)
        self.assertIn("next teaching decision", briefing)
        self.assertIn("continuity", briefing)

    def test_generate_daily_briefing_for_quiz_module(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"briefing":"Today we use a quiz to understand which ideas are solid and which still need attention. Move carefully and use every hesitation as evidence for how your preparation is developing."}'
        )

        briefing = service.generate_daily_briefing(self._knowledge(), 1)

        self.assertIn("quiz", briefing.lower())
        self.assertIn("attention", briefing)

    def test_daily_briefing_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        briefing = service.generate_daily_briefing(
            self._knowledge(study_language="Italian"),
            1,
        )

        self.assertIn("Oggi", briefing)
        self.assertIn("quiz", briefing.lower())

    def test_daily_briefing_validation_rejects_flashcard_for_quiz_only_module(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_daily_briefing(
                "Today we will use flashcards to strengthen recall.",
                self._knowledge(module_activity_type="QUIZ"),
                1,
            )
        )

    def test_daily_briefing_validation_rejects_topic_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_daily_briefing(
                "The selected topics include Ownership and Agreement.",
                self._knowledge(),
                1,
            )
        )

    def test_daily_briefing_validation_rejects_long_output(self):
        validator = ProfessorVoiceValidator()
        long_text = " ".join(["careful"] * 91)

        self.assertFalse(
            validator.validate_daily_briefing(
                long_text,
                self._knowledge(),
                1,
            )
        )

    def test_daily_briefing_validation_rejects_repetitive_opening(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_daily_briefing(
                "We are studying this module now because the quiz will help reveal what is stable.",
                self._knowledge(),
                1,
            )
        )

    def test_daily_briefing_prompt_contains_v4_rules(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"briefing":"Today we use a quiz to clarify what is stable and what needs attention. Work slowly and treat each uncertainty as useful evidence."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_daily_briefing(self._knowledge(), 1)

        self.assertIn("Daily Briefing", captured["prompt"])
        self.assertIn("Why are we studying this module now", captured["prompt"])
        self.assertIn("What should you focus on", captured["prompt"])
        self.assertIn("Why is today's activity type appropriate", captured["prompt"])
        self.assertIn("Speak directly to the learner", captured["prompt"])
        self.assertIn("Vary the opening", captured["prompt"])
        self.assertIn("40-80 words", captured["prompt"])

    def test_daily_briefing_prompt_uses_teaching_context(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"briefing":"Today we use a quiz to clarify what is stable and what needs attention. Work slowly and treat each uncertainty as useful evidence."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_daily_briefing(self._knowledge(), 1)

        self.assertIn("current_module.teaching_context", captured["prompt"])
        self.assertIn("TEST TEACHING CONTEXT", captured["prompt"])
        self.assertIn("TEST PROGRESSION", captured["prompt"])

    def test_daily_briefing_fallback_consumes_teaching_context(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        briefing = service.generate_daily_briefing(self._knowledge(), 1)

        self.assertIn("verify conceptual stability", briefing)

    def test_generate_module_objective_for_quiz_module(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"objective":"By the end of this module, you should be able to distinguish the central legal relationships, explain the reasoning that connects them, and recognise which ideas are secure."}'
        )

        objective = service.generate_module_objective(self._knowledge(), 1)

        self.assertIn("By the end of this module", objective)
        self.assertIn("distinguish", objective)

    def test_module_objective_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        objective = service.generate_module_objective(
            self._knowledge(study_language="Italian"),
            1,
        )

        self.assertIn("Al termine di questo modulo", objective)
        self.assertIn("ragionamento", objective)

    def test_module_objective_validation_rejects_category_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_objective(
                "By the end of this module, you should understand Property and Contracts.",
                self._knowledge(include_second_module_category=True),
                1,
            )
        )

    def test_module_objective_validation_rejects_topic_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_objective(
                "By the end of this module, you should understand Ownership and Agreement.",
                self._knowledge(include_second_module_category=True),
                1,
            )
        )

    def test_module_objective_validation_rejects_visible_statistics(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_objective(
                "This module includes one quiz and ten questions.",
                self._knowledge(),
                1,
            )
        )

    def test_module_objective_validation_rejects_generic_objective(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_objective(
                "By the end of this module, you should learn more and improve your knowledge.",
                self._knowledge(),
                1,
            )
        )

    def test_module_objective_validation_rejects_long_output(self):
        validator = ProfessorVoiceValidator()
        long_text = " ".join(["reasoning"] * 91)

        self.assertFalse(
            validator.validate_module_objective(
                long_text,
                self._knowledge(),
                1,
            )
        )

    def test_module_objective_validation_rejects_institutional_voice(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_objective(
                "Students should be able to distinguish central relationships and explain the reasoning behind them.",
                self._knowledge(),
                1,
            )
        )

    def test_module_objective_validation_requires_direct_address(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_objective(
                "By the end of this module, central relationships become clearer and can be used in reasoning.",
                self._knowledge(),
                1,
            )
        )

    def test_module_objective_prompt_contains_v5_rules(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"objective":"By the end of this module, you should explain the central distinctions and use them in a coherent line of reasoning."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_module_objective(self._knowledge(), 1)

        self.assertIn("Module Objective", captured["prompt"])
        self.assertIn("What will you be able to understand or master", captured["prompt"])
        self.assertIn("expected learning outcome", captured["prompt"])
        self.assertIn("address the learner directly in second person", captured["prompt"])
        self.assertIn("avoid enumerating categories or topics", captured["prompt"])
        self.assertIn("40-80 words", captured["prompt"])

    def test_module_objective_prompt_uses_teaching_context(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"objective":"By the end of this module, you should explain the central distinctions and use them in a coherent line of reasoning."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_module_objective(self._knowledge(), 1)

        self.assertIn("current_module.teaching_context", captured["prompt"])
        self.assertIn("TEST MASTERY", captured["prompt"])

    def test_flashcard_module_teaching_context_uses_flashcard_rationale(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        briefing = service.generate_daily_briefing(
            self._knowledge(
                quiz_count=0,
                flashcard_count=2,
                module_activity_type="FLASHCARDS",
            ),
            1,
        )

        self.assertIn("long-term consolidation", briefing)

    def test_mixed_module_teaching_context_uses_mixed_rationale(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        briefing = service.generate_daily_briefing(
            self._knowledge(
                quiz_count=2,
                flashcard_count=2,
                module_activity_type="QUIZ_PLUS_FLASHCARDS",
            ),
            1,
        )

        self.assertIn("retrieval practice with reinforcement", briefing)

    def test_generate_activity_debrief_for_high_performance(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"debrief":"Your answers suggest that the fundamental relationships are becoming stable. This gives you a secure basis for the next activity, where you can test whether that stability holds in more demanding reasoning."}'
        )

        debrief = service.generate_activity_debrief(
            self._knowledge(),
            1,
            {
                "activity_type": "quiz",
                "accuracy": 0.9,
                "completed": True,
            },
        )

        self.assertIn("Your answers", debrief)
        self.assertIn("stable", debrief)

    def test_activity_debrief_prompt_contains_v7_rules(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"debrief":"Your work suggests that the main concepts are present, but some distinctions still need reinforcement before you continue."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_activity_debrief(
            self._knowledge(),
            1,
            {
                "activity_type": "quiz",
                "correct": 7,
                "total": 10,
            },
        )

        self.assertIn("ACTIVITY_DEBRIEF", captured["prompt"])
        self.assertIn("not a quiz review", captured["prompt"])
        self.assertIn("activity_context", captured["prompt"])
        self.assertIn("performance_level", captured["prompt"])
        self.assertIn("Speak directly to the learner", captured["prompt"])
        self.assertIn("60-120 words", captured["prompt"])

    def test_activity_debrief_fallback_varies_by_high_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_activity_debrief(
            self._knowledge(),
            1,
            {
                "activity_type": "quiz",
                "accuracy": 0.85,
            },
        )

        self.assertIn("Your work", debrief)
        self.assertIn("stable", debrief)

    def test_activity_debrief_fallback_varies_by_medium_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_activity_debrief(
            self._knowledge(),
            1,
            {
                "activity_type": "quiz",
                "accuracy": 0.65,
            },
        )

        self.assertIn("You have identified", debrief)
        self.assertIn("reinforcement", debrief)

    def test_activity_debrief_fallback_varies_by_low_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_activity_debrief(
            self._knowledge(),
            1,
            {
                "activity_type": "quiz",
                "accuracy": 0.3,
            },
        )

        self.assertIn("Your result", debrief)
        self.assertIn("strengthen these foundations", debrief)

    def test_activity_debrief_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_activity_debrief(
            self._knowledge(study_language="Italian"),
            1,
            {
                "activity_type": "quiz",
                "accuracy": 0.35,
            },
        )

        self.assertIn("Il risultato", debrief)
        self.assertIn("dovresti", debrief)

    def test_activity_debrief_validation_rejects_score_repetition(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_activity_debrief(
                "Your score was 90%, so you did very well.",
                self._knowledge(),
                1,
            )
        )

    def test_activity_debrief_validation_rejects_question_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_activity_debrief(
                "You missed question 2, but question 4 shows better reasoning.",
                self._knowledge(),
                1,
            )
        )

    def test_activity_debrief_validation_requires_direct_address(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_activity_debrief(
                "The result suggests that the fundamental relationships are becoming stable.",
                self._knowledge(),
                1,
            )
        )

    def test_activity_debrief_validation_rejects_long_output(self):
        validator = ProfessorVoiceValidator()
        long_text = " ".join(["you"] * 121)

        self.assertFalse(
            validator.validate_activity_debrief(
                long_text,
                self._knowledge(),
                1,
            )
        )

    def test_generate_module_debrief_for_high_mastery(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"debrief":"Your work across this module suggests that the fundamental concepts are now well connected. This gives you a reliable framework for the following module, where more advanced reasoning can build on what you have consolidated."}'
        )

        debrief = service.generate_module_debrief(
            self._knowledge(),
            1,
            {
                "activity_results": [
                    {"activity_type": "quiz", "accuracy": 0.9},
                    {"activity_type": "quiz", "accuracy": 0.85},
                ],
            },
        )

        self.assertIn("Your work", debrief)
        self.assertIn("following module", debrief)

    def test_module_debrief_prompt_contains_v8_rules(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"debrief":"Your work across this module shows a usable foundation, although some relationships should remain active as you move into the next step."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_module_debrief(
            self._knowledge(),
            1,
            {
                "activity_results": [
                    {"activity_type": "quiz", "correct": 7, "total": 10},
                ],
            },
        )

        self.assertIn("MODULE_DEBRIEF", captured["prompt"])
        self.assertIn("not an Activity Debrief", captured["prompt"])
        self.assertIn("module_debrief_context", captured["prompt"])
        self.assertIn("overall_accuracy", captured["prompt"])
        self.assertIn("following module", captured["prompt"])
        self.assertIn("80-140 words", captured["prompt"])

    def test_module_debrief_fallback_varies_by_high_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_module_debrief(
            self._knowledge(),
            1,
            {
                "activity_results": [
                    {"activity_type": "quiz", "accuracy": 0.9},
                ],
            },
        )

        self.assertIn("your fundamental concepts", debrief)
        self.assertIn("next learning step", debrief)

    def test_module_debrief_fallback_varies_by_medium_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_module_debrief(
            self._knowledge(),
            1,
            {
                "activity_results": [
                    {"activity_type": "quiz", "accuracy": 0.65},
                ],
            },
        )

        self.assertIn("you still have", debrief)
        self.assertIn("broader context", debrief)

    def test_module_debrief_fallback_varies_by_low_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_module_debrief(
            self._knowledge(),
            1,
            {
                "activity_results": [
                    {"activity_type": "quiz", "accuracy": 0.35},
                ],
            },
        )

        self.assertIn("underlying concepts", debrief)
        self.assertIn("should be strengthened", debrief)

    def test_module_debrief_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_module_debrief(
            self._knowledge(study_language="Italian"),
            1,
            {
                "activity_results": [
                    {"activity_type": "quiz", "accuracy": 0.9},
                ],
            },
        )

        self.assertIn("Il lavoro", debrief)
        self.assertIn("ti dà", debrief)

    def test_module_debrief_validation_rejects_visible_statistics(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_debrief(
                "This module contains 2 quizzes and 20 questions.",
                self._knowledge(),
                1,
            )
        )

    def test_module_debrief_validation_rejects_category_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_debrief(
                "Your work in Property and Contracts shows that the main ideas are connected.",
                self._knowledge(include_second_module_category=True),
                1,
            )
        )

    def test_module_debrief_validation_rejects_score_repetition(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_debrief(
                "Your accuracy was 80%, so the module went well.",
                self._knowledge(),
                1,
            )
        )

    def test_module_debrief_validation_requires_direct_address(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_debrief(
                "The completed module suggests that the fundamental concepts are stable.",
                self._knowledge(),
                1,
            )
        )

    def test_module_debrief_validation_rejects_long_output(self):
        validator = ProfessorVoiceValidator()
        long_text = " ".join(["you"] * 141)

        self.assertFalse(
            validator.validate_module_debrief(
                long_text,
                self._knowledge(),
                1,
            )
        )

    def test_generate_homework_recommendation_for_medium_performance(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: (
                '{"homework":"Choose one uncertain distinction and write a short two-column comparison from memory, then check the material only to correct the weakest link in your reasoning."}'
            )
        )

        homework = service.generate_homework_recommendation(
            self._knowledge(),
            1,
            {"accuracy": 0.65},
        )

        self.assertIn("you", homework.lower())
        self.assertIn("comparison", homework)

    def test_homework_recommendation_prompt_contains_v1_rules(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return (
                '{"homework":"Spend ten minutes writing one compact explanation from memory, then check only the point where your reasoning becomes least precise."}'
            )

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_homework_recommendation(
            self._knowledge(),
            1,
            {"accuracy": 0.9},
        )

        self.assertIn("HOMEWORK_RECOMMENDATION", captured["prompt"])
        self.assertIn("ONE concrete action", captured["prompt"])
        self.assertIn("5-15 minutes", captured["prompt"])
        self.assertIn("Do not simply say", captured["prompt"])
        self.assertIn("homework_context", captured["prompt"])

    def test_homework_recommendation_fallback_varies_by_high_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        homework = service.generate_homework_recommendation(
            self._knowledge(),
            1,
            {"accuracy": 0.9},
        )

        self.assertIn("ten minutes", homework)
        self.assertIn("explanation", homework)

    def test_homework_recommendation_fallback_varies_by_low_performance(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        homework = service.generate_homework_recommendation(
            self._knowledge(),
            1,
            {"accuracy": 0.25},
        )

        self.assertIn("ten focused minutes", homework)
        self.assertIn("own words", homework)

    def test_homework_recommendation_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        homework = service.generate_homework_recommendation(
            self._knowledge(study_language="Italian"),
            1,
            {"accuracy": 0.25},
        )

        self.assertIn("dieci minuti", homework)
        self.assertIn("parole tue", homework)

    def test_homework_recommendation_validation_rejects_generic_text(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_homework_recommendation(
                "You should review the material again.",
                self._knowledge(),
                1,
            )
        )

    def test_homework_recommendation_validation_rejects_score_repetition(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_homework_recommendation(
                "You scored 80%, so spend ten minutes writing a compact explanation.",
                self._knowledge(),
                1,
            )
        )

    def test_homework_recommendation_validation_requires_direct_address(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_homework_recommendation(
                "The learner should write a compact explanation from memory.",
                self._knowledge(),
                1,
            )
        )

    def test_homework_recommendation_validation_rejects_repeated_debrief(self):
        validator = ProfessorVoiceValidator()
        debrief = (
            "You have developed a usable understanding of the main ideas, and the next module can now revisit them in a broader context."
        )

        self.assertFalse(
            validator.validate_homework_recommendation(
                "You have developed a usable understanding of the main ideas, and the next module can now revisit them in a broader context.",
                self._knowledge(),
                1,
                module_debrief=debrief,
            )
        )

    def test_homework_recommendation_validation_rejects_unrelated_multiple_tasks(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_homework_recommendation(
                "You should write a short explanation from memory. In addition, also complete a separate task with ten new examples.",
                self._knowledge(),
                1,
            )
        )

    def test_generate_study_plan_debrief_for_high_mastery(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"debrief":"Throughout this Study Plan, you have developed a coherent understanding of the underlying concepts. The knowledge you have built is now connected enough to support more demanding reasoning in the next Study Plan."}'
        )

        debrief = service.generate_study_plan_debrief(
            self._knowledge(),
            {
                "module_results": [
                    {"activity_results": [{"activity_type": "quiz", "accuracy": 0.9}]},
                    {"activity_results": [{"activity_type": "quiz", "accuracy": 0.85}]},
                ],
            },
        )

        self.assertIn("Throughout this Study Plan", debrief)
        self.assertIn("next Study Plan", debrief)

    def test_study_plan_debrief_prompt_contains_v9_rules(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"debrief":"Throughout this Study Plan, you have connected the main ideas more clearly. The next Study Plan can build on that foundation while reinforcing the relationships that still need attention."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_study_plan_debrief(
            self._knowledge(),
            {
                "module_results": [
                    {"activity_results": [{"activity_type": "quiz", "correct": 7, "total": 10}]},
                ],
            },
        )

        self.assertIn("STUDY_PLAN_DEBRIEF", captured["prompt"])
        self.assertIn("not a Weekly Debrief", captured["prompt"])
        self.assertIn("study_plan_debrief_context", captured["prompt"])
        self.assertIn("overall_accuracy", captured["prompt"])
        self.assertIn("next Study Plan", captured["prompt"])
        self.assertIn("100-180 words", captured["prompt"])

    def test_study_plan_debrief_fallback_varies_by_high_mastery(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_study_plan_debrief(
            self._knowledge(),
            {
                "module_results": [
                    {"activity_results": [{"activity_type": "quiz", "accuracy": 0.9}]},
                ],
            },
        )

        self.assertIn("Throughout this Study Plan", debrief)
        self.assertIn("more demanding reasoning", debrief)

    def test_study_plan_debrief_fallback_varies_by_medium_mastery(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_study_plan_debrief(
            self._knowledge(),
            {
                "module_results": [
                    {"activity_results": [{"activity_type": "quiz", "accuracy": 0.65}]},
                ],
            },
        )

        self.assertIn("clear progress", debrief)
        self.assertIn("further consolidation", debrief)

    def test_study_plan_debrief_fallback_varies_by_low_mastery(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_study_plan_debrief(
            self._knowledge(),
            {
                "module_results": [
                    {"activity_results": [{"activity_type": "quiz", "accuracy": 0.3}]},
                ],
            },
        )

        self.assertIn("fundamental concepts are still developing", debrief)
        self.assertIn("next Study Plan", debrief)

    def test_study_plan_debrief_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        debrief = service.generate_study_plan_debrief(
            self._knowledge(study_language="Italian"),
            {
                "module_results": [
                    {"activity_results": [{"activity_type": "quiz", "accuracy": 0.9}]},
                ],
            },
        )

        self.assertIn("Piano di Studio", debrief)
        self.assertIn("hai sviluppato", debrief)

    def test_study_plan_debrief_validation_rejects_module_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_debrief(
                "In Module 1 you built foundations, and in Module 2 you applied them.",
                self._knowledge(),
            )
        )

    def test_study_plan_debrief_validation_rejects_category_enumeration(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_debrief(
                "Your work in Property and Contracts shows that the main ideas are connected.",
                self._knowledge(),
            )
        )

    def test_study_plan_debrief_validation_rejects_visible_statistics(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_debrief(
                "This Study Plan contains 2 modules and 2 quizzes.",
                self._knowledge(),
            )
        )

    def test_study_plan_debrief_validation_rejects_score_repetition(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_debrief(
                "Your accuracy was 80%, so the Study Plan went well.",
                self._knowledge(),
            )
        )

    def test_study_plan_debrief_validation_requires_direct_address(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_study_plan_debrief(
                "The completed Study Plan shows coherent progress across the core ideas.",
                self._knowledge(),
            )
        )

    def test_study_plan_debrief_validation_rejects_long_output(self):
        validator = ProfessorVoiceValidator()
        long_text = " ".join(["you"] * 181)

        self.assertFalse(
            validator.validate_study_plan_debrief(
                long_text,
                self._knowledge(),
            )
        )

    def test_generate_module_question_answer(self):
        service = ProfessorVoiceService(
            llm_generate=lambda _prompt: '{"answer":"You should connect the distinction to the concrete example from this module, because that is where the reasoning becomes clearer."}'
        )

        answer = service.generate_module_question_answer(
            self._knowledge(),
            1,
            {
                "activity_results": [{"activity_type": "quiz", "accuracy": 0.75}],
                "professor_debrief": "You have identified the main ideas.",
                "homework_recommendation": "Write one short comparison from memory.",
            },
            "Can you explain the main distinction again?",
            [],
        )

        self.assertIn("You should connect", answer)

    def test_module_question_prompt_contains_context(self):
        captured = {}

        def capture_prompt(prompt):
            captured["prompt"] = prompt
            return '{"answer":"You should focus on the central relationship from this module and use the example to test whether the distinction is clear."}'

        service = ProfessorVoiceService(llm_generate=capture_prompt)

        service.generate_module_question_answer(
            self._knowledge(),
            1,
            {
                "activity_results": [{"activity_type": "quiz", "accuracy": 0.75}],
                "professor_debrief": "You have identified the main ideas.",
                "homework_recommendation": "Write one short comparison from memory.",
            },
            "What should I focus on?",
            [{"role": "student", "content": "I am unsure about the example."}],
        )

        self.assertIn("MODULE_QUESTION", captured["prompt"])
        self.assertIn("module_question_context", captured["prompt"])
        self.assertIn("professor_debrief", captured["prompt"])
        self.assertIn("homework_recommendation", captured["prompt"])
        self.assertIn("What should I focus on?", captured["prompt"])

    def test_module_question_fallback_uses_study_language(self):
        service = ProfessorVoiceService(llm_generate=lambda _prompt: "")

        answer = service.generate_module_question_answer(
            self._knowledge(study_language="Italian"),
            1,
            {"activity_results": [{"activity_type": "quiz", "accuracy": 0.75}]},
            "Può chiarire questo punto?",
            [],
        )

        self.assertIn("La domanda", answer)
        self.assertIn("modulo", answer)

    def test_module_question_validation_rejects_implementation_language(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_question_answer(
                "The software generated this answer from the Planner prompt.",
                self._knowledge(),
                1,
            )
        )

    def test_module_question_validation_rejects_score_repetition(self):
        validator = ProfessorVoiceValidator()

        self.assertFalse(
            validator.validate_module_question_answer(
                "Your accuracy was 75%, so you should continue.",
                self._knowledge(),
                1,
            )
        )


if __name__ == "__main__":
    unittest.main()
