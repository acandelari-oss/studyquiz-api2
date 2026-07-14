"""Professor Voice service for Study Plan narratives.

This module generates only Professor Voice text. It does not make planning,
strategy, activity, quiz, flashcard, homework, debrief, or conversation
decisions.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, is_dataclass
from typing import Any, Callable, Optional

from .professor_identity import DEFAULT_PROFESSOR_IDENTITY, ProfessorIdentity
from .professor_knowledge import ProfessorKnowledge


COMMUNICATION_TYPE_STUDY_PLAN_BRIEFING = "STUDY_PLAN_BRIEFING"
COMMUNICATION_TYPE_DAILY_BRIEFING = "DAILY_BRIEFING"
COMMUNICATION_TYPE_MODULE_OBJECTIVE = "MODULE_OBJECTIVE"
COMMUNICATION_TYPE_ACTIVITY_DEBRIEF = "ACTIVITY_DEBRIEF"
COMMUNICATION_TYPE_MODULE_DEBRIEF = "MODULE_DEBRIEF"
COMMUNICATION_TYPE_STUDY_PLAN_DEBRIEF = "STUDY_PLAN_DEBRIEF"
COMMUNICATION_TYPE_HOMEWORK_RECOMMENDATION = "HOMEWORK_RECOMMENDATION"
COMMUNICATION_TYPE_MODULE_QUESTION = "MODULE_QUESTION"


class ProfessorVoiceValidator:
    """Lightweight validation for Professor Voice outputs."""

    FORBIDDEN_TERMS = (
        "software",
        "system",
        "application",
        "gpt",
        "chatgpt",
        "openai",
        "planner",
        "algorithm",
        "implementation",
        "prompt",
        "generated plan",
        "generated study plan",
        "language model",
        "ai model",
    )
    CATEGORY_LIST_PATTERNS = (
        r"\bselected categories (are|include)\b",
        r"\bcategories (are|include)\b",
        r"\bthe categories selected\b",
        r"\ble categorie selezionate\b",
        r"\ble categorie sono\b",
        r"\ble categorie includono\b",
    )
    VISIBLE_STAT_PATTERNS = (
        r"\b(contains|has|includes|consists of)\s+\d+\s+(modules?|quizzes|flashcards|questions?)\b",
        r"\b\d+\s+(modules?|quizzes|flashcards|questions?)\b",
        r"\b(the\s+)?study plan (contains|has|includes|consists of|is divided into)\b",
        r"\bthere are\s+\d+\b",
        r"\b(contiene|ha|include|comprende)\s+\d+\s+(moduli|quiz|flashcard|domande)\b",
        r"\b\d+\s+(moduli|quiz|flashcard|domande)\b",
        r"\bil piano di studio (contiene|ha|include|comprende|è diviso in)\b",
        r"\bci sono\s+\d+\b",
    )
    TOPIC_LIST_PATTERNS = (
        r"\btopics (are|include)\b",
        r"\bselected topics\b",
        r"\btopic list\b",
        r"\bgli argomenti sono\b",
        r"\bgli argomenti includono\b",
        r"\belenco degli argomenti\b",
    )
    DOCUMENTATION_TONE_PATTERNS = (
        r"\bthis (study )?plan (shows|displays|presents|lists|summarizes|describes)\b",
        r"\bthe (study )?plan is (organized|structured|generated|created)\b",
        r"\bthis module (shows|displays|presents|lists|summarizes|describes|contains|includes)\b",
        r"\bthe module (shows|displays|presents|lists|summarizes|describes|contains|includes)\b",
        r"\bthis module includes? (one|two|three|\d+)\b",
        r"\bthis module contains? (one|two|three|\d+)\b",
        r"\byou can see\b",
        r"\bas shown\b",
        r"\bvisible on (the )?(screen|dashboard|page)\b",
        r"\bil piano (mostra|presenta|elenca|riassume|descrive)\b",
        r"\bil piano di studio è (organizzato|strutturato|generato|creato)\b",
        r"\bquesto modulo (mostra|presenta|elenca|riassume|descrive|contiene|include)\b",
        r"\bil modulo (mostra|presenta|elenca|riassume|descrive|contiene|include)\b",
        r"\bcome mostrato\b",
        r"\bvisibile (sullo schermo|nella dashboard|nella pagina)\b",
    )
    GENERIC_OBJECTIVE_PATTERNS = (
        r"\blearn more\b",
        r"\bimprove (your )?knowledge\b",
        r"\bdeepen (your )?understanding\b",
        r"\bunderstand better\b",
        r"\bstudy the material\b",
        r"\bcomplete (today'?s )?activities\b",
        r"\bcomplete the module\b",
        r"\bfinish the module\b",
        r"\bwe will study\b",
        r"\bwe will cover\b",
        r"\bimparare di più\b",
        r"\bmigliorare (la )?conoscenza\b",
        r"\bcapire meglio\b",
        r"\bstudiare il materiale\b",
        r"\bcompletare (le )?attività\b",
        r"\bcompletare il modulo\b",
        r"\bstudieremo\b",
        r"\btratteremo\b",
    )
    GENERIC_HOMEWORK_PATTERNS = (
        r"\breview\b",
        r"\bstudy again\b",
        r"\bstudy the material\b",
        r"\blearn more\b",
        r"\bgo over\b",
        r"\bripassa\b",
        r"\brivedi\b",
        r"\bstudia di nuovo\b",
        r"\bstudia il materiale\b",
        r"\bapprofondisci\b",
    )
    MULTI_TASK_HOMEWORK_PATTERNS = (
        r"\bin addition\b",
        r"\balso complete\b",
        r"\balso do\b",
        r"\bseparate task\b",
        r"\bsecond task\b",
        r"\bthen do another\b",
        r"\binoltre\b",
        r"\bfai anche\b",
        r"\bsecondo compito\b",
        r"\bun compito separato\b",
    )
    INSTITUTIONAL_OBJECTIVE_PATTERNS = (
        r"\bstudents? should\b",
        r"\bstudents? will\b",
        r"\blearners? should\b",
        r"\blearners? will\b",
        r"\bthe student should\b",
        r"\bthe student will\b",
        r"\bthe student is expected to\b",
        r"\ba successful student should\b",
        r"\bgli studenti dovrebbero\b",
        r"\bgli studenti sapranno\b",
        r"\blo studente dovrebbe\b",
        r"\blo studente saprà\b",
    )
    DIRECT_ADDRESS_PATTERNS = (
        r"\byou\b",
        r"\byour\b",
        r"\byou'll\b",
        r"\byou will\b",
        r"\byou should\b",
        r"\bdovresti\b",
        r"\bsarai\b",
        r"\bsaprai\b",
        r"\bpotrai\b",
        r"\briuscirai\b",
        r"\bti\b",
    )
    REPETITIVE_DAILY_OPENING_PATTERNS = (
        r"^we are studying this module now\b",
        r"^we're studying this module now\b",
        r"^today we are working to see\b",
        r"^oggi lavoriamo per capire quanto\b",
    )
    SCORE_REPETITION_PATTERNS = (
        r"\b(score|scored|accuracy|result)\s+(is|was|of|:)?\s*\d+([.,]\d+)?\s*(%|/)",
        r"\b\d+([.,]\d+)?\s*(%|/)\s*(score|accuracy|correct|result)\b",
        r"\b(punteggio|accuratezza|risultato)\s+(è|era|di|:)?\s*\d+([.,]\d+)?\s*(%|/)",
    )
    QUESTION_ENUMERATION_PATTERNS = (
        r"\bquestion\s+\d+\b",
        r"\bquestions?\s+\d+\s*(and|,)\s*\d+\b",
        r"\bdomanda\s+\d+\b",
        r"\bdomande\s+\d+\s*(e|,)\s*\d+\b",
    )
    MODULE_ENUMERATION_PATTERNS = (
        r"\bmodule\s+\d+\b",
        r"\bmodules?\s+\d+\s*(and|,)\s*\d+\b",
        r"\bmodulo\s+\d+\b",
        r"\bmoduli\s+\d+\s*(e|,)\s*\d+\b",
    )

    def validate_study_plan_briefing(
        self,
        briefing: str,
        knowledge: ProfessorKnowledge,
    ) -> bool:
        """Return whether a Study Plan Briefing respects V1 constraints."""

        text = " ".join(str(briefing or "").split())

        if not text:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if knowledge.activity_mix.quiz_count == 0 and self._mentions_quiz(lower_text):
            return False

        if knowledge.activity_mix.flashcard_count == 0 and self._mentions_flashcards(lower_text):
            return False

        if not self._has_mixed_activity_types(knowledge) and self._claims_activity_mix(lower_text):
            return False

        if knowledge.additional_modules_remain and self._claims_everything_is_included(lower_text):
            return False

        if not knowledge.additional_modules_remain and self._claims_additional_modules(lower_text):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        return True

    def validate_daily_briefing(
        self,
        briefing: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        """Return whether a Daily Briefing respects Professor Voice constraints."""

        text = " ".join(str(briefing or "").split())

        if not text:
            return False

        if len(text.split()) > 90:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        if self._uses_repetitive_daily_opening(lower_text):
            return False

        module_mix = self._module_activity_mix(knowledge, module_index)

        if module_mix["quiz"] == 0 and self._mentions_quiz(lower_text):
            return False

        if module_mix["flashcards"] == 0 and self._mentions_flashcards(lower_text):
            return False

        if (
            module_mix["quiz"] == 0
            and module_mix["flashcards"] == 0
            and (
                self._mentions_quiz(lower_text)
                or self._mentions_flashcards(lower_text)
            )
        ):
            return False

        return True

    def validate_module_objective(
        self,
        objective: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        """Return whether a Module Objective respects Professor Voice constraints."""

        text = " ".join(str(objective or "").split())

        if not text:
            return False

        if len(text.split()) > 90:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        if self._looks_like_module_category_enumeration(lower_text, knowledge, module_index):
            return False

        if self._looks_like_module_topic_enumeration(lower_text, knowledge, module_index):
            return False

        if self._is_generic_objective(lower_text):
            return False

        if self._is_institutional_objective(lower_text):
            return False

        if not self._addresses_learner_directly(lower_text):
            return False

        return True

    def validate_activity_debrief(
        self,
        debrief: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        """Return whether an Activity Debrief respects Professor Voice constraints."""

        text = " ".join(str(debrief or "").split())

        if not text:
            return False

        if len(text.split()) > 120:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        if self._looks_like_module_category_enumeration(lower_text, knowledge, module_index):
            return False

        if self._looks_like_module_topic_enumeration(lower_text, knowledge, module_index):
            return False

        if self._repeats_score(lower_text):
            return False

        if self._enumerates_questions(lower_text):
            return False

        if not self._addresses_learner_directly(lower_text):
            return False

        return True

    def validate_module_debrief(
        self,
        debrief: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        """Return whether a Module Debrief respects Professor Voice constraints."""

        text = " ".join(str(debrief or "").split())

        if not text:
            return False

        if len(text.split()) > 140:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        if self._looks_like_module_category_enumeration(lower_text, knowledge, module_index):
            return False

        if self._looks_like_module_topic_enumeration(lower_text, knowledge, module_index):
            return False

        if self._repeats_score(lower_text):
            return False

        if self._enumerates_questions(lower_text):
            return False

        if not self._addresses_learner_directly(lower_text):
            return False

        return True

    def validate_study_plan_debrief(
        self,
        debrief: str,
        knowledge: ProfessorKnowledge,
    ) -> bool:
        """Return whether a Study Plan Debrief respects Professor Voice constraints."""

        text = " ".join(str(debrief or "").split())

        if not text:
            return False

        if len(text.split()) > 180:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        if self._repeats_score(lower_text):
            return False

        if self._enumerates_questions(lower_text):
            return False

        if self._enumerates_modules(lower_text):
            return False

        if not self._addresses_learner_directly(lower_text):
            return False

        return True

    def validate_homework_recommendation(
        self,
        homework: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_debrief: str = "",
    ) -> bool:
        """Return whether a Homework recommendation respects Professor Voice constraints."""

        text = " ".join(str(homework or "").split())

        if not text:
            return False

        if len(text.split()) > 80:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_category_list(lower_text, knowledge):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._looks_like_topic_list(lower_text):
            return False

        if self._sounds_like_documentation(lower_text):
            return False

        if self._looks_like_module_category_enumeration(lower_text, knowledge, module_index):
            return False

        if self._looks_like_module_topic_enumeration(lower_text, knowledge, module_index):
            return False

        if self._repeats_score(lower_text):
            return False

        if not self._addresses_learner_directly(lower_text):
            return False

        if self._is_generic_homework(lower_text):
            return False

        if self._contains_multiple_unrelated_homework_tasks(lower_text):
            return False

        if module_debrief and self._substantially_repeats(lower_text, module_debrief):
            return False

        return True

    def validate_module_question_answer(
        self,
        answer: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        """Return whether an Ask-the-Professor answer respects Voice constraints."""

        text = " ".join(str(answer or "").split())

        if not text:
            return False

        if len(text.split()) > 180:
            return False

        lower_text = text.lower()

        if any(term in lower_text for term in self.FORBIDDEN_TERMS):
            return False

        if self._looks_like_visible_statistics(lower_text):
            return False

        if self._repeats_score(lower_text):
            return False

        if not self._addresses_learner_directly(lower_text):
            return False

        return True

    def _mentions_quiz(self, lower_text: str) -> bool:
        return any(term in lower_text for term in ("quiz", "questionari"))

    def _mentions_flashcards(self, lower_text: str) -> bool:
        return any(term in lower_text for term in ("flashcard", "flashcards", "carte"))

    def _has_mixed_activity_types(self, knowledge: ProfessorKnowledge) -> bool:
        return (
            knowledge.activity_mix.quiz_count > 0
            and knowledge.activity_mix.flashcard_count > 0
        ) or knowledge.activity_mix.mixed_count > 0

    def _claims_activity_mix(self, lower_text: str) -> bool:
        return any(
            term in lower_text
            for term in (
                "activity mix",
                "mix of activities",
                "mix of quizzes",
                "mixed activities",
                "combination of activities",
                "combines assessment",
                "combines quizzes",
                "combines flashcards",
                "variety of activities",
                "alternating assessment",
                "alternating quizzes",
                "mix di attività",
                "mix di quiz",
                "combinazione di attività",
                "combina quiz",
                "combina flashcard",
                "varietà di attività",
                "alternanza tra",
            )
        )

    def _claims_everything_is_included(self, lower_text: str) -> bool:
        return any(
            phrase in lower_text
            for phrase in (
                "all material is included",
                "all the material is included",
                "everything is included",
                "tutto il materiale rientra",
                "tutti gli argomenti rientrano",
            )
        )

    def _claims_additional_modules(self, lower_text: str) -> bool:
        return any(
            phrase in lower_text
            for phrase in (
                "additional modules",
                "more modules",
                "future study plans",
                "moduli aggiuntivi",
                "altri moduli",
                "piani successivi",
            )
        )

    def _looks_like_category_list(
        self,
        lower_text: str,
        knowledge: ProfessorKnowledge,
    ) -> bool:
        if any(re.search(pattern, lower_text) for pattern in self.CATEGORY_LIST_PATTERNS):
            return True

        selected_categories = [
            str(category).strip().lower()
            for category in knowledge.selected_categories
            if str(category).strip()
        ]

        if len(selected_categories) < 2:
            return False

        mentioned_categories = [
            category
            for category in selected_categories
            if category in lower_text
        ]

        return len(mentioned_categories) >= min(3, len(selected_categories))

    def _looks_like_visible_statistics(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.VISIBLE_STAT_PATTERNS
        )

    def _looks_like_topic_list(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.TOPIC_LIST_PATTERNS
        )

    def _sounds_like_documentation(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.DOCUMENTATION_TONE_PATTERNS
        )

    def _is_generic_objective(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.GENERIC_OBJECTIVE_PATTERNS
        )

    def _is_generic_homework(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.GENERIC_HOMEWORK_PATTERNS
        )

    def _contains_multiple_unrelated_homework_tasks(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.MULTI_TASK_HOMEWORK_PATTERNS
        )

    def _is_institutional_objective(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.INSTITUTIONAL_OBJECTIVE_PATTERNS
        )

    def _addresses_learner_directly(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.DIRECT_ADDRESS_PATTERNS
        )

    def _uses_repetitive_daily_opening(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.REPETITIVE_DAILY_OPENING_PATTERNS
        )

    def _repeats_score(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.SCORE_REPETITION_PATTERNS
        )

    def _enumerates_questions(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.QUESTION_ENUMERATION_PATTERNS
        )

    def _enumerates_modules(self, lower_text: str) -> bool:
        return any(
            re.search(pattern, lower_text)
            for pattern in self.MODULE_ENUMERATION_PATTERNS
        )

    def _substantially_repeats(
        self,
        lower_text: str,
        comparison_text: str,
    ) -> bool:
        comparison_words = {
            word
            for word in re.findall(r"\b[\wÀ-ÿ']{5,}\b", str(comparison_text or "").lower())
        }
        homework_words = {
            word
            for word in re.findall(r"\b[\wÀ-ÿ']{5,}\b", lower_text)
        }

        if len(homework_words) < 6 or len(comparison_words) < 6:
            return False

        overlap = homework_words.intersection(comparison_words)
        return len(overlap) / len(homework_words) >= 0.65

    def _looks_like_module_category_enumeration(
        self,
        lower_text: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        categories = self._module_categories(knowledge, module_index)

        if len(categories) < 2:
            return False

        mentioned = [
            category
            for category in categories
            if category.lower() in lower_text
        ]

        return len(mentioned) >= 2

    def _looks_like_module_topic_enumeration(
        self,
        lower_text: str,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> bool:
        topics = self._module_topics(knowledge, module_index)

        if len(topics) < 2:
            return False

        mentioned = [
            topic
            for topic in topics
            if topic.lower() in lower_text
        ]

        return len(mentioned) >= 2

    def _module_categories(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> tuple[str, ...]:
        categories = []
        module_strategy = next(
            (
                strategy
                for strategy in knowledge.module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )

        if module_strategy:
            categories.extend(
                str(activity.category)
                for activity in module_strategy.activities
                if activity.category
            )

        categories.extend(
            str(activity.category)
            for activity in knowledge.activity_sizes
            if activity.module_index == module_index and activity.category
        )

        return tuple(dict.fromkeys(categories))

    def _module_topics(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> tuple[str, ...]:
        categories = self._module_categories(knowledge, module_index)
        topics = []

        for category in categories:
            topics.extend(
                str(topic.title)
                for topic in knowledge.selected_topics_by_category.get(category, ())
                if topic.title
            )

        return tuple(dict.fromkeys(topics))

    def _module_activity_mix(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> dict[str, int]:
        module_strategy = next(
            (
                strategy
                for strategy in knowledge.module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )
        quiz_count = 0
        flashcard_count = 0

        if module_strategy:
            for activity in module_strategy.activities:
                activity_type = str(activity.activity_type or "").upper()

                if "QUIZ_PLUS_FLASHCARDS" in activity_type:
                    quiz_count += 1
                    flashcard_count += 1
                elif "QUIZ" in activity_type:
                    quiz_count += 1
                elif "FLASHCARD" in activity_type:
                    flashcard_count += 1

        if quiz_count or flashcard_count:
            return {
                "quiz": quiz_count,
                "flashcards": flashcard_count,
            }

        for activity in knowledge.activity_sizes:
            if activity.module_index != module_index:
                continue

            activity_type = str(activity.activity_type or "").upper()

            if "QUIZ" in activity_type:
                quiz_count += 1

            if "FLASHCARD" in activity_type:
                flashcard_count += 1

        return {
            "quiz": quiz_count,
            "flashcards": flashcard_count,
        }


class ProfessorVoiceService:
    """Generate Professor Voice text from identity and ProfessorKnowledge only."""

    def __init__(
        self,
        *,
        identity: ProfessorIdentity = DEFAULT_PROFESSOR_IDENTITY,
        llm_generate: Optional[Callable[[str], str]] = None,
        validator: Optional[ProfessorVoiceValidator] = None,
    ) -> None:
        self.identity = identity
        self.llm_generate = llm_generate
        self.validator = validator or ProfessorVoiceValidator()

    def generate_study_plan_briefing(
        self,
        knowledge: ProfessorKnowledge,
    ) -> str:
        """Generate only the Study Plan Briefing.

        GPT/LLM failures and validation failures always fall back to a
        deterministic briefing. The method never returns an empty string.
        """

        fallback = self._fallback_study_plan_briefing(knowledge)

        try:
            briefing = self._generate_with_llm(knowledge).strip()
        except Exception as error:
            print("⚠️ PROFESSOR VOICE FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_study_plan_briefing(briefing, knowledge):
            print("⚠️ PROFESSOR VOICE VALIDATION FAILED — USING FALLBACK")
            return fallback

        return briefing or fallback

    def generate_daily_briefing(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        """Generate the Professor Daily Briefing for one study module."""

        fallback = self._fallback_daily_briefing(knowledge, module_index)

        try:
            briefing = self._generate_daily_with_llm(knowledge, module_index).strip()
        except Exception as error:
            print("⚠️ PROFESSOR DAILY VOICE FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_daily_briefing(
            briefing,
            knowledge,
            module_index,
        ):
            print("⚠️ PROFESSOR DAILY VOICE VALIDATION FAILED — USING FALLBACK")
            return fallback

        return briefing or fallback

    def generate_module_objective(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        """Generate the Professor Module Objective for one study module."""

        fallback = self._fallback_module_objective(knowledge, module_index)

        try:
            objective = self._generate_module_objective_with_llm(
                knowledge,
                module_index,
            ).strip()
        except Exception as error:
            print("⚠️ PROFESSOR MODULE OBJECTIVE FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_module_objective(
            objective,
            knowledge,
            module_index,
        ):
            print("⚠️ PROFESSOR MODULE OBJECTIVE VALIDATION FAILED — USING FALLBACK")
            return fallback

        return objective or fallback

    def generate_activity_debrief(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_result: Any,
    ) -> str:
        """Generate the Professor Activity Debrief after one completed activity."""

        fallback = self._fallback_activity_debrief(
            knowledge,
            module_index,
            activity_result,
        )

        try:
            debrief = self._generate_activity_debrief_with_llm(
                knowledge,
                module_index,
                activity_result,
            ).strip()
        except Exception as error:
            print("⚠️ PROFESSOR ACTIVITY DEBRIEF FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_activity_debrief(
            debrief,
            knowledge,
            module_index,
        ):
            print("⚠️ PROFESSOR ACTIVITY DEBRIEF VALIDATION FAILED — USING FALLBACK")
            return fallback

        return debrief or fallback

    def generate_module_debrief(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        """Generate the Professor Module Debrief after all module activities."""

        fallback = self._fallback_module_debrief(
            knowledge,
            module_index,
            module_results,
        )

        try:
            debrief = self._generate_module_debrief_with_llm(
                knowledge,
                module_index,
                module_results,
            ).strip()
        except Exception as error:
            print("⚠️ PROFESSOR MODULE DEBRIEF FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_module_debrief(
            debrief,
            knowledge,
            module_index,
        ):
            print("⚠️ PROFESSOR MODULE DEBRIEF VALIDATION FAILED — USING FALLBACK")
            return fallback

        return debrief or fallback

    def generate_homework_recommendation(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        """Generate one Professor Homework recommendation after a module debrief."""

        fallback = self._fallback_homework_recommendation(
            knowledge,
            module_index,
            module_results,
        )

        try:
            homework = self._generate_homework_recommendation_with_llm(
                knowledge,
                module_index,
                module_results,
            ).strip()
        except Exception as error:
            print("⚠️ PROFESSOR HOMEWORK FALLBACK:", repr(error))
            return fallback

        module_debrief = self._activity_result_value(module_results, "professor_debrief") or ""

        if not self.validator.validate_homework_recommendation(
            homework,
            knowledge,
            module_index,
            module_debrief,
        ):
            print("⚠️ PROFESSOR HOMEWORK VALIDATION FAILED — USING FALLBACK")
            return fallback

        return homework or fallback

    def generate_module_question_answer(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
        question: str,
        conversation: Optional[list[dict[str, str]]] = None,
    ) -> str:
        """Answer an optional learner question about a completed module."""

        fallback = self._fallback_module_question_answer(
            knowledge,
            module_index,
            question,
        )

        try:
            answer = self._generate_module_question_answer_with_llm(
                knowledge,
                module_index,
                module_results,
                question,
                conversation or [],
            ).strip()
        except Exception as error:
            print("⚠️ PROFESSOR MODULE QUESTION FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_module_question_answer(
            answer,
            knowledge,
            module_index,
        ):
            print("⚠️ PROFESSOR MODULE QUESTION VALIDATION FAILED — USING FALLBACK")
            return fallback

        return answer or fallback

    def generate_study_plan_debrief(
        self,
        knowledge: ProfessorKnowledge,
        study_plan_results: Any,
    ) -> str:
        """Generate the Professor Study Plan Debrief after plan completion."""

        fallback = self._fallback_study_plan_debrief(
            knowledge,
            study_plan_results,
        )

        try:
            debrief = self._generate_study_plan_debrief_with_llm(
                knowledge,
                study_plan_results,
            ).strip()
        except Exception as error:
            print("⚠️ PROFESSOR STUDY PLAN DEBRIEF FALLBACK:", repr(error))
            return fallback

        if not self.validator.validate_study_plan_debrief(
            debrief,
            knowledge,
        ):
            print("⚠️ PROFESSOR STUDY PLAN DEBRIEF VALIDATION FAILED — USING FALLBACK")
            return fallback

        return debrief or fallback

    def _generate_with_llm(self, knowledge: ProfessorKnowledge) -> str:
        prompt = self._build_study_plan_briefing_prompt(knowledge)
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "briefing")

    def _generate_daily_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        prompt = self._build_daily_briefing_prompt(knowledge, module_index)
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "briefing")

    def _generate_module_objective_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        prompt = self._build_module_objective_prompt(knowledge, module_index)
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "objective")

    def _generate_activity_debrief_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_result: Any,
    ) -> str:
        prompt = self._build_activity_debrief_prompt(
            knowledge,
            module_index,
            activity_result,
        )
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "debrief")

    def _generate_module_debrief_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        prompt = self._build_module_debrief_prompt(
            knowledge,
            module_index,
            module_results,
        )
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "debrief")

    def _generate_homework_recommendation_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        prompt = self._build_homework_recommendation_prompt(
            knowledge,
            module_index,
            module_results,
        )
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "homework")

    def _generate_module_question_answer_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
        question: str,
        conversation: list[dict[str, str]],
    ) -> str:
        prompt = self._build_module_question_prompt(
            knowledge,
            module_index,
            module_results,
            question,
            conversation,
        )
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "answer")

    def _generate_study_plan_debrief_with_llm(
        self,
        knowledge: ProfessorKnowledge,
        study_plan_results: Any,
    ) -> str:
        prompt = self._build_study_plan_debrief_prompt(
            knowledge,
            study_plan_results,
        )
        raw_output = self._generate_raw_output(prompt)
        return self._extract_json_field(raw_output, "debrief")

    def _generate_raw_output(self, prompt: str) -> str:
        return (
            self.llm_generate(prompt)
            if self.llm_generate
            else self._default_llm_generate(prompt)
        )

    def _build_study_plan_briefing_prompt(self, knowledge: ProfessorKnowledge) -> str:
        payload = {
            "communication_type": COMMUNICATION_TYPE_STUDY_PLAN_BRIEFING,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
        }

        return f"""
You are writing the Study Plan Briefing as the Professor.

COMMUNICATION GOAL:
Interpret the educational reasoning behind the Study Plan. Do not describe
what was generated.

The student can already see the modules, categories, activities, counts,
durations, and sequence on screen. Your role is to explain the educational
reasoning behind the plan, not to repeat what is visible.

ProfessorKnowledge.teaching_contexts contains deterministic educational context
for the modules. Use it as the primary source for conceptual interpretation,
learning progression, expected mastery, and activity rationale. Do not rebuild
those concepts independently when TeachingContext already provides them.

Write as an experienced university professor introducing the learning path.
The student should feel guided by a teacher, not informed by software.

Your explanation must cover these educational questions naturally:
1. WHY WE START HERE:
   Explain why this is the correct starting point, what foundation it builds,
   and why beginning elsewhere would be less effective. Do not list
   categories.
2. STUDY PLAN OBJECTIVE:
   Explain what the student should be able to achieve by the end of this Study
   Plan. Focus on learning outcomes such as conceptual foundations, connecting
   isolated ideas, identifying weak areas, consolidating essential knowledge,
   or preparing for more advanced work. Do not merely describe activities.
3. WHAT COMES NEXT:
   Explain educational continuity. After this Study Plan, the next teaching
   decision can be based on evidence from the work: whether reinforcement is
   needed, progression is appropriate, harder material can be introduced, or
   earlier concepts require consolidation.
4. ACTIVITY REASONING:
   Explicitly explain why the chosen activity type is pedagogically appropriate
   for this Study Plan.

Every sentence must explain a deterministic educational decision that is
grounded in ProfessorKnowledge.
If another Study Plan may follow, explain it as pedagogical continuity:
remaining work receives priority and the next step continues from the evidence
created here. Never frame this as a software or generation process.

ACTIVITY GROUNDING RULE:
- If ProfessorKnowledge.activity_mix contains only quizzes, explain why quizzes
  were chosen: they provide an objective picture of current preparation. Do not
  mention a mix, combination, variety, alternation, or flashcards.
- If ProfessorKnowledge.activity_mix contains only flashcards, explain why
  flashcards were chosen: they reinforce long-term retention for concepts that
  should be stabilised. Do not mention quizzes.
- Only if both quizzes and flashcards are present may you explain the reason
  for combining activity types: some concepts need assessment while others
  benefit from reinforcement.

STYLE:
- Experienced university professor.
- Calm, concise, evidence-based, natural.
- Confident and encouraging, without empty praise.
- One coherent explanation, not a report.
- Target length: 80-140 words.
- Do not enumerate categories, topics, modules, durations, activity counts, or
  question counts.
- Avoid repeating information already visible in the UI.
- Avoid generic educational phrases unless they are directly supported by the
  actual Study Plan.
- Avoid course-brochure language such as "future applications" or "deeper
  understanding" unless the deterministic plan specifically justifies it.
- Never sound like documentation.
- Never use phrases such as "the system", "the planner", "the algorithm",
  "the application", "the generated plan", or "the generated Study Plan".

Use ONLY the provided Professor Identity and ProfessorKnowledge.
Do not add categories, activities, results, future plans, or decisions that are
not present in ProfessorKnowledge.
Do not claim activity variety unless ProfessorKnowledge actually contains more
than one activity type.
Do not invent weaknesses, strengths, previous performance, or future certainty.
Do not mention software, GPT, Planner, algorithms, implementation, prompts, or
internal codes.
Do not expose enum/code names directly.
Write entirely in ProfessorKnowledge.study_language.
Return ONLY valid JSON with this shape:
{{"briefing": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_daily_briefing_prompt(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        module_context = self._module_context(knowledge, module_index)
        payload = {
            "communication_type": COMMUNICATION_TYPE_DAILY_BRIEFING,
            "module_index": module_index,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "current_module": self._to_jsonable(module_context),
        }

        return f"""
You are writing the Daily Briefing as the Professor immediately before a study
module begins.

This is not another Study Plan briefing. It is a short conversation before
today's lesson starts.

Use only Professor Identity, ProfessorKnowledge, and current_module.
Do not inspect or invent anything else.
current_module.teaching_context is the deterministic teaching interpretation
for this module. Use it as the primary source for the conceptual focus,
progression, expected mastery, and activity rationale.

The Daily Briefing must answer four questions naturally:
1. Why are we studying this module now?
2. What should you focus on?
3. Why is today's activity type appropriate?
4. What attitude should you have while studying?

Speak directly to the learner. Use second-person language such as "you",
"your", and "you should". Avoid institutional formulations such as "the
student", "students should", or "learners should".
Do not always begin with "We are studying this module now". Vary the opening
naturally, for example: "Today we'll focus on...", "In this session you'll...",
"We'll begin by...", "This module introduces...", or "Today's work centres on...".

ACTIVITY GROUNDING RULE:
- If the current module contains only quizzes, explain that quizzes help reveal
  what is already solid and what still requires attention. Do not mention
  flashcards.
- If the current module contains only flashcards, explain that flashcards help
  strengthen active recall and long-term retention. Do not mention quizzes.
- If the current module contains both, explain that the work both checks and
  reinforces preparation.

STYLE:
- Experienced, calm, encouraging, academically rigorous.
- One coherent paragraph.
- Target length: 40-80 words.
- No bullet lists.
- Do not enumerate topics or categories.
- Do not describe visible dashboard data.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Do not invent previous mistakes, previous scores, previous sessions,
  strengths, weaknesses, or future certainty.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"briefing": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_module_objective_prompt(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        module_context = self._module_context(knowledge, module_index)
        payload = {
            "communication_type": COMMUNICATION_TYPE_MODULE_OBJECTIVE,
            "module_index": module_index,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "current_module": self._to_jsonable(module_context),
        }

        return f"""
You are writing the Module Objective as the Professor.

The objective is not a briefing and not a description of today's activities.
It must answer one question:
"What will you be able to understand or master by the end of this module?"

Use only Professor Identity, ProfessorKnowledge, and current_module.
Do not inspect or invent anything else.
current_module.teaching_context is the deterministic teaching interpretation
for this module. Use it as the primary source for the expected mastery,
conceptual focus, progression, and activity rationale.

The Module Objective must:
- express the expected learning outcome
- describe the understanding, reasoning ability, or mastery you should develop
- be grounded in the module categories, module topics, and activity type
- address the learner directly in second person
- prefer formulations such as "By the end of this module, you will be able
  to...", "When you complete this module, you will be able to...", or "After
  this session, you should be able to..."
- avoid enumerating categories or topics
- avoid describing the module contents
- avoid describing visible dashboard data
- avoid institutional formulations such as "students should", "learners
  should", "the student will", or "the student should"
- avoid saying "we will study", "this module contains", "this module includes",
  or "complete today's activities"
- avoid generic objectives such as "learn more", "improve knowledge", or
  "understand better"

STYLE:
- Experienced university lecturer.
- Natural, educational, evidence-based, concrete.
- One coherent paragraph.
- Target length: 40-80 words.
- No bullet lists.
- No motivational coach language.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Do not invent previous mistakes, previous scores, previous sessions,
  strengths, weaknesses, or future certainty.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"objective": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_activity_debrief_prompt(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_result: Any,
    ) -> str:
        module_context = self._module_context(knowledge, module_index)
        activity_context = self._activity_debrief_context(
            knowledge,
            module_index,
            activity_result,
        )
        payload = {
            "communication_type": COMMUNICATION_TYPE_ACTIVITY_DEBRIEF,
            "module_index": module_index,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "current_module": self._to_jsonable(module_context),
            "activity_context": self._to_jsonable(activity_context),
        }

        return f"""
You are writing the Activity Debrief as the Professor immediately after one
learning activity has been completed.

This is not a quiz review. The detailed question-by-question review already
exists elsewhere. Your task is to interpret the educational meaning of the
activity outcome.

Use only Professor Identity, ProfessorKnowledge, current_module, and
activity_context. Do not inspect or invent anything else.
current_module.teaching_context is the deterministic teaching interpretation
for this module. activity_context contains the available runtime result.

The Activity Debrief must explain naturally:
1. What this activity demonstrates about the learner's preparation.
2. What appears solid, if the result supports that.
3. What still needs reinforcement, if the result supports that.
4. Why the next activity or continuation makes educational sense.

Speak directly to the learner. Always use second-person language such as
"you", "your", and "you should".

PERFORMANCE GROUNDING RULE:
- High performance means the answer pattern suggests stable understanding.
  Explain that the foundation can support more demanding application.
- Medium performance means the main ideas are present but distinctions still
  need reinforcement. Explain why consolidation or careful continuation helps.
- Low performance means the core ideas are not yet stable. Explain why
  strengthening the foundation should come before adding complexity.
- If no score or accuracy is available, interpret completion cautiously and
  avoid claims about strengths or weaknesses.

STYLE:
- Experienced university professor.
- Calm, encouraging but objective, concise, evidence-based.
- One coherent paragraph.
- Target length: 60-120 words.
- Do not enumerate quiz questions, categories, or topics.
- Do not repeat the score, accuracy, number of questions, or visible
  statistics.
- Do not congratulate only because of a high score.
- Do not criticize only because of a low score.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"debrief": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_module_debrief_prompt(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        module_context = self._module_context(knowledge, module_index)
        module_debrief_context = self._module_debrief_context(
            knowledge,
            module_index,
            module_results,
        )
        payload = {
            "communication_type": COMMUNICATION_TYPE_MODULE_DEBRIEF,
            "module_index": module_index,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "current_module": self._to_jsonable(module_context),
            "module_debrief_context": self._to_jsonable(module_debrief_context),
        }

        return f"""
You are writing the Module Debrief as the Professor after the final learning
activity of a module has been completed.

This is not an Activity Debrief and not a quiz review. The detailed
question-by-question review already exists elsewhere. Your task is to interpret
the educational meaning of the whole completed module.

Use only Professor Identity, ProfessorKnowledge, current_module, and
module_debrief_context. Do not inspect or invent anything else.
current_module.teaching_context is the deterministic teaching interpretation
for this module. module_debrief_context contains the available runtime module
results.

The Module Debrief must explain naturally:
1. What has been consolidated during this module.
2. What appears conceptually stable, if the results support that.
3. What may still require reinforcement, if the results support that.
4. Why the following module or next learning step naturally builds on this one.

Speak directly to the learner. Always use second-person language such as
"you", "your", and "you should".

PERFORMANCE GROUNDING RULE:
- High module performance means the module outcome suggests connected and
  reliable foundations. Explain why the following module can introduce more
  advanced reasoning without losing coherence.
- Medium module performance means the main ideas are present but some
  relationships need reinforcement. Explain how the next module can revisit
  them in a broader context.
- Low module performance means the underlying concepts still need
  consolidation. Explain that the next step should strengthen the foundation
  while gradually introducing new material.
- If no score or mastery signal is available, interpret completion cautiously
  and avoid claims about strengths or weaknesses.

STYLE:
- Experienced university professor.
- Calm, encouraging but objective, concise, evidence-based.
- One coherent paragraph.
- Target length: 80-140 words.
- Do not enumerate quiz questions, categories, or topics.
- Do not repeat the score, accuracy, activity count, question count, or visible
  dashboard statistics.
- Do not simply summarize completed activities.
- Do not congratulate based only on score.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"debrief": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_homework_recommendation_prompt(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        module_context = self._module_context(knowledge, module_index)
        homework_context = self._homework_recommendation_context(
            knowledge,
            module_index,
            module_results,
        )
        payload = {
            "communication_type": COMMUNICATION_TYPE_HOMEWORK_RECOMMENDATION,
            "module_index": module_index,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "current_module": self._to_jsonable(module_context),
            "homework_context": self._to_jsonable(homework_context),
        }

        return f"""
You are writing ONE Homework recommendation as the Professor immediately after
the Module Debrief.

This is not a new activity and not a second debrief. It is one short,
practical learning action the learner can do independently in approximately
5-15 minutes.

Use only Professor Identity, ProfessorKnowledge, current_module, and
homework_context. Do not inspect or invent anything else.
current_module.teaching_context is the deterministic teaching interpretation
for this module. homework_context contains the runtime module result and
performance level.

The Homework recommendation must:
1. Address the learner directly in second person.
2. Describe exactly ONE concrete action.
3. Take approximately 5-15 minutes.
4. Reinforce weak areas when performance is low.
5. Consolidate understanding when performance is high.
6. Be educationally meaningful and connected to the module's teaching context.

Do not simply say "review", "study again", "go over the material", or similar
generic instructions. The recommendation must tell the learner what to do and
how to do it, without enumerating categories or topics.

STYLE:
- Experienced university professor.
- Calm, concrete, concise, evidence-based.
- One coherent paragraph.
- Target length: 40-70 words.
- Do not enumerate module categories or topics.
- Do not repeat the Module Debrief.
- Do not mention scores, accuracy, activity counts, question counts, or visible
  dashboard statistics.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"homework": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_module_question_prompt(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
        question: str,
        conversation: list[dict[str, str]],
    ) -> str:
        module_context = self._module_context(knowledge, module_index)
        module_question_context = self._module_question_context(
            knowledge,
            module_index,
            module_results,
            question,
            conversation,
        )
        payload = {
            "communication_type": COMMUNICATION_TYPE_MODULE_QUESTION,
            "module_index": module_index,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "current_module": self._to_jsonable(module_context),
            "module_question_context": self._to_jsonable(module_question_context),
        }

        return f"""
You are answering a learner's optional question after a completed Study Plan
module.

This is Ask the Professor for the Module Debrief. It is not a new planning
decision, not a new activity, and not a general chatbot conversation.

Use only Professor Identity, ProfessorKnowledge, current_module, and
module_question_context. Do not inspect or invent anything else.
current_module.teaching_context is the deterministic teaching interpretation
for this module. module_question_context contains the completed module result,
Professor Debrief, Homework recommendation, short conversation history, and
the learner's latest question.

The answer must:
1. Address the learner directly in second person.
2. Answer concisely as a university professor.
3. Use only the material and runtime evidence from the completed module.
4. Explain concepts educationally, not procedurally.
5. If the question is unrelated to the completed module, answer briefly and
   gently bring the focus back to the studied material.

STYLE:
- Experienced university professor.
- Calm, clear, encouraging, academically precise.
- One or two short paragraphs.
- Target length: 60-140 words.
- Do not enumerate quiz questions, categories, or topics.
- Do not repeat scores, accuracy, question counts, or dashboard statistics.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"answer": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _build_study_plan_debrief_prompt(
        self,
        knowledge: ProfessorKnowledge,
        study_plan_results: Any,
    ) -> str:
        study_plan_debrief_context = self._study_plan_debrief_context(
            knowledge,
            study_plan_results,
        )
        payload = {
            "communication_type": COMMUNICATION_TYPE_STUDY_PLAN_DEBRIEF,
            "professor_identity": self._to_jsonable(self.identity),
            "professor_knowledge": self._to_jsonable(knowledge),
            "study_plan_debrief_context": self._to_jsonable(study_plan_debrief_context),
        }

        return f"""
You are writing the Study Plan Debrief as the Professor after the full Study
Plan has been completed.

This is not a Weekly Debrief. The Planner is not calendar-driven. Refer to the
completed Study Plan, not to a week.

This is not an Activity Debrief and not a Module Debrief. Your task is to
interpret the educational meaning of the entire completed Study Plan.

Use only Professor Identity, ProfessorKnowledge, and
study_plan_debrief_context. Do not inspect or invent anything else.
ProfessorKnowledge.teaching_contexts and study_plan_debrief_context contain the
available deterministic planning and runtime results.

The Study Plan Debrief must explain naturally:
1. What has been genuinely consolidated across the Study Plan.
2. How the learner's understanding has evolved.
3. Which conceptual areas appear stable, if the results support that.
4. Which areas may still deserve attention, if the results support that.
5. How these results will influence the next Study Plan.

The final part should prepare the learner for continuation of the learning
journey. Speak directly to the learner. Always use second-person language such
as "you", "your", and "you should".

PERFORMANCE GROUNDING RULE:
- High overall mastery means the Study Plan outcome suggests coherent and
  interconnected foundations. Explain why the next Study Plan can focus more on
  demanding application and less on establishing foundations.
- Medium overall mastery means clear progress exists, but some relationships
  still deserve consolidation. Explain why the next Study Plan should reinforce
  these concepts while extending them.
- Low overall mastery means the fundamental concepts are still developing.
  Explain why the next Study Plan should continue strengthening foundations
  before moving toward more advanced material.
- If no score or mastery signal is available, interpret completion cautiously
  and avoid claims about strengths or weaknesses.

STYLE:
- Experienced university professor.
- Reflective, encouraging but objective, concise, evidence-based.
- One or two coherent paragraphs.
- Target length: 100-180 words.
- Do not enumerate modules, categories, or topics.
- Do not repeat quiz scores, activity counts, module counts, question counts,
  or visible dashboard statistics.
- Do not simply summarize completed activities.
- Do not congratulate based only on score.
- Do not mention Planner, algorithm, GPT, AI, system, application, software,
  implementation, prompts, or internal codes.
- Write entirely in ProfessorKnowledge.study_language.

Return ONLY valid JSON with this shape:
{{"debrief": "..."}}

INPUT:
{json.dumps(payload, ensure_ascii=False, indent=2)}
"""

    def _default_llm_generate(self, prompt: str) -> str:
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")

        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not configured")

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=os.getenv("PROFESSOR_VOICE_MODEL", "gpt-4o-mini"),
            temperature=0.2,
            messages=[
                {
                    "role": "system",
                    "content": "You are the Professor voice. Explain only deterministic planning knowledge.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or ""

    def _extract_json_field(self, raw_output: str, field: str) -> str:
        content = str(raw_output or "").strip()
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return content

        return str(data.get(field, "")).strip()

    def _fallback_study_plan_briefing(self, knowledge: ProfessorKnowledge) -> str:
        if self._is_italian(knowledge.study_language):
            return self._fallback_italian(knowledge)

        return self._fallback_english(knowledge)

    def _fallback_daily_briefing(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        if self._is_italian(knowledge.study_language):
            return self._fallback_daily_italian(knowledge, module_index)

        return self._fallback_daily_english(knowledge, module_index)

    def _fallback_module_objective(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        if self._is_italian(knowledge.study_language):
            return self._fallback_module_objective_italian(knowledge, module_index)

        return self._fallback_module_objective_english(knowledge, module_index)

    def _fallback_activity_debrief(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_result: Any,
    ) -> str:
        activity_context = self._activity_debrief_context(
            knowledge,
            module_index,
            activity_result,
        )

        if self._is_italian(knowledge.study_language):
            return self._fallback_activity_debrief_italian(activity_context)

        return self._fallback_activity_debrief_english(activity_context)

    def _fallback_module_debrief(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        module_context = self._module_debrief_context(
            knowledge,
            module_index,
            module_results,
        )

        if self._is_italian(knowledge.study_language):
            return self._fallback_module_debrief_italian(module_context)

        return self._fallback_module_debrief_english(module_context)

    def _fallback_homework_recommendation(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> str:
        homework_context = self._homework_recommendation_context(
            knowledge,
            module_index,
            module_results,
        )

        if self._is_italian(knowledge.study_language):
            return self._fallback_homework_recommendation_italian(homework_context)

        return self._fallback_homework_recommendation_english(homework_context)

    def _fallback_module_question_answer(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        question: str,
    ) -> str:
        teaching_context = self._teaching_context(knowledge, module_index)
        focus = getattr(teaching_context, "expected_mastery", "") or ""
        focus = self._second_person_mastery(focus)

        if self._is_italian(knowledge.study_language):
            if focus:
                return (
                    f"La domanda è utile perché riguarda il modo in cui dovresti orientare il ragionamento dopo questo modulo. "
                    f"Il punto centrale è questo: {focus} "
                    "Se qualcosa resta incerto, torna al collegamento tra idea principale ed esempio, perché lì di solito si chiarisce la distinzione più importante."
                )

            return (
                "La domanda è utile, ma posso rispondere solo restando nel perimetro del modulo appena concluso. "
                "Concentrati sul rapporto tra ciò che ti è sembrato stabile e ciò che richiede ancora una distinzione più precisa: è da lì che il lavoro successivo diventa più efficace."
            )

        if focus:
            return (
                f"Your question is useful because it goes to how you should organise the reasoning after this module. "
                f"The central point is this: {focus} "
                "If something still feels uncertain, return to the link between the main idea and a concrete example, because that is usually where the important distinction becomes clearer."
            )

        return (
            "Your question is useful, but I can answer it only within the scope of the module you have just completed. "
            "Focus on the relationship between what felt stable and what still needs a sharper distinction; that is where the next step becomes more effective."
        )

    def _fallback_study_plan_debrief(
        self,
        knowledge: ProfessorKnowledge,
        study_plan_results: Any,
    ) -> str:
        study_plan_context = self._study_plan_debrief_context(
            knowledge,
            study_plan_results,
        )

        if self._is_italian(knowledge.study_language):
            return self._fallback_study_plan_debrief_italian(study_plan_context)

        return self._fallback_study_plan_debrief_english(study_plan_context)

    def _fallback_study_plan_debrief_english(
        self,
        study_plan_context: dict[str, Any],
    ) -> str:
        performance_level = study_plan_context.get("performance_level")
        activity_profile = study_plan_context.get("activity_profile") or {}
        has_flashcards = int(activity_profile.get("flashcards") or 0) > 0

        if performance_level == "high":
            return (
                "Throughout this Study Plan, you have developed a more coherent understanding of the underlying concepts. "
                "The knowledge you have built is now sufficiently connected to support more demanding reasoning, because the central relationships no longer stand as isolated facts. "
                "The next Study Plan can therefore focus less on establishing foundations and more on applying them in broader, more complex situations."
            )

        if performance_level == "medium":
            reinforcement_phrase = (
                "The reinforcement work has helped make several ideas more stable, although"
                if has_flashcards
                else "The work has shown clear progress, although"
            )
            return (
                f"{reinforcement_phrase} some relationships still deserve further consolidation. "
                "You have made progress in connecting the principal ideas, and that progress gives the next Study Plan a useful starting point. "
                "Rather than simply moving on, the continuation should reinforce these concepts while gradually extending them into broader contexts."
            )

        if performance_level == "low":
            return (
                "This Study Plan has highlighted that some fundamental concepts are still developing. "
                "That is a normal part of learning, and it gives us useful direction rather than a final judgement. "
                "Before you move toward more advanced material, the next Study Plan should continue strengthening these foundations so that future concepts become progressively easier to connect and integrate."
            )

        return (
            "You have completed this Study Plan, and the important result is the clearer direction it gives for what should come next. "
            "Use the ideas that now feel stable as anchors, and treat uncertainty as evidence for where reinforcement will be most useful. "
            "The next Study Plan should continue from this point, preserving continuity while helping you integrate the remaining material more securely."
        )

    def _fallback_study_plan_debrief_italian(
        self,
        study_plan_context: dict[str, Any],
    ) -> str:
        performance_level = study_plan_context.get("performance_level")
        activity_profile = study_plan_context.get("activity_profile") or {}
        has_flashcards = int(activity_profile.get("flashcards") or 0) > 0

        if performance_level == "high":
            return (
                "Nel corso di questo Piano di Studio hai sviluppato una comprensione più coerente dei concetti di base. "
                "La conoscenza che hai costruito è ora abbastanza collegata da sostenere un ragionamento più esigente, perché le relazioni centrali non restano più come fatti isolati. "
                "Il prossimo Piano di Studio potrà quindi concentrarsi meno sulla costruzione delle basi e più sulla loro applicazione in situazioni più ampie e complesse."
            )

        if performance_level == "medium":
            reinforcement_phrase = (
                "Il lavoro di rinforzo ha reso più stabili diverse idee, anche se"
                if has_flashcards
                else "Il lavoro svolto mostra un progresso chiaro, anche se"
            )
            return (
                f"{reinforcement_phrase} alcune relazioni meritano ancora consolidamento. "
                "Hai fatto progressi nel collegare le idee principali, e questo progresso offre un punto di partenza utile per il prossimo Piano di Studio. "
                "Invece di passare semplicemente oltre, la continuazione dovrebbe rinforzare questi concetti mentre li estende gradualmente a contesti più ampi."
            )

        if performance_level == "low":
            return (
                "Questo Piano di Studio ha messo in evidenza che alcuni concetti fondamentali sono ancora in costruzione. "
                "È una parte normale dell’apprendimento, e offre una direzione utile più che un giudizio definitivo. "
                "Prima di passare a materiale più avanzato, il prossimo Piano di Studio dovrebbe continuare a rafforzare queste basi perché i concetti futuri diventino progressivamente più facili da collegare e integrare."
            )

        return (
            "Hai completato questo Piano di Studio, e il risultato più importante è la direzione più chiara che offre per il passaggio successivo. "
            "Usa le idee che ora ti sembrano stabili come punti di appoggio, e considera le incertezze come indicazioni su dove il rinforzo sarà più utile. "
            "Il prossimo Piano di Studio dovrebbe proseguire da qui, mantenendo continuità e aiutandoti a integrare il materiale rimanente con maggiore sicurezza."
        )

    def _fallback_module_debrief_english(
        self,
        module_context: dict[str, Any],
    ) -> str:
        performance_level = module_context.get("performance_level")
        activity_profile = module_context.get("activity_profile") or {}
        has_flashcards = int(activity_profile.get("flashcards") or 0) > 0

        if performance_level == "high":
            return (
                "The work completed in this module suggests that your fundamental concepts are now well connected. "
                "That gives you a reliable framework for the next learning step, where the reasoning can become more demanding without losing conceptual coherence. "
                "If the following module introduces new relationships, you should use the stability built here as the reference point for judging them."
            )

        if performance_level == "medium":
            reinforcement_phrase = (
                "The reinforcement work has helped, but"
                if has_flashcards
                else "The assessment has shown that"
            )
            return (
                f"{reinforcement_phrase} you still have a few relationships that deserve closer attention. "
                "You have developed a usable understanding of the main ideas, and the next module can now revisit them in a broader context. "
                "Your task is to keep those distinctions active while gradually connecting them to new material."
            )

        if performance_level == "low":
            return (
                "This module shows that some underlying concepts still need consolidation before they can support more complex reasoning. "
                "Treat that as useful evidence rather than a setback: it tells us where your foundation should be strengthened. "
                "The following module should build from this point carefully, reinforcing the central relationships while introducing new material gradually."
            )

        return (
            "You have completed this module, and the important step now is to turn that work into orientation for what follows. "
            "Use the ideas that felt secure as anchors, and treat the uncertain ones as signals for reinforcement. "
            "The next module should build continuity from this point rather than feel like a separate block of work."
        )

    def _fallback_module_debrief_italian(
        self,
        module_context: dict[str, Any],
    ) -> str:
        performance_level = module_context.get("performance_level")
        activity_profile = module_context.get("activity_profile") or {}
        has_flashcards = int(activity_profile.get("flashcards") or 0) > 0

        if performance_level == "high":
            return (
                "Il lavoro completato in questo modulo indica che i concetti fondamentali sono ora più ben collegati. "
                "Questo ti dà una struttura affidabile per il passaggio successivo, dove il ragionamento potrà diventare più esigente senza perdere coerenza concettuale. "
                "Se il modulo seguente introduce nuove relazioni, dovresti usare la stabilità costruita qui come punto di riferimento."
            )

        if performance_level == "medium":
            reinforcement_phrase = (
                "Il lavoro di rinforzo ha aiutato, ma"
                if has_flashcards
                else "La verifica mostra che"
            )
            return (
                f"{reinforcement_phrase} alcune relazioni meritano ancora attenzione. "
                "Hai sviluppato una comprensione utilizzabile delle idee principali, e il modulo successivo potrà riprenderle in un contesto più ampio. "
                "Il tuo compito sarà mantenere attive queste distinzioni mentre le colleghi gradualmente a nuovo materiale."
            )

        if performance_level == "low":
            return (
                "Questo modulo mostra che alcuni concetti di base richiedono ancora consolidamento prima di sostenere un ragionamento più complesso. "
                "Consideralo un’indicazione utile, non un arretramento: chiarisce dove la tua base deve essere rafforzata. "
                "Il modulo successivo dovrebbe partire da qui con cautela, rinforzando le relazioni centrali e introducendo il nuovo materiale gradualmente."
            )

        return (
            "Hai completato questo modulo, e ora il passaggio importante è trasformare il lavoro svolto in orientamento per ciò che segue. "
            "Usa le idee che ti sono sembrate sicure come punti di appoggio, e considera quelle incerte come segnali di rinforzo. "
            "Il modulo successivo dovrebbe costruire continuità da qui, non apparire come un blocco separato."
        )

    def _fallback_homework_recommendation_english(
        self,
        homework_context: dict[str, Any],
    ) -> str:
        performance_level = homework_context.get("performance_level")
        teaching_context = homework_context.get("teaching_context")
        activity_profile = homework_context.get("activity_profile") or {}
        has_flashcards = int(activity_profile.get("flashcards") or 0) > 0
        focus_phrase = (
            getattr(teaching_context, "expected_mastery", "") if teaching_context else ""
        )
        focus_phrase = self._second_person_mastery(focus_phrase).strip()

        if performance_level == "high":
            return (
                "Spend ten minutes writing one compact explanation of the central reasoning from this module without looking at the material. "
                "Then compare it with your notes and add only the missing link that would make your explanation clearer."
            )

        if performance_level == "medium":
            return (
                "Choose one distinction that felt uncertain and build a brief two-column comparison from memory. "
                "In one column write the idea itself; in the other, write when you would recognise or apply it. "
                "Use the material only at the end to correct the comparison."
            )

        if performance_level == "low":
            return (
                "Take ten focused minutes to rebuild the foundation slowly: write the main idea in your own words, then add one simple example and one reason why it matters. "
                "Do not aim for completeness; aim for a clearer starting point."
            )

        if has_flashcards:
            return (
                "Before the next module, spend a short interval recalling the most important relationships without looking at the answers first. "
                "Write down the connections that come easily and mark the ones that still need a slower return."
            )

        if focus_phrase:
            return (
                f"Spend ten minutes turning the objective into a short explanation in your own words: {focus_phrase} "
                "Keep it concise, then check where your reasoning becomes vague and refine only that passage."
            )

        return (
            "Spend ten minutes reconstructing the main line of reasoning from memory. "
            "Write three connected sentences, then check the material only to identify the one point that needs the most precise correction."
        )

    def _fallback_homework_recommendation_italian(
        self,
        homework_context: dict[str, Any],
    ) -> str:
        performance_level = homework_context.get("performance_level")
        teaching_context = homework_context.get("teaching_context")
        activity_profile = homework_context.get("activity_profile") or {}
        has_flashcards = int(activity_profile.get("flashcards") or 0) > 0
        focus_phrase = (
            getattr(teaching_context, "expected_mastery", "") if teaching_context else ""
        )
        focus_phrase = self._second_person_mastery(focus_phrase).strip()

        if performance_level == "high":
            return (
                "Dedica dieci minuti a scrivere una spiegazione compatta del ragionamento centrale del modulo senza guardare il materiale. "
                "Poi confrontala con gli appunti e aggiungi soltanto il collegamento mancante che renderebbe la spiegazione più chiara."
            )

        if performance_level == "medium":
            return (
                "Scegli una distinzione rimasta incerta e costruisci a memoria un breve confronto in due colonne. "
                "In una colonna scrivi l’idea; nell’altra, quando sapresti riconoscerla o applicarla. "
                "Usa il materiale solo alla fine per correggere il confronto."
            )

        if performance_level == "low":
            return (
                "Prenditi dieci minuti per ricostruire lentamente la base: scrivi l’idea principale con parole tue, poi aggiungi un esempio semplice e una ragione per cui è importante. "
                "Non cercare completezza; cerca un punto di partenza più chiaro."
            )

        if has_flashcards:
            return (
                "Prima del prossimo modulo, dedica un breve intervallo a richiamare le relazioni più importanti senza guardare subito le risposte. "
                "Annota i collegamenti che emergono facilmente e segna quelli che richiedono un ritorno più lento."
            )

        if focus_phrase:
            return (
                f"Dedica dieci minuti a trasformare l’obiettivo in una breve spiegazione con parole tue: {focus_phrase} "
                "Tienila concisa, poi controlla dove il ragionamento diventa vago e correggi solo quel passaggio."
            )

        return (
            "Dedica dieci minuti a ricostruire a memoria la linea principale del ragionamento. "
            "Scrivi tre frasi collegate, poi usa il materiale solo per individuare il punto che richiede la correzione più precisa."
        )

    def _fallback_activity_debrief_english(
        self,
        activity_context: dict[str, Any],
    ) -> str:
        performance_level = activity_context.get("performance_level")
        activity_type = str(activity_context.get("activity_type") or "").lower()
        activity_phrase = (
            "this retrieval work"
            if "flashcard" in activity_type
            else "this activity"
        )

        if performance_level == "high":
            return (
                f"Your work in {activity_phrase} suggests that the fundamental relationships are becoming stable. "
                "That matters because secure foundations allow you to approach the next step with more precise reasoning, not simply faster answers. "
                "Continue by using the next activity to test whether that stability holds when the ideas are applied in a slightly more demanding way."
            )

        if performance_level == "medium":
            return (
                f"You have identified the main direction of {activity_phrase}, but some distinctions still need reinforcement. "
                "That is a useful result: it shows that the foundation is present, while also indicating where your attention should become more precise. "
                "The next step should consolidate those relationships before you build on them."
            )

        if performance_level == "low":
            return (
                f"Your result in {activity_phrase} indicates that the core ideas are not yet stable enough to support more complex work. "
                "This is not a failure; it tells us where the teaching should slow down. "
                "Before adding new material, you should strengthen these foundations so later concepts become easier to understand."
            )

        return (
            f"You have completed {activity_phrase}, and the useful point now is to turn that work into direction. "
            "Use what felt secure and what felt uncertain as evidence for the next step. "
            "The continuation should help you make the central relationships clearer and more stable."
        )

    def _fallback_activity_debrief_italian(
        self,
        activity_context: dict[str, Any],
    ) -> str:
        performance_level = activity_context.get("performance_level")
        activity_type = str(activity_context.get("activity_type") or "").lower()
        activity_phrase = (
            "questo lavoro di richiamo"
            if "flashcard" in activity_type
            else "questa attività"
        )

        if performance_level == "high":
            return (
                f"Il tuo lavoro in {activity_phrase} indica che le relazioni fondamentali stanno diventando più stabili. "
                "Questo è importante perché una base sicura ti permette di affrontare il passaggio successivo con un ragionamento più preciso, non solo con risposte più rapide. "
                "Prosegui usando la prossima attività per verificare se questa stabilità regge quando le idee vengono applicate in modo più esigente."
            )

        if performance_level == "medium":
            return (
                f"Hai individuato la direzione principale di {activity_phrase}, ma alcune distinzioni richiedono ancora rinforzo. "
                "È un risultato utile: mostra che la base è presente, ma indica anche dove la tua attenzione deve diventare più precisa. "
                "Il passaggio successivo dovrebbe consolidare queste relazioni prima di costruirci sopra."
            )

        if performance_level == "low":
            return (
                f"Il risultato di {activity_phrase} indica che le idee centrali non sono ancora abbastanza stabili per sostenere un lavoro più complesso. "
                "Non è un fallimento: ci dice dove l’insegnamento deve rallentare. "
                "Prima di aggiungere nuovo materiale, dovresti rafforzare queste basi perché i concetti successivi diventino più comprensibili."
            )

        return (
            f"Hai completato {activity_phrase}, e ora il punto utile è trasformare il lavoro svolto in una direzione. "
            "Usa ciò che ti è sembrato sicuro e ciò che è rimasto incerto come indicazione per il passaggio successivo. "
            "La continuazione dovrebbe aiutarti a rendere più chiare e stabili le relazioni centrali."
        )

    def _fallback_module_objective_italian(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        module_mix = self.validator._module_activity_mix(knowledge, module_index)

        if module_mix["quiz"] > 0 and module_mix["flashcards"] == 0:
            return (
                "Al termine di questo modulo dovresti saper riconoscere le distinzioni centrali di questa parte del programma, spiegare il ragionamento che le collega e individuare con maggiore precisione quali passaggi sono già solidi."
            )

        if module_mix["flashcards"] > 0 and module_mix["quiz"] == 0:
            return (
                "Al termine di questo modulo dovresti richiamare con maggiore sicurezza i concetti essenziali, collegarli senza dipendere dal materiale davanti a te e usarli come base stabile per il lavoro successivo."
            )

        if module_mix["quiz"] > 0 and module_mix["flashcards"] > 0:
            return (
                "Al termine di questo modulo dovresti saper collegare richiamo e applicazione: recuperare i concetti essenziali, usarli nel ragionamento e riconoscere quali passaggi richiedono ancora consolidamento."
            )

        return (
            "Al termine di questo modulo dovresti avere una comprensione più ordinata dei concetti centrali, saper distinguere i passaggi principali e spiegare perché sono importanti nel percorso di studio."
        )

    def _fallback_module_objective_english(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        module_mix = self.validator._module_activity_mix(knowledge, module_index)

        if module_mix["quiz"] > 0 and module_mix["flashcards"] == 0:
            return (
                "By the end of this module, you should be able to recognise the central distinctions in this part of the programme, explain the reasoning that connects them, and identify more precisely which ideas are already secure."
            )

        if module_mix["flashcards"] > 0 and module_mix["quiz"] == 0:
            return (
                "By the end of this module, you should recall the essential concepts with greater confidence, connect them without relying on the material in front of you, and use them as a stable base for later work."
            )

        if module_mix["quiz"] > 0 and module_mix["flashcards"] > 0:
            return (
                "By the end of this module, you should connect recall with application: retrieve the essential concepts, use them in reasoning, and recognise which steps still need consolidation."
            )

        return (
            "By the end of this module, you should have a more ordered understanding of the central concepts, distinguish the main steps, and explain why they matter within the study path."
        )

    def _fallback_daily_italian(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        module_mix = self.validator._module_activity_mix(knowledge, module_index)

        if module_mix["quiz"] > 0 and module_mix["flashcards"] == 0:
            return (
                "Oggi lavoriamo per capire quanto sono solide le basi di questo passaggio. "
                "Il quiz non serve solo a rispondere correttamente: serve a distinguere ciò che sai già usare da ciò che richiede attenzione. "
                "Procedi con calma e considera ogni esitazione come un indizio utile."
            )

        if module_mix["flashcards"] > 0 and module_mix["quiz"] == 0:
            return (
                "Oggi l’obiettivo è rendere più pronti e stabili i concetti che stai consolidando. "
                "Prima di scoprire ogni risposta, prova a richiamarla attivamente: il valore dell’esercizio sta nello sforzo di recupero, non nella velocità."
            )

        if module_mix["quiz"] > 0 and module_mix["flashcards"] > 0:
            return (
                "Oggi combiniamo verifica e consolidamento. "
                "Alcuni passaggi devono essere messi alla prova, altri resi più stabili attraverso il richiamo. "
                "Affronta il modulo con attenzione: l’obiettivo è capire meglio come procede la preparazione."
            )

        return (
            "Oggi lavoriamo su un passaggio circoscritto del percorso. "
            "Concentrati sul ragionamento, non sulla velocità: ciò che conta è capire quali idee sono già chiare e quali meritano un ritorno più attento."
        )

    def _fallback_daily_english(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> str:
        teaching_context = self._teaching_context(knowledge, module_index)

        if teaching_context:
            return (
                f"In this session you'll focus on a contained step in the learning path. "
                f"{teaching_context.activity_rationale} "
                "Move carefully and treat each uncertainty as useful evidence for the next teaching decision."
            )

        module_mix = self.validator._module_activity_mix(knowledge, module_index)

        if module_mix["quiz"] > 0 and module_mix["flashcards"] == 0:
            return (
                "Today we'll focus on how solid this part of your preparation really is. "
                "The quiz is not only about answering correctly: it helps separate what you can already use from what still needs attention. "
                "Move carefully and treat each hesitation as useful evidence."
            )

        if module_mix["flashcards"] > 0 and module_mix["quiz"] == 0:
            return (
                "Today’s work is meant to make the concepts you are consolidating more stable and easier to recall. "
                "Before revealing each answer, try to retrieve it actively: the value of the exercise is in the act of recall, not in speed."
            )

        if module_mix["quiz"] > 0 and module_mix["flashcards"] > 0:
            return (
                "Today we combine assessment and consolidation. "
                "Some ideas need to be tested, while others benefit from active recall. "
                "Approach the module carefully: the objective is to understand how your preparation is developing."
            )

        return (
            "Today we focus on one contained step in the path. "
            "Give priority to reasoning rather than speed: what matters is seeing which ideas are already clear and which ones deserve more careful attention."
        )

    def _fallback_italian(self, knowledge: ProfessorKnowledge) -> str:
        parts = []

        parts.append(
            "Iniziamo dal punto che offre la base più utile per orientare il percorso: prima di avanzare, è importante capire quali passaggi sostengono davvero il resto del programma."
        )

        if knowledge.activity_mix.quiz_count > 0 and knowledge.activity_mix.flashcard_count == 0:
            parts.append(
                "I quiz sono la scelta più adatta perché danno un quadro oggettivo della preparazione attuale, distinguendo ciò che è già stabile da ciò che richiederà un intervento mirato."
            )
        elif knowledge.activity_mix.flashcard_count > 0 and knowledge.activity_mix.quiz_count == 0:
            parts.append(
                "Le flashcard sono la scelta più adatta perché l’obiettivo è rendere più pronta e duratura la memoria dei concetti essenziali prima di aumentare il carico di verifica."
            )
        elif knowledge.activity_mix.quiz_count > 0 and knowledge.activity_mix.flashcard_count > 0:
            parts.append(
                "La combinazione tra verifica e consolidamento serve a distinguere ciò che sai già applicare da ciò che deve ancora diventare stabile."
            )

        parts.append(
            "Alla fine dovresti avere una base più leggibile: non una certezza definitiva, ma indicazioni sufficienti per decidere se rinforzare, procedere o tornare sui passaggi meno sicuri."
        )

        if knowledge.additional_modules_remain:
            parts.append(
                "Ciò che resta fuori non viene ignorato: sarà ripreso con continuità quando avremo evidenze migliori su cui fondare la prossima scelta didattica."
            )

        return " ".join(parts)

    def _fallback_english(self, knowledge: ProfessorKnowledge) -> str:
        parts = []

        first_context = self._teaching_context(knowledge, 1)

        parts.append(
            first_context.learning_progression
            if first_context
            else "We begin where the work can build the most useful foundation: before moving further, we need to see which ideas can support the rest of the programme and which ones still need attention."
        )

        if knowledge.activity_mix.quiz_count > 0 and knowledge.activity_mix.flashcard_count == 0:
            parts.append(
                "Quizzes are the right instrument here because they give an objective picture of your current preparation, separating what is already stable from what will need targeted reinforcement."
            )
        elif knowledge.activity_mix.flashcard_count > 0 and knowledge.activity_mix.quiz_count == 0:
            parts.append(
                "Flashcards are the right instrument here because the immediate goal is to make essential concepts easier to recall before increasing the pressure of assessment."
            )
        elif knowledge.activity_mix.quiz_count > 0 and knowledge.activity_mix.flashcard_count > 0:
            parts.append(
                "Combining assessment and consolidation helps distinguish what you can already apply from what still needs to become stable."
            )

        parts.append(
            self._second_person_mastery(first_context.expected_mastery)
            if first_context
            else "By the end, the goal is not a final judgement, but a clearer basis for the next teaching decision: reinforce, progress, or return to concepts that remain uncertain."
        )

        if knowledge.additional_modules_remain:
            parts.append(
                "The material left outside this path is not being ignored; it can be taken up with continuity once the evidence from this work shows where the next priority should be."
            )

        return " ".join(parts)

    def _format_list(self, values, language: str) -> str:
        items = [str(value) for value in values if value]

        if not items:
            return ""

        if len(items) == 1:
            return items[0]

        conjunction = " e " if language == "Italian" else " and "
        return ", ".join(items[:-1]) + conjunction + items[-1]

    def _is_italian(self, language: Optional[str]) -> bool:
        return str(language or "").strip().lower().startswith("italian")

    def _second_person_mastery(self, text: str) -> str:
        value = str(text or "").strip()

        if not value:
            return value

        replacements = (
            ("the student should", "you should"),
            ("The student should", "You should"),
            ("a successful student should", "you should"),
            ("A successful student should", "You should"),
            ("the student will", "you will"),
            ("The student will", "You will"),
            ("students should", "you should"),
            ("Students should", "You should"),
            ("learners should", "you should"),
            ("Learners should", "You should"),
        )

        for source, target in replacements:
            value = value.replace(source, target)

        return value

    def _activity_debrief_context(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_result: Any,
    ) -> dict[str, Any]:
        accuracy = self._normalized_accuracy(activity_result)
        score = self._activity_result_value(activity_result, "score")
        correct = self._activity_result_value(activity_result, "correct")
        total = (
            self._activity_result_value(activity_result, "total")
            or self._activity_result_value(activity_result, "questions")
            or self._activity_result_value(activity_result, "num_questions")
            or self._activity_result_value(activity_result, "cards")
            or self._activity_result_value(activity_result, "num_cards")
        )
        activity_type = (
            self._activity_result_value(activity_result, "activity_type")
            or self._activity_result_value(activity_result, "type")
            or self._first_module_activity_type(knowledge, module_index)
        )

        return {
            "activity_type": activity_type,
            "completed": bool(
                self._activity_result_value(activity_result, "completed")
                if self._activity_result_value(activity_result, "completed") is not None
                else True
            ),
            "accuracy": accuracy,
            "score": score,
            "correct": correct,
            "total": total,
            "performance_level": self._performance_level(accuracy),
            "teaching_context": self._teaching_context(knowledge, module_index),
            "module_objective": self._second_person_mastery(
                getattr(self._teaching_context(knowledge, module_index), "expected_mastery", "") or ""
            ),
            "current_activity_strategy": self._first_module_activity_strategy(
                knowledge,
                module_index,
                activity_type,
            ),
        }

    def _module_debrief_context(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> dict[str, Any]:
        activity_results = self._module_activity_results(module_results)
        activity_contexts = tuple(
            self._activity_debrief_context(
                knowledge,
                module_index,
                activity_result,
            )
            for activity_result in activity_results
        )
        accuracies = [
            context["accuracy"]
            for context in activity_contexts
            if context.get("accuracy") is not None
        ]
        overall_accuracy = (
            sum(accuracies) / len(accuracies)
            if accuracies
            else self._normalized_accuracy(module_results)
        )
        activity_profile = self._module_activity_profile(
            knowledge,
            module_index,
            activity_contexts,
        )

        return {
            "completed_activities": activity_contexts,
            "activity_profile": activity_profile,
            "overall_accuracy": overall_accuracy,
            "performance_level": self._performance_level(overall_accuracy),
            "mastery_level": self._performance_level(overall_accuracy),
            "teaching_context": self._teaching_context(knowledge, module_index),
            "module_objective": self._second_person_mastery(
                getattr(self._teaching_context(knowledge, module_index), "expected_mastery", "") or ""
            ),
            "current_activity_strategy": self._module_strategy(knowledge, module_index),
            "next_module_teaching_context": self._teaching_context(
                knowledge,
                module_index + 1,
            ),
        }

    def _homework_recommendation_context(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
    ) -> dict[str, Any]:
        module_context = self._module_debrief_context(
            knowledge,
            module_index,
            module_results,
        )

        return {
            "performance_level": module_context.get("performance_level"),
            "mastery_level": module_context.get("mastery_level"),
            "overall_accuracy": module_context.get("overall_accuracy"),
            "activity_profile": module_context.get("activity_profile"),
            "completed_activities": module_context.get("completed_activities"),
            "teaching_context": module_context.get("teaching_context"),
            "module_objective": module_context.get("module_objective"),
            "current_activity_strategy": module_context.get("current_activity_strategy"),
        }

    def _module_question_context(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        module_results: Any,
        question: str,
        conversation: list[dict[str, str]],
    ) -> dict[str, Any]:
        module_context = self._module_debrief_context(
            knowledge,
            module_index,
            module_results,
        )
        safe_history = tuple(
            {
                "role": str(item.get("role") or "")[:20],
                "content": str(item.get("content") or "")[:1200],
            }
            for item in (conversation or [])[-8:]
            if isinstance(item, dict) and str(item.get("content") or "").strip()
        )

        return {
            "learner_question": str(question or "").strip(),
            "conversation": safe_history,
            "module_results": module_context,
            "professor_debrief": self._activity_result_value(
                module_results,
                "professor_debrief",
            ) or "",
            "homework_recommendation": self._activity_result_value(
                module_results,
                "homework_recommendation",
            ) or "",
            "module_categories": self.validator._module_categories(
                knowledge,
                module_index,
            ),
            "module_topics": self.validator._module_topics(
                knowledge,
                module_index,
            ),
            "teaching_context": module_context.get("teaching_context"),
            "module_objective": module_context.get("module_objective"),
            "activity_profile": module_context.get("activity_profile"),
            "performance_level": module_context.get("performance_level"),
        }

    def _study_plan_debrief_context(
        self,
        knowledge: ProfessorKnowledge,
        study_plan_results: Any,
    ) -> dict[str, Any]:
        module_results = self._study_plan_module_results(study_plan_results)
        module_contexts = tuple(
            self._module_debrief_context(
                knowledge,
                module_index,
                module_result,
            )
            for module_index, module_result in enumerate(module_results, start=1)
        )
        accuracies = [
            context["overall_accuracy"]
            for context in module_contexts
            if context.get("overall_accuracy") is not None
        ]
        overall_accuracy = (
            sum(accuracies) / len(accuracies)
            if accuracies
            else self._normalized_accuracy(study_plan_results)
        )
        activity_profile = self._study_plan_activity_profile(knowledge, module_contexts)

        return {
            "completed_modules": module_contexts,
            "overall_accuracy": overall_accuracy,
            "performance_level": self._performance_level(overall_accuracy),
            "mastery_level": self._performance_level(overall_accuracy),
            "activity_profile": activity_profile,
            "teaching_contexts": getattr(knowledge, "teaching_contexts", ()) or (),
            "learning_progression": tuple(
                context.learning_progression
                for context in getattr(knowledge, "teaching_contexts", ()) or ()
                if getattr(context, "learning_progression", None)
            ),
            "expected_mastery": tuple(
                self._second_person_mastery(context.expected_mastery)
                for context in getattr(knowledge, "teaching_contexts", ()) or ()
                if getattr(context, "expected_mastery", None)
            ),
            "additional_modules_remain": knowledge.additional_modules_remain,
            "remaining_categories": knowledge.remaining_categories,
        }

    def _study_plan_module_results(self, study_plan_results: Any) -> tuple[Any, ...]:
        if study_plan_results is None:
            return ()

        if isinstance(study_plan_results, dict):
            for key in ("module_results", "modules", "daily_plans", "results", "completed_modules"):
                value = study_plan_results.get(key)
                if isinstance(value, (list, tuple)):
                    return tuple(value)

            return (study_plan_results,)

        if isinstance(study_plan_results, (list, tuple)):
            return tuple(study_plan_results)

        for key in ("module_results", "modules", "daily_plans", "results", "completed_modules"):
            value = getattr(study_plan_results, key, None)
            if isinstance(value, (list, tuple)):
                return tuple(value)

        return (study_plan_results,)

    def _study_plan_activity_profile(
        self,
        knowledge: ProfessorKnowledge,
        module_contexts: tuple[dict[str, Any], ...],
    ) -> dict[str, int]:
        quiz_count = 0
        flashcard_count = 0

        for context in module_contexts:
            activity_profile = context.get("activity_profile") or {}
            quiz_count += int(activity_profile.get("quiz") or 0)
            flashcard_count += int(activity_profile.get("flashcards") or 0)

        if quiz_count or flashcard_count:
            return {
                "quiz": quiz_count,
                "flashcards": flashcard_count,
                "mixed": int(quiz_count > 0 and flashcard_count > 0),
            }

        return {
            "quiz": knowledge.activity_mix.quiz_count,
            "flashcards": knowledge.activity_mix.flashcard_count,
            "mixed": int(
                knowledge.activity_mix.mixed_count > 0
                or (
                    knowledge.activity_mix.quiz_count > 0
                    and knowledge.activity_mix.flashcard_count > 0
                )
            ),
        }

    def _module_activity_results(self, module_results: Any) -> tuple[Any, ...]:
        if module_results is None:
            return ()

        if isinstance(module_results, dict):
            for key in ("activity_results", "activities", "results", "completed_activities"):
                value = module_results.get(key)
                if isinstance(value, (list, tuple)):
                    return tuple(value)

            return (module_results,)

        if isinstance(module_results, (list, tuple)):
            return tuple(module_results)

        value = getattr(module_results, "activity_results", None)
        if isinstance(value, (list, tuple)):
            return tuple(value)

        value = getattr(module_results, "activities", None)
        if isinstance(value, (list, tuple)):
            return tuple(value)

        return (module_results,)

    def _module_activity_profile(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_contexts: tuple[dict[str, Any], ...],
    ) -> dict[str, int]:
        quiz_count = 0
        flashcard_count = 0

        for context in activity_contexts:
            activity_type = str(context.get("activity_type") or "").upper()
            if "QUIZ" in activity_type:
                quiz_count += 1
            if "FLASHCARD" in activity_type:
                flashcard_count += 1

        if quiz_count or flashcard_count:
            return {
                "quiz": quiz_count,
                "flashcards": flashcard_count,
                "mixed": int(quiz_count > 0 and flashcard_count > 0),
            }

        module_mix = self.validator._module_activity_mix(knowledge, module_index)

        return {
            "quiz": module_mix.get("quiz", 0),
            "flashcards": module_mix.get("flashcards", 0),
            "mixed": int(
                module_mix.get("quiz", 0) > 0
                and module_mix.get("flashcards", 0) > 0
            ),
        }

    def _activity_result_value(self, activity_result: Any, key: str) -> Any:
        if activity_result is None:
            return None

        if isinstance(activity_result, dict):
            return activity_result.get(key)

        return getattr(activity_result, key, None)

    def _normalized_accuracy(self, activity_result: Any) -> Optional[float]:
        raw_accuracy = (
            self._activity_result_value(activity_result, "accuracy")
            or self._activity_result_value(activity_result, "percentage")
        )

        if raw_accuracy is not None:
            try:
                accuracy = float(raw_accuracy)
            except (TypeError, ValueError):
                return None

            return accuracy / 100 if accuracy > 1 else accuracy

        correct = self._activity_result_value(activity_result, "correct")
        total = (
            self._activity_result_value(activity_result, "total")
            or self._activity_result_value(activity_result, "questions")
            or self._activity_result_value(activity_result, "num_questions")
            or self._activity_result_value(activity_result, "cards")
            or self._activity_result_value(activity_result, "num_cards")
        )

        try:
            correct_number = float(correct)
            total_number = float(total)
        except (TypeError, ValueError):
            return None

        if total_number <= 0:
            return None

        return correct_number / total_number

    def _performance_level(self, accuracy: Optional[float]) -> str:
        if accuracy is None:
            return "unknown"

        if accuracy >= 0.8:
            return "high"

        if accuracy >= 0.5:
            return "medium"

        return "low"

    def _first_module_activity_type(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> Optional[str]:
        for activity in knowledge.activity_sizes:
            if activity.module_index == module_index:
                return activity.activity_type

        module_strategy = next(
            (
                strategy
                for strategy in knowledge.module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )

        if module_strategy and module_strategy.activities:
            return module_strategy.activities[0].activity_type

        return None

    def _first_module_activity_strategy(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
        activity_type: Any,
    ) -> Any:
        module_strategy = next(
            (
                strategy
                for strategy in knowledge.module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )

        if not module_strategy:
            return None

        normalized_type = str(activity_type or "").upper()

        for activity in module_strategy.activities:
            if not normalized_type or normalized_type in str(activity.activity_type or "").upper():
                return activity

        return module_strategy.activities[0] if module_strategy.activities else None

    def _module_strategy(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> Any:
        return next(
            (
                strategy
                for strategy in knowledge.module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )

    def _module_context(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> dict[str, Any]:
        module_strategy = next(
            (
                strategy
                for strategy in knowledge.module_strategies
                if strategy.module_index == module_index
            ),
            None,
        )
        activity_sizes = tuple(
            activity
            for activity in knowledge.activity_sizes
            if activity.module_index == module_index
        )
        categories = tuple(
            dict.fromkeys(
                [
                    str(activity.category)
                    for activity in activity_sizes
                    if activity.category
                ]
                + (
                    [
                        str(activity.category)
                        for activity in module_strategy.activities
                        if activity.category
                    ]
                    if module_strategy
                    else []
                )
            )
        )

        return {
            "module_index": module_index,
            "categories": categories,
            "selected_topics_by_category": {
                category: knowledge.selected_topics_by_category.get(category, ())
                for category in categories
            },
            "activity_sizes": activity_sizes,
            "module_strategy": module_strategy,
            "activity_mix": self.validator._module_activity_mix(
                knowledge,
                module_index,
            ),
            "teaching_context": self._teaching_context(knowledge, module_index),
            "planning_constraints": knowledge.planning_constraints,
            "study_language": knowledge.study_language,
        }

    def _teaching_context(
        self,
        knowledge: ProfessorKnowledge,
        module_index: int,
    ) -> Any:
        return next(
            (
                context
                for context in getattr(knowledge, "teaching_contexts", ()) or ()
                if context.module_index == module_index
            ),
            None,
        )

    def _to_jsonable(self, value: Any) -> Any:
        if is_dataclass(value):
            return {
                key: self._to_jsonable(item)
                for key, item in asdict(value).items()
            }

        if isinstance(value, dict):
            return {
                str(key): self._to_jsonable(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [self._to_jsonable(item) for item in value]

        return value
