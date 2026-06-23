import unittest
from unittest.mock import MagicMock, patch

import main


class HardQuizDifficultyTests(unittest.TestCase):
    def _valid_explanation(self, connector):
        return (
            "The first supported proposition establishes one mechanism, "
            f"{connector} the second proposition establishes a different "
            "effect, and applying both facts identifies the correct "
            "interpretation without adding external information."
        )

    def _evaluate_phrase(self, question, connector="therefore"):
        return main.evaluate_hard_question_reasoning({
            "question": question,
            "explanation": self._valid_explanation(connector),
            "source_chunk_ids": ["1", "2"],
        })

    def test_detects_distinction_between(self):
        result = self._evaluate_phrase(
            "Which distinction between mechanism A and mechanism B explains "
            "their different explicitly supported outcomes in the material?"
        )

        self.assertTrue(result["valid"])

    def test_detects_differ_from(self):
        result = self._evaluate_phrase(
            "How does mechanism X differ from mechanism Y when both supported "
            "effects and their source conditions are considered together?"
        )

        self.assertTrue(result["valid"])

    def test_detects_synthesizes_concepts(self):
        result = self._evaluate_phrase(
            "Which interpretation synthesizes the concepts of mechanism A "
            "and mechanism B while preserving both source-supported "
            "relationships and their distinct implications?"
        )

        self.assertTrue(result["valid"])

    def test_detects_compare_to(self):
        result = self._evaluate_phrase(
            "How does mechanism X compare to mechanism Y when the two "
            "source-supported processes and their explicitly stated effects "
            "are evaluated together?"
        )

        self.assertTrue(result["valid"])

    def test_accepts_thus_as_explanation_connector(self):
        result = self._evaluate_phrase(
            "Considering both supported mechanisms, which distinction best "
            "explains why their explicitly described outcomes differ under "
            "the source conditions?",
            connector="thus",
        )

        self.assertTrue(result["valid"])

    def test_accepts_consequently_as_explanation_connector(self):
        result = self._evaluate_phrase(
            "Considering both supported mechanisms, which distinction best "
            "explains why their explicitly described outcomes differ under "
            "the source conditions?",
            connector="consequently",
        )

        self.assertTrue(result["valid"])

    def test_accepts_long_explanation_without_configured_connector(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "Which distinction between mechanism A and mechanism B "
                "accounts for their different outcomes under the supplied "
                "source conditions?"
            ),
            "explanation": (
                "Mechanism A changes the first supported process. Mechanism "
                "B changes the second supported process. Their distinct "
                "effects establish the answer from the supplied evidence."
            ),
            "source_chunk_ids": ["1", "2"],
        })

        self.assertTrue(result["valid"])
        self.assertFalse(result["explanation_connector_present"])

    def test_accepts_long_non_recall_stem_without_relational_marker(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "How do the supplied facts about mechanism A and mechanism B "
                "support the resulting conclusion described in the source "
                "material?"
            ),
            "explanation": (
                "Mechanism A establishes the initial condition. Mechanism B "
                "establishes the resulting effect. Reading those supported "
                "facts together identifies the conclusion without external "
                "information."
            ),
            "source_chunk_ids": ["1", "2"],
        })

        self.assertTrue(result["valid"])
        self.assertFalse(result["relational_reasoning"])

    def test_uses_good_explanation_when_explanation_long_is_bad(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "Considering both supported mechanisms, which distinction "
                "best explains why their explicitly described outcomes "
                "differ under the source conditions?"
            ),
            "explanation": self._valid_explanation("therefore"),
            "explanation_long": "Too short and weak.",
            "source_chunk_ids": ["1", "2"],
        })

        self.assertTrue(result["valid"])

    def test_generator_contract_uses_shared_specification(self):
        contract = main.render_hard_quiz_specification()

        self.assertIn(
            str(main.HARD_QUIZ_SPEC["minimum_supported_propositions"]),
            contract,
        )
        self.assertIn(
            str(main.HARD_QUIZ_SPEC["minimum_stem_words"]),
            contract,
        )
        self.assertIn(
            str(main.HARD_QUIZ_SPEC["minimum_explanation_words"]),
            contract,
        )

        for mode in main.HARD_QUIZ_SPEC["reasoning_modes"]:
            self.assertIn(mode, contract)

        for forbidden_form in main.HARD_QUIZ_SPEC[
            "forbidden_question_forms"
        ]:
            self.assertIn(forbidden_form, contract)

        self.assertIn(
            "Comparison and distinction are intermediate reasoning "
            "operations",
            contract,
        )
        self.assertIn(
            "which implication follows",
            contract,
        )

    def test_validator_uses_shared_word_thresholds(self):
        custom_spec = {
            **main.HARD_QUIZ_SPEC,
            "minimum_stem_words": 100,
            "minimum_explanation_words": 100,
        }
        result = main.evaluate_hard_question_reasoning(
            {
                "question": (
                    "Considering both supported mechanisms, which distinction "
                    "best explains why their explicitly described outcomes "
                    "differ under the source conditions?"
                ),
                "explanation": (
                    "The first mechanism changes the sequence, whereas the "
                    "second terminates it; therefore their supported outcomes "
                    "differ."
                ),
                "source_chunk_ids": ["1", "2"],
            },
            spec=custom_spec,
        )

        self.assertFalse(result["valid"])
        self.assertIn(
            "stem_too_short_for_hard_reasoning",
            result["reasons"],
        )
        self.assertIn(
            "explanation_does_not_show_two_step_reasoning",
            result["reasons"],
        )

    def test_requested_count_invariant_accepts_exact_count(self):
        main.validate_quiz_requested_count(
            2,
            [{"question": "A"}, {"question": "B"}],
        )

    def test_requested_count_invariant_rejects_underfill(self):
        with self.assertRaisesRegex(
            ValueError,
            "requested 10, accepted 2",
        ):
            main.validate_quiz_requested_count(
                10,
                [{"question": "A"}, {"question": "B"}],
            )

    def test_single_source_topic_attribution(self):
        result = main.resolve_quiz_question_topic(
            {"source_chunk_ids": ["101"], "topic": None},
            {"101": "Canonical A"},
            {"Canonical A"},
        )

        self.assertEqual(result["topic"], "Canonical A")
        self.assertEqual(result["resolution"], "source_same_topic")

    def test_multi_source_same_topic_attribution(self):
        result = main.resolve_quiz_question_topic(
            {"source_chunk_ids": ["101", "102"]},
            {"101": "Canonical A", "102": "Canonical A"},
            {"Canonical A"},
        )

        self.assertEqual(result["topic"], "Canonical A")
        self.assertEqual(result["resolution"], "source_same_topic")

    def test_multi_source_majority_topic_attribution(self):
        result = main.resolve_quiz_question_topic(
            {"source_chunk_ids": ["101", "102", "103"]},
            {
                "101": "Canonical A",
                "102": "Canonical B",
                "103": "Canonical A",
            },
            {"Canonical A", "Canonical B"},
        )

        self.assertEqual(result["topic"], "Canonical A")
        self.assertEqual(result["resolution"], "source_majority_topic")

    def test_multi_source_tie_uses_first_valid_source(self):
        result = main.resolve_quiz_question_topic(
            {"source_chunk_ids": ["101", "102"]},
            {"101": "Canonical B", "102": "Canonical A"},
            {"Canonical A", "Canonical B"},
        )

        self.assertEqual(result["topic"], "Canonical B")
        self.assertEqual(
            result["resolution"],
            "source_first_topic_tiebreak",
        )

    def test_missing_model_topic_uses_valid_source_topic(self):
        result = main.resolve_quiz_question_topic(
            {"source_chunk_ids": ["101"]},
            {"101": "Canonical A"},
            {"Canonical A"},
        )

        self.assertEqual(result["topic"], "Canonical A")

    def test_incorrect_model_topic_cannot_override_source_topic(self):
        result = main.resolve_quiz_question_topic(
            {
                "source_chunk_ids": ["101"],
                "topic": "Canonical B",
            },
            {"101": "Canonical A"},
            {"Canonical A", "Canonical B"},
        )

        self.assertEqual(result["topic"], "Canonical A")

    def test_noncanonical_model_topic_is_ignored(self):
        result = main.resolve_quiz_question_topic(
            {
                "source_chunk_ids": [],
                "topic": "Invented Topic",
            },
            {},
            {"Canonical A"},
        )

        self.assertIsNone(result["topic"])
        self.assertEqual(result["resolution"], "unattributed")

    def test_canonical_model_topic_is_used_without_source(self):
        result = main.resolve_quiz_question_topic(
            {
                "source_chunk_ids": [],
                "topic": "Canonical A",
            },
            {},
            {"Canonical A"},
        )

        self.assertEqual(result["topic"], "Canonical A")
        self.assertEqual(
            result["resolution"],
            "canonical_model_fallback",
        )

    def test_selected_topic_scope_attribution_still_works(self):
        result = main.resolve_quiz_question_topic(
            {
                "source_chunk_ids": ["selected-1", "selected-2"],
                "topic": "Canonical A",
            },
            {
                "selected-1": "Canonical A",
                "selected-2": "Canonical A",
            },
            {"Canonical A"},
        )

        self.assertEqual(result["topic"], "Canonical A")

    def test_whole_project_scope_attribution_still_works(self):
        result = main.resolve_quiz_question_topic(
            {
                "source_chunk_ids": ["project-2", "project-1"],
                "topic": "Invented Combined Topic",
            },
            {
                "project-1": "Canonical A",
                "project-2": "Canonical B",
            },
            {"Canonical A", "Canonical B"},
        )

        self.assertEqual(result["topic"], "Canonical B")

    def test_quiz_reload_returns_persisted_topic(self):
        result_proxy = MagicMock()
        result_proxy.fetchall.return_value = [
            (
                "Question",
                ["A", "B"],
                0,
                "Explanation",
                "Long explanation",
                "Document",
                "12",
                "Canonical A",
            )
        ]
        session = MagicMock()
        session.execute.return_value = result_proxy

        with patch.object(main, "SessionLocal", return_value=session):
            result = __import__("asyncio").run(
                main.get_quiz("quiz-id")
            )

        self.assertEqual(
            result["questions"][0]["topic"],
            "Canonical A",
        )
        session.close.assert_called_once()

    def test_builds_hard_generation_metrics(self):
        metrics = main.build_hard_generation_metrics(
            requested_questions=10,
            generated_questions=14,
            rejected_questions=4,
            rejection_reasons=[
                "direct_recall_opening",
                "stem_too_short_for_hard_reasoning",
                "direct_recall_opening",
                "explanation_does_not_show_two_step_reasoning",
            ],
        )

        self.assertEqual(metrics["requested_questions"], 10)
        self.assertEqual(metrics["generated_questions"], 14)
        self.assertEqual(metrics["accepted_questions"], 10)
        self.assertEqual(metrics["rejected_questions"], 4)
        self.assertEqual(metrics["acceptance_rate"], 0.7143)
        self.assertEqual(
            metrics["rejection_reasons_breakdown"],
            {
                "direct_recall_opening": 2,
                "explanation_does_not_show_two_step_reasoning": 1,
                "stem_too_short_for_hard_reasoning": 1,
            },
        )

    def test_hard_generation_metrics_handles_zero_generated(self):
        metrics = main.build_hard_generation_metrics(
            requested_questions=10,
            generated_questions=0,
            rejected_questions=0,
            rejection_reasons=[],
        )

        self.assertEqual(metrics["rejected_questions"], 0)
        self.assertEqual(metrics["acceptance_rate"], 0.0)
        self.assertEqual(metrics["rejection_reasons_breakdown"], {})

    def test_builds_rejected_hard_question_sample(self):
        sample = main.build_hard_question_diagnostic_sample(
            question={
                "question": "How do A and B support the conclusion?",
                "topic": "Mechanism A",
            },
            outcome="rejected",
            question_type="reasoning",
            rejection_reasons=["stem_too_short_for_hard_reasoning"],
            topic_category_by_name={
                "mechanism a": "CORE MECHANISMS",
            },
        )

        self.assertEqual(sample["outcome"], "rejected")
        self.assertEqual(sample["question_type"], "reasoning")
        self.assertEqual(sample["topic"], "Mechanism A")
        self.assertEqual(sample["category"], "CORE MECHANISMS")
        self.assertEqual(
            sample["rejection_reasons"],
            ["stem_too_short_for_hard_reasoning"],
        )

    def test_builds_accepted_hard_question_sample(self):
        sample = main.build_hard_question_diagnostic_sample(
            question={
                "question": "A sufficiently developed reasoning question.",
                "topic": "Mechanism B",
                "category": "EXPLICIT CATEGORY",
            },
            outcome="accepted",
            question_type="exam",
        )

        self.assertEqual(sample["outcome"], "accepted")
        self.assertEqual(sample["question_type"], "exam")
        self.assertEqual(sample["category"], "EXPLICIT CATEGORY")
        self.assertEqual(sample["rejection_reasons"], [])

    def test_diagnostic_storage_is_fail_open(self):
        class FailingDiagnosticSession:
            def __init__(self):
                self.rolled_back = False
                self.closed = False

            def execute(self, *args, **kwargs):
                raise RuntimeError("diagnostic database unavailable")

            def rollback(self):
                self.rolled_back = True

            def close(self):
                self.closed = True

        session = FailingDiagnosticSession()
        metrics = {
            "project_id": "00000000-0000-0000-0000-000000000001",
            "project_name": "Test project",
            "quiz_id": "00000000-0000-0000-0000-000000000002",
            "difficulty": "hard",
            "question_style": "reasoning",
            "requested_questions": 10,
            "generated_questions": 12,
            "accepted_questions": 10,
            "rejected_questions": 2,
            "acceptance_rate": 0.8333,
            "rejection_reasons_breakdown": {
                "direct_recall_opening": 2,
            },
        }

        with patch.object(
            main,
            "SessionLocal",
            return_value=session,
        ):
            result = main.persist_hard_generation_diagnostics(
                metrics,
                [],
            )

        self.assertIsNone(result)
        self.assertTrue(session.rolled_back)
        self.assertTrue(session.closed)

    def test_rejects_direct_fact_lookup(self):
        result = main.evaluate_hard_question_reasoning({
            "question": "Which mutation introduces a premature stop codon?",
            "explanation": (
                "A nonsense mutation introduces a premature stop codon in "
                "the coding sequence."
            ),
            "source_chunk_ids": ["1"],
        })

        self.assertFalse(result["valid"])
        self.assertIn("direct_recall_opening", result["reasons"])

    def test_rejects_true_false_recognition(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "Which statement is true about chromosomal abnormalities "
                "described in the material?"
            ),
            "explanation": (
                "The correct option repeats the definition stated in the "
                "retrieved source material."
            ),
            "source_chunk_ids": ["1"],
        })

        self.assertFalse(result["valid"])
        self.assertIn("direct_recall_opening", result["reasons"])

    def test_explanation_cannot_rescue_a_recall_stem(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "What is an implication of a balanced chromosomal "
                "rearrangement for potential offspring outcomes?"
            ),
            "explanation": (
                "The carrier may be balanced, whereas meiotic segregation "
                "can produce unbalanced gametes; therefore offspring can "
                "inherit an unbalanced chromosomal complement."
            ),
            "source_chunk_ids": ["1", "2"],
        })

        self.assertFalse(result["valid"])
        self.assertIn("direct_recall_opening", result["reasons"])

    def test_rejects_padded_direct_recall(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "What is the named mutation described by the supplied "
                "material as producing a premature stop codon in the final "
                "protein sequence?"
            ),
            "explanation": (
                "The supplied material names the mutation that produces the "
                "premature stop codon in the final protein sequence and "
                "identifies that named mutation as the answer."
            ),
            "source_chunk_ids": ["1"],
        })

        self.assertFalse(result["valid"])
        self.assertIn("direct_recall_opening", result["reasons"])

    def test_allows_direct_recall_opening_with_relational_reasoning(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "What is the distinction between mechanism A and mechanism B "
                "that explains their different outcomes under the supplied "
                "source conditions?"
            ),
            "explanation": (
                "Mechanism A changes the initial supported process. Mechanism "
                "B changes the later supported process. Their distinct effects "
                "identify the answer from the supplied evidence."
            ),
            "source_chunk_ids": ["1", "2"],
        })

        self.assertTrue(result["valid"])
        self.assertTrue(result["relational_reasoning"])

    def test_rejects_short_stem_without_relational_marker(self):
        result = main.evaluate_hard_question_reasoning({
            "question": "How do the supplied facts support this conclusion?",
            "explanation": (
                "The first supported fact establishes the initial condition. "
                "The second supported fact establishes the resulting effect "
                "and completes the requested reasoning."
            ),
            "source_chunk_ids": ["1", "2"],
        })

        self.assertFalse(result["valid"])
        self.assertIn(
            "stem_too_short_for_hard_reasoning",
            result["reasons"],
        )

    def test_rejects_short_explanation_without_connector_requirement(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "How do the supplied facts about mechanism A and mechanism B "
                "support the resulting conclusion described in the source "
                "material?"
            ),
            "explanation": "The two supported facts establish the answer.",
            "source_chunk_ids": ["1", "2"],
        })

        self.assertFalse(result["valid"])
        self.assertIn(
            "explanation_does_not_show_two_step_reasoning",
            result["reasons"],
        )

    def test_accepts_grounded_comparison_and_synthesis(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "Considering that frameshift mutations alter the reading "
                "frame while nonsense mutations introduce a premature stop "
                "codon, which distinction best explains why their effects on "
                "the resulting protein sequence differ?"
            ),
            "explanation": (
                "Frameshift mutations change the grouping of downstream "
                "codons, whereas nonsense mutations terminate translation at "
                "one premature codon; therefore the first can alter the "
                "entire downstream sequence while the second truncates it."
            ),
            "source_chunk_ids": ["11", "12"],
        })

        self.assertTrue(result["valid"])
        self.assertEqual(result["reasons"], [])
        self.assertEqual(result["source_chunk_count"], 2)

    def test_rejects_comparison_with_one_fact_explanation(self):
        result = main.evaluate_hard_question_reasoning({
            "question": (
                "Compared with genomic DNA, which distinction best explains "
                "the organization of mitochondrial DNA described by the "
                "retrieved material?"
            ),
            "explanation": "Mitochondrial DNA is circular.",
            "source_chunk_ids": ["7"],
        })

        self.assertFalse(result["valid"])
        self.assertIn(
            "explanation_does_not_show_two_step_reasoning",
            result["reasons"],
        )


if __name__ == "__main__":
    unittest.main()
