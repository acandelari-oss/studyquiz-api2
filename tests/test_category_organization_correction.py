"""Offline semantic-correction contract tests; no model or database."""
import copy
import json
import unittest
import taxonomy_organization as org


class CorrectionTests(unittest.TestCase):
    def setUp(self):
        titles = [
            "Definizione e principi del concorso di reati", "Imputato",
            "Cumulo materiale e giuridico", "Struttura Organizzativa del Pubblico Ministero"]
        self.topics = [dict(assignment_id=f"uuid-{i}", title=title, description=title,
                           document_id="soggetti", evidence_ids=[i], chunk_ids=[i, i+10])
                       for i, title in enumerate(titles)]
        self.data = {"categories": [{"name": "Original", "topics": self.topics}]}
        self.categories = [
            {"name": "Giudici e Funzioni Giudiziarie", "description": "Organi giudicanti"},
            {"name": "Competenza e Incompetenza", "description": "Ripartizione della competenza"},
            {"name": "Ruolo della Polizia Giudiziaria", "description": "Funzioni di polizia"}]
        self.rows = [{"topic_id": f"T{i+1:04d}", "category_id": f"C{c:04d}"}
                     for i, c in enumerate([0, 0, 1, 2])]
        self.initial = {"assignments": self.rows, "quality_issues": [
            {"code": "incoherent", "topic_ids": [r["topic_id"] for r in self.rows],
             "category_ids": ["C0000", "C0001", "C0002"]}]}
        self.corrected = {
            "categories": [
                {"name": "Concorso di reati e cumulo delle pene", "description": "Pluralità di reati e determinazione della pena"},
                {"name": "Imputato e posizione processuale", "description": "Ruolo e diritti dell'imputato"},
                {"name": "Pubblico Ministero", "description": "Organizzazione dell'organo requirente"}],
            "assignments": [{"topic_id": f"T{i+1:04d}", "category_id": f"C{c:04d}"}
                            for i, c in enumerate([0, 1, 0, 2])], "quality_issues": []}
        self.calls = []

    def request(self, prompt, stage):
        self.calls.append((prompt, stage))
        return copy.deepcopy({"category_proposal": {"categories": self.categories},
                              "category_assignment": self.initial,
                              "category_correction": self.corrected}[stage])

    def run_organization(self):
        return org.organize_categories(self.data, "infer", [], self.request)

    def test_soggetti_replaces_structure_and_reassigns_all_four_topics(self):
        before = copy.deepcopy(self.data)
        result = self.run_organization()
        self.assertEqual([c["name"] for c in result["categories"]],
                         [c["name"] for c in self.corrected["categories"]])
        self.assertEqual([[t["assignment_id"] for t in c["topics"]] for c in result["categories"]],
                         [["uuid-0", "uuid-2"], ["uuid-1"], ["uuid-3"]])
        self.assertEqual(self.data, before)
        self.assertTrue(all(any(t is old for old in self.topics)
                            for c in result["categories"] for t in c["topics"]))
        self.assertEqual([stage for _, stage in self.calls],
                         ["category_proposal", "category_assignment", "category_correction"])
        prompt = self.calls[-1][0]
        payload = json.loads(prompt.split("ORGANIZATION INPUT:\n")[1])
        self.assertEqual([t["topic_id"] for t in payload["topics"]], [r["topic_id"] for r in self.rows])
        self.assertEqual(payload["initial_assignment"], self.initial)
        self.assertIn("categories are not fixed", prompt)
        self.assertIn("incidental relationships", self.calls[1][0])
        self.assertLessEqual(len(prompt), org.MAX_PROMPT)

    def test_correction_can_split_proposal_and_change_category_count(self):
        self.categories = [{"name": "Soggetti e competenze", "description": "Organi e competenze processuali"}]
        for row in self.initial["assignments"]:
            row["category_id"] = "C0000"
        result = self.run_organization()
        self.assertEqual(len(result["categories"]), 3)
        self.assertEqual(sum(len(c["topics"]) for c in result["categories"]), 4)
        self.assertEqual(len(self.calls), 3)

    def test_clean_review_uses_only_two_calls(self):
        # Represents an acceptable distribution; no extra quality call is made.
        self.categories = self.corrected["categories"]
        self.initial = {k: v for k, v in self.corrected.items() if k != "categories"}
        self.run_organization()
        self.assertEqual(len(self.calls), 2)

    def test_invalid_correction_fails_without_second_attempt_or_input_mutation(self):
        changes = [
            lambda r: r["assignments"].pop(),
            lambda r: r["assignments"].append(r["assignments"][0]),
            lambda r: r["assignments"][0].update(topic_id="UNKNOWN"),
            lambda r: r["assignments"][0].update(category_id="C9999"),
            lambda r: r["assignments"][0].update(category_id=None),
            lambda r: r.update(quality_issues=[{"code": "overlap"}]),
            lambda r: r.pop("quality_issues"),
            lambda r: r.update(categories=[{"name": "General", "description": "All"}]),
        ]
        good = copy.deepcopy(self.corrected)
        before = copy.deepcopy(self.data)
        for change in changes:
            self.calls.clear()
            self.corrected = copy.deepcopy(good)
            change(self.corrected)
            with self.assertRaises(ValueError):
                self.run_organization()
            self.assertEqual(len(self.calls), 3)
            self.assertEqual(self.data, before)

    def test_null_assignment_without_report_still_requires_correction(self):
        self.initial["assignments"][0]["category_id"] = None
        self.initial["quality_issues"] = []
        self.run_organization()
        self.assertEqual(len(self.calls), 3)

    def test_correction_context_overflow_fails_before_additional_request(self):
        # Initial prompts fit; oversized issue details cannot bypass the limit.
        self.initial["quality_issues"][0]["reason"] = "x" * org.MAX_PROMPT
        with self.assertRaisesRegex(ValueError, "bounded context"):
            self.run_organization()
        self.assertEqual(len(self.calls), 2)

    def test_correction_request_failure_not_retried(self):
        original = self.request
        def request(prompt, stage):
            if stage == "category_correction":
                self.calls.append((prompt, stage))
                raise RuntimeError("mock transport failure")
            return original(prompt, stage)
        with self.assertRaisesRegex(RuntimeError, "transport"):
            org.organize_categories(self.data, "infer", [], request)
        self.assertEqual(len(self.calls), 3)
