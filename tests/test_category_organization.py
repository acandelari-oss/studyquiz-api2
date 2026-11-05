import copy
import json
import unittest
import uuid
import io
import ast
from pathlib import Path
from types import SimpleNamespace
from contextlib import redirect_stdout
from taxonomy_organization import (organize_categories, category_distribution,
                                   REVIEW_CATEGORY, OrganizationValidationError,
                                   parse_organization_response)


class OrganizationTests(unittest.TestCase):
    def setUp(self):
        self.data = {"categories": [{"name": "SECRET_OLD_LABEL", "topics": [
            {"assignment_id": str(i), "title": "Objective " + str(i),
             "description": "Full source explanation", "evidence_ids": [i, i + 3],
             "candidate_ids": [str(i)], "embedding": [1., 0.]}
            for i in range(3)]}]}
        self.calls = []
        self.assignments = [{"topic_id": f"T{i + 1:04d}", "category_id": "C0000"} for i in range(3)]

    def request(self, prompt, stage):
        self.calls.append((prompt, stage))
        self.assertNotIn("SECRET_OLD_LABEL", prompt)
        self.assertLessEqual(len(prompt), 48000)
        if stage == "category_proposal":
            return {"categories": [{"name": "Educational area", "description": "Scope"}]}
        return {"assignments": self.assignments, "quality_issues": []}

    def test_infer_separate_calls_and_exact_evidence_preserved(self):
        before = copy.deepcopy(self.data)
        result = organize_categories(self.data, "infer", [], self.request)
        self.assertEqual([c[1] for c in self.calls], ["category_proposal", "category_assignment"])
        self.assertEqual(result["categories"][0]["topics"], self.data["categories"][0]["topics"])
        self.assertEqual(self.data, before)
        self.assertIs(result["categories"][0]["topics"][0], self.data["categories"][0]["topics"][0])

    def test_manual_unmatched_preserved_and_empty_student_category_kept(self):
        self.assignments[1]["category_id"] = None
        result = organize_categories(self.data, "manual",
            [{"name": "Student Case"}, {"name": "Unused Student Area"}], self.request)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual([c["name"] for c in result["categories"]],
                         ["Student Case", "Unused Student Area", REVIEW_CATEGORY])
        self.assertEqual(result["categories"][1]["topics"], [])
        self.assertTrue(result["categories"][2]["system_review"])
        self.assertEqual(result["categories"][2]["topics"][0]["assignment_id"], "1")

    def test_material_relative_guidance_is_only_in_automatic_proposal(self):
        organize_categories(self.data, "infer", [], self.request)
        proposal, assignment = [prompt for prompt, stage in self.calls]
        for instruction in (
            'meaningful major learning areas within the supplied material',
            'rather than simply naming its broader academic discipline',
            'umbrella category hides distinct, substantial learning areas',
            'A single category is valid',
            'Do not impose a minimum or target category count',
            'artificially fragment coherent areas',
            'or split categories based on their size',
        ):
            self.assertIn(instruction, proposal)
            self.assertNotIn(instruction, assignment)
        self.assertIn('Do not assign topics in this step', proposal)
        self.assertIn('Category names and hierarchy are fixed', assignment)
        for mode in ('manual', 'syllabus'):
            self.calls.clear()
            organize_categories(self.data, mode, [{"name": "Fixed area"}], self.request)
            self.assertEqual([stage for _, stage in self.calls], ['category_assignment'])
            self.assertNotIn('Before finalizing the proposal', self.calls[0][0])

    def test_syllabus_hierarchy_and_unmatched(self):
        self.assignments[0]["category_id"] = None
        result = organize_categories(self.data, "syllabus",
            [{"name": "Part / Section"}], self.request, {"syllabus_text": "Part\nSection"})
        self.assertEqual(result["categories"][0]["name"], "Part / Section")
        self.assertEqual(result["categories"][1]["name"], REVIEW_CATEGORY)
        self.assertIn("syllabus_text", self.calls[0][0])

    def test_reject_incomplete_duplicate_unknown_and_invalid_categories(self):
        for rows in (self.assignments[:2], self.assignments + [self.assignments[0]],
                     [{"topic_id": "unknown", "category_id": "C0000"}],
                     [{"topic_id": f"T{i + 1:04d}", "category_id": "invented"} for i in range(3)],
                     [{"topic_id": f"T{i + 1:04d}", "category_id": None} for i in range(3)]):
            self.assignments = rows
            with self.assertRaises(ValueError):
                organize_categories(self.data, "infer", [], self.request)

    def test_oversized_mandatory_input_fails_before_request(self):
        self.data["categories"][0]["topics"][0]["title"] = "x" * 48000
        with self.assertRaisesRegex(ValueError, "bounded context"):
            organize_categories(self.data, "infer", [], self.request)
        self.assertEqual(self.calls, [])

    def test_no_count_based_split(self):
        self.data["categories"][0]["topics"] = [
            {"assignment_id": str(i), "title": f"Objective {i}"} for i in range(80)]
        self.assignments = [{"topic_id": f"T{i + 1:04d}", "category_id": "C0000"} for i in range(80)]
        result = organize_categories(self.data, "infer", [], self.request)
        self.assertEqual(len(result["categories"]), 1)
        self.assertEqual(category_distribution(result), {"Educational area": 80})

    def test_455_topics_map_back_to_exact_immutable_ids_and_evidence(self):
        topics = [{"assignment_id": str(uuid.UUID(int=i+1)), "title": f"Learning objective {i}",
                   "description": "Synthetic explanation " * 20,
                   "evidence_ids": [i, i+455], "candidate_ids": [str(i)],
                   "embedding": [1., 0.]} for i in range(455)]
        self.data["categories"][0]["topics"] = topics
        original = copy.deepcopy(self.data)
        payloads = []
        def request(prompt, stage):
            payload = json.loads(prompt.split("ORGANIZATION INPUT:\n")[1])
            payloads.append(payload)
            self.assertLessEqual(len(prompt), 48000)
            self.assertEqual([t["topic_id"] for t in payload["topics"]],
                             [f"T{i+1:04d}" for i in range(455)])
            self.assertNotIn(topics[0]["assignment_id"], prompt)
            if stage == "category_proposal":
                # Provider-supplied category IDs must never override local IDs.
                return {"categories": [{"name": "Area A", "description": "Scope A", "category_id": "wrong"},
                                       {"name": "Area B", "description": "Scope B", "category_id": "wrong"}]}
            self.assertEqual([c["category_id"] for c in payload["categories"]], ["C0000", "C0001"])
            return {"quality_issues": [], "assignments": [
                {"topic_id": t["topic_id"], "category_id": f"C{i % 2:04d}"}
                for i, t in reversed(list(enumerate(payload["topics"])))]}
        result = organize_categories(self.data, "infer", [], request)
        output = [t for c in result["categories"] for t in c["topics"]]
        self.assertEqual(len(output), 455)
        self.assertEqual({t["assignment_id"] for t in output}, {t["assignment_id"] for t in topics})
        self.assertEqual({t["assignment_id"]: t for t in output}, {t["assignment_id"]: t for t in topics})
        self.assertTrue(all(any(t is original_topic for original_topic in topics) for t in output))
        self.assertEqual(self.data, original)
        self.assertEqual([t["topic_id"] for t in payloads[0]["topics"]],
                         [t["topic_id"] for t in payloads[1]["topics"]])

    def test_rejected_id_diagnostics_distinguish_failure_types_without_material(self):
        cases = [
            (None, "non_string_topic_id"),
            ([], "non_string_topic_id"),
            ("PRIVATE SOURCE PASSAGE", "unknown_topic_id"),
            ("T9999", "unknown_topic_id"),
            ("T0001", "duplicate_topic_id"),
        ]
        for bad_id, code in cases:
            with self.subTest(code=code, bad_id=bad_id):
                self.assignments = [{"topic_id": "T0001", "category_id": "C0000"},
                                    {"topic_id": bad_id, "category_id": "C0000"},
                                    {"topic_id": "T0003", "category_id": "C0000"}]
                before = copy.deepcopy(self.data)
                output = io.StringIO()
                with redirect_stdout(output), self.assertRaises(OrganizationValidationError) as raised:
                    organize_categories(self.data, "infer", [], self.request)
                self.assertIn(code, raised.exception.diagnostics["codes"])
                self.assertEqual(raised.exception.diagnostics["missing_ids"], ["T0002"])
                self.assertNotIn("PRIVATE SOURCE PASSAGE", output.getvalue())
                self.assertEqual(self.data, before)

    def test_455_duplicate_response_is_not_repaired_or_retried(self):
        topics = [{"assignment_id": str(uuid.UUID(int=i+1)), "title": f"Objective {i}"}
                  for i in range(455)]
        self.data["categories"][0]["topics"] = topics
        self.assignments = [{"topic_id": f"T{i+1:04d}", "category_id": "C0000"} for i in range(455)]
        self.assignments[-1]["topic_id"] = "T0001"
        with self.assertRaises(OrganizationValidationError) as raised:
            organize_categories(self.data, "infer", [], self.request)
        diagnostics = raised.exception.diagnostics
        self.assertEqual(diagnostics["duplicate_ids"], ["T0001"])
        self.assertEqual(diagnostics["missing_ids"], ["T0455"])
        self.assertEqual(diagnostics["expected_count"], 455)
        self.assertEqual(diagnostics["returned_count"], 455)
        self.assertEqual(len(self.calls), 2)  # proposal and assignment only
        self.assertEqual(self.data["categories"][0]["topics"], topics)

    def test_malformed_response_shape_is_reported(self):
        def request(prompt, stage):
            return {"categories": [{"name": "Area", "description": "Scope"}]} if stage == "category_proposal" else []
        with self.assertRaises(OrganizationValidationError) as raised:
            organize_categories(self.data, "infer", [], request)
        self.assertEqual(raised.exception.diagnostics["code"], "invalid_assignment_object")

    def test_json_parser_never_silently_overwrites_duplicate_id_fields(self):
        content = '{"assignments":[{"topic_id":"unknown","topic_id":"T0001","category_id":"C0000"}]}'
        with self.assertRaises(OrganizationValidationError) as raised:
            parse_organization_response(content)
        self.assertEqual(raised.exception.diagnostics["code"], "duplicate_json_field")
        with self.assertRaises(OrganizationValidationError) as raised:
            parse_organization_response('{"assignments":')
        self.assertEqual(raised.exception.diagnostics["code"], "malformed_json")
        self.assertEqual(parse_organization_response('{"assignments":[]}'), {"assignments": []})

    def test_production_request_records_finish_and_usage_without_content(self):
        tree = ast.parse((Path(__file__).resolve().parents[1] / "main.py").read_text())
        node = next(n for n in ast.walk(tree)
                    if isinstance(n, ast.FunctionDef) and n.name == "request_organization")
        calls = []
        options = []
        reply = SimpleNamespace(model="gpt-4o-mini", usage=SimpleNamespace(prompt_tokens=100, completion_tokens=50),
            choices=[SimpleNamespace(finish_reason="stop",
                message=SimpleNamespace(content='{"assignments":[]}'))])
        def call(fn, **kwargs):
            calls.append(kwargs)
            return reply
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))
        client.with_options = lambda **kw: (options.append(kw) or client)
        ns = {"cost": SimpleNamespace(call=call), "client": client, "json": json,
              "project_id": "test-project", "module_id": "test-module",
              "parse_organization_response": parse_organization_response}
        exec(compile(ast.Module(body=[node], type_ignores=[]), "main.py", "exec"), ns)
        output = io.StringIO()
        with redirect_stdout(output):
            self.assertEqual(ns["request_organization"]("PRIVATE MATERIAL", "category_assignment"),
                             {"assignments": []})
        self.assertEqual(options, [{"max_retries": 0}])
        self.assertEqual(len(calls), 1)
        self.assertIn('"input_tokens": 100', output.getvalue())
        self.assertIn('"output_tokens": 50', output.getvalue())
        self.assertIn('"finish_reason": "stop"', output.getvalue())
        self.assertNotIn("PRIVATE MATERIAL", output.getvalue())
        reply.choices[0].finish_reason = "length"
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            ns["request_organization"]("PRIVATE MATERIAL", "category_assignment")
