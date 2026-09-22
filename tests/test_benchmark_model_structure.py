import importlib.util
import contextlib
import base64
import io
import sys
import tempfile
import types
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "benchmark_model_structure.py"
)

spec = importlib.util.spec_from_file_location("benchmark_model_structure", SCRIPT_PATH)
benchmark = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = benchmark
spec.loader.exec_module(benchmark)


class BenchmarkModelStructureTests(unittest.TestCase):
    def test_pdf_input_file_data_is_mime_prefixed_data_url(self):
        pdf_bytes = b"%PDF-1.7\nbenchmark pdf bytes\n%%EOF"

        with tempfile.NamedTemporaryFile(suffix=".pdf") as temp_pdf:
            temp_pdf.write(pdf_bytes)
            temp_pdf.flush()
            pdf_path = Path(temp_pdf.name)

            file_data = benchmark.encode_pdf_file_data(pdf_path)
            payload = benchmark.build_response_input(pdf_path)

        prefix = "data:application/pdf;base64,"
        self.assertTrue(file_data.startswith(prefix))
        self.assertEqual(base64.b64decode(file_data.removeprefix(prefix)), pdf_bytes)

        input_file = payload[0]["content"][1]
        self.assertEqual(input_file["type"], "input_file")
        self.assertEqual(input_file["filename"], pdf_path.name)
        self.assertEqual(input_file["file_data"], file_data)
        self.assertEqual(input_file["detail"], "high")

    def test_parse_structured_result_requires_machine_readable_object(self):
        result = benchmark.parse_structured_result(
            """
            {
              "document_title": "Example",
              "document_type": "lecture notes",
              "structure_confidence": "HIGH",
              "sections": [],
              "local_structure_summary": {
                "detected": false,
                "types": [],
                "representative_examples": []
              },
              "notes": []
            }
            """
        )

        self.assertEqual(result["document_title"], "Example")

        with self.assertRaises(ValueError):
            benchmark.parse_structured_result("not json")

        with self.assertRaises(ValueError):
            benchmark.parse_structured_result('{"document_title": "Incomplete"}')

    def test_format_tree_uses_levels_pages_and_reasons(self):
        result = {
            "sections": [
                {
                    "title": "Main Section",
                    "level": 1,
                    "parent": None,
                    "start_page": 1,
                    "end_page": 3,
                    "confidence": "HIGH",
                },
                {
                    "title": "Subsection",
                    "level": 2,
                    "parent": "Main Section",
                    "start_page": 2,
                    "end_page": 2,
                    "confidence": "MEDIUM",
                },
            ]
        }

        tree = benchmark.format_tree(result)

        self.assertIn("- Main Section [HIGH, pages 1-3]", tree)
        self.assertIn("  - Subsection [MEDIUM, page 2]", tree)
        self.assertNotIn("reason:", tree)

    def test_format_local_structure_summary(self):
        result = {
            "local_structure_summary": {
                "detected": True,
                "types": ["enumeration", "table"],
                "representative_examples": [
                    {
                        "page": 12,
                        "type": "enumeration",
                        "title": "Local mechanisms",
                        "confidence": "HIGH",
                    }
                ],
            }
        }

        formatted = benchmark.format_local_structures(result)

        self.assertIn("types: enumeration, table", formatted)
        self.assertIn("enumeration | page 12 | Local mechanisms [HIGH]", formatted)

    def test_compact_schema_removes_reasons_and_caps_local_summaries(self):
        schema = benchmark.DOCUMENT_STRUCTURE_SCHEMA
        section_schema = schema["properties"]["sections"]["items"]
        local_summary = schema["properties"]["local_structure_summary"]
        examples = local_summary["properties"]["representative_examples"]

        self.assertIn("local_structure_summary", schema["required"])
        self.assertNotIn("local_structures", schema["required"])
        self.assertNotIn("reason", section_schema["required"])
        self.assertNotIn("reason", section_schema["properties"])
        self.assertEqual(examples["maxItems"], 10)
        self.assertEqual(schema["properties"]["notes"]["maxItems"], 5)

    def test_prompt_requests_compact_document_agnostic_output(self):
        self.assertIn("canonical document hierarchy", benchmark.SYSTEM_INSTRUCTIONS)
        self.assertIn("Do not list ordinary body content", benchmark.SYSTEM_INSTRUCTIONS)
        self.assertIn("capped at 10", benchmark.SYSTEM_INSTRUCTIONS)
        self.assertIn("at most 10 representative", benchmark.USER_TASK)
        self.assertNotIn("Physiology", benchmark.SYSTEM_INSTRUCTIONS)
        self.assertNotIn("Genetics", benchmark.SYSTEM_INSTRUCTIONS)

    def test_usage_extraction_supports_object_shapes(self):
        response = types.SimpleNamespace(
            usage=types.SimpleNamespace(
                input_tokens=1000,
                output_tokens=250,
                total_tokens=1250,
                output_tokens_details=types.SimpleNamespace(reasoning_tokens=40),
            )
        )

        usage = benchmark.extract_usage(response)

        self.assertEqual(usage.input_tokens, 1000)
        self.assertEqual(usage.output_tokens, 250)
        self.assertEqual(usage.total_tokens, 1250)
        self.assertEqual(usage.reasoning_tokens, 40)

    def test_usage_extraction_supports_dict_shapes(self):
        response = {
            "usage": {
                "input_tokens": 10,
                "output_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 5},
            }
        }

        usage = benchmark.extract_usage(response)

        self.assertEqual(usage.input_tokens, 10)
        self.assertEqual(usage.output_tokens, 20)
        self.assertEqual(usage.total_tokens, 30)
        self.assertEqual(usage.reasoning_tokens, 5)

    def test_cost_calculation_is_optional_and_configurable(self):
        usage = benchmark.UsageSummary(
            input_tokens=1_000_000,
            output_tokens=500_000,
            total_tokens=1_500_000,
            reasoning_tokens=None,
        )

        without_rates = benchmark.estimate_cost(usage, "unknown-model")
        self.assertIsNone(without_rates.total_cost)

        with_rates = benchmark.estimate_cost(
            usage,
            "unknown-model",
            input_cost_per_1m=2.0,
            output_cost_per_1m=8.0,
        )
        self.assertEqual(with_rates.input_cost, 2.0)
        self.assertEqual(with_rates.output_cost, 4.0)
        self.assertEqual(with_rates.total_cost, 6.0)

    def test_response_text_falls_back_to_output_content(self):
        response = types.SimpleNamespace(
            output=[
                types.SimpleNamespace(
                    content=[
                        types.SimpleNamespace(text='{"document_title":"A"}')
                    ]
                )
            ]
        )

        self.assertEqual(
            benchmark.response_text(response),
            '{"document_title":"A"}',
        )

    def test_response_diagnostics_preserve_metadata(self):
        response = types.SimpleNamespace(
            status="incomplete",
            incomplete_details=types.SimpleNamespace(reason="max_output_tokens"),
            max_output_tokens=6000,
            output=[
                types.SimpleNamespace(status="incomplete", content=[])
            ],
            usage=types.SimpleNamespace(
                input_tokens=100,
                output_tokens=6000,
                total_tokens=6100,
                output_tokens_details=types.SimpleNamespace(reasoning_tokens=25),
            ),
        )

        diagnostics = benchmark.extract_response_diagnostics(response, "abc")

        self.assertEqual(diagnostics.response_status, "incomplete")
        self.assertEqual(diagnostics.incomplete_reason, "max_output_tokens")
        self.assertEqual(diagnostics.response_max_output_tokens, 6000)
        self.assertEqual(diagnostics.input_tokens, 100)
        self.assertEqual(diagnostics.output_tokens, 6000)
        self.assertEqual(diagnostics.total_tokens, 6100)
        self.assertEqual(diagnostics.reasoning_tokens, 25)
        self.assertEqual(diagnostics.raw_output_chars, 3)
        self.assertEqual(diagnostics.output_item_statuses, ["incomplete"])

    def test_incomplete_max_output_tokens_reports_clear_parse_error(self):
        diagnostics = benchmark.ResponseDiagnostics(
            response_status="incomplete",
            incomplete_reason="max_output_tokens",
            response_max_output_tokens=6000,
            input_tokens=10,
            output_tokens=6000,
            total_tokens=6010,
            reasoning_tokens=None,
            raw_output_chars=20,
            output_item_statuses=["incomplete"],
        )

        with self.assertRaises(benchmark.BenchmarkResponseError) as raised:
            benchmark.parse_structured_result('{"notes": ["unterminated', diagnostics)

        self.assertIn("Response incomplete", str(raised.exception))
        self.assertIs(raised.exception.diagnostics, diagnostics)

    def test_completed_invalid_json_reports_distinct_parse_error(self):
        diagnostics = benchmark.ResponseDiagnostics(
            response_status="completed",
            incomplete_reason=None,
            response_max_output_tokens=6000,
            input_tokens=10,
            output_tokens=20,
            total_tokens=30,
            reasoning_tokens=None,
            raw_output_chars=8,
            output_item_statuses=["completed"],
        )

        with self.assertRaises(benchmark.BenchmarkResponseError) as raised:
            benchmark.parse_structured_result("not json", diagnostics)

        self.assertIn("Completed response", str(raised.exception))
        self.assertIs(raised.exception.diagnostics, diagnostics)

    def test_main_fails_clearly_without_api_key(self):
        original = benchmark.os.environ.pop("OPENAI_API_KEY", None)
        try:
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                status = benchmark.main([
                    "/tmp/example.pdf",
                    "--model",
                    "example-model",
                    "--env-file",
                    "/tmp/douno-benchmark-missing.env",
                ])
            self.assertEqual(status, 2)
            self.assertIn("OPENAI_API_KEY", stderr.getvalue())
        finally:
            if original is not None:
                benchmark.os.environ["OPENAI_API_KEY"] = original


if __name__ == "__main__":
    unittest.main()
