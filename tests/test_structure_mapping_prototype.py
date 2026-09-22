import json
import sys
import types
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from document_extractors import ExtractedBlock, ExtractedDocument
from model_structure_interpreter import (
    DOCUMENT_STRUCTURE_SCHEMA,
    InterpretationResult,
    build_response_input,
    interpret_pdf_structure,
    parse_model_response,
    structure_from_model_json,
)
from structure_anchor import anchor_document_structure
from structure_segmenter import (
    diagnostic_chunks_from_segments,
    production_compatible_chunk_text,
    segment_extracted_document,
)


def extracted(blocks):
    return ExtractedDocument(
        filename="fixture.pdf",
        file_format="pdf",
        file_size_bytes=100,
        blocks=blocks,
        pages_detected=len(blocks),
    )


def structure(sections, confidence="HIGH"):
    return structure_from_model_json(
        {
            "document_title": "Fixture",
            "document_type": "lecture notes",
            "structure_confidence": confidence,
            "sections": sections,
            "local_structure_summary": {
                "detected": False,
                "types": [],
                "representative_examples": [],
            },
            "notes": [],
        }
    )


class FakeResponses:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.response


class FakeClient:
    def __init__(self, response=None, error=None):
        self.responses = FakeResponses(response=response, error=error)


class StructureMappingPrototypeTests(unittest.TestCase):
    def test_interpreter_sends_pdf_as_real_input_file(self):
        pdf_bytes = b"%PDF-1.7\nx\n%%EOF"
        payload = build_response_input(pdf_bytes, "example.pdf")
        input_file = payload[0]["content"][1]

        self.assertEqual(input_file["type"], "input_file")
        self.assertEqual(input_file["filename"], "example.pdf")
        self.assertEqual(input_file["detail"], "high")
        self.assertTrue(input_file["file_data"].startswith("data:application/pdf;base64,"))

    def test_interpreter_uses_compact_strict_schema(self):
        section_schema = DOCUMENT_STRUCTURE_SCHEMA["properties"]["sections"]["items"]
        self.assertEqual(
            DOCUMENT_STRUCTURE_SCHEMA["properties"]["local_structure_summary"]
            ["properties"]["representative_examples"]["maxItems"],
            10,
        )
        self.assertNotIn("reason", section_schema["properties"])
        self.assertIn("source_title", section_schema["required"])
        self.assertIn("semantic_title", section_schema["required"])

    def test_interpreter_parses_mocked_response(self):
        raw = json.dumps(
            {
                "document_title": "Doc",
                "document_type": "notes",
                "structure_confidence": "HIGH",
                "sections": [
                    {
                        "source_title": "1 Introduction",
                        "semantic_title": "Introduction",
                        "level": 1,
                        "parent": None,
                        "start_page": 1,
                        "end_page": 2,
                        "confidence": "HIGH",
                        "scope": "global",
                    }
                ],
                "local_structure_summary": {
                    "detected": False,
                    "types": [],
                    "representative_examples": [],
                },
                "notes": [],
            }
        )
        response = types.SimpleNamespace(
            output_text=raw,
            status="completed",
            incomplete_details=None,
            max_output_tokens=6000,
            output=[types.SimpleNamespace(status="completed", content=[])],
            usage=types.SimpleNamespace(
                input_tokens=10,
                output_tokens=20,
                total_tokens=30,
                output_tokens_details=types.SimpleNamespace(reasoning_tokens=2),
            ),
        )
        client = FakeClient(response=response)

        result = interpret_pdf_structure(
            pdf_bytes=b"%PDF-1.7",
            filename="doc.pdf",
            model="mock-model",
            client=client,
        )

        self.assertIsInstance(result, InterpretationResult)
        self.assertEqual(result.structure.sections[0].source_title, "1 Introduction")
        self.assertEqual(client.responses.calls[0]["store"], False)

    def test_model_api_mocked_failure(self):
        with self.assertRaises(Exception):
            interpret_pdf_structure(
                pdf_bytes=b"%PDF-1.7",
                filename="doc.pdf",
                model="mock-model",
                client=FakeClient(error=RuntimeError("boom")),
            )

    def test_parse_model_response_distinguishes_invalid_json(self):
        with self.assertRaises(Exception):
            parse_model_response("not json")

    def test_exact_title_anchoring(self):
        doc = extracted([
            ExtractedBlock(
                text="1 Introduction\nBody",
                raw_text="1 Introduction\nBody",
                page=1,
                block_index=1,
            )
        ])
        model = structure([
            {
                "source_title": "1 Introduction",
                "semantic_title": "Introduction",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.exact, 1)
        self.assertEqual(anchored.sections[0].anchor.start_offset, 0)

    def test_whitespace_newline_normalized_anchoring(self):
        doc = extracted([
            ExtractedBlock(
                text="2.3   Epigenetic\nRegulation\nBody",
                raw_text="2.3   Epigenetic\nRegulation\nBody",
                page=17,
                block_index=1,
            )
        ])
        model = structure([
            {
                "source_title": "2.3 Epigenetic Regulation",
                "semantic_title": "Epigenetic Regulation",
                "level": 2,
                "parent": None,
                "start_page": 17,
                "end_page": 17,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.normalized, 1)
        self.assertEqual(anchored.sections[0].anchor.source_title_fidelity, "normalized")

    def test_unicode_normalization(self):
        doc = extracted([
            ExtractedBlock(
                text="3 Café regulation",
                raw_text="3 Café regulation",
                page=1,
                block_index=1,
            )
        ])
        model = structure([
            {
                "source_title": "3 Café regulation",
                "semantic_title": "Cafe regulation",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.normalized, 1)
        self.assertEqual(anchored.sections[0].anchor.method, "unicode")

    def test_punctuation_normalization(self):
        doc = extracted([
            ExtractedBlock(
                text="4 Regulation: transcription",
                raw_text="4 Regulation: transcription",
                page=1,
                block_index=1,
            )
        ])
        model = structure([
            {
                "source_title": "4 Regulation - transcription",
                "semantic_title": "Regulation transcription",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.normalized, 1)
        self.assertEqual(anchored.sections[0].anchor.method, "punctuation")

    def test_fuzzy_page_constrained_anchoring(self):
        doc = extracted([
            ExtractedBlock(
                text="5 Prokaryotic gene regulation: operons\nBody",
                raw_text="5 Prokaryotic gene regulation: operons\nBody",
                page=5,
                block_index=1,
            )
        ])
        model = structure([
            {
                "source_title": "5 Prokaryotic regulation operons",
                "semantic_title": "Prokaryotic gene regulation",
                "level": 1,
                "parent": None,
                "start_page": 5,
                "end_page": 5,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.fuzzy, 1)
        self.assertEqual(anchored.sections[0].anchor.source_title_fidelity, "fuzzy")

    def test_unresolved_model_only_heading(self):
        doc = extracted([
            ExtractedBlock(text="Body without heading", raw_text="Body without heading", page=1, block_index=1)
        ])
        model = structure([
            {
                "source_title": None,
                "semantic_title": "Synthesized Introduction",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "MEDIUM",
                "scope": "global",
            }
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.unresolved, 1)
        self.assertEqual(summary.model_only, 1)
        self.assertIsNone(anchored.sections[0].anchor)

    def test_multiple_headings_on_same_page_are_distinct(self):
        raw = (
            "end 2.2\n"
            "2.3 Epigenetic Regulation\n"
            "body A\n"
            "2.3.1 DNA methylation\n"
            "body B\n"
            "2.4 Post-transcriptional Regulation\n"
            "body C"
        )
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=17, block_index=1)])
        model = structure([
            {
                "source_title": "2.3 Epigenetic Regulation",
                "semantic_title": "Epigenetic Regulation",
                "level": 2,
                "parent": None,
                "start_page": 17,
                "end_page": 17,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "2.3.1 DNA methylation",
                "semantic_title": "DNA methylation",
                "level": 3,
                "parent": "2.3 Epigenetic Regulation",
                "start_page": 17,
                "end_page": 17,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "2.4 Post-transcriptional Regulation",
                "semantic_title": "Post-transcriptional Regulation",
                "level": 2,
                "parent": None,
                "start_page": 17,
                "end_page": 18,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])

        anchored, summary = anchor_document_structure(model, doc)
        segments = segment_extracted_document(doc, anchored)

        self.assertEqual(summary.anchored, 3)
        self.assertEqual([s.source_start_offset for s in segments], sorted(s.source_start_offset for s in segments))
        self.assertEqual(segments[1].raw_text.splitlines()[0], "2.3 Epigenetic Regulation")
        self.assertEqual(segments[2].raw_text.splitlines()[0], "2.3.1 DNA methylation")
        self.assertEqual(segments[3].raw_text.splitlines()[0], "2.4 Post-transcriptional Regulation")
        self.assertIn("2.3 Epigenetic Regulation", segments[1].section_title)
        self.assertIn("2.3.1 DNA methylation", segments[2].section_title)
        self.assertIn("2.4 Post-transcriptional Regulation", segments[3].section_title)

    def test_nested_headings_use_canonical_path_policy(self):
        raw = "5 Parent\nBody\n5.1 Child\nChild body"
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=1, block_index=1)])
        model = structure([
            {
                "source_title": "5 Parent",
                "semantic_title": "Parent",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "5.1 Child",
                "semantic_title": "Child",
                "level": 2,
                "parent": "5 Parent",
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])

        anchored, _ = anchor_document_structure(model, doc)
        segments = segment_extracted_document(doc, anchored)

        self.assertEqual(segments[-1].section_title, "5 Parent > 5.1 Child")

    def test_page_continuation_keeps_active_section(self):
        doc = extracted([
            ExtractedBlock(text="1 Intro\nBody page 1", raw_text="1 Intro\nBody page 1", page=1, block_index=1),
            ExtractedBlock(text="Continuation page 2", raw_text="Continuation page 2", page=2, block_index=2),
        ])
        model = structure([
            {
                "source_title": "1 Intro",
                "semantic_title": "Intro",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 2,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, _ = anchor_document_structure(model, doc)
        segments = segment_extracted_document(doc, anchored)

        self.assertEqual(segments[-1].section_title, "1 Intro")
        self.assertEqual(segments[-1].page, 2)

    def test_content_before_first_heading_uses_fallback(self):
        raw = "Preface body\n1 Intro\nBody"
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=1, block_index=1)])
        model = structure([
            {
                "source_title": "1 Intro",
                "semantic_title": "Intro",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            }
        ])

        anchored, _ = anchor_document_structure(model, doc)
        segments = segment_extracted_document(doc, anchored, fallback_section_title="FIXTURE")

        self.assertEqual(segments[0].section_title, "FIXTURE")
        self.assertEqual(segments[0].provenance, "fallback_before_first_heading")

    def test_two_sections_sharing_one_page_split(self):
        raw = "1 One\n" + ("A " * 80) + "\n2 Two\n" + ("B " * 80)
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=1, block_index=1)])
        model = structure([
            {
                "source_title": "1 One",
                "semantic_title": "One",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "2 Two",
                "semantic_title": "Two",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])
        anchored, _ = anchor_document_structure(model, doc)
        segments = segment_extracted_document(doc, anchored)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].section_title, "1 One")
        self.assertEqual(segments[1].section_title, "2 Two")

    def test_no_chunk_crosses_anchored_section_boundary(self):
        raw = "1 One\n" + ("Alpha " * 80) + "\n2 Two\n" + ("Beta " * 80)
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=1, block_index=1)])
        model = structure([
            {
                "source_title": "1 One",
                "semantic_title": "One",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "2 Two",
                "semantic_title": "Two",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])
        anchored, _ = anchor_document_structure(model, doc)
        chunks = diagnostic_chunks_from_segments(segment_extracted_document(doc, anchored))

        self.assertTrue(chunks)
        for chunk in chunks:
            if chunk.section_title == "1 One":
                self.assertNotIn("2 Two", chunk.text)
            if chunk.section_title == "2 Two":
                self.assertNotIn("Alpha", chunk.text)

    def test_duplicate_ambiguous_title_occurrence_does_not_reuse_same_anchor(self):
        raw = "1 Repeat\nBody\n1 Repeat\nMore"
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=1, block_index=1)])
        model = structure([
            {
                "source_title": "1 Repeat",
                "semantic_title": "Repeat",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "1 Repeat",
                "semantic_title": "Repeat again",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.anchored, 1)
        self.assertEqual(summary.unresolved, 1)
        self.assertIsNone(anchored.sections[1].anchor)

    def test_source_order_conflict_is_rejected(self):
        raw = "2 Later\nBody\n1 Earlier\nBody"
        doc = extracted([ExtractedBlock(text=raw, raw_text=raw, page=1, block_index=1)])
        model = structure([
            {
                "source_title": "1 Earlier",
                "semantic_title": "Earlier",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "2 Later",
                "semantic_title": "Later",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])

        anchored, summary = anchor_document_structure(model, doc)

        self.assertEqual(summary.source_order_violations, 1)
        self.assertIsNone(anchored.sections[1].anchor)

    def test_low_confidence_structure_can_be_detected_for_fallback(self):
        model = structure([], confidence="LOW")
        self.assertEqual(model.confidence, "LOW")

    def test_partial_safe_mapping(self):
        doc = extracted([
            ExtractedBlock(text="1 Good\nBody", raw_text="1 Good\nBody", page=1, block_index=1)
        ])
        model = structure([
            {
                "source_title": "1 Good",
                "semantic_title": "Good",
                "level": 1,
                "parent": None,
                "start_page": 1,
                "end_page": 1,
                "confidence": "HIGH",
                "scope": "global",
            },
            {
                "source_title": "2 Missing",
                "semantic_title": "Missing",
                "level": 1,
                "parent": None,
                "start_page": 2,
                "end_page": 2,
                "confidence": "HIGH",
                "scope": "global",
            },
        ])

        anchored, summary = anchor_document_structure(model, doc)
        segments = segment_extracted_document(doc, anchored)

        self.assertEqual(summary.anchored, 1)
        self.assertEqual(summary.unresolved, 1)
        self.assertEqual(segments[-1].section_title, "1 Good")

    def test_existing_chunker_algorithm_is_reused_by_adapter(self):
        chunks = production_compatible_chunk_text("A" * 1200, max_chars=1000, overlap=200)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(len(chunks[0]), 1000)


if __name__ == "__main__":
    unittest.main()
