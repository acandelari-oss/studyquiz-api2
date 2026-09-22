import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_DOCX,
)
from document_interpreter_chunk_adapter import chunks_from_interpreted_document
from document_interpreter_contract import (
    CONTENT_ROLE_BODY,
    CONTENT_ROLE_HEADING,
    DocumentInterpreterResult,
    InterpretedContentBlock,
    InterpretedDocumentSection,
    INTERPRETER_STATUS_ACCEPTED,
    INTERPRETER_STATUS_FALLBACK,
    source_span,
)


def block(
    block_id,
    text,
    order,
    *,
    section_id=None,
    role=CONTENT_ROLE_BODY,
    page=None,
):
    return InterpretedContentBlock(
        block_id=block_id,
        text=text,
        raw_text=text,
        source_order=order,
        source_span=source_span("docx_block", order, order, end_offset=len(text)),
        role=role,
        page=page,
        section_id=section_id,
        provenance=ORIGIN_EXPLICIT_SOURCE,
        confidence=CONFIDENCE_HIGH if section_id else CONFIDENCE_LOW,
    )


def section(section_id, title, order, *, level=1, parent_id=None, owned_blocks=None):
    return InterpretedDocumentSection(
        section_id=section_id,
        title=title,
        logical_title=title,
        level=level,
        parent_id=parent_id,
        source_order=order,
        source_span=source_span("docx_block", order, order, end_offset=len(title)),
        confidence=CONFIDENCE_HIGH,
        origin=ORIGIN_EXPLICIT_SOURCE,
        owned_blocks=owned_blocks or [],
    )


class DocumentInterpreterChunkAdapterTests(unittest.TestCase):
    def test_chunks_do_not_cross_section_boundaries(self):
        intro = section(
            "s1",
            "1 Introduction",
            1,
            owned_blocks=[
                block("b1", "1 Introduction", 1, section_id="s1", role=CONTENT_ROLE_HEADING),
                block("b2", "Intro body " * 40, 2, section_id="s1", page=1),
            ],
        )
        methods = section(
            "s2",
            "2 Methods",
            3,
            owned_blocks=[
                block("b3", "2 Methods", 3, section_id="s2", role=CONTENT_ROLE_HEADING),
                block("b4", "Methods body " * 40, 4, section_id="s2", page=2),
            ],
        )
        result = DocumentInterpreterResult(
            document_id="doc-1",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Study Notes.docx",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[intro, methods],
        )

        chunks = chunks_from_interpreted_document(result, min_chars=0)

        self.assertEqual([chunk.section_id for chunk in chunks], ["s1", "s2"])
        self.assertIn("Intro body", chunks[0].chunk_text)
        self.assertNotIn("Methods body", chunks[0].chunk_text)
        self.assertIn("Methods body", chunks[1].chunk_text)
        self.assertNotIn("Intro body", chunks[1].chunk_text)

    def test_section_path_preserves_parent_hierarchy(self):
        parent = section(
            "s1",
            "5 Gene regulation",
            1,
            owned_blocks=[
                block("b1", "5 Gene regulation", 1, section_id="s1", role=CONTENT_ROLE_HEADING),
                block("b2", "Parent body " * 30, 2, section_id="s1"),
            ],
        )
        child = section(
            "s2",
            "5.1 Operons",
            3,
            level=2,
            parent_id="s1",
            owned_blocks=[
                block("b3", "5.1 Operons", 3, section_id="s2", role=CONTENT_ROLE_HEADING),
                block("b4", "Child body " * 30, 4, section_id="s2"),
            ],
        )
        result = DocumentInterpreterResult(
            document_id="doc-1",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Biology.docx",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[parent, child],
        )

        chunks = chunks_from_interpreted_document(result, min_chars=0)

        self.assertEqual(chunks[1].section_path, "5 Gene regulation > 5.1 Operons")
        self.assertEqual(chunks[1].section_title, "5.1 Operons")

    def test_unresolved_blocks_remain_unowned_fallback_chunks(self):
        result = DocumentInterpreterResult(
            document_id="doc-1",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Unstructured.docx",
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[
                block("b1", "Unresolved preface " * 30, 1, section_id=None),
            ],
        )

        chunks = chunks_from_interpreted_document(result, min_chars=0)

        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].section_id)
        self.assertTrue(chunks[0].unresolved)
        self.assertEqual(chunks[0].section_path, "Unstructured.docx")

    def test_short_chunks_are_filtered_like_diagnostics(self):
        result = DocumentInterpreterResult(
            document_id="doc-1",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Short.docx",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section(
                    "s1",
                    "Short",
                    1,
                    owned_blocks=[block("b1", "Short body", 1, section_id="s1")],
                )
            ],
        )

        chunks = chunks_from_interpreted_document(result)

        self.assertEqual(chunks, [])

    def test_invalid_interpreter_result_fails_before_chunking(self):
        shared = block("b1", "Shared body " * 30, 2, section_id="s1")
        result = DocumentInterpreterResult(
            document_id="doc-1",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Invalid.docx",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section("s1", "One", 1, owned_blocks=[shared]),
                section("s2", "Two", 3, owned_blocks=[shared]),
            ],
        )

        with self.assertRaisesRegex(ValueError, "duplicate_content_block_id"):
            chunks_from_interpreted_document(result, min_chars=0)

    def test_custom_chunker_can_split_long_section_without_crossing_sections(self):
        result = DocumentInterpreterResult(
            document_id="doc-1",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Split.docx",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section(
                    "s1",
                    "Long",
                    1,
                    owned_blocks=[block("b1", "abcdefghij" * 20, 1, section_id="s1")],
                )
            ],
        )

        chunks = chunks_from_interpreted_document(
            result,
            chunk_text=lambda text: [text[:50], text[50:]],
            min_chars=0,
        )

        self.assertEqual(len(chunks), 2)
        self.assertEqual({chunk.section_id for chunk in chunks}, {"s1"})
        self.assertEqual([chunk.chunk_index for chunk in chunks], [1, 2])


if __name__ == "__main__":
    unittest.main()
