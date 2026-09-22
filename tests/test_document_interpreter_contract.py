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
    SOURCE_FORMAT_PDF,
)
from document_interpreter_contract import (
    CONTENT_ROLE_BODY,
    DocumentInterpreterResult,
    InterpretedContentBlock,
    InterpretedDocumentSection,
    INTERPRETER_STATUS_ACCEPTED,
    INTERPRETER_STATUS_FALLBACK,
    source_span,
    validate_document_interpreter_result,
)


def block(
    block_id: str,
    text: str,
    order: int,
    *,
    section_id=None,
    page=1,
) -> InterpretedContentBlock:
    return InterpretedContentBlock(
        block_id=block_id,
        text=text,
        raw_text=text,
        source_order=order,
        source_span=source_span("page", page, order, end_offset=len(text)),
        role=CONTENT_ROLE_BODY,
        page=page,
        section_id=section_id,
    )


def section(
    section_id: str,
    title: str,
    order: int,
    *,
    level=1,
    parent_id=None,
    owned_blocks=None,
) -> InterpretedDocumentSection:
    return InterpretedDocumentSection(
        section_id=section_id,
        title=title,
        level=level,
        parent_id=parent_id,
        source_order=order,
        source_span=source_span("page", 1, order, end_offset=len(title)),
        confidence=CONFIDENCE_HIGH,
        origin=ORIGIN_EXPLICIT_SOURCE,
        owned_blocks=owned_blocks or [],
    )


class DocumentInterpreterContractTests(unittest.TestCase):
    def test_valid_high_confidence_result_can_be_authoritative(self):
        result = DocumentInterpreterResult(
            document_id="doc",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Biology.pdf",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section(
                    "s1",
                    "5 Prokaryotic gene regulation",
                    1,
                    owned_blocks=[
                        block("b1", "5 Prokaryotic gene regulation", 1, section_id="s1"),
                        block("b2", "Body content", 2, section_id="s1"),
                    ],
                ),
                section(
                    "s2",
                    "5.1 Why operons are useful",
                    3,
                    level=2,
                    parent_id="s1",
                    owned_blocks=[
                        block("b3", "5.1 Why operons are useful", 3, section_id="s2"),
                        block("b4", "Subsection body", 4, section_id="s2"),
                    ],
                ),
            ],
        )

        report = validate_document_interpreter_result(result)

        self.assertTrue(report.is_valid)
        self.assertTrue(result.has_authoritative_structure)
        self.assertEqual([b.block_id for b in result.all_content_blocks], ["b1", "b2", "b3", "b4"])

    def test_duplicate_content_ownership_is_invalid(self):
        shared = block("b1", "Shared body", 2, section_id="s1")
        result = DocumentInterpreterResult(
            document_id="doc",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Biology.pdf",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section("s1", "One", 1, owned_blocks=[shared]),
                section("s2", "Two", 3, owned_blocks=[shared]),
            ],
        )

        report = validate_document_interpreter_result(result)

        self.assertFalse(report.is_valid)
        self.assertIn("duplicate_content_block_id", {issue.code for issue in report.issues})

    def test_owned_block_must_reference_owning_section(self):
        result = DocumentInterpreterResult(
            document_id="doc",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Biology.pdf",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section(
                    "s1",
                    "One",
                    1,
                    owned_blocks=[block("b1", "Body", 2, section_id="other")],
                )
            ],
        )

        report = validate_document_interpreter_result(result)

        self.assertFalse(report.is_valid)
        self.assertIn("owned_block_section_mismatch", {issue.code for issue in report.issues})

    def test_parent_must_precede_child_and_have_higher_level(self):
        result = DocumentInterpreterResult(
            document_id="doc",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Biology.pdf",
            status=INTERPRETER_STATUS_ACCEPTED,
            confidence=CONFIDENCE_HIGH,
            sections=[
                section("child", "Child", 1, level=1, parent_id="parent"),
                section("parent", "Parent", 2, level=1),
            ],
        )

        report = validate_document_interpreter_result(result)

        self.assertFalse(report.is_valid)
        issue_codes = {issue.code for issue in report.issues}
        self.assertIn("parent_not_before_child", issue_codes)

    def test_fallback_result_keeps_unresolved_content_without_claiming_structure(self):
        result = DocumentInterpreterResult(
            document_id="doc",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Unstructured.pdf",
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[block("b1", "Unresolved body", 1)],
        )

        report = validate_document_interpreter_result(result)

        self.assertTrue(report.is_valid)
        self.assertFalse(result.has_authoritative_structure)
        self.assertEqual(result.all_content_blocks[0].block_id, "b1")

    def test_unresolved_block_must_not_claim_section_ownership(self):
        result = DocumentInterpreterResult(
            document_id="doc",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Unstructured.pdf",
            status=INTERPRETER_STATUS_FALLBACK,
            confidence=CONFIDENCE_LOW,
            unresolved_blocks=[block("b1", "Unresolved body", 1, section_id="s1")],
        )

        report = validate_document_interpreter_result(result)

        self.assertFalse(report.is_valid)
        self.assertIn("unresolved_block_has_section", {issue.code for issue in report.issues})


if __name__ == "__main__":
    unittest.main()
