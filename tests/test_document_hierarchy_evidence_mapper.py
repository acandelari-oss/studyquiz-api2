import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import (
    ROLE_BODY,
    ROLE_HEADING,
    ROLE_LIST,
    ROLE_TITLE,
    CanonicalBlock,
    CanonicalDocumentInput,
    CanonicalUnit,
    SourcePosition,
)
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONTENT_KIND_BODY,
    CONTENT_KIND_DOCUMENT_BOUNDARY,
    CONTENT_KIND_LIST_ITEM,
    EVIDENCE_ROLE_BODY,
    EVIDENCE_ROLE_HEADING,
    EVIDENCE_ROLE_LIST,
    EVIDENCE_ROLE_TITLE,
)
from document_hierarchy_evidence_mapper import (
    map_block_to_structural_evidence,
    map_document_to_structural_evidence,
)


def source_position(unit_type="document_flow", unit_index=1, block_index=1, start=0, end=None):
    return SourcePosition(unit_type, unit_index, block_index, start, end)


def block(
    block_id,
    text,
    *,
    source_order=1,
    position=None,
    role=ROLE_BODY,
    style=None,
    hierarchy=None,
    metadata=None,
):
    return CanonicalBlock(
        block_id=block_id,
        text=" ".join(text.split()),
        raw_text=text,
        source_order=source_order,
        source_position=position or source_position(end=len(text)),
        role_hint=role,
        style_hint=style,
        native_hierarchy_hint=hierarchy,
        metadata=metadata or {},
    )


def document(file_type, blocks, *, unit_type="document_flow", unit_index=1, label="Document flow"):
    return CanonicalDocumentInput(
        file_type=file_type,
        filename=f"fixture.{file_type}",
        title=f"fixture.{file_type}",
        units=[
            CanonicalUnit(
                unit_id=f"{unit_type}-{unit_index:04d}",
                unit_type=unit_type,
                unit_index=unit_index,
                display_label=label,
                blocks=blocks,
            )
        ],
    )


class DocumentHierarchyEvidenceMapperTests(unittest.TestCase):
    def test_docx_native_heading_evidence_is_preserved_without_final_hierarchy(self):
        doc = document(
            "docx",
            [
                block(
                    "paragraph-0002",
                    "5.1 Operons",
                    source_order=2,
                    position=source_position(block_index=2, end=11),
                    role=ROLE_HEADING,
                    style="Heading 2",
                    hierarchy=2,
                    metadata={"paragraph_index": 2},
                )
            ],
        )

        evidence = map_document_to_structural_evidence(doc)[0]

        self.assertEqual(evidence.raw_text, "5.1 Operons")
        self.assertEqual(evidence.normalized_text, "5.1 Operons")
        self.assertEqual(evidence.source_order, 2)
        self.assertEqual(evidence.source_span.start, doc.blocks[0].source_position)
        self.assertEqual(evidence.evidence_role, EVIDENCE_ROLE_HEADING)
        self.assertEqual(evidence.content_kind, CONTENT_KIND_DOCUMENT_BOUNDARY)
        self.assertEqual(evidence.native_evidence["style_hint"], "Heading 2")
        self.assertEqual(evidence.native_evidence["native_hierarchy_hint"], 2)
        self.assertEqual(evidence.native_evidence["paragraph_index"], 2)
        self.assertEqual(evidence.confidence_hint, CONFIDENCE_HIGH)
        self.assertFalse(hasattr(evidence, "parent_id"))
        self.assertFalse(hasattr(evidence, "level"))

    def test_docx_body_and_list_content_remain_source_evidence_not_canonical_hierarchy(self):
        doc = document(
            "docx",
            [
                block(
                    "paragraph-0003",
                    "Ordinary explanation text.",
                    source_order=3,
                    position=source_position(block_index=3, end=26),
                    role=ROLE_BODY,
                    metadata={"paragraph_index": 3},
                ),
                block(
                    "paragraph-0004",
                    "Example list item",
                    source_order=4,
                    position=source_position(block_index=4, end=17),
                    role=ROLE_LIST,
                    hierarchy=1,
                    metadata={"paragraph_index": 4},
                ),
            ],
        )

        body, list_item = map_document_to_structural_evidence(doc)

        self.assertEqual(body.evidence_role, EVIDENCE_ROLE_BODY)
        self.assertEqual(body.content_kind, CONTENT_KIND_BODY)
        self.assertEqual(body.confidence_hint, CONFIDENCE_LOW)
        self.assertEqual(list_item.evidence_role, EVIDENCE_ROLE_LIST)
        self.assertEqual(list_item.content_kind, CONTENT_KIND_LIST_ITEM)
        self.assertEqual(list_item.numbering_evidence["list_level"], 1)
        self.assertFalse(hasattr(list_item, "parent_id"))

    def test_pptx_title_placeholder_role_is_preserved_without_global_section_assertion(self):
        slide_block = block(
            "slide-0001-block-0001",
            "Cell metabolism",
            source_order=1,
            position=source_position("slide", 1, 1, end=15),
            role=ROLE_TITLE,
            hierarchy=0,
            metadata={"shape_index": 1, "placeholder_type": "TITLE (1)", "bullet_level": 0},
        )
        doc = document("pptx", [slide_block], unit_type="slide", label="Cell metabolism")

        evidence = map_document_to_structural_evidence(doc)[0]

        self.assertEqual(evidence.evidence_role, EVIDENCE_ROLE_TITLE)
        self.assertEqual(evidence.content_kind, CONTENT_KIND_DOCUMENT_BOUNDARY)
        self.assertEqual(evidence.native_evidence["placeholder_type"], "TITLE (1)")
        self.assertEqual(evidence.native_evidence["shape_index"], 1)
        self.assertEqual(evidence.structural_context["unit_type"], "slide")
        self.assertEqual(evidence.structural_context["unit_display_label"], "Cell metabolism")
        self.assertFalse(hasattr(evidence, "parent_id"))

    def test_pptx_bullet_depth_is_preserved_as_local_source_evidence(self):
        bullet = block(
            "slide-0001-block-0002",
            "ATP production",
            source_order=2,
            position=source_position("slide", 1, 2, end=14),
            role=ROLE_LIST,
            hierarchy=1,
            metadata={"shape_index": 2, "placeholder_type": "BODY (2)", "bullet_level": 1},
        )
        doc = document("pptx", [bullet], unit_type="slide", label="Cell metabolism")

        evidence = map_document_to_structural_evidence(doc)[0]

        self.assertEqual(evidence.evidence_role, EVIDENCE_ROLE_LIST)
        self.assertEqual(evidence.content_kind, CONTENT_KIND_LIST_ITEM)
        self.assertEqual(evidence.native_evidence["bullet_level"], 1)
        self.assertEqual(evidence.numbering_evidence["list_level"], 1)
        self.assertNotEqual(evidence.content_kind, CONTENT_KIND_DOCUMENT_BOUNDARY)

    def test_pdf_block_maps_only_currently_available_page_order_and_text_evidence(self):
        pdf_block = block(
            "page-0003-block-0001",
            "Raw PDF page text",
            source_order=3,
            position=source_position("page", 3, 1, end=17),
            role=ROLE_BODY,
        )
        doc = document("pdf", [pdf_block], unit_type="page", unit_index=3, label="Page 3")

        evidence = map_document_to_structural_evidence(doc)[0]

        self.assertEqual(evidence.source_format, "pdf")
        self.assertEqual(evidence.source_span.start.unit_type, "page")
        self.assertEqual(evidence.source_span.start.unit_index, 3)
        self.assertEqual(evidence.source_order, 3)
        self.assertEqual(evidence.raw_text, "Raw PDF page text")
        self.assertEqual(evidence.visual_evidence, {})
        self.assertEqual(evidence.numbering_evidence, {})
        self.assertEqual(evidence.content_kind, CONTENT_KIND_BODY)

    def test_missing_evidence_maps_to_neutral_empty_fields(self):
        minimal = block(
            "unknown-0001",
            "Minimal text",
            role="unknown",
            position=source_position("page", 1, 1, end=12),
        )
        doc = document("pdf", [minimal], unit_type="page", label="Page 1")

        evidence = map_block_to_structural_evidence(doc, minimal, unit=doc.units[0])

        self.assertEqual(evidence.evidence_role, "unknown")
        self.assertEqual(evidence.content_kind, "unknown")
        self.assertEqual(evidence.native_evidence["role_hint"], "unknown")
        self.assertEqual(evidence.visual_evidence, {})
        self.assertEqual(evidence.numbering_evidence, {})
        self.assertEqual(evidence.confidence_hint, CONFIDENCE_LOW)


if __name__ == "__main__":
    unittest.main()
