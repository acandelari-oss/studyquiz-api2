import sys
import unittest
from pathlib import Path
from typing import Optional


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import SourcePosition
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONTENT_KIND_BODY,
    CONTENT_KIND_DOCUMENT_BOUNDARY,
    CONTENT_KIND_LIST_ITEM,
    EVIDENCE_ROLE_BODY,
    EVIDENCE_ROLE_HEADING,
    EVIDENCE_ROLE_LIST,
    EVIDENCE_ROLE_TITLE,
    EvidenceProvenance,
    CanonicalDocumentNode,
    CanonicalDocumentStructure,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_DOCX,
    SOURCE_FORMAT_PDF,
    SOURCE_FORMAT_PPTX,
    STATUS_ACCEPTED,
    SourceSpan,
    StructuralEvidence,
)
from document_hierarchy_preserver import (
    preserve_document_hierarchy,
    render_document_hierarchy_diagnostics,
    validate_canonical_document_structure,
)


def span(order: int, *, unit_type: str = "document_flow", unit_index: int = 1) -> SourceSpan:
    position = SourcePosition(unit_type, unit_index, order, 0, 10)
    return SourceSpan(start=position, end=position)


def docx_heading(order: int, level: int, *, ilvl: Optional[int] = None) -> StructuralEvidence:
    numbering = {}
    if ilvl is not None:
        numbering = {"source": "direct", "num_id": 5, "abstract_num_id": 7, "ilvl": ilvl}
    return StructuralEvidence(
        evidence_id=f"docx-{order}",
        source_format=SOURCE_FORMAT_DOCX,
        source_order=order,
        source_span=span(order),
        raw_text=f"Heading {order}",
        normalized_text=f"Heading {order}",
        evidence_role=EVIDENCE_ROLE_HEADING,
        content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
        native_evidence={"native_hierarchy_hint": level, "style_hint": f"Heading {level}"},
        numbering_evidence=numbering,
        confidence_hint=CONFIDENCE_HIGH,
    )


def docx_numbered(order: int, ilvl: int) -> StructuralEvidence:
    return StructuralEvidence(
        evidence_id=f"docx-{order}",
        source_format=SOURCE_FORMAT_DOCX,
        source_order=order,
        source_span=span(order),
        raw_text=f"List {order}",
        normalized_text=f"List {order}",
        evidence_role=EVIDENCE_ROLE_LIST,
        content_kind=CONTENT_KIND_LIST_ITEM,
        native_evidence={"native_hierarchy_hint": ilvl},
        numbering_evidence={"source": "direct", "num_id": 5, "ilvl": ilvl},
    )


def pptx_title(order: int, text: str, *, slide: int, layout: str = "Title and Content") -> StructuralEvidence:
    return StructuralEvidence(
        evidence_id=f"pptx-{order}",
        source_format=SOURCE_FORMAT_PPTX,
        source_order=order,
        source_span=span(order, unit_type="slide", unit_index=slide),
        raw_text=text,
        normalized_text=text,
        evidence_role=EVIDENCE_ROLE_TITLE,
        content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
        native_evidence={
            "placeholder_type": "TITLE",
            "layout_name": layout,
            "layout_index": 1,
        },
        visual_evidence={"x": 1, "y": 1, "width": 100, "height": 20},
        confidence_hint=CONFIDENCE_HIGH,
    )


def pptx_bullet(order: int, text: str, *, slide: int, level: int = 1) -> StructuralEvidence:
    return StructuralEvidence(
        evidence_id=f"pptx-{order}",
        source_format=SOURCE_FORMAT_PPTX,
        source_order=order,
        source_span=span(order, unit_type="slide", unit_index=slide),
        raw_text=text,
        normalized_text=text,
        evidence_role=EVIDENCE_ROLE_LIST,
        content_kind=CONTENT_KIND_LIST_ITEM,
        native_evidence={"bullet_level": level, "placeholder_type": "BODY"},
        numbering_evidence={"list_level": level},
    )


def pptx_body(order: int, text: str, *, slide: int) -> StructuralEvidence:
    return StructuralEvidence(
        evidence_id=f"pptx-{order}",
        source_format=SOURCE_FORMAT_PPTX,
        source_order=order,
        source_span=span(order, unit_type="slide", unit_index=slide),
        raw_text=text,
        normalized_text=text,
        evidence_role=EVIDENCE_ROLE_BODY,
        content_kind=CONTENT_KIND_BODY,
        native_evidence={"layout_name": "Blank"},
    )


def pdf_text(
    order: int,
    text: str,
    *,
    page: int = 1,
    size: int = 12,
    y: int = 650,
    font: str = "Helvetica",
) -> StructuralEvidence:
    return StructuralEvidence(
        evidence_id=f"pdf-{order}",
        source_format=SOURCE_FORMAT_PDF,
        source_order=order,
        source_span=span(order, unit_type="page", unit_index=page),
        raw_text=text,
        normalized_text=text,
        evidence_role=EVIDENCE_ROLE_BODY,
        content_kind=CONTENT_KIND_BODY,
        native_evidence={"page_index": page},
        visual_evidence={
            "font_size": float(size),
            "font_name": font,
            "font_subtype": "Type1",
            "x0": 72.0,
            "y0": float(y),
            "page_height": 792.0,
            "page_width": 612.0,
        },
    )


class CrossFormatHierarchyPreserverTests(unittest.TestCase):
    def test_docx_heading_1_2_3_hierarchy_is_preserved(self):
        structure = preserve_document_hierarchy(
            [docx_heading(1, 1), docx_heading(2, 2), docx_heading(3, 3)],
            document_id="doc",
            document_title="docx",
            source_format=SOURCE_FORMAT_DOCX,
        )

        self.assertEqual([node.level for node in structure.nodes], [1, 2, 3])
        self.assertEqual(structure.nodes[1].parent_id, structure.nodes[0].node_id)
        self.assertEqual(structure.nodes[2].parent_id, structure.nodes[1].node_id)

    def test_docx_malformed_heading_jump_remains_unresolved(self):
        structure = preserve_document_hierarchy(
            [docx_heading(1, 1), docx_heading(2, 3)],
            document_id="doc",
            source_format=SOURCE_FORMAT_DOCX,
        )

        self.assertEqual(structure.nodes[1].status, "unresolved")
        self.assertIsNone(structure.nodes[1].parent_id)

    def test_docx_numbered_normal_paragraphs_are_not_promoted(self):
        structure = preserve_document_hierarchy(
            [docx_heading(1, 1), docx_numbered(2, 0), docx_numbered(3, 1)],
            document_id="doc",
            source_format=SOURCE_FORMAT_DOCX,
        )

        self.assertEqual(len(structure.nodes), 1)
        self.assertEqual(len(structure.diagnostics["ignored_local_items"]), 2)

    def test_docx_numbering_conflict_does_not_rewrite_hierarchy(self):
        structure = preserve_document_hierarchy(
            [docx_heading(1, 1, ilvl=2)],
            document_id="doc",
            source_format=SOURCE_FORMAT_DOCX,
        )

        self.assertEqual(structure.nodes[0].level, 1)
        self.assertEqual(structure.diagnostics["conflicts"][0]["reason"], "heading_numbering_conflict")

    def test_pptx_normal_title_content_deck_preserves_slide_titles(self):
        structure = preserve_document_hierarchy(
            [
                pptx_title(1, "Respiration", slide=1),
                pptx_bullet(2, "ATP", slide=1),
                pptx_title(3, "Photosynthesis", slide=2),
            ],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual([node.source_title for node in structure.nodes], ["Respiration", "Photosynthesis"])
        self.assertEqual([node.level for node in structure.nodes], [1, 1])
        self.assertEqual(len(structure.diagnostics["ignored_local_items"]), 1)

    def test_pptx_title_placeholders_preserved_in_slide_order(self):
        structure = preserve_document_hierarchy(
            [pptx_title(5, "Later", slide=2), pptx_title(2, "Earlier", slide=1)],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual([node.source_title for node in structure.nodes], ["Earlier", "Later"])

    def test_pptx_bullet_hierarchy_remains_local_content(self):
        structure = preserve_document_hierarchy(
            [pptx_title(1, "Slide", slide=1), pptx_bullet(2, "Nested", slide=1, level=2)],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual(len(structure.nodes), 1)
        self.assertEqual(structure.diagnostics["ignored_local_items"][0]["reason"], "pptx_bullet_local_content")

    def test_pptx_repeated_slide_titles_remain_distinct(self):
        structure = preserve_document_hierarchy(
            [pptx_title(1, "Action potential", slide=1), pptx_title(2, "Action potential", slide=2)],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual(len(structure.nodes), 2)
        self.assertNotEqual(structure.nodes[0].node_id, structure.nodes[1].node_id)

    def test_pptx_section_header_layout_groups_following_slide_title(self):
        structure = preserve_document_hierarchy(
            [
                pptx_title(1, "Gene regulation", slide=1, layout="Section Header"),
                pptx_title(2, "Operons", slide=2),
            ],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual([node.level for node in structure.nodes], [1, 2])
        self.assertEqual(structure.nodes[1].parent_id, structure.nodes[0].node_id)

    def test_pptx_arbitrary_text_boxes_are_not_promoted(self):
        structure = preserve_document_hierarchy(
            [pptx_body(1, "Side annotation", slide=1)],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual(structure.nodes, [])
        self.assertEqual(structure.diagnostics["ignored_local_items"][0]["reason"], "pptx_non_title_content_not_promoted")

    def test_pdf_clear_recurring_visual_heading_pattern_is_preserved(self):
        structure = preserve_document_hierarchy(
            [
                pdf_text(1, "Introduction", page=1, size=24),
                pdf_text(2, "Body text one", page=1, size=12),
                pdf_text(3, "More body", page=1, size=12),
                pdf_text(4, "Methods", page=2, size=24),
                pdf_text(5, "Body text two", page=2, size=12),
                pdf_text(6, "More body two", page=2, size=12),
            ],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual([node.source_title for node in structure.nodes], ["Introduction", "Methods"])
        self.assertEqual([node.level for node in structure.nodes], [1, 1])

    def test_pdf_hierarchical_numbered_headings_need_matching_visual_evidence(self):
        structure = preserve_document_hierarchy(
            [
                pdf_text(1, "1 Introduction", page=1, size=24),
                pdf_text(2, "Body text one", page=1, size=12),
                pdf_text(3, "1.1 Background", page=1, size=18),
                pdf_text(4, "More body", page=1, size=12),
                pdf_text(5, "2 Methods", page=2, size=24),
                pdf_text(6, "Body text two", page=2, size=12),
                pdf_text(7, "2.1 Design", page=2, size=18),
                pdf_text(8, "More body two", page=2, size=12),
            ],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual([node.level for node in structure.nodes], [1, 2, 1, 2])
        self.assertEqual(structure.nodes[1].parent_id, structure.nodes[0].node_id)
        self.assertEqual(structure.nodes[3].parent_id, structure.nodes[2].node_id)

    def test_pdf_local_numbered_list_is_not_promoted(self):
        structure = preserve_document_hierarchy(
            [
                pdf_text(1, "Procedure", size=12),
                pdf_text(2, "1 Add reagent", size=12),
                pdf_text(3, "2 Mix", size=12),
            ],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual(structure.nodes, [])
        self.assertTrue(
            any(item["reason"] == "pdf_numbered_list_without_visual_structure" for item in structure.diagnostics["ignored_local_items"])
        )

    def test_pdf_larger_isolated_text_without_recurrence_is_unresolved(self):
        structure = preserve_document_hierarchy(
            [pdf_text(1, "Standalone large line", size=24), pdf_text(2, "Body", size=12), pdf_text(3, "More body", size=12)],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual(structure.nodes, [])
        self.assertEqual(structure.diagnostics["unresolved_items"][0]["reason"], "pdf_insufficient_recurring_visual_structure")

    def test_pdf_repeated_page_header_is_not_promoted(self):
        structure = preserve_document_hierarchy(
            [
                pdf_text(1, "Course Header", page=1, size=18, y=760),
                pdf_text(2, "Body one", page=1, size=12),
                pdf_text(3, "Body one b", page=1, size=12),
                pdf_text(4, "Course Header", page=2, size=18, y=760),
                pdf_text(5, "Body two", page=2, size=12),
            ],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual(structure.nodes, [])
        self.assertTrue(
            any(item["reason"] == "pdf_repeated_running_header" for item in structure.diagnostics["ignored_local_items"])
        )

    def test_pdf_multiple_visual_tiers_can_create_parent_child_hierarchy(self):
        structure = preserve_document_hierarchy(
            [
                pdf_text(1, "Main A", size=24),
                pdf_text(2, "Body", size=12),
                pdf_text(3, "Sub A", size=18),
                pdf_text(4, "More body", size=12),
                pdf_text(5, "Sub B", size=18),
                pdf_text(6, "Body b", size=12),
                pdf_text(7, "Main B", size=24),
                pdf_text(8, "Body c", size=12),
            ],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual([node.level for node in structure.nodes], [1, 2, 2, 1])
        self.assertEqual(structure.nodes[1].parent_id, structure.nodes[0].node_id)
        self.assertEqual(structure.nodes[2].parent_id, structure.nodes[0].node_id)

    def test_pdf_ambiguous_fragment_remains_unresolved(self):
        structure = preserve_document_hierarchy(
            [pdf_text(1, "Possible Heading", size=16), pdf_text(2, "Body", size=12), pdf_text(3, "More body", size=12)],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertEqual(structure.nodes, [])
        self.assertEqual(len(structure.diagnostics["unresolved_items"]), 1)

    def test_all_accepted_nodes_retain_provenance(self):
        structure = preserve_document_hierarchy(
            [docx_heading(1, 1)],
            document_id="doc",
            source_format=SOURCE_FORMAT_DOCX,
        )

        self.assertEqual(structure.nodes[0].evidence_ids, ["docx-1"])
        self.assertEqual(structure.nodes[0].provenance.supporting_evidence_ids, ["docx-1"])

    def test_source_order_remains_stable(self):
        structure = preserve_document_hierarchy(
            [pptx_title(3, "Three", slide=3), pptx_title(1, "One", slide=1), pptx_title(2, "Two", slide=2)],
            document_id="ppt",
            source_format=SOURCE_FORMAT_PPTX,
        )

        self.assertEqual([node.source_order for node in structure.nodes], [1, 2, 3])

    def test_invalid_parent_relationships_are_diagnosed(self):
        bad = CanonicalDocumentStructure(
            document_id="bad",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="bad",
            structure_status=STATUS_ACCEPTED,
            structure_confidence=CONFIDENCE_HIGH,
            nodes=[
                CanonicalDocumentNode(
                    node_id="child",
                    source_title="Child",
                    logical_title="Child",
                    level=2,
                    parent_id="missing",
                    source_order=1,
                    source_span=span(1),
                    origin=ORIGIN_EXPLICIT_SOURCE,
                    confidence=CONFIDENCE_HIGH,
                    status=STATUS_ACCEPTED,
                    evidence_ids=["docx-1"],
                    provenance=EvidenceProvenance(supporting_evidence_ids=["docx-1"]),
                )
            ],
        )

        report = validate_canonical_document_structure(bad, [docx_heading(1, 2)])
        self.assertFalse(report.is_valid)
        self.assertEqual(report.issues[0].code, "missing_parent")

    def test_cross_format_entry_point_returns_canonical_structure_for_all_formats(self):
        docx = preserve_document_hierarchy([docx_heading(1, 1)], document_id="docx", source_format=SOURCE_FORMAT_DOCX)
        pptx = preserve_document_hierarchy([pptx_title(1, "Title", slide=1)], document_id="pptx", source_format=SOURCE_FORMAT_PPTX)
        pdf = preserve_document_hierarchy(
            [pdf_text(1, "A", size=24), pdf_text(2, "body", size=12), pdf_text(3, "B", size=24), pdf_text(4, "body", size=12)],
            document_id="pdf",
            source_format=SOURCE_FORMAT_PDF,
        )

        self.assertIsInstance(docx, CanonicalDocumentStructure)
        self.assertIsInstance(pptx, CanonicalDocumentStructure)
        self.assertIsInstance(pdf, CanonicalDocumentStructure)

    def test_no_model_or_api_call_dependency(self):
        import document_hierarchy_preserver

        self.assertFalse(hasattr(document_hierarchy_preserver, "openai"))

    def test_renderer_exposes_accepted_unresolved_local_and_conflict(self):
        structure = preserve_document_hierarchy(
            [docx_heading(1, 1, ilvl=2), docx_numbered(2, 0)],
            document_id="doc",
            document_title="example.docx",
            source_format=SOURCE_FORMAT_DOCX,
        )

        rendered = render_document_hierarchy_diagnostics(structure)
        self.assertIn("DOCUMENT: example.docx", rendered)
        self.assertIn("[accepted][HIGH] L1 Heading 1", rendered)
        self.assertIn("[ignored-local] List 2", rendered)
        self.assertIn("[conflict] docx-1", rendered)


if __name__ == "__main__":
    unittest.main()
