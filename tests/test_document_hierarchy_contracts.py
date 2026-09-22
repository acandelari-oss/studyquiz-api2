import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import SourcePosition
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CONFIDENCE_LOW,
    CONFIDENCE_MEDIUM,
    CONTENT_KIND_DOCUMENT_BOUNDARY,
    CONTENT_KIND_LIST_ITEM,
    CONTENT_KIND_LOCAL_STRUCTURE,
    EVIDENCE_ROLE_HEADING,
    EVIDENCE_ROLE_LIST,
    EVIDENCE_ROLE_TITLE,
    EVIDENCE_ROLE_VISUAL_HEADING,
    EvidenceProvenance,
    CanonicalDocumentNode,
    CanonicalDocumentStructure,
    NODE_KIND_DOCUMENT_SECTION,
    NODE_KIND_UNRESOLVED,
    ORIGIN_EXPLICIT_SOURCE,
    ORIGIN_UNRESOLVED,
    SOURCE_FORMAT_DOCX,
    SOURCE_FORMAT_PDF,
    SOURCE_FORMAT_PPTX,
    STATUS_ACCEPTED,
    STATUS_UNRESOLVED,
    SourceSpan,
    StructuralEvidence,
)


def pos(unit_type="document_flow", unit_index=1, block_index=1, start=0, end=None):
    return SourcePosition(unit_type, unit_index, block_index, start, end)


class DocumentHierarchyContractsTests(unittest.TestCase):
    def test_clean_docx_native_heading_evidence_can_support_final_nodes(self):
        chapter = StructuralEvidence(
            evidence_id="ev-docx-001",
            source_format=SOURCE_FORMAT_DOCX,
            source_order=1,
            source_span=SourceSpan(pos(block_index=1, end=9)),
            raw_text="Chapter 5",
            normalized_text="Chapter 5",
            evidence_role=EVIDENCE_ROLE_HEADING,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            native_evidence={"style": "Heading 1", "outline_level": 1},
            confidence_hint=CONFIDENCE_HIGH,
        )
        operons = StructuralEvidence(
            evidence_id="ev-docx-002",
            source_format=SOURCE_FORMAT_DOCX,
            source_order=2,
            source_span=SourceSpan(pos(block_index=2, end=11)),
            raw_text="5.1 Operons",
            normalized_text="5.1 Operons",
            evidence_role=EVIDENCE_ROLE_HEADING,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            native_evidence={"style": "Heading 2", "outline_level": 2},
            numbering_evidence={"numbering": "5.1", "level": 2},
            confidence_hint=CONFIDENCE_HIGH,
        )
        structure = CanonicalDocumentStructure(
            document_id="docx-clean",
            source_format=SOURCE_FORMAT_DOCX,
            document_title="Clean DOCX",
            structure_status=STATUS_ACCEPTED,
            structure_confidence=CONFIDENCE_HIGH,
            nodes=[
                CanonicalDocumentNode(
                    node_id="node-001",
                    source_title="Chapter 5",
                    logical_title="Chapter 5",
                    level=1,
                    parent_id=None,
                    source_order=1,
                    source_span=chapter.source_span,
                    origin=ORIGIN_EXPLICIT_SOURCE,
                    confidence=CONFIDENCE_HIGH,
                    status=STATUS_ACCEPTED,
                    evidence_ids=[chapter.evidence_id],
                    provenance=EvidenceProvenance(
                        supporting_evidence_ids=[chapter.evidence_id],
                        decision_source=ORIGIN_EXPLICIT_SOURCE,
                    ),
                ),
                CanonicalDocumentNode(
                    node_id="node-002",
                    source_title="5.1 Operons",
                    logical_title="5.1 Operons",
                    level=2,
                    parent_id="node-001",
                    source_order=2,
                    source_span=operons.source_span,
                    origin=ORIGIN_EXPLICIT_SOURCE,
                    confidence=CONFIDENCE_HIGH,
                    status=STATUS_ACCEPTED,
                    evidence_ids=[operons.evidence_id],
                    provenance=EvidenceProvenance(
                        supporting_evidence_ids=[operons.evidence_id],
                        decision_source=ORIGIN_EXPLICIT_SOURCE,
                    ),
                ),
            ],
        )

        self.assertEqual(structure.nodes[1].parent_id, "node-001")
        self.assertEqual(structure.nodes[1].evidence_ids, ["ev-docx-002"])
        self.assertEqual(operons.native_evidence["outline_level"], 2)

    def test_pdf_visual_heading_is_evidence_not_hierarchy_truth(self):
        evidence = StructuralEvidence(
            evidence_id="ev-pdf-visual-001",
            source_format=SOURCE_FORMAT_PDF,
            source_order=8,
            source_span=SourceSpan(pos("page", 3, 2, start=120, end=148)),
            raw_text="Ribosome structure",
            normalized_text="Ribosome structure",
            evidence_role=EVIDENCE_ROLE_VISUAL_HEADING,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            visual_evidence={
                "font_size": 18,
                "font_weight": "bold",
                "x": 72,
                "y": 110,
                "prominence": "high",
            },
            confidence_hint=CONFIDENCE_MEDIUM,
        )

        self.assertEqual(evidence.source_span.start.unit_type, "page")
        self.assertEqual(evidence.source_span.start.unit_index, 3)
        self.assertEqual(evidence.evidence_role, EVIDENCE_ROLE_VISUAL_HEADING)
        self.assertEqual(evidence.confidence_hint, CONFIDENCE_MEDIUM)

    def test_pptx_title_and_bullets_keep_document_vs_local_structure_distinct(self):
        title = StructuralEvidence(
            evidence_id="ev-pptx-title-001",
            source_format=SOURCE_FORMAT_PPTX,
            source_order=1,
            source_span=SourceSpan(pos("slide", 1, 1, end=20)),
            raw_text="Cell metabolism",
            normalized_text="Cell metabolism",
            evidence_role=EVIDENCE_ROLE_TITLE,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            native_evidence={"placeholder_role": "title", "slide_index": 1},
            confidence_hint=CONFIDENCE_HIGH,
        )
        bullet = StructuralEvidence(
            evidence_id="ev-pptx-bullet-001",
            source_format=SOURCE_FORMAT_PPTX,
            source_order=2,
            source_span=SourceSpan(pos("slide", 1, 2, end=18)),
            raw_text="ATP production",
            normalized_text="ATP production",
            evidence_role=EVIDENCE_ROLE_LIST,
            content_kind=CONTENT_KIND_LIST_ITEM,
            native_evidence={"bullet_level": 0, "slide_index": 1},
            confidence_hint=CONFIDENCE_MEDIUM,
        )
        sub_bullet = StructuralEvidence(
            evidence_id="ev-pptx-bullet-002",
            source_format=SOURCE_FORMAT_PPTX,
            source_order=3,
            source_span=SourceSpan(pos("slide", 1, 3, end=22)),
            raw_text="Mitochondrial steps",
            normalized_text="Mitochondrial steps",
            evidence_role=EVIDENCE_ROLE_LIST,
            content_kind=CONTENT_KIND_LOCAL_STRUCTURE,
            native_evidence={"bullet_level": 1, "slide_index": 1},
            confidence_hint=CONFIDENCE_MEDIUM,
        )
        structure = CanonicalDocumentStructure(
            document_id="pptx-title-bullets",
            source_format=SOURCE_FORMAT_PPTX,
            document_title="Lecture deck",
            structure_status=STATUS_ACCEPTED,
            structure_confidence=CONFIDENCE_HIGH,
            nodes=[
                CanonicalDocumentNode(
                    node_id="node-slide-001",
                    source_title="Cell metabolism",
                    logical_title="Cell metabolism",
                    level=1,
                    parent_id=None,
                    source_order=1,
                    source_span=title.source_span,
                    origin=ORIGIN_EXPLICIT_SOURCE,
                    confidence=CONFIDENCE_HIGH,
                    status=STATUS_ACCEPTED,
                    evidence_ids=[title.evidence_id],
                    provenance=EvidenceProvenance(
                        supporting_evidence_ids=[title.evidence_id],
                        decision_source=ORIGIN_EXPLICIT_SOURCE,
                    ),
                    node_kind=NODE_KIND_DOCUMENT_SECTION,
                )
            ],
        )

        self.assertEqual(bullet.content_kind, CONTENT_KIND_LIST_ITEM)
        self.assertEqual(sub_bullet.content_kind, CONTENT_KIND_LOCAL_STRUCTURE)
        self.assertEqual([node.source_title for node in structure.nodes], ["Cell metabolism"])

    def test_compound_heading_can_reference_multiple_evidence_items_later(self):
        marker = StructuralEvidence(
            evidence_id="ev-compound-001",
            source_format=SOURCE_FORMAT_DOCX,
            source_order=12,
            source_span=SourceSpan(pos(block_index=12, end=11)),
            raw_text="CAPITOLO III",
            normalized_text="CAPITOLO III",
            evidence_role=EVIDENCE_ROLE_HEADING,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            native_evidence={"style": "Heading 1"},
            confidence_hint=CONFIDENCE_HIGH,
        )
        title = StructuralEvidence(
            evidence_id="ev-compound-002",
            source_format=SOURCE_FORMAT_DOCX,
            source_order=13,
            source_span=SourceSpan(pos(block_index=13, end=60)),
            raw_text="Il soggetto giuridico ed il soggetto economico d'impresa",
            normalized_text="Il soggetto giuridico ed il soggetto economico d'impresa",
            evidence_role=EVIDENCE_ROLE_HEADING,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            native_evidence={"style": "Heading 1"},
            structural_context={"adjacent_to_previous": True},
            confidence_hint=CONFIDENCE_HIGH,
        )
        node = CanonicalDocumentNode(
            node_id="node-compound-001",
            source_title=title.raw_text,
            logical_title=title.raw_text,
            level=1,
            parent_id=None,
            source_order=12,
            source_span=SourceSpan(marker.source_span.start, title.source_span.end or title.source_span.start),
            origin=ORIGIN_EXPLICIT_SOURCE,
            confidence=CONFIDENCE_HIGH,
            status=STATUS_ACCEPTED,
            evidence_ids=[marker.evidence_id, title.evidence_id],
            provenance=EvidenceProvenance(
                supporting_evidence_ids=[marker.evidence_id, title.evidence_id],
                decision_source=ORIGIN_EXPLICIT_SOURCE,
            ),
        )

        self.assertEqual(node.evidence_ids, ["ev-compound-001", "ev-compound-002"])
        self.assertEqual(node.source_title, title.raw_text)

    def test_unresolved_structure_can_be_represented_without_inventing_parent_or_level(self):
        evidence = StructuralEvidence(
            evidence_id="ev-ambiguous-001",
            source_format=SOURCE_FORMAT_PDF,
            source_order=44,
            source_span=SourceSpan(pos("page", 7, 4, end=32)),
            raw_text="Potential heading-like phrase",
            normalized_text="Potential heading-like phrase",
            evidence_role=EVIDENCE_ROLE_VISUAL_HEADING,
            content_kind=CONTENT_KIND_DOCUMENT_BOUNDARY,
            visual_evidence={"prominence": "weak"},
            confidence_hint=CONFIDENCE_LOW,
        )
        unresolved = CanonicalDocumentNode(
            node_id="node-unresolved-001",
            source_title=evidence.raw_text,
            logical_title=evidence.normalized_text,
            level=None,
            parent_id=None,
            source_order=evidence.source_order,
            source_span=evidence.source_span,
            origin=ORIGIN_UNRESOLVED,
            confidence=CONFIDENCE_LOW,
            status=STATUS_UNRESOLVED,
            evidence_ids=[evidence.evidence_id],
            provenance=EvidenceProvenance(
                supporting_evidence_ids=[evidence.evidence_id],
                decision_source=ORIGIN_UNRESOLVED,
                repair_reason="insufficient evidence to assign hierarchy safely",
            ),
            node_kind=NODE_KIND_UNRESOLVED,
        )

        self.assertIsNone(unresolved.level)
        self.assertIsNone(unresolved.parent_id)
        self.assertEqual(unresolved.status, STATUS_UNRESOLVED)
        self.assertEqual(unresolved.origin, ORIGIN_UNRESOLVED)


if __name__ == "__main__":
    unittest.main()
