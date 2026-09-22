import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import SourcePosition
from document_extractors import ExtractedBlock
from document_hierarchy_contracts import (
    CONFIDENCE_HIGH,
    CanonicalDocumentNode,
    CanonicalDocumentStructure,
    EvidenceProvenance,
    ORIGIN_EXPLICIT_SOURCE,
    SOURCE_FORMAT_PDF,
    STATUS_ACCEPTED,
    SourceSpan,
)
from document_hierarchy_ingestion_bridge import apply_preserved_hierarchy_to_blocks


def node(
    node_id: str,
    title: str,
    order: int,
    *,
    level: int = 1,
    parent_id=None,
    page: int = 1,
) -> CanonicalDocumentNode:
    position = SourcePosition("page", page, order, 0, len(title))
    return CanonicalDocumentNode(
        node_id=node_id,
        source_title=title,
        logical_title=title,
        level=level,
        parent_id=parent_id,
        source_order=order,
        source_span=SourceSpan(start=position, end=position),
        origin=ORIGIN_EXPLICIT_SOURCE,
        confidence=CONFIDENCE_HIGH,
        status=STATUS_ACCEPTED,
        evidence_ids=[f"evidence-{node_id}"],
        provenance=EvidenceProvenance(
            supporting_evidence_ids=[f"evidence-{node_id}"],
            decision_source=ORIGIN_EXPLICIT_SOURCE,
        ),
    )


def structure(nodes):
    return CanonicalDocumentStructure(
        document_id="document-id",
        source_format=SOURCE_FORMAT_PDF,
        document_title="Benchmark.pdf",
        structure_status=STATUS_ACCEPTED,
        structure_confidence=CONFIDENCE_HIGH,
        nodes=nodes,
    )


class DocumentHierarchyIngestionBridgeTests(unittest.TestCase):
    def test_splits_block_at_preserved_section_boundaries(self):
        blocks = [
            ExtractedBlock(
                text="Preface text\n\n5 Prokaryotic gene regulation\nBody A\n\n5.1 Why operons are useful\nBody B",
                raw_text="Preface text\n\n5 Prokaryotic gene regulation\nBody A\n\n5.1 Why operons are useful\nBody B",
                page=1,
                block_index=1,
            )
        ]
        result, split_count, matched_count = apply_preserved_hierarchy_to_blocks(
            blocks,
            structure([
                node("n5", "5 Prokaryotic gene regulation", 1),
                node("n51", "5.1 Why operons are useful", 2, level=2, parent_id="n5"),
            ]),
            fallback_section="Benchmark.pdf",
        )

        self.assertEqual(matched_count, 2)
        self.assertGreaterEqual(split_count, 2)
        self.assertEqual(result[0].section, "Benchmark.pdf")
        self.assertEqual(result[1].section, "5 Prokaryotic gene regulation")
        self.assertEqual(
            result[2].section,
            "5 Prokaryotic gene regulation > 5.1 Why operons are useful",
        )
        self.assertEqual("".join(block.text for block in result).replace("\n", ""), blocks[0].text.replace("\n", ""))

    def test_section_continues_across_following_blocks(self):
        blocks = [
            ExtractedBlock(
                text="5 Prokaryotic gene regulation\nBody A",
                raw_text="5 Prokaryotic gene regulation\nBody A",
                page=1,
                block_index=1,
            ),
            ExtractedBlock(
                text="Continuation body on next page",
                raw_text="Continuation body on next page",
                page=2,
                block_index=2,
            ),
        ]
        result, _, _ = apply_preserved_hierarchy_to_blocks(
            blocks,
            structure([node("n5", "5 Prokaryotic gene regulation", 1, page=1)]),
            fallback_section="Benchmark.pdf",
        )

        self.assertEqual(result[0].section, "5 Prokaryotic gene regulation")
        self.assertEqual(result[1].section, "5 Prokaryotic gene regulation")

    def test_unmatched_node_falls_back_without_losing_content(self):
        blocks = [
            ExtractedBlock(
                text="Only body content",
                raw_text="Only body content",
                page=1,
                section=None,
                block_index=1,
            )
        ]
        result, split_count, matched_count = apply_preserved_hierarchy_to_blocks(
            blocks,
            structure([node("n5", "Missing heading", 1, page=1)]),
            fallback_section="Benchmark.pdf",
        )

        self.assertEqual(split_count, 0)
        self.assertEqual(matched_count, 0)
        self.assertEqual(result[0].text, "Only body content")
        self.assertEqual(result[0].section, "Benchmark.pdf")

    def test_unresolved_structure_returns_original_blocks(self):
        unresolved = CanonicalDocumentStructure(
            document_id="document-id",
            source_format=SOURCE_FORMAT_PDF,
            document_title="Benchmark.pdf",
            structure_status="unresolved",
            structure_confidence="LOW",
            nodes=[node("n5", "5 Prokaryotic gene regulation", 1)],
        )
        blocks = [
            ExtractedBlock(
                text="5 Prokaryotic gene regulation\nBody A",
                raw_text="5 Prokaryotic gene regulation\nBody A",
                page=1,
                section=None,
                block_index=1,
            )
        ]
        result, split_count, matched_count = apply_preserved_hierarchy_to_blocks(
            blocks,
            unresolved,
            fallback_section="Benchmark.pdf",
        )

        self.assertEqual(result, blocks)
        self.assertEqual(split_count, 0)
        self.assertEqual(matched_count, 0)


if __name__ == "__main__":
    unittest.main()
