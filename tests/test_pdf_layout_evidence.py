import io
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from canonical_document import ROLE_BODY, adapt_pdf_document
from document_hierarchy_evidence_mapper import map_document_to_structural_evidence


def _require_pypdf():
    try:
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    except ImportError as exc:  # pragma: no cover - dependency check.
        raise unittest.SkipTest("pypdf is not installed") from exc
    return PdfWriter, DecodedStreamObject, DictionaryObject, NameObject


def _literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_bytes(pages) -> bytes:
    PdfWriter, DecodedStreamObject, DictionaryObject, NameObject = _require_pypdf()
    writer = PdfWriter()
    for page_commands in pages:
        page = writer.add_blank_page(width=612, height=792)
        fonts = DictionaryObject({
            NameObject("/F1"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica"),
            }),
            NameObject("/F2"): DictionaryObject({
                NameObject("/Type"): NameObject("/Font"),
                NameObject("/Subtype"): NameObject("/Type1"),
                NameObject("/BaseFont"): NameObject("/Helvetica-Bold"),
            }),
        })
        page[NameObject("/Resources")] = DictionaryObject({
            NameObject("/Font"): fonts,
        })
        stream = DecodedStreamObject()
        stream.set_data("\n".join(page_commands).encode("utf-8"))
        page[NameObject("/Contents")] = stream
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _text_command(text: str, *, x: int, y: int, size: int = 12, font: str = "F1") -> str:
    return f"BT /{font} {size} Tf {x} {y} Td ({_literal(text)}) Tj ET"


class PdfLayoutEvidenceTests(unittest.TestCase):
    def test_different_font_sizes_are_preserved_without_heading_classification(self):
        pdf = _pdf_bytes([[
            _text_command("Large title", x=72, y=700, size=24),
            _text_command("Normal body text", x=72, y=660, size=12),
        ]])

        canonical = adapt_pdf_document(pdf, "font-sizes.pdf")
        evidence = map_document_to_structural_evidence(canonical)

        self.assertEqual([block.text for block in canonical.blocks], ["Large title", "Normal body text"])
        self.assertEqual(canonical.blocks[0].metadata["font_size"], 24.0)
        self.assertEqual(canonical.blocks[1].metadata["font_size"], 12.0)
        self.assertEqual(evidence[0].visual_evidence["font_size"], 24.0)
        self.assertEqual(evidence[1].visual_evidence["font_size"], 12.0)
        self.assertEqual(evidence[0].evidence_role, "body")
        self.assertEqual(evidence[0].content_kind, "body_content")
        self.assertFalse(hasattr(evidence[0], "parent_id"))
        self.assertFalse(hasattr(evidence[0], "level"))

    def test_geometry_and_source_positions_remain_distinguishable(self):
        pdf = _pdf_bytes([[
            _text_command("Top left", x=72, y=720, size=12),
            _text_command("Lower right", x=300, y=320, size=12),
        ]])

        canonical = adapt_pdf_document(pdf, "geometry.pdf")
        first, second = canonical.blocks
        evidence = map_document_to_structural_evidence(canonical)

        self.assertEqual(first.source_position.unit_type, "page")
        self.assertEqual(first.source_position.unit_index, 1)
        self.assertEqual(first.source_position.block_index, 1)
        self.assertEqual(second.source_position.block_index, 2)
        self.assertEqual(first.metadata["x0"], 72.0)
        self.assertEqual(first.metadata["y0"], 720.0)
        self.assertEqual(second.metadata["x0"], 300.0)
        self.assertEqual(second.metadata["y0"], 320.0)
        self.assertEqual(evidence[0].visual_evidence["page_width"], 612.0)
        self.assertEqual(evidence[0].visual_evidence["page_height"], 792.0)
        self.assertNotEqual(evidence[0].visual_evidence["x0"], evidence[1].visual_evidence["x0"])
        self.assertNotEqual(evidence[0].visual_evidence["y0"], evidence[1].visual_evidence["y0"])

    def test_multiple_physical_text_fragments_are_distinct_and_ordered(self):
        pdf = _pdf_bytes([[
            _text_command("First physical element", x=72, y=700, size=12),
            _text_command("Second physical element", x=72, y=680, size=12),
            _text_command("Third physical element", x=72, y=660, size=12),
        ]])

        canonical = adapt_pdf_document(pdf, "fragments.pdf")

        self.assertEqual([block.text for block in canonical.blocks], [
            "First physical element",
            "Second physical element",
            "Third physical element",
        ])
        self.assertEqual([block.source_order for block in canonical.blocks], [1, 2, 3])
        self.assertEqual([block.source_position.block_index for block in canonical.blocks], [1, 2, 3])
        self.assertEqual({block.metadata["pdf_text_unit"] for block in canonical.blocks}, {"text_fragment"})

    def test_same_text_on_different_pages_remains_distinct_by_provenance(self):
        pdf = _pdf_bytes([
            [_text_command("Repeated header", x=72, y=740, size=10)],
            [_text_command("Repeated header", x=72, y=740, size=10)],
        ])

        canonical = adapt_pdf_document(pdf, "repeated.pdf")
        evidence = map_document_to_structural_evidence(canonical)

        self.assertEqual([block.text for block in canonical.blocks], ["Repeated header", "Repeated header"])
        self.assertEqual([block.source_position.unit_index for block in canonical.blocks], [1, 2])
        self.assertEqual([item.source_span.start.unit_index for item in evidence], [1, 2])
        self.assertEqual([item.native_evidence["page_index"] for item in evidence], [1, 2])

    def test_mixed_typography_fragments_preserve_font_source_information(self):
        pdf = _pdf_bytes([[
            _text_command("Mixed regular", x=72, y=700, size=12, font="F1"),
            _text_command("Mixed bold", x=170, y=700, size=12, font="F2"),
        ]])

        canonical = adapt_pdf_document(pdf, "mixed-font.pdf")
        evidence = map_document_to_structural_evidence(canonical)

        self.assertEqual(canonical.blocks[0].metadata["font_name"], "Helvetica")
        self.assertEqual(canonical.blocks[1].metadata["font_name"], "Helvetica-Bold")
        self.assertEqual(canonical.blocks[0].metadata["font_subtype"], "Type1")
        self.assertEqual(canonical.blocks[1].metadata["font_subtype"], "Type1")
        self.assertEqual(evidence[0].visual_evidence["font_name"], "Helvetica")
        self.assertEqual(evidence[1].visual_evidence["font_name"], "Helvetica-Bold")
        self.assertNotIn("heading", evidence[1].native_evidence)

    def test_minimal_body_text_maps_to_neutral_structural_evidence(self):
        pdf = _pdf_bytes([[
            _text_command("Ordinary body text", x=72, y=700, size=12),
        ]])

        canonical = adapt_pdf_document(pdf, "body.pdf")
        evidence = map_document_to_structural_evidence(canonical)[0]

        self.assertEqual(canonical.blocks[0].role_hint, ROLE_BODY)
        self.assertEqual(evidence.evidence_role, "body")
        self.assertEqual(evidence.content_kind, "body_content")
        self.assertEqual(evidence.numbering_evidence, {})
        self.assertFalse(hasattr(evidence, "section"))

    def test_page_without_native_text_does_not_invoke_ocr(self):
        pdf = _pdf_bytes([[]])

        canonical = adapt_pdf_document(pdf, "blank.pdf")

        self.assertEqual(len(canonical.units), 1)
        self.assertEqual(canonical.blocks, [])
        self.assertEqual(canonical.units[0].metadata["native_text_available"], False)
        self.assertEqual(canonical.units[0].metadata["page_width"], 612.0)
        self.assertEqual(canonical.units[0].metadata["page_height"], 792.0)


if __name__ == "__main__":
    unittest.main()
