import io
import sys
import unittest
from pathlib import Path


BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from document_hierarchy_contracts import CONFIDENCE_HIGH, CONFIDENCE_LOW, CONFIDENCE_MEDIUM
from document_interpreter_contract import (
    CONTENT_ROLE_BODY,
    CONTENT_ROLE_HEADING,
    INTERPRETER_STATUS_ACCEPTED,
    INTERPRETER_STATUS_FALLBACK,
)
from document_interpreter_pdf import interpret_pdf_document


def _require_pypdf():
    try:
        from pypdf import PdfWriter
        from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject
    except ImportError as exc:
        raise unittest.SkipTest("pypdf is not installed") from exc
    return PdfWriter, DecodedStreamObject, DictionaryObject, NameObject


def _literal(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _text_command(text: str, *, x: int, y: int, size: int = 12, font: str = "F1") -> str:
    return f"BT /{font} {size} Tf {x} {y} Td ({_literal(text)}) Tj ET"


def _pdf_bytes(pages) -> bytes:
    PdfWriter, DecodedStreamObject, DictionaryObject, NameObject = _require_pypdf()
    writer = PdfWriter()
    for commands in pages:
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
        stream.set_data("\n".join(commands).encode("utf-8"))
        page[NameObject("/Contents")] = writer._add_object(stream)
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


class DocumentInterpreterPdfTests(unittest.TestCase):
    def test_explicit_numbered_headings_create_hierarchy(self):
        pdf = _pdf_bytes([[
            _text_command("5 Prokaryotic gene regulation", x=72, y=720, size=16, font="F2"),
            _text_command("Operons regulate bacterial transcription. " * 8, x=72, y=690, size=12),
            _text_command("5.1 Why operons are useful", x=72, y=650, size=14, font="F2"),
            _text_command("They coordinate genes with related functions. " * 8, x=72, y=620, size=12),
        ]])

        result = interpret_pdf_document(pdf, "biology.pdf", document_id="doc-1")

        self.assertEqual(result.status, INTERPRETER_STATUS_ACCEPTED)
        self.assertEqual(result.confidence, CONFIDENCE_HIGH)
        self.assertEqual([section.title for section in result.sections], [
            "5 Prokaryotic gene regulation",
            "5.1 Why operons are useful",
        ])
        self.assertEqual(result.sections[1].parent_id, result.sections[0].section_id)
        self.assertEqual([block.role for block in result.sections[0].owned_blocks], [
            CONTENT_ROLE_HEADING,
            CONTENT_ROLE_BODY,
        ])
        self.assertTrue(result.diagnostics["contract_validation"]["is_valid"])

    def test_visual_heading_contrast_creates_conservative_section(self):
        pdf = _pdf_bytes([[
            _text_command("Cellular Respiration", x=72, y=720, size=22, font="F2"),
            _text_command("Respiration converts nutrients into ATP. " * 10, x=72, y=680, size=12),
        ]])

        result = interpret_pdf_document(pdf, "visual.pdf", document_id="doc-1")

        self.assertEqual(result.status, INTERPRETER_STATUS_ACCEPTED)
        self.assertEqual(result.sections[0].title, "Cellular Respiration")
        self.assertEqual(result.sections[0].confidence, CONFIDENCE_MEDIUM)
        self.assertEqual(result.sections[0].metadata["detection_method"], "visual_heading")
        self.assertEqual(result.diagnostics["visual_heading_count"], 1)

    def test_plain_body_pdf_falls_back_without_inventing_structure(self):
        pdf = _pdf_bytes([[
            _text_command("This is ordinary body text with no reliable heading evidence. " * 4, x=72, y=720, size=12),
            _text_command("It should remain unresolved rather than becoming a section. " * 4, x=72, y=690, size=12),
        ]])

        result = interpret_pdf_document(pdf, "plain.pdf", document_id="doc-1")

        self.assertEqual(result.status, INTERPRETER_STATUS_FALLBACK)
        self.assertEqual(result.confidence, CONFIDENCE_LOW)
        self.assertEqual(result.sections, [])
        self.assertEqual(len(result.unresolved_blocks), 2)
        self.assertEqual(result.diagnostics["fallback_reason"], "pdf_contains_no_high_confidence_headings")
        self.assertTrue(result.diagnostics["contract_validation"]["is_valid"])

    def test_numbered_prose_and_page_like_numbers_are_not_promoted(self):
        pdf = _pdf_bytes([[
            _text_command("2024 This line should not become a heading", x=72, y=720, size=16),
            _text_command("95 Appendix-looking page number should not become a heading", x=72, y=690, size=16),
            _text_command("1 This ordinary sentence contains many words and ends as prose.", x=72, y=660, size=16),
        ]])

        result = interpret_pdf_document(pdf, "false-positive.pdf", document_id="doc-1")

        self.assertEqual(result.status, INTERPRETER_STATUS_FALLBACK)
        self.assertEqual(result.sections, [])

    def test_split_visual_title_is_reconstructed_before_heading_detection(self):
        pdf = _pdf_bytes([[
            _text_command("I", x=72, y=720, size=22, font="F2"),
            _text_command("NDAGINI PRELIMINARI", x=84, y=720, size=22, font="F2"),
            _text_command("Il libro V disciplina la prima fase del procedimento penale. " * 6, x=72, y=680, size=12),
        ]])

        result = interpret_pdf_document(pdf, "law.pdf", document_id="doc-1")

        self.assertEqual(result.status, INTERPRETER_STATUS_ACCEPTED)
        self.assertEqual(result.sections[0].title, "INDAGINI PRELIMINARI")
        self.assertEqual(result.sections[0].metadata["detection_method"], "visual_heading")

    def test_legal_references_and_durations_are_not_promoted_to_headings(self):
        pdf = _pdf_bytes([[
            _text_command("INDAGINI PRELIMINARI", x=72, y=720, size=22, font="F2"),
            _text_command("L'art. 326 cpp apre il Libro V. " * 8, x=72, y=690, size=12),
            _text_command("26 cpp", x=72, y=650, size=16, font="F2"),
            _text_command("21 Cost.", x=72, y=620, size=16, font="F2"),
            _text_command("8 CEDU", x=72, y=590, size=16, font="F2"),
            _text_command("6 mesi per le contravvenzioni;", x=72, y=560, size=16, font="F2"),
            _text_command("31 agosto", x=72, y=530, size=16, font="F2"),
            _text_command("La sospensione feriale non opera in alcuni casi. " * 8, x=72, y=500, size=12),
        ]])

        result = interpret_pdf_document(pdf, "law.pdf", document_id="doc-1")

        self.assertEqual([section.title for section in result.sections], ["INDAGINI PRELIMINARI"])
        owned_text = " ".join(block.text for block in result.sections[0].owned_blocks)
        self.assertIn("26 cpp", owned_text)
        self.assertIn("21 Cost.", owned_text)
        self.assertIn("8 CEDU", owned_text)
        self.assertIn("6 mesi per le contravvenzioni", owned_text)

    def test_split_heading_continuation_is_merged_into_one_section(self):
        pdf = _pdf_bytes([[
            _text_command("INDAGINI PRELIMINARI", x=72, y=720, size=22, font="F2"),
            _text_command("Intro body " * 20, x=72, y=690, size=12),
            _text_command("v NOTIZIE DI REATO CONTRO", x=72, y=640, size=12, font="F2"),
            _text_command("IGNOTI", x=72, y=624, size=12, font="F2"),
            _text_command("Body under the joined heading. " * 20, x=72, y=590, size=12),
        ]])

        result = interpret_pdf_document(pdf, "split-heading.pdf", document_id="doc-1")

        self.assertEqual(
            [section.title for section in result.sections],
            ["INDAGINI PRELIMINARI", "v NOTIZIE DI REATO CONTRO IGNOTI"],
        )

    def test_centered_diagram_labels_are_not_promoted_to_sections(self):
        pdf = _pdf_bytes([[
            _text_command("INDAGINI PRELIMINARI", x=72, y=720, size=22, font="F2"),
            _text_command("Main body before diagram. " * 20, x=72, y=690, size=12),
            _text_command("MODELLO 21 MODELLO 44 MODELLO 45 MODELLO 46", x=220, y=560, size=12, font="F2"),
            _text_command("MINIMA", x=250, y=530, size=12, font="F2"),
            _text_command("Main body after diagram. " * 20, x=72, y=500, size=12),
        ]])

        result = interpret_pdf_document(pdf, "diagram.pdf", document_id="doc-1")

        self.assertEqual([section.title for section in result.sections], ["INDAGINI PRELIMINARI"])
        owned_text = " ".join(block.text for block in result.sections[0].owned_blocks)
        self.assertIn("MODELLO 21", owned_text)
        self.assertIn("MINIMA", owned_text)

    def test_slide_export_pdf_uses_pages_as_sections(self):
        pages = []
        for index in range(1, 8):
            pages.append([
                _text_command(f"Slide Topic {index}", x=72, y=720, size=22, font="F2"),
                _text_command(f"Short slide body {index}. " * 8, x=72, y=650, size=12),
                _text_command(f"Label {index}", x=260, y=500, size=12, font="F2"),
            ])
        pdf = _pdf_bytes(pages)

        result = interpret_pdf_document(pdf, "slides-export.pdf", document_id="doc-1")

        self.assertEqual(result.status, INTERPRETER_STATUS_ACCEPTED)
        self.assertEqual(result.diagnostics["pdf_mode"], "slide_export")
        self.assertEqual(len(result.sections), 7)
        self.assertEqual(result.sections[0].title, "Slide Topic 1")
        self.assertEqual(result.sections[-1].title, "Slide Topic 7")
        self.assertEqual({section.metadata["pdf_mode"] for section in result.sections}, {"slide_export"})

    def test_local_decimal_numbered_items_stay_inside_parent_section(self):
        pdf = _pdf_bytes([[
            _text_command("BUSINESS ADMINISTRATION", x=72, y=720, size=22, font="F2"),
            _text_command("Business administration body. " * 12, x=72, y=690, size=12),
            _text_command("1. Planning", x=72, y=640, size=16, font="F2"),
            _text_command("Planning is a function of management. " * 8, x=72, y=610, size=12),
            _text_command("2. Organizing", x=72, y=560, size=16, font="F2"),
            _text_command("Organizing arranges resources. " * 8, x=72, y=530, size=12),
        ]])

        result = interpret_pdf_document(pdf, "business.pdf", document_id="doc-1")

        self.assertEqual([section.title for section in result.sections], ["BUSINESS ADMINISTRATION"])
        owned_text = " ".join(block.text for block in result.sections[0].owned_blocks)
        self.assertIn("1. Planning", owned_text)
        self.assertIn("2. Organizing", owned_text)

    def test_explicit_numbered_skeleton_recovers_dot_delimited_global_parents(self):
        pdf = _pdf_bytes([[
            _text_command("1. Big picture", x=72, y=720, size=16),
            _text_command("Opening regulation overview. " * 8, x=72, y=690, size=12),
            _text_command("2. Long non-coding RNAs", x=72, y=650, size=16),
            _text_command("lncRNA overview. " * 8, x=72, y=620, size=12),
            _text_command("2.1 Definition and general properties", x=72, y=590, size=14, font="F2"),
            _text_command("Definition body. " * 8, x=72, y=560, size=12),
            _text_command("2.2 Classification by genomic position", x=72, y=530, size=14, font="F2"),
            _text_command("Classification body. " * 8, x=72, y=500, size=12),
            _text_command("3. RNA-based diagnosis and therapy", x=72, y=470, size=16),
            _text_command("Diagnosis body. " * 8, x=72, y=440, size=12),
            _text_command("3.1 RNA molecules as biomarkers", x=72, y=410, size=14, font="F2"),
            _text_command("Biomarker body. " * 8, x=72, y=380, size=12),
            _text_command("4. Genome editing", x=72, y=350, size=16),
            _text_command("Genome editing body. " * 8, x=72, y=320, size=12),
        ]])

        result = interpret_pdf_document(pdf, "numbered.pdf", document_id="doc-1")

        titles = [section.title for section in result.sections]
        self.assertIn("1. Big picture", titles)
        self.assertIn("2. Long non-coding RNAs", titles)
        self.assertIn("3. RNA-based diagnosis and therapy", titles)
        by_title = {section.title: section for section in result.sections}
        self.assertEqual(
            by_title["2.1 Definition and general properties"].parent_id,
            by_title["2. Long non-coding RNAs"].section_id,
        )
        self.assertEqual(
            by_title["3.1 RNA molecules as biomarkers"].parent_id,
            by_title["3. RNA-based diagnosis and therapy"].section_id,
        )

    def test_local_integer_enumeration_inside_dotted_span_is_not_global_skeleton(self):
        pdf = _pdf_bytes([[
            _text_command("2. lncRNAs", x=72, y=720, size=16),
            _text_command("2.1 Definition", x=72, y=690, size=14, font="F2"),
            _text_command("Definition body. " * 8, x=72, y=660, size=12),
            _text_command("2.2 Classification", x=72, y=630, size=14, font="F2"),
            _text_command("Classification body. " * 8, x=72, y=600, size=12),
            _text_command("2.3 How lncRNAs regulate gene expression", x=72, y=570, size=14, font="F2"),
            _text_command("1. Transcriptional interference", x=72, y=540, size=16),
            _text_command("2. Chromatin remodeling", x=72, y=510, size=16),
            _text_command("3. Histone modification", x=72, y=480, size=16),
            _text_command("4. Alternative splicing", x=72, y=450, size=16),
            _text_command("2.4 XIST as a key example", x=72, y=420, size=14, font="F2"),
            _text_command("3. RNA diagnosis", x=72, y=390, size=16),
            _text_command("3.1 RNA molecules as biomarkers", x=72, y=370, size=14, font="F2"),
            _text_command("4. CRISPR", x=72, y=340, size=16),
            _text_command("5. Operons", x=72, y=310, size=16),
        ]])

        result = interpret_pdf_document(pdf, "local-enum.pdf", document_id="doc-1")

        titles = [section.title for section in result.sections]
        self.assertIn("2. lncRNAs", titles)
        self.assertIn("3. RNA diagnosis", titles)
        self.assertNotIn("3. Histone modification", titles)
        self.assertNotIn("4. Alternative splicing", titles)

    def test_integer_content_run_without_dotted_hierarchy_is_not_global_skeleton(self):
        pdf = _pdf_bytes([[
            _text_command("BONES AND MUSCLES HISTOLOGY", x=72, y=720, size=18),
            _text_command("Cartilage introduction. " * 8, x=72, y=690, size=12),
            _text_command("2 kinds of growth:", x=72, y=650, size=16),
            _text_command("Growth body. " * 8, x=72, y=620, size=12),
            _text_command("3 major classes of ECM molecules:", x=72, y=590, size=16),
            _text_command("ECM body. " * 8, x=72, y=560, size=12),
            _text_command("4. The calcified matrix inhibits diffusion of nutrients", x=72, y=530, size=16),
            _text_command("Matrix body. " * 8, x=72, y=500, size=12),
            _text_command("5. Mesenchymal stem cells migrate into the cavity", x=72, y=470, size=16),
            _text_command("Stem cell body. " * 8, x=72, y=440, size=12),
        ]])

        result = interpret_pdf_document(pdf, "histology-like.pdf", document_id="doc-1")

        titles = [section.title for section in result.sections]
        self.assertEqual(titles, ["BONES AND MUSCLES HISTOLOGY"])
        owned_text = " ".join(block.text for block in result.sections[0].owned_blocks)
        self.assertIn("3 major classes of ECM molecules", owned_text)
        self.assertIn("Mesenchymal stem cells", owned_text)

    def test_measurement_ranges_and_molecule_labels_are_not_sections(self):
        pdf = _pdf_bytes([[
            _text_command("CARDIAC MUSCLE", x=72, y=720, size=18),
            _text_command("Cardiac muscle body. " * 8, x=72, y=690, size=12),
            _text_command("0.3 to 0.4 μm in diameter are also concentrated", x=72, y=650, size=16, font="F2"),
            _text_command("Granule body. " * 8, x=72, y=620, size=12),
            _text_command("rRNA.", x=72, y=590, size=16, font="F2"),
            _text_command("Molecule label body. " * 8, x=72, y=560, size=12),
            _text_command("tRNA.", x=72, y=530, size=16, font="F2"),
            _text_command("More body. " * 8, x=72, y=500, size=12),
        ]])

        result = interpret_pdf_document(pdf, "labels.pdf", document_id="doc-1")

        titles = [section.title for section in result.sections]
        self.assertEqual(titles, ["CARDIAC MUSCLE"])
        owned_text = " ".join(block.text for block in result.sections[0].owned_blocks)
        self.assertIn("0.3 to 0.4", owned_text)
        self.assertIn("rRNA", owned_text)
        self.assertIn("tRNA", owned_text)

    def test_dangling_visual_heading_continuation_stays_with_previous_section(self):
        pdf = _pdf_bytes([[
            _text_command("EFFECTS OF OLDER AGE ON ARTERY FUNCTION", x=72, y=720, size=18),
            _text_command("AND STRUCTURE", x=72, y=690, size=18),
            _text_command("Artery function body. " * 8, x=72, y=660, size=12),
        ]])

        result = interpret_pdf_document(pdf, "continuation.pdf", document_id="doc-1")

        self.assertEqual([section.title for section in result.sections], [
            "EFFECTS OF OLDER AGE ON ARTERY FUNCTION",
        ])
        owned_text = " ".join(block.text for block in result.sections[0].owned_blocks)
        self.assertIn("AND STRUCTURE", owned_text)

    def test_repeated_slide_course_header_does_not_steal_specific_title(self):
        pages = []
        for index in range(1, 8):
            pages.append([
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command(f"Specific Slide Topic {index}", x=72, y=700, size=22, font="F2"),
                _text_command(f"Short slide body {index}. " * 8, x=72, y=650, size=12),
            ])
        pdf = _pdf_bytes(pages)

        result = interpret_pdf_document(pdf, "organic-slides.pdf", document_id="doc-1")

        self.assertEqual(result.diagnostics["pdf_mode"], "slide_export")
        self.assertEqual(result.sections[0].title, "Specific Slide Topic 1")
        self.assertEqual(result.sections[-1].title, "Specific Slide Topic 7")

    def test_unreliable_slide_formula_fragments_use_neutral_page_title(self):
        pages = []
        for index in range(1, 7):
            pages.append([
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command("HC N", x=250, y=650, size=18, font="F2"),
                _text_command("RHOH H", x=250, y=620, size=18, font="F2"),
                _text_command(f"Short slide body {index}. " * 8, x=72, y=500, size=12),
            ])
        pages.append([
            _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
            _text_command("Reazioni: addizione al carbonile", x=72, y=650, size=18, font="F2"),
            _text_command("Useful slide body. " * 8, x=72, y=500, size=12),
        ])
        pdf = _pdf_bytes(pages)

        result = interpret_pdf_document(pdf, "formula-slides.pdf", document_id="doc-1")

        self.assertEqual(result.diagnostics["pdf_mode"], "slide_export")
        self.assertEqual(result.sections[0].title, "Page 1")
        self.assertEqual(result.sections[5].title, "Page 6")
        self.assertEqual(result.sections[6].title, "Reazioni: addizione al carbonile")

    def test_slide_fragments_starting_with_punctuation_articles_or_chemical_tokens_are_neutral(self):
        pages = [
            [
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command(": descrive il movimento degli elettroni", x=180, y=650, size=18, font="F2"),
                _text_command("Slide body. " * 8, x=72, y=500, size=12),
            ],
            [
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command("le conoscenze di chimica organica,", x=180, y=650, size=18, font="F2"),
                _text_command("Slide body. " * 8, x=72, y=500, size=12),
            ],
            [
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command("Br Br OH OH", x=180, y=650, size=18, font="F2"),
                _text_command("Slide body. " * 8, x=72, y=500, size=12),
            ],
            [
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command("Addizioni b. Radicaliche", x=72, y=650, size=18, font="F2"),
                _text_command("Useful slide body. " * 8, x=72, y=500, size=12),
            ],
            [
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command("Sostituzioni c. Radicaliche", x=72, y=650, size=18, font="F2"),
                _text_command("Useful slide body. " * 8, x=72, y=500, size=12),
            ],
            [
                _text_command("Chimica Organica II", x=72, y=740, size=12, font="F2"),
                _text_command("Struttura atomica: gli orbitali", x=72, y=650, size=18, font="F2"),
                _text_command("Useful slide body. " * 8, x=72, y=500, size=12),
            ],
        ]
        pdf = _pdf_bytes(pages)

        result = interpret_pdf_document(pdf, "fragment-slides.pdf", document_id="doc-1")

        self.assertEqual(result.diagnostics["pdf_mode"], "slide_export")
        self.assertEqual([section.title for section in result.sections[:3]], ["Page 1", "Page 2", "Page 3"])
        self.assertEqual(result.sections[3].title, "Addizioni b. Radicaliche")
        self.assertEqual(result.sections[4].title, "Sostituzioni c. Radicaliche")
        self.assertEqual(result.sections[5].title, "Struttura atomica: gli orbitali")

    def test_page_metadata_is_preserved_on_owned_blocks(self):
        pdf = _pdf_bytes([
            [
                _text_command("1 First section", x=72, y=720, size=16, font="F2"),
                _text_command("First page body " * 10, x=72, y=690, size=12),
            ],
            [
                _text_command("2 Second section", x=72, y=720, size=16, font="F2"),
                _text_command("Second page body " * 10, x=72, y=690, size=12),
            ],
        ])

        result = interpret_pdf_document(pdf, "pages.pdf", document_id="doc-1")

        self.assertEqual([section.title for section in result.sections], ["1 First section", "2 Second section"])
        self.assertEqual([section.owned_blocks[0].page for section in result.sections], [1, 2])


if __name__ == "__main__":
    unittest.main()
