import io
import unittest
from unittest.mock import patch

from document_acquisition import acquire_document, finish_report, _unit


class AcquisitionTests(unittest.TestCase):
    def pdf(self, pages):
        from pypdf import PdfWriter
        from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject, NumberObject
        writer = PdfWriter()
        for text, visual in pages:
            page = writer.add_blank_page(width=600, height=800)
            font = DictionaryObject({NameObject('/Type'):NameObject('/Font'), NameObject('/Subtype'):NameObject('/Type1'), NameObject('/BaseFont'):NameObject('/Helvetica')})
            page[NameObject('/Resources')] = DictionaryObject({NameObject('/Font'):DictionaryObject({NameObject('/F1'):font})})
            stream=DecodedStreamObject()
            content = f'BT /F1 12 Tf 50 700 Td ({text}) Tj ET' if text else ''
            if visual:
                image=DecodedStreamObject()
                image.set_data(b'\x00\x00\x00')
                image.update({NameObject('/Type'):NameObject('/XObject'),NameObject('/Subtype'):NameObject('/Image'),NameObject('/Width'):NumberObject(1),NameObject('/Height'):NumberObject(1),NameObject('/BitsPerComponent'):NumberObject(8),NameObject('/ColorSpace'):NameObject('/DeviceRGB')})
                page['/Resources'][NameObject('/XObject')]=DictionaryObject({NameObject('/Im0'):writer._add_object(image)})
                content += ' q 100 0 0 100 50 600 cm /Im0 Do Q'
            stream.set_data(content.encode())
            page[NameObject('/Contents')] = writer._add_object(stream)
        output=io.BytesIO(); writer.write(output)
        return output.getvalue()

    def test_readable_short_pdf_and_blank(self):
        document, report = acquire_document(self.pdf([('Cell division',False),('',False)]), 'notes.pdf')
        self.assertEqual((report['acquired'],report['blank'],report['failed']), (1,1,0))
        self.assertEqual(report['status'],'completed')
        self.assertEqual(document.blocks[0].page,1)
        self.assertIn('Cell division',document.blocks[0].text)

    def test_scanned_and_mixed_are_partial_even_when_ocr_succeeds(self):
        with patch('document_acquisition.ocr_pdf',return_value='Recovered image text') as ocr:
            document, report = acquire_document(self.pdf([('',True),('Digital content',True)]),'notes.pdf')
        self.assertEqual(ocr.call_count,2)
        self.assertEqual(report['partial'],2)
        self.assertEqual(report['acquired'],0)
        self.assertEqual(report['status'],'completed_with_warnings')
        self.assertIn('Digital content',document.blocks[1].text)
        self.assertIn('Recovered image text',document.blocks[1].text)
        self.assertNotIn('Recovered image text', document.blocks[1].digital_text)
        self.assertIn('Digital content', document.blocks[1].digital_text)
        self.assertEqual(document.blocks[1].raw_text, document.blocks[1].text)
        from pdf_source_alignment import align_digital_pdf
        from test_source_chunk_alignment import CLEAN, CHUNK
        result = align_digital_pdf(self.pdf([('',True),('Digital content',True)]),
            'notes.pdf', 'doc', clean=CLEAN, chunk=CHUNK, extracted_document=document)
        self.assertIn('no_digital_text_provenance', result.blocks[0].diagnostics)
        self.assertEqual(result.blocks[1].diagnostics, ())
        self.assertTrue(any(i.source_spans for c in result.blocks[1].alignment.chunks for i in c.intervals))

    def test_ocr_failure_has_real_page_and_reason(self):
        with patch('document_acquisition.ocr_pdf',side_effect=RuntimeError('missing OCR binary')):
            _, report = acquire_document(self.pdf([('Readable',False),('',True)]),'notes.pdf')
        self.assertFalse(report['accepted'])
        self.assertEqual(report['units'][1]['number'],2)
        self.assertIn('ocr_error',report['units'][1]['reasons'])
        self.assertFalse(report['units'][1]['better_copy'])

    def test_exact_threshold_and_partial_not_counted_as_acquired(self):
        good = _unit(1,'Text')
        bad = _unit(2,'',visual=True)
        partial = _unit(3,'Text',visual=True)
        report = finish_report('file.pdf','page',[good]*8+[partial,bad])
        self.assertEqual(report['status'],'completed_with_warnings')
        self.assertEqual(report['acquired'],8)
        self.assertFalse(finish_report('file.pdf','page',[good]*8+[bad])['accepted'])
        self.assertFalse(finish_report('file.docx','document',[bad])['accepted'])

    def test_docx_short_text_and_table(self):
        from docx import Document
        doc = Document()
        doc.add_paragraph('Mitosis divides a cell.')
        table = doc.add_table(rows=1,cols=2)
        table.cell(0,0).text='DNA'; table.cell(0,1).text='Genetic material'
        output=io.BytesIO(); doc.save(output)
        document, report = acquire_document(output.getvalue(),'short.docx')
        self.assertIsNone(report['total'])
        self.assertEqual(report['status'],'completed')
        self.assertIn('Genetic material',' '.join(b.text for b in document.blocks))

    def test_docx_image_only_recovers_text_with_warning(self):
        from docx import Document
        from PIL import Image
        img=io.BytesIO(); Image.new('RGB',(10,10)).save(img,format='PNG'); img.seek(0)
        doc=Document(); doc.add_picture(img)
        output=io.BytesIO(); doc.save(output)
        with patch('document_acquisition.ocr_image',return_value='Short image lesson'):
            document, report=acquire_document(output.getvalue(),'image.docx')
        self.assertTrue(report['accepted'])
        self.assertEqual(report['partial'],1)
        self.assertIn('Short image lesson',' '.join(b.text for b in document.blocks))

    def test_pptx_slide_provenance_image_and_blank(self):
        from pptx import Presentation
        from pptx.util import Inches
        from PIL import Image
        deck=Presentation()
        slide=deck.slides.add_slide(deck.slide_layouts[6])
        slide.shapes.add_textbox(0,0,Inches(5),Inches(1)).text='Slide lesson'
        slide=deck.slides.add_slide(deck.slide_layouts[6])
        img=io.BytesIO(); Image.new('RGB',(10,10)).save(img,format='PNG'); img.seek(0)
        slide.shapes.add_picture(img,0,0)
        deck.slides.add_slide(deck.slide_layouts[6])
        output=io.BytesIO(); deck.save(output)
        with patch('document_acquisition.ocr_image',return_value='Image lesson'):
            document, report=acquire_document(output.getvalue(),'deck.pptx')
        self.assertEqual(report['total'],3)
        self.assertEqual([u['status'] for u in report['units']],['acquired','partial','blank'])
        self.assertTrue(any(b.source_unit==2 and b.text=='Image lesson' for b in document.blocks))
        self.assertFalse(any(b.source_unit==3 and b.text.strip() for b in document.blocks))

if __name__ == '__main__': unittest.main()

class AcquisitionEdgeTests(unittest.TestCase):
    def test_unreadable_document_has_no_invented_page_count(self):
        _, report = acquire_document(b'not a document','bad.docx')
        self.assertFalse(report['accepted'])
        self.assertIsNone(report['total'])
        self.assertEqual(report['units'][0]['reasons'],['parser_error'])

    def test_parser_failure_counts_even_when_fallback_recovers_some_text(self):
        report=finish_report('notes.pdf','page',[_unit(1,'Partial fallback',errors=['parser_error'])])
        self.assertEqual(report['failed'],1)
        self.assertFalse(report['accepted'])

    def test_text_with_failed_visual_ocr_is_partial_not_acquired(self):
        report=finish_report('notes.pdf','page',[_unit(1,'Digital text',visual=True,errors=['ocr_error'])])
        self.assertEqual(report['acquired'],0)
        self.assertEqual(report['partial'],1)
        self.assertEqual(report['status'],'completed_with_warnings')


class CharacterDamageTests(AcquisitionTests):
    def test_control_damage_attempts_ocr_once_without_erasing_original(self):
        from document_extractors import ExtractedDocument, ExtractedBlock
        body = "Il valore è \x7f metri, non un valore verificabile."
        block = ExtractedBlock(body, body, 1)
        doc = ExtractedDocument('damaged.pdf','pdf',10,[block],1)
        with patch('document_acquisition.extract_uploaded_document',return_value=doc),              patch('document_acquisition.ocr_pdf',return_value='Il valore è 50 metri.') as ocr:
            result, report=acquire_document(self.pdf([('Digital text',False)]),'damaged.pdf')
        ocr.assert_called_once()
        self.assertEqual(result.blocks[0].digital_text,body)
        self.assertIn(body,result.blocks[0].text)
        self.assertIn('50 metri',result.blocks[0].text)
        self.assertIn('text_encoding_anomaly',report['units'][0]['reasons'])
        self.assertEqual(report['status'],'completed_with_warnings')

    def test_math_and_multilingual_text_do_not_trigger_damage_heuristic(self):
        from document_acquisition import has_extraction_character_errors
        self.assertFalse(has_extraction_character_errors("Velocità v₀ = 0; Δs ≥ 2. English 中文.\n\t"))
        self.assertTrue(has_extraction_character_errors("Unavailable \ufffd"))
        self.assertTrue(has_extraction_character_errors("Broken \x7f"))
