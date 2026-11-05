"""Real, in-memory PDF fixtures using the repository's existing pypdf builder."""
import contextlib
from dataclasses import replace
import io
import unittest
from unittest.mock import patch
from test_document_interpreter_pdf import _pdf_bytes, _text_command
from test_source_chunk_alignment import CLEAN, CHUNK
from document_extractors import extract_pdf_document
from pdf_source_alignment import align_digital_pdf


def pdf(lines):
    return _pdf_bytes([[_text_command(text, x=x, y=y, size=size) for text, x, y, size in lines]])


def run(data, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return align_digital_pdf(data, 'fixture.pdf', 'document-A', clean=CLEAN, chunk=CHUNK, **kw)


class PdfAlignmentTests(unittest.TestCase):
    def test_boundary_conflict_retains_positions_and_safe_fallback(self):
        from pdf_source_alignment import chapter_metadata
        data = _pdf_bytes([[_text_command(title,x=72,y=740,size=20)] +
                           [_text_command(f'Body line {i}.',x=72,y=700-i*30,size=12) for i in range(3)]
                           for title in ('1 Chapter Alpha', '2 Chapter Beta')])
        result = run(data)
        result.canonical_document.blocks[1].metadata['y'] = 780
        metadata, provenance = chapter_metadata(result)
        self.assertEqual(metadata['reason'], 'ambiguous_boundary_order')
        self.assertEqual(metadata['chapters'], [])
        diagnostic = metadata['boundary_diagnostics'][0]
        self.assertEqual(diagnostic['page'], 1)
        self.assertIn('Chapter', diagnostic['heading_text'])
        self.assertIn('Body line', diagnostic['conflicting_text'])
        self.assertGreater(diagnostic['conflicting_y'], diagnostic['heading_y'])
        self.assertTrue(all(i['chapter_id'] is None for group in provenance for c in group for i in c['intervals']))

    def test_metadata_retains_extraction_diagnostics_without_changing_fallback(self):
        from pdf_source_alignment import chapter_metadata
        from source_chunk_alignment import metadata_revision
        result = run(pdf([('Body text',72,700,12)]))
        result = replace(result, blocks=(replace(result.blocks[0],
            diagnostics=('extraction_identity_mismatch', 'no_digital_text_provenance')),))
        metadata, provenance = chapter_metadata(result)
        self.assertEqual(metadata['status'], 'unavailable')
        self.assertEqual(metadata['reason'], 'unverified_extraction')
        self.assertEqual(metadata['chapters'], [])
        self.assertEqual(metadata['extraction_diagnostics'], [{
            'page': 1, 'extraction_block_index': 1,
            'codes': ['extraction_identity_mismatch', 'no_digital_text_provenance']}])
        self.assertEqual(metadata['revision'], metadata_revision(metadata))
        self.assertTrue(provenance)

    def test_supplement_preserves_digital_spans_and_isolates_bad_page(self):
        from document_acquisition import acquire_document, _supplement
        data = _pdf_bytes([[_text_command('Digital body', x=72, y=700, size=12)],
                           [_text_command('Other body', x=72, y=700, size=12)]])
        original = extract_pdf_document(data, 'fixture.pdf', inspect_acquisition=True)
        a = original.blocks[0]
        a.digital_text = a.text
        a.text = a.raw_text = _supplement(a.text, 'OCR appendix')
        a.digital_text_start = len(a.digital_text)-len(a.digital_text.lstrip())
        original.blocks[1].text = 'Invalid replacement'
        result = run(data, extracted_document=original)
        self.assertEqual(result.blocks[0].diagnostics, ())
        self.assertIn('extraction_identity_mismatch', result.blocks[1].diagnostics)
        intervals = result.blocks[0].alignment.chunks[0].intervals
        self.assertTrue(any(i.source_spans for i in intervals))
        self.assertTrue(any(i.status == 'unresolved' for i in intervals))
        self.assertEqual([c.text for c in result.blocks[0].alignment.chunks],
                         [CLEAN(c) for c in CHUNK(CLEAN(a.text)) if CLEAN(c)])

    def test_scaled_nominal_fonts_recognize_headings_without_promoting_confidence(self):
        from canonical_document import adapt_pdf_document
        from document_hierarchy_evidence_mapper import map_document_to_structural_evidence
        from document_hierarchy_preserver import preserve_document_hierarchy
        commands = []
        for title, y, scale in [('Alpha',740,30),('Body text',700,12),
                                ('Beta',600,30),('More text',560,12),('Last body',520,12)]:
            commands.append(f'BT /F1 1 Tf {scale} 0 0 {scale} 72 {y} Tm ({title}) Tj ET')
        data = _pdf_bytes([commands])
        document = adapt_pdf_document(data, 'scaled.pdf')
        self.assertEqual([b.metadata['font_size'] for b in document.blocks], [1]*5)
        self.assertEqual([b.metadata['effective_font_size'] for b in document.blocks], [30,12,30,12,12])
        structure = preserve_document_hierarchy(map_document_to_structural_evidence(document),
            document_id='doc', document_title='scaled.pdf', source_format='pdf')
        self.assertEqual([n.source_title for n in structure.nodes], ['Alpha','Beta'])
        self.assertTrue(all(n.status == 'accepted' and n.confidence == 'MEDIUM' for n in structure.nodes))
        from pdf_source_alignment import chapter_metadata
        metadata, _ = chapter_metadata(run(data))
        self.assertEqual(metadata['status'], 'verified')
        self.assertNotEqual(metadata.get('reason'), 'transformed_text')

    def test_chapters_nested_headings_and_cross_fragment_chunk(self):
        data = pdf([('1 Chapter Alpha',72,740,20), ('First body text.',72,710,12),
                    ('1.1 Detail',72,680,16), ('Nested body text.',72,650,12),
                    ('2 Chapter Beta',72,610,20), ('Second body text.',72,580,12)])
        before = extract_pdf_document(data, 'fixture.pdf', inspect_acquisition=True)
        result = run(data)
        self.assertEqual(before, extract_pdf_document(data, 'fixture.pdf', inspect_acquisition=True))
        self.assertEqual(len(result.blocks), 1)
        aligned = result.blocks[0].alignment
        self.assertNotIn('ambiguous', aligned.diagnostics)
        self.assertTrue(all(c.text[i.chunk_start:i.chunk_end].isspace() for c in aligned.chunks for i in c.intervals if i.status == 'unresolved'))
        with contextlib.redirect_stdout(io.StringIO()):
            expected = [CLEAN(c) for c in CHUNK(CLEAN(before.blocks[0].text)) if CLEAN(c)]
        self.assertEqual([c.text for c in aligned.chunks], expected)
        spans = [s for c in aligned.chunks for i in c.intervals for s in i.source_spans]
        self.assertEqual({s.start.block_index for s in spans}, {1,2,3,4,5,6})
        self.assertTrue(result.structural_evidence)
        self.assertFalse(result.section_extents_verified)
        # Source evidence identifies heading fragments; it is never promoted to
        # an authoritative section extent by this extraction adapter.
        for e, b in zip(result.structural_evidence, result.canonical_document.blocks):
            self.assertEqual(e.source_span.start, b.source_position)

    def test_repeated_text_retains_event_identity_and_offsets(self):
        data = pdf([('Repeated heading',72,740,18), ('Repeated heading',72,700,18)])
        result = run(data)
        source_blocks = result.canonical_document.blocks
        exact = [s for i in result.blocks[0].alignment.chunks[0].intervals for s in i.source_spans]
        self.assertEqual(len(source_blocks), 2)
        self.assertEqual({s.start.block_index for s in exact}, {1,2})
        self.assertEqual([s.start.start_offset for s in exact if s.start.start_offset == 0], [0,0])

    def test_normalization_and_overlap_preserve_source_characters(self):
        data = pdf([('alphaBeta  12C.Test ' * 90,72,700,12)])
        result = run(data, max_chars=90, overlap=20)
        blocks = {b.source_position.block_index: b for b in result.canonical_document.blocks}
        self.assertGreater(len(result.blocks[0].alignment.chunks), 1)
        statuses = set()
        for c in result.blocks[0].alignment.chunks:
            for interval in c.intervals:
                statuses.add(interval.status)
                if interval.status == 'exact':
                    text = ''.join(blocks[s.start.block_index].raw_text[s.start.start_offset:s.end.start_offset] for s in interval.source_spans)
                    self.assertEqual(' ' if text.isspace() else text, c.text[interval.chunk_start:interval.chunk_end])
        self.assertIn('generated', statuses)
        self.assertNotIn('ambiguous', statuses)

    def test_unreliable_layout_does_not_authorize_chapter_ownership(self):
        # Content-stream order goes bottom-right -> top-left -> overlapping text.
        data = pdf([('Bottom first',330,100,12), ('Top second',72,740,12), ('Top second',72,740,12)])
        result = run(data)
        self.assertFalse(result.section_extents_verified)
        self.assertIn('semantic_reading_order_and_section_extents_unverified', result.diagnostics)
        self.assertFalse(hasattr(result, 'chapter_assignments'))

    def test_ocr_only_and_blank_pages_stay_unresolved(self):
        data = _pdf_bytes([['q 100 0 0 100 72 600 cm BI /W 1 /H 1 /CS /RGB /BPC 8 /F /AHx ID 000000> EI Q']])
        blank = run(data)
        self.assertIn('no_digital_text_provenance', blank.blocks[0].diagnostics)
        original = extract_pdf_document(data, 'fixture.pdf', inspect_acquisition=True)
        original.blocks[0].text = original.blocks[0].raw_text = 'OCR recovered text'
        result = run(data, extracted_document=original)
        self.assertIn('extraction_identity_mismatch', result.blocks[0].diagnostics)
        self.assertTrue(all(i.status == 'unresolved' and not i.source_spans for c in result.blocks[0].alignment.chunks for i in c.intervals))

    def test_split_extraction_blocks_cannot_be_matched_by_page(self):
        data = pdf([('First body Second body',72,700,12)])
        original = extract_pdf_document(data, 'fixture.pdf', inspect_acquisition=True)
        a = original.blocks[0]
        original.blocks = [replace(a, text='First body', raw_text='First body'), replace(a, text='Second body', raw_text='Second body', block_index=2)]
        result = run(data, extracted_document=original)
        self.assertEqual(len(result.blocks), 2)
        self.assertTrue(all(not i.source_spans for b in result.blocks for c in b.alignment.chunks for i in c.intervals))

    def test_event_tape_disagreement_is_ambiguous_without_associations(self):
        data = pdf([('Same',72,700,12)])
        from canonical_document import adapt_pdf_document
        canonical = adapt_pdf_document(data, 'fixture.pdf')
        canonical.units[0].blocks[0] = replace(canonical.blocks[0], raw_text='Changed')
        with patch('pdf_source_alignment.adapt_pdf_document', return_value=canonical):
            result = run(data)
        self.assertIn('ambiguous_extraction_event_tape', result.blocks[0].diagnostics)
        self.assertTrue(all(i.status == 'ambiguous' and not i.source_spans for c in result.blocks[0].alignment.chunks for i in c.intervals))

    def test_multiple_pages_keep_block_order_and_document_identity(self):
        data = _pdf_bytes([[_text_command('Repeated',x=72,y=700)], [_text_command('Repeated',x=72,y=700)]])
        result = run(data)
        self.assertEqual([b.page for b in result.blocks], [1,2])
        for page, b in enumerate(result.blocks,1):
            self.assertEqual(b.alignment.document_id, 'document-A')
            self.assertEqual({s.start.unit_index for c in b.alignment.chunks for i in c.intervals for s in i.source_spans}, {page})


if __name__ == '__main__':
    unittest.main()
