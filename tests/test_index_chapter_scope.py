"""Index ranges only authorize scope when corroborated by source provenance."""
import copy
import unittest
from test_chapter_provenance import fixture
from source_chunk_alignment import verified_index_chapters, metadata_revision
from taxonomy_organization import organize_verified_chapters

class IndexScopeTests(unittest.TestCase):
    def setUp(self):
        self.structure, self.chunks = fixture()
        # Retain coverage of metadata written before page-range mapping existed.
        self.structure.pop('index_page_provenance_version', None)
        self.structure['revision'] = metadata_revision(self.structure)
        for chunk in self.chunks:
            chunk['structural_provenance']['revision'] = self.structure['revision']
        self.items = [{'display_title': '1 Chapter Alpha', 'page_range': [1, 1]},
                      {'display_title': '2 Chapter Beta', 'page_range': [2, 2]}]

    def resolve(self, **kwargs):
        return verified_index_chapters(kwargs.get('structure', self.structure),
             kwargs.get('chunks', self.chunks), kwargs.get('document_id', 'document-A'),
             kwargs.get('items', self.items))

    def test_verified_ranges_and_repeated_subjects(self):
        before = copy.deepcopy((self.structure, self.chunks, self.items))
        structure, owners, reason = self.resolve()
        self.assertEqual(reason, 'verified')
        topics = [{'assignment_id': str(i), 'title': 'Shared subject', 'evidence_ids': [c['id']]}
                  for i, c in enumerate(self.chunks)]
        result = organize_verified_chapters({'categories': [{'name': 'Old', 'topics': topics}]}, structure, owners)
        self.assertEqual([c['name'] for c in result['categories']], [i['display_title'] for i in self.items])
        self.assertEqual([len(c['topics']) for c in result['categories']], [1, 1])
        self.assertEqual((self.structure, self.chunks, self.items), before)

    def test_wrong_pdf_page_offsets_do_not_authorize_scope(self):
        self.items[0]['page_range'] = [2, 2]
        self.assertEqual(self.resolve()[2], 'index_source_page_mismatch')

    def test_matching_pages_without_matching_headings_are_insufficient(self):
        self.items[0]['display_title'] = 'Different subject'
        self.assertEqual(self.resolve()[2], 'index_source_title_mismatch')

    def test_missing_ranges_or_structure_falls_back(self):
        self.items[0]['page_range'] = None
        self.assertEqual(self.resolve()[2], 'missing_or_invalid_index_range')
        self.assertEqual(self.resolve(structure=None)[2], 'unverified_source_ownership')

    def test_foreign_or_modified_chunk_falls_back(self):
        self.assertEqual(self.resolve(document_id='other')[2], 'invalid_index_source_revision')
        self.chunks[0]['chunk_text'] += 'changed'
        self.assertEqual(self.resolve()[2], 'invalid_index_chunk_revision')


class IndependentIndexScopeTests(IndexScopeTests):
    def setUp(self):
        super().setUp()
        # Automatic chapters unavailable; independently checked starts remain.
        self.structure['status'] = 'unavailable'
        self.structure['reason'] = 'no_reliable_main_headings'
        self.structure['chapters'] = []
        self.structure['revision'] = metadata_revision(self.structure)
        for chunk in self.chunks:
            chunk['structural_provenance']['revision'] = self.structure['revision']
            for interval in chunk['structural_provenance']['intervals']:
                interval['chapter_id'] = None

    def test_wrong_pdf_page_offsets_do_not_authorize_scope(self):
        self.items[0]['page_range'] = [2, 2]
        self.assertIsNone(self.resolve()[1])

    def test_matching_pages_without_matching_headings_are_insufficient(self):
        self.items[0]['display_title'] = 'Different subject'
        self.assertEqual(self.resolve()[2], 'unverified_or_ambiguous_index_heading')

    def test_foreign_or_modified_chunk_falls_back(self):
        self.assertIsNone(self.resolve(document_id='other')[1])
        self.chunks[0]['chunk_text'] += 'changed'
        self.assertIsNone(self.resolve()[1])

    def test_ambiguous_heading_is_rejected(self):
        self.structure['index_boundaries'].append(copy.deepcopy(self.structure['index_boundaries'][0]))
        self.structure['revision'] = metadata_revision(self.structure)
        self.assertEqual(self.resolve()[2], 'unverified_or_ambiguous_index_heading')

    def test_unresolved_content_is_not_discarded(self):
        interval = self.chunks[0]['structural_provenance']['intervals'][0]
        interval['source_spans'] = []
        interval['status'] = 'unresolved'
        self.assertEqual(self.resolve()[2], 'incomplete_index_source_ownership')

    def test_cross_boundary_chunk_cannot_be_forced_into_one_category(self):
        interval = self.chunks[0]['structural_provenance']['intervals'][0]
        interval['source_spans'].extend(copy.deepcopy(
            self.chunks[1]['structural_provenance']['intervals'][0]['source_spans']))
        self.assertEqual(self.resolve()[2], 'incomplete_index_source_ownership')

    def test_midpage_start_uses_exact_block_positions(self):
        for boundary in self.structure['index_boundaries']:
            if boundary['start']['unit_index'] == 2:
                boundary['start']['unit_index'] = 1
                boundary['start']['block_index'] += 10
        self.structure['source_end']['unit_index'] = 2
        self.structure['revision'] = metadata_revision(self.structure)
        for chunk in self.chunks:
            p = chunk['structural_provenance']
            p['revision'] = self.structure['revision']
            for interval in p['intervals']:
                for span in interval['source_spans']:
                    for position in (span['start'], span['end']):
                        if position['unit_index'] == 2:
                            position['unit_index'] = 1
                            position['block_index'] += 10
        self.items[1]['page_range'] = [1, 1]
        mapped, owners, reason = self.resolve()
        self.assertEqual(reason, 'verified')
        self.assertNotEqual(owners[1], owners[2])
        interval = self.chunks[1]['structural_provenance']['intervals'][0]
        interval.update(status='unresolved', source_spans=[], source_page=1)
        self.assertEqual(self.resolve()[2], 'incomplete_index_source_ownership')

    def test_ocr_interior_page_uses_page_evidence_without_coordinates(self):
        from source_chunk_alignment import text_digest
        for boundary in self.structure['index_boundaries']:
            if boundary['start']['unit_index'] == 2:
                boundary['start']['unit_index'] = 3
        self.structure['source_end']['unit_index'] = 4
        self.structure['revision'] = metadata_revision(self.structure)
        for chunk in self.chunks:
            p = chunk['structural_provenance']
            p['revision'] = self.structure['revision']
            for interval in p['intervals']:
                for span in interval['source_spans']:
                    for position in (span['start'], span['end']):
                        if position['unit_index'] == 2:
                            position['unit_index'] = 3
        self.items[0]['page_range'] = [1, 2]
        self.items[1]['page_range'] = [3, 3]
        self.chunks.append(dict(id=3, document_id='document-A', chunk_text='OCR',
            structural_provenance=dict(version=1,document_id='document-A',
                revision=self.structure['revision'],chunk_sha256=text_digest('OCR'),
                intervals=[dict(start=0,end=3,status='unresolved',source_spans=[],source_page=2)])))
        before = copy.deepcopy(self.chunks)
        mapped, owners, reason = self.resolve()
        self.assertEqual(reason, 'verified')
        self.assertEqual(owners[3], owners[1])
        self.assertEqual(self.chunks, before)
