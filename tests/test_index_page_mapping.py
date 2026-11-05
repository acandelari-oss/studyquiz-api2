"""Index labels, page authority, local review and immutable source contracts."""
import copy
import unittest
from test_chapter_provenance import fixture
from test_syllabus_outline import ns, org
from source_chunk_alignment import verified_index_chapters, metadata_revision
from taxonomy_organization import organize_verified_chapters, REVIEW_CATEGORY
from taxonomy_coverage import consolidate_candidates


class PageMappingTests(unittest.TestCase):
    def setUp(self):
        self.structure, self.chunks = fixture()
        self.items = [dict(display_title='Student area A',page_range=[1,1]),
                      dict(display_title='Student area B',page_range=[2,2])]
        self.structure.update(status='unavailable',chapters=[],reason='ambiguous_boundary_order')
        self.seal()

    def seal(self):
        self.structure['revision'] = metadata_revision(self.structure)
        for c in self.chunks:
            c['structural_provenance']['revision'] = self.structure['revision']

    def resolve(self):
        return verified_index_chapters(self.structure,self.chunks,'document-A',self.items)

    def shared_page(self):
        for b in self.structure['index_boundaries']:
            if b['start']['unit_index'] == 2:
                b['start']['unit_index'] = 1
                b['start']['block_index'] += 10
        self.structure['source_end']['unit_index'] = 2
        for c in self.chunks:
            p = c['structural_provenance']
            p['source_page'] = 1
            for interval in p['intervals']:
                for span in interval['source_spans']:
                    for position in (span['start'],span['end']):
                        if position['unit_index'] == 2:
                            position['unit_index'] = 1
                            position['block_index'] += 10
        self.items[1].update(page_range=[1,1],source_heading='2 Chapter Beta')
        self.seal()

    def test_labels_independent_from_headings_and_no_mutation(self):
        before=copy.deepcopy((self.structure,self.chunks,self.items))
        mapped,owners,status=self.resolve()
        self.assertEqual(status,'mapped')
        self.assertEqual([c['title'] for c in mapped['chapters']],['Student area A','Student area B'])
        self.assertEqual(len(set(owners.values())),2)
        self.assertEqual((self.structure,self.chunks,self.items),before)

    def test_shared_page_exact_anchor_separates_owners(self):
        self.shared_page()
        _,owners,status=self.resolve()
        self.assertEqual(status,'mapped')
        self.assertNotEqual(owners[1],owners[2])

    def test_ambiguous_anchor_is_local_review_not_global_fallback(self):
        self.shared_page()
        self.structure['index_boundaries'].append(copy.deepcopy(self.structure['index_boundaries'][-1]))
        self.seal()
        mapped,owners,status=self.resolve()
        self.assertEqual(status,'mapped_with_review')
        self.assertEqual(set(owners),{1,2})
        self.assertNotEqual(owners[1],owners[2]) # cannot consolidate unrelated review sources
        self.assertEqual(mapped['index_mapping']['review_chunks'],2)

    def test_boundary_ocr_requires_review_even_with_anchor(self):
        self.shared_page()
        interval=self.chunks[1]['structural_provenance']['intervals'][0]
        interval.update(source_spans=[],status='unresolved')
        mapped,owners,status=self.resolve()
        self.assertEqual(status,'mapped_with_review')
        self.assertEqual(owners[1],'index-0')
        self.assertTrue(owners[2].startswith('index-review'))
        self.assertEqual(mapped['index_mapping']['review_chunks'],1)

    def test_ocr_exclusive_page_keeps_all_material_without_coordinates(self):
        for interval in self.chunks[0]['structural_provenance']['intervals']:
            interval.update(source_spans=[],status='unresolved')
        _,owners,status=self.resolve()
        self.assertEqual(status,'mapped')
        self.assertEqual(owners[1],'index-0')

    def test_cross_boundary_evidence_goes_to_review(self):
        self.shared_page()
        self.chunks[0]['structural_provenance']['intervals'][0]['source_spans'].extend(
            copy.deepcopy(self.chunks[1]['structural_provenance']['intervals'][0]['source_spans']))
        mapped,owners,status=self.resolve()
        self.assertEqual(status,'mapped_with_review')
        self.assertTrue(owners[1].startswith('index-review'))
        self.assertEqual(owners[2],'index-1')

    def test_no_source_page_is_review_not_an_inferred_page(self):
        self.chunks[0]['structural_provenance']['source_page']=None
        mapped,owners,status=self.resolve()
        self.assertEqual(status,'mapped_with_review')
        self.assertEqual(mapped['index_mapping']['review_details'][0]['reason'],'missing_source_page')
        self.assertEqual(owners[2],'index-1')

    def test_invalid_source_revision_foreign_document_or_changed_text_rejected(self):
        for field,value in [('document_id','foreign'),('chunk_text','modified')]:
            before=copy.deepcopy(self.chunks)
            self.chunks[0][field]=value
            self.assertIsNone(self.resolve()[1])
            self.chunks=before
        self.structure['source_end']['unit_index']=999
        self.assertIsNone(self.resolve()[1])

    def test_duplicate_chunk_ids_or_broken_intervals_rejected(self):
        self.chunks.append(copy.deepcopy(self.chunks[0]))
        self.assertEqual(self.resolve()[2],'duplicate_index_chunk_id')
        self.chunks.pop()
        self.chunks[0]['structural_provenance']['intervals'][0]['start']=1
        self.assertEqual(self.resolve()[2],'invalid_index_chunk_intervals')

    def test_source_page_and_exact_spans_cannot_disagree(self):
        self.chunks[0]['structural_provenance']['source_page']=2
        self.assertEqual(self.resolve()[2],'inconsistent_index_source_page')

    def test_invalid_ranges_fail_before_any_assignment(self):
        self.items[0]['page_range']=[1,10]
        self.assertIsNone(self.resolve()[1])
        self.items[0]['page_range']=[2,2]
        self.items[1]['page_range']=[1,1]
        self.assertEqual(self.resolve()[2],'overlapping_or_unordered_index_ranges')

    def test_review_category_preserves_topics_and_evidence(self):
        self.shared_page()
        self.items[1]['source_heading']='Unverified heading'
        mapped,owners,_=self.resolve()
        topics=[dict(assignment_id=str(c['id']),title='Repeated subject',description='unchanged',
                     evidence_ids=[c['id']],chapter_id=owners[c['id']]) for c in self.chunks]
        before=copy.deepcopy(topics)
        result=organize_verified_chapters({'categories':[{'name':'Old','topics':topics}]},mapped,owners)
        self.assertEqual(result['categories'][-1]['name'],REVIEW_CATEGORY)
        self.assertTrue(result['categories'][-1]['system_review'])
        self.assertEqual(result['categories'][-1]['topics'],before)
        self.assertIs(result['categories'][-1]['topics'][0],topics[0])

    def test_review_scopes_cannot_merge_across_ambiguous_sources(self):
        self.shared_page()
        self.items[1]['source_heading']='Unknown'
        _,owners,_=self.resolve()
        candidates=[dict(candidate_id=str(i),assignment_id=str(i),topic='Repeated subject',
                         description='Same words',importance=5,category=REVIEW_CATEGORY,embedding=[1.,0.],
                         evidence_ids=[c['id']],chapter_id=owners[c['id']]) for i,c in enumerate(self.chunks)]
        result=consolidate_candidates(candidates,lambda _:self.fail('Cross-scope review call'),'')
        self.assertEqual(sum(len(c['topics']) for c in result['categories']),2)

    def test_anchor_metadata_is_not_an_extra_topic_or_category(self):
        data=org('### Section 1 — Student label (pp. 1–1)\n- Source heading: "Printed heading"\n- Learning subject')
        outline=ns['_parse_syllabus_outline'](data)
        self.assertEqual(outline[0]['source_heading'],'Printed heading')
        self.assertEqual(len(outline),2)
        self.assertEqual(ns['_global_organization_categories'](data)[0]['name'],'Student label')
        with self.assertRaises(ValueError):
            ns['_parse_syllabus_outline'](org('- Source heading: Unattached'))
