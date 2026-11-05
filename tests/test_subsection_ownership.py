"""Exact source ownership without extraction, AI or database calls."""
import copy
import unittest
from source_chunk_alignment import metadata_revision, text_digest, verified_chunk_chapters
from taxonomy_organization import organize_verified_chapters


def position(page, block, offset=0):
    return dict(unit_index=page, block_index=block, start_offset=offset)


def fixture():
    structure = dict(version=1, document_id='d', status='verified', chapters=[], nodes=[], index_boundaries=[])
    for page in (1, 2):
        root = f'root{page}'
        structure['chapters'].append(dict(id=root, title=f'Chapter {page}', order=page,
            extent=dict(start=position(page,1), end=position(page+1,1))))
        for block, title in [(1,f'Chapter {page}'), (3,'Definitions'), (6,'Applications')]:
            node = root if block == 1 else f'{root}-{block}'
            structure['nodes'].append(dict(node_id=node, parent_id=None if block==1 else root,
                source_title=title, status='accepted', provenance={}))
            structure['index_boundaries'].append(dict(id=node,title=title,start=position(page,block)))
    structure['revision'] = metadata_revision(structure)
    chunks = []
    for page, block in [(1,2),(1,4),(1,7),(2,4)]:
        body='Source passage'
        chunks.append(dict(id=len(chunks)+1, document_id='d', chunk_text=body,
            structural_provenance=dict(version=1,document_id='d',revision=structure['revision'],
                chunk_sha256=text_digest(body), intervals=[dict(start=0,end=len(body),status='exact',
                    chapter_id=f'root{page}', source_spans=[dict(start=position(page,block),
                                                               end=position(page,block,len(body)))])])))
    topics=[dict(assignment_id=str(c['id']),title='Repeated subject',description='Unchanged',
                 evidence_ids=[c['id']],document_id='d') for c in chunks]
    return structure,chunks,{'categories':[{'name':'Old','topics':topics}]}


class SubsectionTests(unittest.TestCase):
    def organize(self, structure, chunks, data):
        owners=verified_chunk_chapters(structure,chunks,'d')
        self.assertIsNotNone(owners)
        return organize_verified_chapters(data,structure,owners,chunks=chunks)

    def refresh(self, structure, chunks):
        structure['revision']=metadata_revision(structure)
        for chunk in chunks:
            chunk['structural_provenance']['revision']=structure['revision']

    def test_mid_page_sections_preserve_intro_and_repeated_titles(self):
        structure,chunks,data=fixture(); before=copy.deepcopy((structure,chunks,data))
        result=self.organize(structure,chunks,data)
        self.assertEqual([c['name'] for c in result['categories']],
            ['Chapter 1','Chapter 1 — Definitions','Chapter 1 — Applications','Chapter 2 — Definitions'])
        for category in result['categories']:
            for topic in category['topics']:
                self.assertIs(topic,data['categories'][0]['topics'][int(topic['assignment_id'])-1])
        self.assertEqual((structure,chunks,data),before)

    def test_topic_supported_in_two_subsections_stays_in_chapter(self):
        structure,chunks,data=fixture()
        data['categories'][0]['topics'][1]['evidence_ids']=[2,3]
        result=self.organize(structure,chunks,data)
        self.assertIn(data['categories'][0]['topics'][1],result['categories'][0]['topics'])

    def test_chunk_crossing_section_start_stays_in_chapter(self):
        structure,chunks,data=fixture()
        chunks[1]['structural_provenance']['intervals'][0]['source_spans'][0]=dict(
            start=position(1,2),end=position(1,4))
        result=self.organize(structure,chunks,data)
        self.assertIn(data['categories'][0]['topics'][1],result['categories'][0]['topics'])

    def test_unresolved_sibling_disables_subsections_only_in_its_chapter(self):
        structure,chunks,data=fixture()
        structure['nodes'][2]['status']='unresolved'; self.refresh(structure,chunks)
        result=self.organize(structure,chunks,data)
        self.assertEqual(len(result['categories'][0]['topics']),3)
        self.assertEqual(result['categories'][1]['name'],'Definitions')
        self.assertEqual(result['categories'][1]['macrocategory'],'Chapter 2')

    def test_missing_boundary_does_not_extend_previous_section(self):
        structure,chunks,data=fixture()
        structure['index_boundaries'].pop(2); self.refresh(structure,chunks)
        result=self.organize(structure,chunks,data)
        self.assertEqual(len(result['categories'][0]['topics']),3)

    def test_modified_snapshot_and_foreign_document_rejected(self):
        for kind in ('text','document'):
            structure,chunks,data=fixture()
            owners=verified_chunk_chapters(structure,chunks,'d')
            if kind=='text': chunks[0]['chunk_text']='Changed'
            else: chunks[0]['document_id']='other'
            with self.assertRaisesRegex(ValueError,'snapshot'):
                organize_verified_chapters(data,structure,owners,chunks=chunks)

    def test_legacy_without_child_nodes_keeps_chapters(self):
        structure,chunks,data=fixture()
        structure['nodes']=[]; self.refresh(structure,chunks)
        result=self.organize(structure,chunks,data)
        self.assertEqual([c['name'] for c in result['categories']],['Chapter 1','Chapter 2'])

    def test_orphan_boundary_cannot_extend_a_verified_sibling(self):
        structure,chunks,data=fixture()
        structure['nodes'].append(dict(node_id='orphan',parent_id=None,source_title='Other',status='unresolved'))
        structure['index_boundaries'].append(dict(id='orphan',title='Other',start=position(1,5)))
        self.refresh(structure,chunks)
        result=self.organize(structure,chunks,data)
        self.assertEqual(len(result['categories'][0]['topics']),3)

    def test_nested_heading_keeps_its_direct_parent_section(self):
        structure,chunks,data=fixture()
        structure['nodes'].append(dict(node_id='nested',parent_id='root1-3',source_title='Example',status='accepted'))
        structure['index_boundaries'].append(dict(id='nested',title='Example',start=position(1,4)))
        self.refresh(structure,chunks)
        result=self.organize(structure,chunks,data)
        self.assertEqual(result['categories'][1]['topics'][0]['assignment_id'],'2')
        self.assertNotIn('Example',[c['name'] for c in result['categories']])

    def test_page_only_ocr_ownership_stays_at_chapter_level(self):
        structure,chunks,data=fixture()
        structure['chapters'][0].update(whole_page_start=1,whole_page_end=1)
        interval=chunks[1]['structural_provenance']['intervals'][0]
        interval.update(status='unresolved',source_spans=[],source_page=1,
                        ownership_basis='chapter_page_interval')
        self.refresh(structure,chunks)
        result=self.organize(structure,chunks,data)
        self.assertIn(data['categories'][0]['topics'][1],result['categories'][0]['topics'])
