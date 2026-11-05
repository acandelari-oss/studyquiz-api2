"""Offline hierarchy contracts: no services, uploads or stored data."""
import copy
import json
import unittest
from taxonomy_organization import organize_categories, translate_final_taxonomy
from test_syllabus_outline import ns, org
from test_index_chapter_scope import IndependentIndexScopeTests


class HierarchyTests(unittest.TestCase):
    def test_contents_page_starts_and_parent_relationships(self):
        outline = ns['_parse_syllabus_outline'](org(
            '1 What is logic? 5\n1.1 Arguments . . . . 5\n'
            'Practice Exercises . . . . 14\n2 Sentential logic 16\n'
            '2.1 Sentence letters . . . . 16\n3 Truth tables 35'))
        categories = ns['_syllabus_likely_category_items'](outline)
        self.assertEqual([c['display_title'] for c in categories],
                         ['What is logic?', 'Sentential logic', 'Truth tables'])
        self.assertEqual([c['page_start'] for c in categories], [5, 16, 35])
        self.assertEqual(outline[1]['parent_index'], 0)
        self.assertEqual(outline[4]['parent_index'], 3)
        self.assertTrue(all(c['page_range'] is None for c in categories))

    def test_macro_groups_preserve_topic_identity_and_translate_parent_once(self):
        topics = [dict(assignment_id=str(i), title='Title '+str(i), description='Description',
                       evidence_ids=[i], document_id='doc') for i in range(2)]
        data = {'categories':[{'name':'Old', 'topics':topics}]}
        before = copy.deepcopy(data)
        def request(prompt, stage):
            if stage == 'category_proposal':
                return {'categories':[{'name':n,'macrocategory':'Logic','description':n}
                                      for n in ['Syntax','Semantics']]}
            return {'assignments':[{'topic_id':f'T{i+1:04d}','category_id':f'C{i:04d}'}
                                   for i in range(2)],'quality_issues':[]}
        result = organize_categories(data,'infer',[],request,document_context={'require_hierarchy':True})
        self.assertEqual(data,before)
        self.assertIs(result['categories'][1]['topics'][0],topics[1])
        requested = []
        def translate(prompt, stage):
            items = json.loads(prompt.split('\n',1)[1])['items']; requested.extend(items)
            return {'items':[dict(i,text='Logica' if i['text']=='Logic' else i['text']) for i in items]}
        final = translate_final_taxonomy(result,'Italian',translate)
        self.assertEqual(sum(i['text']=='Logic' for i in requested),1)
        self.assertEqual([c['macrocategory'] for c in final['categories']],['Logica','Logica'])

    def test_automatic_missing_parent_fails_before_assignment(self):
        calls=[]
        def request(prompt,stage):
            calls.append(stage)
            return {'categories':[{'name':'Area','description':'Scope'}]}
        with self.assertRaisesRegex(ValueError,'omitted macrocategory'):
            organize_categories({'categories':[{'name':'Old','topics':[{'assignment_id':'t','title':'T'}]}]},
                'infer',[],request,document_context={'require_hierarchy':True})
        self.assertEqual(calls,['category_proposal'])


class IndexStartTests(IndependentIndexScopeTests):
    def test_starts_resolve_against_existing_verified_positions(self):
        for item in self.items:
            item['page_start'] = item.pop('page_range')[0]
        structure, owners, status = self.resolve()
        self.assertIsNotNone(owners)
        self.assertEqual(len(owners),len(self.chunks))
