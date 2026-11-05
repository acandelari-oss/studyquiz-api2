import ast
import copy
import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock
from taxonomy_organization import resolve_index_review

class IndexResolutionTests(TestCase):
    def setUp(self):
        self.topic = {'assignment_id':'uuid-original', 'title':'FISH probes', 'description':'Fluorescent probes',
                      'evidence_ids':[4], 'chapter_id':'index-review-4', 'document_id':'d'}
        self.fixed = {'assignment_id':'verified', 'title':'Mitosis', 'evidence_ids':[5], 'chapter_id':'index-1'}
        self.data = {'categories':[{'name':'Molecular Cytogenetics','topics':[]},
                                  {'name':'Cell Division','topics':[self.fixed]},
                                  {'name':'TO ORGANIZE','system_review':True,'topics':[self.topic]}]}
        self.chunks = [{'chunk_id':4,'text':'FISH uses fluorescent probes','page':23}]
        self.response = {'categories':[{'category_id':'C0000','name':'Citogenetica molecolare'},
                                       {'category_id':'C0001','name':'Divisione cellulare'}],
                         'assignments':[{'topic_id':'T0001','category_id':'C0000'}]}

    def test_resolves_without_changing_verified_assignment_or_provenance(self):
        before = copy.deepcopy(self.data)
        request = Mock(return_value=self.response)
        result = resolve_index_review(self.data, request, self.chunks, 'Italian')
        self.assertEqual(self.data, before)
        self.assertIs(result['categories'][0]['topics'][0], self.topic)
        self.assertIs(result['categories'][1]['topics'][0], self.fixed)
        self.assertEqual(self.topic['chapter_id'], 'index-review-4')
        self.assertEqual(request.call_count, 1)
        self.assertIn('FISH uses fluorescent probes', request.call_args.args[0])
        self.assertIn('Italian', request.call_args.args[0])

    def test_rejects_invalid_incomplete_duplicate_assignments_and_labels(self):
        for change in ('missing','duplicate','foreign','labels'):
            response = copy.deepcopy(self.response)
            if change == 'missing': response['assignments'] = []
            if change == 'duplicate': response['assignments'] *= 2
            if change == 'foreign': response['assignments'][0]['category_id'] = 'unknown'
            if change == 'labels': response['categories'].reverse()
            with self.subTest(change=change), self.assertRaises(ValueError):
                resolve_index_review(self.data, Mock(return_value=response), self.chunks, 'Italian')

    def test_uncertain_stays_review_and_no_work_means_no_call(self):
        self.response['assignments'][0]['category_id'] = None
        result = resolve_index_review(self.data, Mock(return_value=self.response), self.chunks, 'Italian')
        self.assertIs(result['categories'][-1]['topics'][0], self.topic)
        request = Mock()
        data = {'categories':self.data['categories'][:2]}
        self.assertIs(resolve_index_review(data, request, self.chunks), data)
        request.assert_not_called()

    def test_cross_document_rejected(self):
        self.fixed['document_id']='other'
        with self.assertRaises(ValueError):
            resolve_index_review(self.data, Mock(), self.chunks, 'Italian')

    def test_language_saved_in_existing_blueprint_for_all_modes(self):
        from fastapi import HTTPException
        tree = ast.parse(Path('main.py').read_text())
        fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_normalize_study_module_organization')
        from typing import Optional
        ns={'Optional':Optional,'HTTPException':HTTPException,'STUDY_MODULE_ORGANIZATION_MODES':{'infer','manual','syllabus'}}
        exec(compile(ast.Module(body=[fn],type_ignores=[]),'main.py','exec'),ns)
        for mode,blueprint in [('infer',{}),('syllabus',{'syllabus_text':'Index'}),('manual',{'categories':[{'name':'Original'}]})]:
            result=ns[fn.name](mode,{**blueprint,'output_language':'Italian'},None)
            self.assertEqual(result['organization_blueprint']['output_language'],'Italian')
        with self.assertRaises(HTTPException):
            ns[fn.name]('infer',{'output_language':'invalid'},None)
