"""Offline upload contract tests; no feature endpoints, network or database."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import Mock, patch
from taxonomy_organization import translate_final_taxonomy
from taxonomy_coverage import validate_discovery_evidence
from taxonomy_source_review import recover_source_links
from taxonomy_assignment import validate_evidence

BODY = "La velocità è la derivata prima dello spazio rispetto al tempo."


class UploadLanguageAdapterTests(unittest.TestCase):
    def data(self):
        return {'categories':[{'name':'Cinematica', 'topics':[{
            'assignment_id':'stable-topic', 'title':'Velocità', 'description':BODY,
            'importance':5, 'evidence_ids':[1], 'candidate_ids':['0'],
            'original_objectives':[BODY], 'embedding':[1,0], 'chapter_id':'chapter-1',
            'document_id':'document'}]}]}

    def request(self, prompt, stage):
        self.assertEqual(stage, 'taxonomy_translation')
        payload = json.loads(prompt.split('\n',1)[1])
        translations = {'Cinematica':'Kinematics', 'Velocità':'Velocity',
                        BODY:'Velocity is the first derivative of displacement with respect to time.'}
        return {'items':[{'id':i['id'], 'text':translations.get(i['text'],i['text'])} for i in payload['items']]}

    def test_final_fields_translate_but_ids_lineage_and_assignments_do_not(self):
        data = self.data(); before = copy.deepcopy(data)
        result = translate_final_taxonomy(data, 'English', self.request)
        self.assertEqual(data,before)
        self.assertEqual(result['categories'][0]['name'],'Kinematics')
        old, new = data['categories'][0]['topics'][0], result['categories'][0]['topics'][0]
        self.assertEqual(new['title'],'Velocity')
        for key in ('assignment_id','evidence_ids','candidate_ids','original_objectives','importance','chapter_id','document_id'):
            self.assertEqual(old[key],new[key])
        self.assertNotIn('embedding',new)
        validate_evidence(result,[{'assignment_id':'stable-topic','evidence_ids':{1},'description':BODY}],{1})

    def test_manual_syllabus_and_system_names_are_not_translated(self):
        for protected, system in [(True,False),(False,True)]:
            data=self.data(); data['categories'][0]['system_review']=system
            result=translate_final_taxonomy(data,'English',self.request,preserve_category_names=protected)
            self.assertEqual(result['categories'][0]['name'],'Cinematica')
            self.assertEqual(result['categories'][0]['topics'][0]['title'],'Velocity')

    def test_legacy_without_output_language_has_no_request(self):
        request=Mock()
        self.assertEqual(translate_final_taxonomy(self.data(),None,request),self.data())
        request.assert_not_called()

    def test_invalid_or_incomplete_response_cannot_mutate_input(self):
        for response in [{'items':[]}, {'items':[{'id':'unknown','text':'X'}]}, {'items':'bad'}]:
            data=self.data();before=copy.deepcopy(data);request=Mock(return_value=response)
            with self.assertRaises(ValueError):
                translate_final_taxonomy(data,'English',request)
            self.assertEqual(data,before)
            self.assertEqual(request.call_count,2 if response == {'items':[]} else 1)

    def test_numeric_change_and_duplicate_ids_are_rejected(self):
        for duplicate in [True,False]:
            data=self.data(); data['categories'][0]['topics'][0]['description']='v = 2'
            def request(prompt, stage):
                response=self.request(prompt, stage)
                if duplicate:
                    response['items'].append(response['items'][0])
                else:
                    response['items'][-1]['text']='v = 3'
                return response
            with self.assertRaises(ValueError):
                translate_final_taxonomy(data,'English',request)

    def test_bounded_requests_cover_every_item_without_truncation(self):
        data=self.data()
        base=data['categories'][0]['topics'][0]
        data['categories'][0]['topics']=[dict(base,assignment_id=str(i),description=('Testo lungo. '*50).strip()) for i in range(8)]
        seen=[];prompts=[]
        def request(prompt, stage):
            prompts.append(prompt)
            items=json.loads(prompt.split('\n',1)[1])['items']
            seen.extend(i['id'] for i in items)
            return {'items':items}
        with patch('taxonomy_organization.MAX_PROMPT',2000):
            result=translate_final_taxonomy(data,'English',request)
        self.assertGreater(len(prompts),1)
        self.assertEqual(len(seen),17)
        self.assertEqual(len(set(seen)),17)
        self.assertTrue(all(len(p)<=2000 for p in prompts))
        self.assertEqual(result,data)

    def test_source_selection_is_resolved_locally_in_original_language(self):
        data={'categories':[{'name':'Area','topics':[{'title':'Velocity','description':'Definition',
              'source_evidence':[{'chunk_id':'1','support':'direct'}]}]}]}
        validate_discovery_evidence(data,[{'chunk_id':1,'text':BODY}])
        self.assertEqual(data['categories'][0]['topics'][0]['source_evidence'],
                         [{'chunk_id':'1','quote':BODY}])

    def test_unknown_source_and_unconfirmed_support_never_authorize_evidence(self):
        for reference in [{'chunk_id':'99','support':'direct'},
                          {'chunk_id':'1','support':'uncertain'},
                          {'chunk_id':True,'support':'direct'}]:
            data={'categories':[{'name':'Area','topics':[{'title':'T','description':'D','source_evidence':[reference]}]}]}
            validate_discovery_evidence(data,[{'chunk_id':1,'text':BODY}])
            self.assertEqual(data['categories'][0]['topics'][0]['source_evidence'],[])

    def test_recovery_reference_and_repair_preserve_original_source(self):
        candidate={'assignment_id':'t','topic':'Velocità','description':BODY,'evidence_ids':set(),
                   'discovery_chunk_ids':[1],'chapter_id':'a'}
        response={'links':[{'source_id':'S0','topic_id':'T0','support':'direct'}],
                  'repairs':[{'template_id':'T0','topic_id':'T0','title':'Velocità','description':BODY,
                              'citations':[{'source_id':'S0','support':'direct'}]}]}
        result,report=recover_source_links([candidate],[{'chunk_id':1,'text':BODY}],Mock(return_value=response),{1:'a'})
        self.assertEqual(result[0]['evidence_ids'],{1})
        self.assertEqual(result[0]['source_evidence'][0]['quote'],BODY)
        self.assertEqual(report['calls'],1)

    def test_reference_cannot_cross_discovery_scope(self):
        candidate={'assignment_id':'t','topic':'T','description':BODY,'evidence_ids':set(),'discovery_chunk_ids':[]}
        result,report=recover_source_links([candidate],[{'chunk_id':1,'text':BODY}],
            Mock(return_value={'links':[{'source_id':'S0','topic_id':'T0','support':'direct'}]}))
        self.assertFalse(result[0]['evidence_ids'])

    def test_upload_translation_precedes_ledger_and_publication(self):
        tree=ast.parse(Path('main.py').read_text())
        task=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='process_topics_task')
        calls=[n for n in ast.walk(task) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name)]
        line=lambda name:min(n.lineno for n in calls if n.func.id==name)
        self.assertLess(line('translate_final_taxonomy'),max(n.lineno for n in calls if n.func.id=='build_immutable_taxonomy_ledger'))
        self.assertLess(line('translate_final_taxonomy'),line('publish_prepared_topic_links') if any(n.func.id=='publish_prepared_topic_links' for n in calls) else line('assign_topics_to_chunks'))
        # Compare against final ledger only: recovery can precompute repaired candidates.

    def test_translation_cannot_introduce_ambiguous_topic_names(self):
        data=self.data()
        other=copy.deepcopy(data['categories'][0]['topics'][0])
        other.update(assignment_id='other',title='Rapidità')
        data['categories'][0]['topics'].append(other)
        def request(prompt,stage):
            response=self.request(prompt,stage)
            for item in response['items']:
                if item['id'].endswith(':title'):
                    item['text']='Velocity'
            return response
        with self.assertRaisesRegex(ValueError,'ambiguous topic title'):
            translate_final_taxonomy(data,'English',request)

    def test_second_translation_batch_failure_does_not_mutate_any_input(self):
        data=self.data()
        data['categories'][0]['topics'][0]['description']='Testo lungo. '*100
        data['categories'][0]['topics'].append(dict(data['categories'][0]['topics'][0],assignment_id='other'))
        before=copy.deepcopy(data); calls=[]
        def request(prompt,stage):
            calls.append(prompt)
            return self.request(prompt,stage) if len(calls)==1 else {'items':[]}
        with patch('taxonomy_organization.MAX_PROMPT',2100):
            with self.assertRaisesRegex(ValueError,'Incomplete'):
                translate_final_taxonomy(data,'English',request)
        self.assertEqual(data,before)
        self.assertEqual(len(calls),3)

    def test_one_recovery_collects_only_missing_items_from_all_batches(self):
        data=self.data(); before=copy.deepcopy(data); calls=[]
        def request(prompt,stage):
            items=json.loads(prompt.split('\n',1)[1])['items']
            calls.append(items)
            return {'items': items[1:] if len(calls)<=2 else items}
        with patch('taxonomy_organization.TRANSLATION_MAX_ITEMS',2):
            result=translate_final_taxonomy(data,'English',request)
        self.assertEqual(len(calls),3)
        self.assertEqual([i['id'] for i in calls[-1]],
                         [calls[0][0]['id'],calls[1][0]['id']])
        self.assertEqual(result,before)
        self.assertEqual(data,before)

    def test_large_omission_does_not_trigger_unbounded_recovery(self):
        request=Mock(return_value={'items':[]})
        with patch('taxonomy_organization.TRANSLATION_MAX_ITEMS',2):
            with self.assertRaisesRegex(ValueError,'recovery budget exceeded'):
                translate_final_taxonomy(self.data(),'English',request)
        self.assertEqual(request.call_count,2)

    def test_recovery_cannot_return_previously_accepted_id(self):
        calls=[]; before=self.data()
        def request(prompt,stage):
            items=json.loads(prompt.split('\n',1)[1])['items']
            calls.append(items)
            return {'items':items[1:]} if len(calls)==1 else {'items':[calls[0][1]]}
        with self.assertRaisesRegex(ValueError,'Unknown or duplicate'):
            translate_final_taxonomy(before,'English',request)
        self.assertEqual(before,self.data())
        self.assertEqual(len(calls),2)

    def test_large_taxonomy_has_bounded_complete_requests(self):
        data=self.data(); base=data['categories'][0]['topics'][0]
        data['categories'][0]['topics']=[dict(base,assignment_id=str(i),
            description='Testo lungo. '*80) for i in range(100)]
        calls=[]
        def request(prompt,stage):
            items=json.loads(prompt.split('\n',1)[1])['items']; calls.append(items)
            self.assertLessEqual(len(prompt),16000)
            self.assertLessEqual(len(items),60)
            return {'items':items}
        result=translate_final_taxonomy(data,'English',request)
        self.assertEqual(len(result['categories'][0]['topics']),100)
        ids=[i['id'] for batch in calls for i in batch]
        self.assertEqual(len(ids),201)
        self.assertEqual(len(set(ids)),201)
