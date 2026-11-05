import copy
import json
import unittest
from unittest.mock import patch
from taxonomy_source_review import recover_source_links

class BatchRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.chunks=[{'chunk_id':i,'text':(f'Source {i} explains its own learning subject. '*75)} for i in range(3)]
        self.candidates=[{'assignment_id':str(i),'topic':f'Topic {i}','description':'A learning objective',
            'evidence_ids':set(),'discovery_chunk_ids':[i],'chapter_id':'a'} for i in range(3)]
        self.calls=[]
    def request(self,prompt,**kw):
        self.calls.append((prompt,kw))
        data=json.loads(prompt.split('\n',1)[1])
        return {'links':[{'source_id':s['id'],'topic_id':s['eligible_topics'][0],'quote':s['text'][:90]}
                         for s in data['sources']]}
    def test_oversized_total_is_batched_without_truncation(self):
        before=copy.deepcopy(self.candidates)
        result,diag=recover_source_links(self.candidates,self.chunks,self.request,max_chars=6000)
        self.assertEqual(diag['calls'],3)
        self.assertEqual(diag['source_outcomes'],{0:'linked',1:'linked',2:'linked'})
        self.assertEqual([t['evidence_ids'] for t in result],[{0},{1},{2}])
        self.assertEqual(before,self.candidates)
        texts=[s['text'] for p,_ in self.calls for s in json.loads(p.split('\n',1)[1])['sources']]
        self.assertEqual(texts,[s['text'] for s in self.chunks])
        self.assertTrue(all(len(p)<=6000 for p,_ in self.calls))
    def test_call_budget_retains_all_deferred_sources(self):
        result,diag=recover_source_links(self.candidates,self.chunks,self.request,max_chars=6000,max_calls=1)
        self.assertEqual(diag['source_outcomes'],{0:'linked',1:'call_budget',2:'call_budget'})
        self.assertEqual(len(result),3)
        self.assertEqual(len(self.calls),1)
    def test_failure_does_not_undo_prior_batch_or_retry(self):
        def request(p):
            if len(self.calls)==1:
                self.calls.append((p,{}));raise RuntimeError('provider error')
            return self.request(p)
        result,diag=recover_source_links(self.candidates,self.chunks,request,max_chars=6000)
        self.assertEqual([t['evidence_ids'] for t in result],[{0},set(),{2}])
        self.assertEqual(diag['source_outcomes'][1],'recovery_failed')
        self.assertEqual(diag['calls'],3)
    def test_remaining_deadline_is_passed_and_late_result_rejected(self):
        now=[0.0]
        def request(p,timeout):
            self.assertEqual(timeout,5.0)
            response=self.request(p,timeout=timeout);now[0]=6
            return response
        with patch('taxonomy_source_review.time.monotonic',side_effect=lambda:now[0]):
            result,diag=recover_source_links(self.candidates,self.chunks,request,max_chars=6000,
                budget_seconds=5,request_accepts_timeout=True)
        self.assertEqual(diag['calls'],1)
        self.assertTrue(all(not t['evidence_ids'] for t in result))
        self.assertEqual(set(diag['source_outcomes'].values()),{'time_budget'})
    def test_oversized_single_source_does_not_block_others(self):
        self.chunks[1]['text']='X'*50000
        _,diag=recover_source_links(self.candidates,self.chunks,self.request,max_chars=6000)
        self.assertEqual(diag['source_outcomes'],{0:'linked',1:'context_limit',2:'linked'})
    def test_chapters_are_not_mixed(self):
        owners={0:'a',1:'b',2:'a'}
        self.candidates[1]['chapter_id']='b'
        _,diag=recover_source_links(self.candidates,self.chunks,self.request,owners)
        self.assertEqual(diag['calls'],3)

    def test_quote_contract_and_diagnostics_are_specific_and_private(self):
        cases = [
            (12, 'quote_not_string'),
            ('Source 0', 'quote_too_short'),
            (' ' + self.chunks[0]['text'][:90], 'quote_surrounding_whitespace'),
            ('An invented statement that is not present in the actual source.', 'quote_not_in_source'),
        ]
        self.chunks[0]['page'] = 7
        for quote, code in cases:
            def request(prompt):
                self.assertIn('40 characters', prompt)
                self.assertIn('Do not translate', prompt)
                return {'links': [{'source_id':'S0', 'topic_id':'T0', 'quote':quote}]}
            before = copy.deepcopy(self.candidates)
            result, diag = recover_source_links(self.candidates, self.chunks, request,
                                                max_chars=6000, max_calls=1)
            batch = diag['batch_diagnostics'][0]
            failure = batch['rejections'][0]
            self.assertEqual(failure['error_code'], code)
            self.assertEqual(failure['failure_stage'], 'link_validation')
            self.assertEqual(failure['chunk_id'], 0)
            self.assertEqual(batch['source_refs'], [{'chunk_id':0, 'page':7}])
            self.assertEqual(result, before)
            self.assertNotIn(self.chunks[0]['text'], json.dumps(diag))

    def test_request_failure_does_not_leak_exception_content(self):
        def request(prompt):
            raise RuntimeError('SECRET credential and private document text')
        _, diag = recover_source_links(self.candidates, self.chunks, request, max_calls=1)
        failure = diag['batch_diagnostics'][0]
        self.assertEqual(failure['failure_stage'], 'request')
        self.assertEqual(failure['error_type'], 'RuntimeError')
        self.assertNotIn('SECRET', json.dumps(diag))

    def test_response_validation_and_empty_valid_result_are_distinct(self):
        for response, code in [(None, 'invalid_recovery_response'),
                               ({'links':[{}]}, 'invalid_recovery_link')]:
            _, diag = recover_source_links(self.candidates, self.chunks,
                                            lambda p: response, max_calls=1)
            self.assertEqual(diag['batch_diagnostics'][0].get('error_code') or diag['batch_diagnostics'][0]['rejections'][0]['error_code'], code)
        _, diag = recover_source_links(self.candidates, self.chunks,
                                        lambda p: {'links':[]}, max_calls=1)
        self.assertNotIn('error_code', diag['batch_diagnostics'][0])
        self.assertEqual(set(diag['source_outcomes'].values()), {'no_validated_topic_link'})

    def test_repair_quote_rejection_uses_same_contract(self):
        response = {'links':[], 'repairs':[{'template_id':'T0', 'topic_id':'T0',
                    'title':'Topic 0', 'description':'Corrected objective',
                    'citations':[{'source_id':'S0', 'quote':'Source 0'}]}]}
        result, diag = recover_source_links(self.candidates, self.chunks,
                                            lambda p: response, max_calls=1)
        self.assertEqual(diag['batch_diagnostics'][0]['rejections'][0]['error_code'], 'quote_too_short')
        self.assertEqual(diag['batch_diagnostics'][0]['rejections'][0]['failure_stage'], 'repair_validation')
        self.assertEqual(result, self.candidates)

    def test_diagnostics_survive_existing_report_serialization(self):
        from unittest.mock import Mock
        from taxonomy_source_review import persist_source_review, decode_review
        _, diag = recover_source_links(self.candidates, self.chunks,
                                        lambda p: {'links':[{}]}, max_calls=1)
        report = {'version':1, 'entries':[], 'recovery':diag}
        db = Mock()
        db.get_bind.return_value.dialect.name = 'postgresql'
        persist_source_review(db, 'project', 'document', report)
        parameters = db.execute.call_args.args[1]
        restored = decode_review(parameters['review'])
        self.assertEqual(restored['recovery']['batch_diagnostics'], diag['batch_diagnostics'])
        db.commit.assert_not_called()
