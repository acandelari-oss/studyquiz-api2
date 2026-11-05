import copy
import unittest
from unittest.mock import Mock
from taxonomy_source_review import recover_source_links
from taxonomy_organization import coalesce_identical_topics
from taxonomy_assignment import validate_evidence

BODY = 'Sanger sequencing uses chain termination to determine the DNA sequence.'
class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.c = [{'assignment_id':'t','topic':'Sanger sequencing','description':BODY,
                   'evidence_ids':set(), 'discovery_chunk_ids':[1], 'chapter_id':'a'}]
        self.chunks = [{'chunk_id':1, 'text':BODY}]
        self.response = {'links':[{'source_id':'S0','topic_id':'T0','quote':BODY}]}

    def test_exact_support_recovers_without_mutating_original(self):
        before=copy.deepcopy(self.c); call=Mock(return_value=self.response)
        result,report=recover_source_links(self.c,self.chunks,call,{1:'a'})
        self.assertEqual(result[0]['evidence_ids'],{1})
        self.assertEqual(self.c,before)
        self.assertEqual(report['added_links'],1)
        self.assertEqual(call.call_count,1)

    def test_invalid_link_does_not_discard_good_link_or_retry(self):
        for field,value in [('quote','invented explanation'),('source_id','S9'),('topic_id','T9')]:
            response=copy.deepcopy(self.response)
            response['links'].append({**response['links'][0],field:value})
            call=Mock(return_value=response)
            result,report=recover_source_links(self.c,self.chunks,call,{1:'a'})
            self.assertEqual(result[0]['evidence_ids'], {1})
            self.assertFalse(self.c[0]['evidence_ids'])
            self.assertEqual(report['rejected_items'], 1)
            self.assertEqual(report['status'], 'partial')
            self.assertEqual(call.call_count,1)

    def test_cross_chapter_or_batch_cannot_link(self):
        for scope in ('chapter','batch'):
            c=copy.deepcopy(self.c)
            if scope=='chapter': c[0]['chapter_id']='b'
            else: c[0]['discovery_chunk_ids']=[]
            result,report=recover_source_links(c,self.chunks,Mock(return_value=self.response),{1:'a'})
            self.assertEqual(report['status'],'unresolved')
            self.assertFalse(result[0]['evidence_ids'])

    def test_budget_and_complete_coverage_skip_request(self):
        call=Mock()
        _,report=recover_source_links(self.c,self.chunks,call,max_chars=10)
        self.assertEqual(report['status'],'context_limit');call.assert_not_called()
        self.c[0]['evidence_ids']={1}
        _,report=recover_source_links(self.c,self.chunks,call)
        self.assertEqual(report['status'],'not_needed');call.assert_not_called()

    def test_empty_response_preserves_unreadable_material(self):
        result,report=recover_source_links(self.c,self.chunks,Mock(return_value={'links':[]}))
        self.assertFalse(result[0]['evidence_ids'])
        self.assertEqual(self.chunks[0]['text'],BODY)
        self.assertEqual(report['added_links'],0)

    def test_duplicate_coalescing_preserves_lineage_and_chapters(self):
        candidates=[dict(self.c[0],assignment_id=str(i),evidence_ids={1}) for i in range(3)]
        topics=[dict(c,title=c['topic'],candidate_ids=[str(i)],original_objectives=[BODY]) for i,c in enumerate(candidates)]
        topics[2]['chapter_id']='b'
        data={'categories':[{'name':'Category','topics':topics}]}
        before=copy.deepcopy(data)
        result=coalesce_identical_topics(data,candidates)
        self.assertEqual(len(result['categories'][0]['topics']),2)
        self.assertEqual(data,before)
        validate_evidence(result,candidates,{1})
        topics[1]['description']='A distinct learning objective'
        self.assertEqual(len(coalesce_identical_topics(data,candidates)['categories'][0]['topics']),3)

class TargetedRepairTests(RecoveryTests):
    def repair(self, topic_id=None):
        return {'template_id':'T0','topic_id':topic_id,'title':'Sanger sequencing' if topic_id else 'Chain termination',
                'description':BODY,'citations':[{'source_id':'S0','quote':BODY}]}

    def test_new_topic_preserves_existing_candidate_and_source(self):
        self.c[0].update(category='Methods',importance=5,embedding=[1,0],document_id='d')
        before=copy.deepcopy(self.c)
        call=Mock(return_value={'links':[],'repairs':[self.repair()]})
        result,report=recover_source_links(self.c,self.chunks,call,{1:'a'},language_instruction='Write in Italian.')
        self.assertEqual(self.c,before)
        self.assertEqual(result[0],before[0])
        self.assertEqual(len(result),2)
        new=result[1]
        self.assertEqual(new['evidence_ids'],{1})
        self.assertEqual(new['chapter_id'],'a')
        self.assertEqual(new['document_id'],'d')
        self.assertEqual(new['category'],'Methods')
        self.assertNotEqual(new['assignment_id'],'t')
        self.assertNotIn('embedding',new)
        self.assertEqual(report['new_topics'],1)
        self.assertIn('Italian',call.call_args.args[0])
        self.assertEqual(call.call_count,1)

    def test_description_correction_keeps_id_title_and_prior_evidence(self):
        self.c[0].update(description='Wrong comparison',embedding=[1,0])
        result,report=recover_source_links(self.c,self.chunks,
            Mock(return_value={'links':[],'repairs':[self.repair('T0')]}),{1:'a'})
        self.assertEqual(result[0]['assignment_id'],'t')
        self.assertEqual(result[0]['topic'],'Sanger sequencing')
        self.assertEqual(result[0]['description'],BODY)
        self.assertEqual(result[0]['evidence_ids'],{1})
        self.assertNotIn('embedding',result[0])
        self.assertEqual(self.c[0]['description'],'Wrong comparison')
        self.assertEqual(report['corrected_topics'],1)

    def test_bad_repair_preserves_good_links(self):
        for key,value in [('template_id','T9'),('topic_id','T9'),('description',''),('citations',[])]:
            repair=self.repair();repair[key]=value
            result,report=recover_source_links(self.c,self.chunks,
                Mock(return_value={**self.response,'repairs':[repair]}),{1:'a'})
            self.assertEqual(result[0]['evidence_ids'], {1})
            self.assertEqual(report['rejected_items'], 1)

    def test_duplicate_correction_and_invented_quote_rejected(self):
        for repairs in ([self.repair('T0'),self.repair('T0')],
                        [{**self.repair(),'citations':[{'source_id':'S0','quote':'Invented quote '*8}]}]):
            result,report=recover_source_links(self.c,self.chunks,
                Mock(return_value={'links':[],'repairs':repairs}),{1:'a'})
            self.assertEqual(result,self.c)
            self.assertEqual(report['status'],'unresolved')
