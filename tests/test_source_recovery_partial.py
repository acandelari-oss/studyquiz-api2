"""Regression coverage for per-item rejection; all AI requests mocked."""
import copy
import json
import unittest
from unittest.mock import Mock
from taxonomy_source_review import recover_source_links, _verify_recovery_quote, RecoveryValidationError

TEXT = "The position of a body changes over time relative to its reference frame."


class PartialRecoveryTests(unittest.TestCase):
    def setUp(self):
        self.topics = [
            {'assignment_id':str(i), 'topic':f'Topic {i}', 'description':'Original objective',
             'evidence_ids':set(), 'discovery_chunk_ids':[i], 'chapter_id':'a',
             'document_id':'doc', 'embedding':[1, 0]} for i in range(3)]
        self.chunks = [{'chunk_id':i, 'page':i+1, 'text':TEXT} for i in range(3)]

    def link(self, i):
        return {'source_id':f'S{i}', 'topic_id':f'T{i}', 'quote':TEXT}

    def repair(self, i):
        return {'template_id':f'T{i}', 'topic_id':f'T{i}', 'title':f'Topic {i}',
                'description':'Corrected supported objective',
                'citations':[{'source_id':f'S{i}', 'quote':TEXT}]}

    def run_response(self, response):
        before = copy.deepcopy(self.topics)
        request = Mock(return_value=response)
        result, report = recover_source_links(self.topics, self.chunks, request)
        self.assertEqual(self.topics, before)
        self.assertEqual(request.call_count, 1)
        return result, report

    def test_first_bad_link_does_not_abort_later_rows(self):
        invalid = dict(self.link(0), quote='Invented statement that is definitely not in the stored source.')
        result, report = self.run_response({'links':[invalid, self.link(1), self.link(2)]})
        self.assertEqual([t['evidence_ids'] for t in result], [set(), {1}, {2}])
        self.assertEqual(report['added_links'], 2)
        self.assertEqual(report['source_outcomes'], {0:'recovery_failed', 1:'linked', 2:'linked'})
        self.assertEqual(report['rejected_items'], 1)

    def test_bad_repair_preserves_links_and_independent_repair(self):
        invalid = self.repair(0)
        invalid['citations'][0]['quote'] = 'Invented statement that is definitely not in the stored source.'
        result, report = self.run_response({'links':[self.link(1)],
                                           'repairs':[invalid, self.repair(2)]})
        self.assertEqual(result[0], self.topics[0])
        self.assertEqual(result[1]['evidence_ids'], {1})
        self.assertEqual(result[2]['description'], 'Corrected supported objective')
        self.assertEqual(result[2]['assignment_id'], '2')
        self.assertEqual(result[2]['document_id'], 'doc')
        self.assertEqual(report['corrected_topics'], 1)
        self.assertEqual(report['added_links'], 2)

    def test_one_bad_citation_rejects_entire_repair_not_its_neighbors(self):
        self.topics[0]['discovery_chunk_ids'] = [0, 1]
        invalid = self.repair(0)
        invalid['citations'].append({'source_id':'S1', 'quote':'Incorrect quotation of more than forty characters.'})
        result, report = self.run_response({'links':[], 'repairs':[invalid, self.repair(2)]})
        self.assertEqual(result[0], self.topics[0])
        self.assertEqual(result[2]['evidence_ids'], {2})
        rejection = report['batch_diagnostics'][0]['rejections'][0]
        self.assertEqual(rejection['citation_index'], 1)
        self.assertEqual(rejection['affected_chunk_ids'], [0, 1])

    def test_duplicate_corrections_all_rejected_without_first_wins(self):
        result, report = self.run_response({'links':[self.link(1)],
                                           'repairs':[self.repair(0), self.repair(0), self.repair(2)]})
        self.assertEqual(result[0], self.topics[0])
        self.assertEqual(report['rejected_items'], 2)
        self.assertEqual(report['corrected_topics'], 1)
        self.assertEqual(result[1]['evidence_ids'], {1})

    def test_unknown_ids_and_scope_failures_do_not_block_valid_link(self):
        bad = [dict(self.link(0), source_id='S99'),
               dict(self.link(0), topic_id='T99'),
               dict(self.link(0), topic_id='T1')]
        result, report = self.run_response({'links':bad+[self.link(2)]})
        self.assertEqual([t['evidence_ids'] for t in result], [set(), set(), {2}])
        self.assertEqual(report['rejected_items'], 3)

    def test_malformed_envelope_rejects_whole_response(self):
        result, report = self.run_response({'links':[self.link(1)], 'repairs':'invalid'})
        self.assertEqual(result, self.topics)
        self.assertEqual(report['status'], 'unresolved')

    def test_formatting_diagnostics_never_authorize_nonexact_quotes(self):
        cases = [
            (TEXT.replace(' ', '\n', 1), TEXT, True, False),
            ('é '+TEXT, 'e\u0301 '+TEXT, False, True),
            (TEXT.replace('changes', 'does not change'), TEXT, False, False),
        ]
        for quote, source, whitespace, unicode in cases:
            with self.assertRaises(RecoveryValidationError) as caught:
                _verify_recovery_quote(quote, source)
            diag = caught.exception.diagnostics
            self.assertEqual(diag['matches_after_whitespace_normalization'], whitespace)
            self.assertEqual(diag['matches_after_unicode_nfc'], unicode)
            self.assertNotIn(TEXT, json.dumps(diag))
            self.assertEqual(diag['quote_characters'], len(quote))
