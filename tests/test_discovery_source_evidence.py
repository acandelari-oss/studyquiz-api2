import copy
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from taxonomy_coverage import validate_discovery_evidence
from taxonomy_assignment import plan_assignments

BODY = 'The stages are prophase, prometaphase, metaphase, anaphase and telophase.'

class DiscoveryEvidenceTests(unittest.TestCase):
    def data(self, citation):
        return {'categories': [{'name': 'Chapter', 'topics': [{'title': 'Fasi della mitosi',
                'description': 'Stages of division', 'source_evidence': citation}]}]}

    def test_exact_quote_in_source_language_preserved(self):
        data = self.data([{'chunk_id': '1', 'quote': BODY}])
        before = copy.deepcopy(data)
        validate_discovery_evidence(data, [{'chunk_id': 1, 'text': BODY}])
        self.assertEqual(data, before)

    def test_invalid_citations_are_quarantined_without_losing_topics(self):
        for citations in ([], None, [None], [{'chunk_id': 2, 'quote': BODY}],
                          [{'chunk_id': 1, 'quote': 'Invented explanation'}]):
            with self.subTest(citations=citations):
                data = self.data(citations)
                validate_discovery_evidence(data, [{'chunk_id': 1, 'text': BODY}])
                topic = data['categories'][0]['topics'][0]
                self.assertEqual(topic['title'], 'Fasi della mitosi')
                self.assertEqual(topic['description'], 'Stages of division')
                self.assertEqual(topic['source_evidence'], [])
                self.assertTrue(topic['source_evidence_rejections'])
                self.assertEqual(self.plan(citations=topic['source_evidence'])[0], {'t': set()})

    def plan(self, quote=BODY, chapter='a', discovery=(1,), citations=None):
        c = dict(assignment_id='t', topic='Fasi della mitosi', source_section=None,
                 embedding=[1.0], chapter_id=chapter, discovery_chunk_ids=list(discovery),
                 source_evidence=citations if citations is not None else [{'chunk_id': 1, 'quote': quote}])
        db = SimpleNamespace(execute=lambda *a, **kw: SimpleNamespace(fetchall=lambda:
            ([(1, BODY, '')] if 'cross join' not in str(a[0]) else [(1, 't', .2)])))
        with patch('taxonomy_assignment.source_snapshot', return_value={1:'fingerprint', 2:'other'} if chapter=='b' else {1:'fingerprint'}):
            return plan_assignments(db,[c],'p',lambda *a:a[-1],
                                    chapter_ownership={1:'a',2:'b'} if chapter=='b' else {1:'a'})

    def test_multiple_and_repeated_quotes_keep_one_link(self):
        citations = [{'chunk_id': 1, 'quote': 'prophase, prometaphase'},
                     {'chunk_id': '1', 'quote': 'metaphase, anaphase and telophase.'},
                     {'chunk_id': 1, 'quote': 'prophase, prometaphase'}]
        data = self.data(citations)
        before = copy.deepcopy(data)
        validate_discovery_evidence(data, [{'chunk_id': 1, 'text': BODY}])
        self.assertEqual(data, before)
        self.assertEqual(self.plan(citations=citations)[0], {'t': {1}})
        citations.append({'chunk_id': 1, 'quote': 'Fabricated extra quote'})
        with self.assertRaises(ValueError):
            self.plan(citations=citations)

    def test_validation_diagnostics_are_specific_and_do_not_echo_content(self):
        import io
        from contextlib import redirect_stdout
        for identity, quote, code in [(True, BODY, 'invalid_source_id_type'),
                (1.0, BODY, 'invalid_source_id_type'), (None, BODY, 'invalid_source_id_type'),
                ('unknown-sensitive', BODY, 'unknown_source_id'),
                (1, 'secret fabricated text', 'quote_not_in_source'),
                (1, ' ' + BODY, 'quote_not_in_source'), (1, '', 'invalid_quote')]:
            with self.subTest(code=code):
                data = self.data([{'chunk_id':identity,'quote':quote}])
                output = io.StringIO()
                with redirect_stdout(output):
                    validate_discovery_evidence(data, [{'chunk_id':1,'text':BODY}])
                diagnostic = data['categories'][0]['topics'][0]['source_evidence_rejections'][0]
                self.assertEqual(diagnostic['validation_code'], code)
                self.assertEqual(diagnostic['citation_index'], 0)
                self.assertNotIn('sensitive', output.getvalue())
                self.assertNotIn('secret', output.getvalue())
                self.assertNotIn(BODY, output.getvalue())

    def test_bad_quote_does_not_retry_and_valid_citation_still_links(self):
        import json
        from unittest.mock import Mock
        from taxonomy_coverage import request_json
        valid = {'chunk_id': 1, 'quote': BODY}
        for citations in ([{'chunk_id':1,'quote':'Paraphrased stages'}],
                          [valid, {'chunk_id':1,'quote':'Paraphrased stages'}]):
            with self.subTest(citations=citations):
                data = self.data(citations)
                create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
                    message=SimpleNamespace(content=json.dumps(data)))]))
                result = request_json(create, 'source request', validate=lambda value:
                    validate_discovery_evidence(value, [{'chunk_id':1,'text':BODY}]))
                self.assertEqual(create.call_count, 1)
                topic = result['categories'][0]['topics'][0]
                expected = [valid] if len(citations) == 2 else []
                self.assertEqual(topic['source_evidence'], expected)
                self.assertEqual(self.plan(citations=expected)[0], {'t': {1} if expected else set()})

    def test_malformed_topic_still_rejected(self):
        data = self.data([{'chunk_id':1,'quote':'bad quote'}])
        data['categories'][0]['topics'][0]['title'] = None
        with self.assertRaises(ValueError):
            validate_discovery_evidence(data, [{'chunk_id':1,'text':BODY}])

    def test_valid_multi_excerpt_response_needs_only_one_request(self):
        import json
        from unittest.mock import Mock
        from taxonomy_coverage import request_json
        data = self.data([{'chunk_id':1,'quote':'prophase, prometaphase'},
                          {'chunk_id':1,'quote':'anaphase and telophase.'}])
        create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=json.dumps(data)))]))
        result = request_json(create, 'JSON source request', validate=lambda value:
            validate_discovery_evidence(value, [{'chunk_id':1,'text':BODY}]))
        self.assertEqual(result, data)
        self.assertEqual(create.call_count, 1)

    def test_verified_citation_survives_low_similarity(self):
        links,_ = self.plan()
        self.assertEqual(links, {'t': {1}})

    def test_changed_quote_cross_chapter_and_cross_batch_fail(self):
        for kwargs in ({'quote':'not in source'}, {'chapter':'b'}, {'discovery':(2,)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.plan(**kwargs)

class ExactOverlapTests(unittest.TestCase):
    def test_only_substantial_unambiguous_same_batch_overlap_inherits(self):
        from taxonomy_assignment import inherit_exact_source_overlaps
        fragment = 'prophase, prometaphase, metaphase, anaphase and telophase.'
        sources = {1: BODY, 2: fragment, 3: 'and telophase.', 4: fragment, 5: fragment}
        candidates = [dict(assignment_id='t', chapter_id='a', discovery_chunk_ids=[1,2,3,4],
                           source_evidence=[{'chunk_id':1, 'quote':BODY}])]
        links = {'t': {1}}
        inherit_exact_source_overlaps(links,candidates,sources,{1:'a',2:'a',3:'a',4:'b',5:'a'})
        self.assertEqual(links, {'t': {1,2}})
        candidates.append(dict(candidates[0], assignment_id='other'))
        links = {'t': {1}, 'other': {1}}
        inherit_exact_source_overlaps(links,candidates,sources,{1:'a',2:'a',3:'a',4:'b',5:'a'})
        self.assertEqual(links, {'t': {1}, 'other': {1}})
