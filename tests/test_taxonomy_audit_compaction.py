"""Audit-only compaction/contract tests; no application import, database or HTTP."""
import ast
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock


def audit_namespace(call):
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'main.py').read_text())
    names = {'_compact_taxonomy_audit_input', 'audit_taxonomy_reorganization'}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == 2
    ns = {'json': json, 'cost': SimpleNamespace(call=call),
          'client': SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=None)))}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'main.py', 'exec'), ns)
    return ns


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(data)))])


class TaxonomyAuditCompactionTests(unittest.TestCase):
    def setUp(self):
        self.call = Mock(side_effect=self.reply)
        self.ns = audit_namespace(self.call)
        self.mutate = lambda data: None
        self.prompts = []

    def reply(self, fn, **kwargs):
        prompt = kwargs['messages'][-1]['content']
        self.prompts.append(prompt)
        payload = json.loads(prompt.split('IMMUTABLE TOPICS:', 1)[1])
        result = {'proposed_categories': [{'name': 'Proposed', 'topic_ids': [t['audit_id'] for t in payload]}],
                  'confidence_score': .9}
        self.mutate(result)
        return response(result)

    def run_audit(self, data):
        with redirect_stdout(io.StringIO()):
            return self.ns['audit_taxonomy_reorganization'](data, 'synthetic')

    def test_all_287_topics_and_full_originals_preserved(self):
        data = {'categories': [{'name': 'CURRENT', 'topics': [
            {'title': f'Complete topic {i}', 'description': ('Detailed explanation. ' * 100) + f'END-{i}',
             'importance': 7, 'custom_metadata': {'keep': i}} for i in range(287)]}]}
        original = copy.deepcopy(data)
        result = self.run_audit(data)
        payload = json.loads(self.prompts[0].split('IMMUTABLE TOPICS:', 1)[1])
        self.assertEqual(len(payload), 287)
        self.assertEqual([p['audit_id'] for p in payload], [f'T{i+1:04d}' for i in range(287)])
        self.assertEqual([p['topic'] for p in payload], [t['title'] for t in data['categories'][0]['topics']])
        self.assertTrue(all(p['current_category'] == 'CURRENT' for p in payload))
        self.assertLessEqual(len(self.prompts[0]), 48000)
        self.assertEqual(data, original)
        self.assertTrue(result['audit_topics'][-1]['description'].endswith('END-286'))
        self.assertEqual(result['proposed_categories'][0]['name'], 'PROPOSED')
        self.assertEqual(result['proposed_categories'][0]['topic_ids'], [p['audit_id'] for p in payload])
        for field in ('current_statistics','proposed_statistics','category_merge_map',
                      'estimated_category_reduction','confidence_score'):
            self.assertIn(field, result)
        self.assertEqual(self.call.call_count, 1)

    def test_excerpts_only_for_deterministically_ambiguous_titles(self):
        titles = ['Short title', 'Trade costs under fixed exchange regimes',
                  'Trade costs under floating exchange regimes', 'Quantum particles exhibit spin measurement uncertainty']
        topics = [dict(audit_id=f'T{i}', topic=t, current_category='Current', description='Evidence '*100)
                  for i,t in enumerate(titles)]
        compact = self.ns['_compact_taxonomy_audit_input']
        prompt, stats = compact(topics, '')
        payload = json.loads(prompt)
        self.assertEqual([('description_excerpt' in p) for p in payload], [True, True, True, False])
        self.assertEqual(stats['description_excerpt_count'], 3)
        self.assertTrue(all(len(p.get('description_excerpt','')) <= 160 for p in payload))
        self.assertEqual((prompt, stats), compact(topics, ''))

    def test_uniform_backoff_preserves_all_mandatory_fields(self):
        topics = [dict(audit_id=f'T{i}',topic=f'Topic {i}', current_category='C',description='detail '*100)
                  for i in range(20)]
        prefix = 'x'*45000
        prompt, stats = self.ns['_compact_taxonomy_audit_input'](topics, prefix)
        self.assertLess(stats['excerpt_character_limit'], 160)
        self.assertLessEqual(len(prompt),48000)
        payload = json.loads(prompt[len(prefix):])
        self.assertEqual([p['topic'] for p in payload], [t['topic'] for t in topics])
        self.assertEqual([p['audit_id'] for p in payload], [t['audit_id'] for t in topics])

    def test_oversized_mandatory_fields_skip_before_ai(self):
        data = {'categories':[{'name':'Current','topics':[{'title':'t'*48000,'description':'short'}]}]}
        with self.assertRaisesRegex(ValueError, 'exceeds bounded context'):
            self.run_audit(data)
        self.call.assert_not_called()

    def test_structural_validation_still_rejects_invalid_assignments(self):
        data = {'categories':[{'name':'C','topics':[{'title':'Alpha'},{'title':'Beta'}]}]}
        for ids in (['T0001'], ['T0001','T0002','T0002'], ['T0001','UNKNOWN'], ['T0001',2], []):
            with self.subTest(ids=ids):
                self.mutate = lambda r: r['proposed_categories'][0].update(topic_ids=ids)
                with self.assertRaises(ValueError):
                    self.run_audit(data)

    def test_structural_validation_still_rejects_invalid_categories_and_confidence(self):
        data = {'categories':[{'name':'C','topics':[{'title':'Alpha'},{'title':'Beta'}]}]}
        mutations = [lambda r: r.update(proposed_categories=[]),
                     lambda r: r['proposed_categories'][0].update(name=''),
                     lambda r: r.update(proposed_categories=[{'name':'SAME','topic_ids':['T0001']},
                                                            {'name':'same','topic_ids':['T0002']}]),
                     lambda r: r.update(confidence_score='invalid')]
        for mutate in mutations:
            self.mutate = mutate
            with self.assertRaises(ValueError):
                self.run_audit(data)

    def test_independent_mode_hides_categories_and_preserves_originals(self):
        data = {'categories': [{'name': 'PRIVATE_CATEGORY_LABEL', 'topics': [
            {'title': 'Short objective', 'description': 'Supporting educational evidence.'}]}]}
        original = copy.deepcopy(data)
        with redirect_stdout(io.StringIO()):
            result = self.ns['audit_taxonomy_reorganization'](
                data, 'synthetic', independent_categories=True, organization_mode='infer')
        prompt = self.prompts[0]
        payload = json.loads(prompt.split('IMMUTABLE TOPICS:', 1)[1])
        self.assertNotIn('PRIVATE_CATEGORY_LABEL', prompt)
        self.assertNotIn('current_category', prompt)
        self.assertEqual(payload[0]['audit_id'], 'T0001')
        self.assertEqual(payload[0]['topic'], 'Short objective')
        self.assertIn('description_excerpt', payload[0])
        self.assertIn('Create the category structure independently', prompt)
        self.assertEqual(result['audit_topics'][0]['current_category'], 'PRIVATE_CATEGORY_LABEL')
        self.assertEqual(data, original)

    def test_independent_mode_requires_explicit_infer(self):
        for mode in (None, 'manual', 'syllabus'):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, 'explicit infer'):
                self.ns['audit_taxonomy_reorganization'](
                    {}, 'synthetic', independent_categories=True, organization_mode=mode)
        self.call.assert_not_called()

    def test_independent_mode_keeps_validation_and_size_limit(self):
        data = {'categories': [{'name': 'C', 'topics': [{'title': 'Alpha'}, {'title': 'Beta'}]}]}
        self.mutate = lambda r: r['proposed_categories'][0].update(topic_ids=['T0001'])
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            self.ns['audit_taxonomy_reorganization'](
                data, 'synthetic', independent_categories=True, organization_mode='infer')
        self.call.reset_mock()
        data['categories'][0]['topics'][0]['title'] = 'x' * 48000
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, 'exceeds bounded context'):
            self.ns['audit_taxonomy_reorganization'](
                data, 'synthetic', independent_categories=True, organization_mode='infer')
        self.call.assert_not_called()

    def test_response_diagnostics_recorded_before_validation(self):
        reply = response({'proposed_categories': [], 'confidence_score': .9})
        reply.choices[0].finish_reason = 'length'
        reply.usage = SimpleNamespace(prompt_tokens=123, completion_tokens=45)
        self.call.side_effect = None
        self.call.return_value = reply
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaises(ValueError):
            self.ns['audit_taxonomy_reorganization'](
                {'categories': [{'name': 'C', 'topics': [{'title': 'Alpha'}]}]}, 'synthetic')
        line = next(line for line in output.getvalue().splitlines()
                    if line.startswith('TAXONOMY AUDIT RESPONSE:'))
        self.assertEqual(json.loads(line.split(':', 1)[1]),
                         {'finish_reason': 'length', 'input_tokens': 123, 'output_tokens': 45})
