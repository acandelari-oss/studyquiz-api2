"""Actual SDK payload and numerical safety checks; all HTTP is mocked."""
import ast
import io
import json
import math
import unittest
import warnings
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import numpy as np
from openai import OpenAI, BadRequestError, InternalServerError, RateLimitError

from taxonomy_coverage import consolidate_candidates, nominate_relationships, request_json
from taxonomy_assignment import validate_evidence
from tests.test_global_duplicate_consolidation import candidate, response


def api_error(status, message='Request rejected', code='invalid_value'):
    cls = BadRequestError if status == 400 else RateLimitError if status == 429 else InternalServerError
    return cls(message, response=httpx.Response(status, request=httpx.Request('POST', 'https://mock.invalid')),
               body={'message': message, 'type': 'invalid_request_error', 'code': code})


class ConsolidationRequestTests(unittest.TestCase):
    def run_review(self, create):
        cs = [candidate('Equivalent objective', i=i) for i in range(2)]
        diagnostics, output = {}, io.StringIO()
        with redirect_stdout(output):
            result = consolidate_candidates(cs, create, 'English',
                source_text={1: 'Validated source one', 2: 'Validated source two'}, diagnostics=diagnostics)
        validate_evidence(result, cs, {1, 2})
        return result, diagnostics, output.getvalue()

    def test_actual_production_sdk_request_and_successful_merge(self):
        # Compile only the production request factory: no application import or live clients.
        tree = ast.parse(Path('main.py').read_text())
        factory = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'create_consolidation')
        calls = []
        def transport(request):
            payload = json.loads(request.content)
            calls.append(payload)
            self.assertEqual(payload['model'], 'gpt-4o-mini')
            self.assertEqual(payload['response_format'], {'type': 'json_object'})
            prompt = payload['messages'][0]['content']
            self.assertIn('Return valid JSON:', prompt)
            data = json.loads(prompt.rsplit('REVIEW INPUT:\n', 1)[1])
            inputs = {c['input_id']: c for c in data['candidates']}
            decisions = [{'input_ids': pair, 'decision': 'duplicate', 'reason': 'Same complete objective',
                'category': inputs[pair[0]]['category'],
                'evidence_checks': {i: [s['chunk_id'] for s in inputs[i]['validated_evidence_samples']] for i in pair}}
                for pair in data['relationships']]
            return httpx.Response(200, json={'id': 'mock', 'object': 'chat.completion', 'created': 0,
                'model': 'gpt-4o-mini', 'choices': [{'index': 0, 'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': json.dumps({'decisions': decisions})}}]})
        with OpenAI(api_key='mock-only', http_client=httpx.Client(transport=httpx.MockTransport(transport))) as client:
            def call(fn, *, cost_feature, cost_suboperation, **kwargs):
                self.assertEqual((cost_feature, cost_suboperation), ('taxonomy', 'consolidation'))
                self.assertEqual(kwargs['timeout'], 240)
                self.assertEqual(fn.__self__._client.max_retries, 0)
                return fn(**kwargs)
            namespace = {'client': client, 'cost': SimpleNamespace(call=call)}
            exec(compile(ast.Module(body=[factory], type_ignores=[]), 'main.py', 'exec'), namespace)
            result, diag, _ = self.run_review(namespace['create_consolidation'])
        self.assertEqual(len(calls), 1)
        self.assertEqual(diag['summary']['confirmed_relationships'], 1)
        topics = result['categories'][0]['topics']
        self.assertEqual(len(topics), 1)
        self.assertEqual(topics[0]['evidence_ids'], [1, 2])
        self.assertEqual(topics[0]['candidate_ids'], ['0', '1'])

    def test_permanent_error_once_and_sanitized_diagnostics(self):
        create = Mock(side_effect=api_error(400,
            "messages must contain JSON; sk-private-secret; Validated source one", 'invalid_value'))
        result, diag, logs = self.run_review(create)
        self.assertEqual(create.call_count, 1)
        self.assertEqual(len(result['categories'][0]['topics']), 2)
        self.assertEqual(diag['summary']['outcomes'], {'request_failure': 1})
        self.assertIn('"http_status": 400', logs)
        self.assertIn('"api_type": "invalid_request_error"', logs)
        self.assertIn('"api_code": "invalid_value"', logs)
        self.assertIn('JSON mode requires messages to explicitly request JSON', logs)
        self.assertNotIn('sk-private-secret', logs)
        self.assertNotIn('Validated source one', logs)

    def test_shared_request_helper_does_not_retry_400(self):
        create = Mock(side_effect=api_error(400))
        with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
            request_json(create, 'JSON')
        self.assertEqual(create.call_count, 1)

    def test_transient_errors_retry_bounded_and_preserve_evidence(self):
        for status in (429, 500):
            with self.subTest(status=status):
                create = Mock(side_effect=api_error(status))
                result, diag, _ = self.run_review(create)
                self.assertEqual(create.call_count, 3)
                self.assertEqual(diag['summary']['outcomes'], {'request_failure': 1})
                self.assertEqual([t['evidence_ids'] for t in result['categories'][0]['topics']], [[1], [2]])

    def test_transient_then_valid_response_recovers(self):
        decision = {'input_ids': ['0', '1'], 'decision': 'duplicate', 'reason': 'Same objective',
                    'category': 'AUTO', 'evidence_checks': {'0': [1], '1': [2]}}
        create = Mock(side_effect=[api_error(500), response({'decisions': [decision]})])
        result, diag, _ = self.run_review(create)
        self.assertEqual(create.call_count, 2)
        self.assertEqual(diag['summary']['confirmed_relationships'], 1)
        self.assertEqual(result['categories'][0]['topics'][0]['evidence_ids'], [1, 2])


class ScreeningNumericsTests(unittest.TestCase):
    def screen(self, vectors, titles=None):
        cs = [candidate(titles[i] if titles else f'Objective {i}', vector=v, i=i) for i, v in enumerate(vectors)]
        with warnings.catch_warnings(), np.errstate(all='raise'):
            warnings.simplefilter('error', RuntimeWarning)
            return nominate_relationships(cs)

    def test_invalid_embeddings_excluded_but_title_review_remains(self):
        bad = [(0., 0.), (float('nan'), 1), (float('inf'), 1), None, [], [[1, 2]], ['bad', 1], [1, 2, 3]]
        pairs, diag = self.screen(bad + [(1., 0.)] * 9)
        self.assertTrue(all(min(p['pair']) >= len(bad) for p in pairs))
        self.assertEqual(diag['invalid_embeddings'], {'zero_norm': 1, 'non_finite': 2, 'malformed': 4, 'dimension_mismatch': 1})
        for title in ('Same title', 'SAME—TITLE'):
            pairs, _ = self.screen([None, [float('nan'), 1]], ['Same title', title])
            self.assertEqual([p['pair'] for p in pairs], [(0, 1)])
            self.assertEqual(pairs[0]['reason'], 'exact_title' if title == 'Same title' else 'normalized_title')

    def test_extreme_finite_vectors_are_stably_normalized(self):
        pairs, diag = self.screen([(1e308, 1e308), (1e-320, 1e-320), (1e308, -1e308)])
        self.assertEqual([p['pair'] for p in pairs], [(0, 1)])
        self.assertEqual(diag['invalid_embeddings'], {})

    def test_high_dimensional_finite_vectors_no_blas_warnings_and_equivalent_nomination(self):
        vectors = np.random.default_rng(19).normal(size=(449, 1536))
        vectors[1] = vectors[0]
        pairs, _ = self.screen(vectors)
        self.assertEqual([p['pair'] for p in pairs], [(0, 1)])
        # Deterministic scalar reference, safely away from the unchanged .82 threshold.
        small = np.random.default_rng(44).normal(size=(20, 8))
        small[1] = small[0]
        normalized = [[float(x) / math.sqrt(math.fsum(float(v)*float(v) for v in row)) for x in row] for row in small]
        expected = {(i,j) for i in range(20) for j in range(i+1,20)
                    if math.fsum(a*b for a,b in zip(normalized[i], normalized[j])) >= .82}
        pairs, _ = self.screen(small)
        self.assertEqual({p['pair'] for p in pairs}, expected)
