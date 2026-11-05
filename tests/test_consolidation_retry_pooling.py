"""Offline request scheduling regressions; no services or database required."""
import copy
import io
import json
import unittest
from collections import Counter
from contextlib import redirect_stdout
from unittest.mock import patch

from taxonomy_coverage import consolidate_candidates
from tests.test_global_duplicate_consolidation import candidate, response
from taxonomy_assignment import validate_evidence


class RetryPoolingTests(unittest.TestCase):
    def run_case(self, reject=False, exhaust=False):
        candidates = [candidate('Objective ' + str(i // 2), i=i) for i in range(72)]
        original = copy.deepcopy(candidates)
        ledger = [{'pair': (i, i + 1), 'reason': 'exact_title'} for i in range(0, 72, 2)]
        attempts, prompts, first_payloads = Counter(), [], []

        def create(prompt):
            payload = json.loads(prompt.rsplit('REVIEW INPUT:\n', 1)[1])
            prompts.append(prompt)
            self.assertLessEqual(len(prompt), 48000)
            self.assertLessEqual(len(payload['candidates']), 24)
            self.assertLessEqual(len(payload['relationships']), 48)
            first = all(attempts[tuple(pair)] == 0 for pair in payload['relationships'])
            if first:
                first_payloads.append(payload)
            records = []
            for index, pair in enumerate(payload['relationships']):
                attempts[tuple(pair)] += 1
                if exhaust or (reject and first and index == 0):
                    continue
                records.append({'input_ids': pair, 'decision': 'duplicate',
                    'reason': 'Equivalent supported objective', 'category': 'AUTO',
                    'evidence_checks': {i: [int(i) + 1] for i in pair}})
            return response({'decisions': records})

        diagnostics = {}
        with patch('taxonomy_coverage.nominate_relationships', return_value=(ledger, {})), redirect_stdout(io.StringIO()):
            result = consolidate_candidates(candidates, create, 'English',
                source_text={i + 1: c['description'] for i, c in enumerate(candidates)},
                diagnostics=diagnostics)
        self.assertEqual(candidates, original)
        validate_evidence(result, candidates, set(range(1, 73)))
        self.assertEqual(set(attempts), {(str(i), str(i + 1)) for i in range(0, 72, 2)})
        self.assertLessEqual(max(attempts.values()), 3)
        return result, diagnostics, prompts, attempts, first_payloads

    def test_sparse_retries_pooled_without_repeating_valid_decisions(self):
        clean, _, _, _, _ = self.run_case()
        result, diag, prompts, attempts, initial = self.run_case(reject=True)
        self.assertEqual(result, clean)
        self.assertEqual(len(initial), 3)
        self.assertEqual(len(prompts), 4)  # Previously 3 first-pass + 3 isolated retries.
        self.assertEqual(Counter(attempts.values()), {1: 33, 2: 3})
        self.assertEqual(diag['summary']['retry_groups'], 1)
        self.assertEqual(diag['summary']['first_pass_calls'], 3)
        self.assertEqual(diag['summary']['retry_calls'], 1)
        self.assertAlmostEqual(diag['summary']['request_seconds'],
            diag['summary']['first_pass_seconds'] + diag['summary']['retry_seconds'])
        self.assertEqual(diag['summary']['review_groups'], 3)
        pooled = json.loads(prompts[-1].rsplit('REVIEW INPUT:\n', 1)[1])
        self.assertEqual(len(pooled['relationships']), 3)
        # Same input context, serialized as the former three separate retries.
        prefix = prompts[-1].split('REVIEW INPUT:\n', 1)[0] + 'REVIEW INPUT:\n'
        isolated_size = 0
        for pair in pooled['relationships']:
            isolated_size += len(prefix + json.dumps({'candidates': [c for c in pooled['candidates']
                if c['input_id'] in pair], 'relationships': [pair]}, ensure_ascii=False))
        self.assertLess(len(prompts[-1]), isolated_size)
        print('OFFLINE RETRY BENCHMARK', json.dumps({'pairs': 36, 'old_calls': 6,
            'new_calls': len(prompts), 'old_retry_characters': isolated_size,
            'new_retry_characters': len(prompts[-1])}))

    def test_exhaustion_preserves_all_candidates_and_sources(self):
        result, diag, prompts, attempts, _ = self.run_case(exhaust=True)
        self.assertEqual(len(prompts), 9)
        self.assertEqual(set(attempts.values()), {3})
        self.assertEqual(diag['summary']['candidates_out'], 72)
        self.assertTrue(all(e['status'] == 'unresolved' for e in diag['relationships']))
        self.assertEqual(sum(len(c['topics']) for c in result['categories']), 72)
