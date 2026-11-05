"""Offline tests of upload preparation contracts; no application startup."""
import ast
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from taxonomy_assignment import plan_assignments


class EmbeddingBatchTests(unittest.TestCase):
    def run_ledger(self, data, corrupt=None):
        tree = ast.parse(Path('main.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'build_immutable_taxonomy_ledger')
        calls = []
        def create(**kwargs):
            bodies = kwargs['input']
            self.assertEqual(kwargs['model'], 'text-embedding-3-small')
            self.assertIsInstance(bodies, list)
            self.assertLessEqual(len(bodies), 32)
            calls.append(bodies)
            items = [NS(index=i, embedding=[float(len(body)), float(i)]) for i, body in enumerate(bodies)]
            if corrupt:
                items = corrupt(items)
            return NS(data=list(reversed(items)))
        def call(fn, cost_feature, cost_suboperation, **kwargs):
            self.assertEqual((cost_feature, cost_suboperation), ('taxonomy', 'topic_embedding'))
            return fn(**kwargs)
        ns = dict(client=NS(embeddings=NS(create=create)), cost=NS(call=call), TaxonomyTopicLedgerEntry=NS)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), 'main.py', 'exec'), ns)
        original = copy.deepcopy(data)
        with redirect_stdout(io.StringIO()):
            result = ns['build_immutable_taxonomy_ledger'](data)
        self.assertEqual(data, original)
        return result, calls

    def test_312_inputs_use_ten_calls_preserve_order_and_exact_text(self):
        data = {'categories': [{'name': ' Area ', 'topics': [
            {'title': f'Topic {i}', 'description': f'Original description {i}', 'importance': i % 10}
            for i in range(312)]}]}
        ledger, calls = self.run_ledger(data)
        self.assertEqual(len(calls), 10)
        self.assertEqual([s for batch in calls for s in batch],
                         [f'AREA - Topic {i}: Original description {i}' for i in range(312)])
        for i, entry in enumerate(ledger):
            self.assertEqual(entry.ledger_id, f'T{i+1:04d}')
            self.assertEqual(entry.source_position, (0, i))
            self.assertEqual(entry.embedding, (float(len(calls[i // 32][i % 32])), float(i % 32)))
            self.assertEqual(entry.importance, i % 10)

    def test_existing_embeddings_reused_and_blank_titles_skipped(self):
        data = {'categories': [{'name': 'a', 'topics': [
            {'title': ' ', 'description': 'skip'},
            {'title': 'Existing', 'embedding': [3, 4]}, {'title': 'New'}]}]}
        ledger, calls = self.run_ledger(data)
        self.assertEqual(calls, [['A - New: ']])
        self.assertEqual(ledger[0].embedding, (3, 4))
        self.assertEqual([e.source_position for e in ledger], [(0, 1), (0, 2)])
        _, calls = self.run_ledger({'categories': [{'name': 'a', 'topics': [{'title': 'Existing', 'embedding': [3, 4]}]}]})
        self.assertEqual(calls, [])

    def test_payload_byte_bound_preserves_unicode(self):
        body = '字' * 10000
        _, calls = self.run_ledger({'categories': [{'name': 'a', 'topics': [
            {'title': str(i), 'description': body} for i in range(5)]}]})
        self.assertEqual(list(map(len, calls)), [2, 2, 1])
        self.assertTrue(all(sum(len(s.encode('utf-8')) for s in batch) <= 64000 for batch in calls))

    def test_missing_duplicate_and_unknown_response_indices_fail(self):
        data = {'categories': [{'name': 'a', 'topics': [{'title': 'a'}, {'title': 'b'}]}]}
        for corrupt in (lambda xs: xs[:-1], lambda xs: [xs[0], xs[0]],
                        lambda xs: [NS(index=99, embedding=[1]), xs[1]]):
            with self.subTest(corrupt=corrupt), self.assertRaises(ValueError):
                self.run_ledger(data, corrupt)


class AssignmentTransferTests(unittest.TestCase):
    def test_pair_scoring_ranking_rescue_and_transfer_contract(self):
        bodies = {i: ('Exact source ' + str(i), 'SECTION') for i in range(1, 206)}
        candidates = [dict(assignment_id=f't{i:03}', topic=f'Topic {i}', source_section='SECTION',
                           embedding=[float(i)]) for i in range(103)]
        reads, queries, seen = [], [], []
        def distance(cid, i):
            if i == 102:
                return None
            if i == 101:
                return -.46  # orphan rescue, tie must select smallest chunk ID
            return -(.8 if i % 10 == cid % 10 else .1)
        def execute(statement, params):
            sql = str(statement)
            self.assertIn('c.document_id=:document_id', sql)
            self.assertEqual(params['document_id'], 'doc')
            self.assertEqual(params['module_id'], 'module')
            ids = sorted(v for k, v in params.items() if k.startswith('c') and k[1:].isdigit())
            if 'cross join' not in sql:
                reads.extend(ids)
                return NS(fetchall=lambda: [(cid, *bodies[cid]) for cid in ids])
            self.assertNotIn('c.chunk_text', sql)
            self.assertNotIn('t.title', sql)
            self.assertIn('c.embedding <#> t.embedding', sql)
            queries.append(sql)
            topics = sorted(v for k, v in params.items() if k.startswith('id') and k[2:].isdigit())
            return NS(fetchall=lambda: [(cid, tid, distance(cid, int(tid[1:]))) for cid in ids for tid in topics])
        def score(body, section, title, topic_section, d):
            self.assertEqual(section, 'SECTION')
            self.assertEqual(topic_section, 'SECTION')
            seen.append((int(body.rsplit(' ', 1)[1]), int(title.split()[1])))
            return None if d is None else -d
        snapshot = {cid: str(cid) for cid in bodies}
        with patch('taxonomy_assignment.source_snapshot', return_value=snapshot), redirect_stdout(io.StringIO()):
            links, actual_snapshot = plan_assignments(NS(execute=execute), candidates, 'project', score,
                module_id='module', document_id='doc', pair_budget=600)
        expected = {c['assignment_id']: set() for c in candidates}
        for cid in bodies:
            eligible = [(distance(cid, i), c['assignment_id']) for i, c in enumerate(candidates)
                        if distance(cid, i) is not None and -distance(cid, i) >= .54]
            for _, tid in sorted(eligible)[:3]:
                expected[tid].add(cid)
        for i, c in enumerate(candidates):
            if not expected[c['assignment_id']]:
                eligible = [(distance(cid, i), cid) for cid in bodies
                            if distance(cid, i) is not None and -distance(cid, i) >= .46]
                if eligible:
                    expected[c['assignment_id']].add(min(eligible)[1])
        self.assertEqual(links, expected)
        self.assertEqual(actual_snapshot, snapshot)
        self.assertEqual(reads, list(bodies))
        self.assertEqual(len(seen), 205 * 103)
        self.assertEqual(len(set(seen)), len(seen))

    def test_source_change_or_missing_source_fails(self):
        candidate = dict(assignment_id='t', topic='Topic', source_section=None, embedding=[1])
        def execute(statement, params):
            rows = [(1, 't', -1)] if 'cross join' in str(statement) else [(1, 'body', None)]
            return NS(fetchall=lambda: rows)
        with patch('taxonomy_assignment.source_snapshot', side_effect=[{1: 'old'}, {1: 'new'}]):
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                plan_assignments(NS(execute=execute), [candidate], 'p', lambda *a: -a[-1])
        with patch('taxonomy_assignment.source_snapshot', return_value={1: 'old'}):
            with self.assertRaisesRegex(ValueError, 'Incomplete.*sources'):
                plan_assignments(NS(execute=lambda *a: NS(fetchall=lambda: [])), [candidate], 'p', lambda *a: 1)
