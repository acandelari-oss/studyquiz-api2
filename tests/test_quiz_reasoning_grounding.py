import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import time
import unittest
from quiz_reasoning import reasoning_evidence_instructions
from unittest.mock import Mock


class ReasoningGroundingTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('main.py').read_text())
        quiz = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_quiz')
        names = {'scenario_source_material', 'generate_reasoning_chain', 'generate_batch', 'fails_basic_reasoning_check'}
        nodes = [n for n in quiz.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names]
        patterns = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'LOW_QUALITY_PATTERNS' for t in n.targets))
        self.ns = {'asyncio': asyncio, 'json': json, 'time': time, 'print': lambda *args: None,
            'req': SimpleNamespace(difficulty='medium', language='English'), 'project_id': 'test-project',
            'cost': SimpleNamespace(bind_call=lambda fn, **kw: fn, outcome=lambda *args: None),
            'refill_rejections': [], 'question_rejections': {}, 'hard_question_type_by_object_id': {},
            'generation_diagnostics': {'initial_generated': 0, 'initial_rejected': 0},
            'quiz_refill_context': lambda *args: '', 'reasoning_evidence_instructions': reasoning_evidence_instructions, 'score_reasoning_quality': lambda text: 4,
            'rewrite_question_opening': lambda text: text}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[patterns] + nodes, type_ignores=[])), 'grounding_test', 'exec'), self.ns)

    def test_source_selection_preserves_explicit_evidence_and_rejects_foreign_ids(self):
        select = self.ns['scenario_source_material']
        chunks = [{'chunk_id': 1, 'chunk_text': 'First'}, {'chunk_id': 20, 'chunk_text': 'Actual evidence'}]
        self.assertEqual(select({'source_chunk_ids': [20, '20']}, chunks), [chunks[1]])
        for ids in ([], ['foreign'], [20, 'foreign'], [True], '20'):
            self.assertEqual(select({'source_chunk_ids': ids}, chunks), [])

    def run_batch(self, question_sources=('20',), blueprint=None):
        chunks = [{'chunk_id': i, 'chunk_text': f'Unrelated material {i}'} for i in range(20)]
        chunks.append({'chunk_id': 20, 'chunk_text': 'Object A has component X. Object B has component Y. Both perform function Z.'})
        self.ns['retrieved_chunks'] = chunks
        async def scenarios(*args):
            return [{'scenario': 'Compare A and B', 'source_chunk_ids': ['20']}]
        async def validate(*args): return True
        self.ns.update(generate_scenarios=scenarios, validate_question=validate)
        chain = blueprint if blueprint is not None else {'observable_change': 'A and B perform Z', 'hidden_cause': 'A uses X, B uses Y', 'required_inference': 'Shared function does not imply identical components'}
        question = {'question': 'Which of the following comparisons explains why a shared function does not establish identical components in objects A and B?', 'options': ['one', 'two', 'three', 'four', 'five'], 'correct': 0, 'explanation': 'A uses X and B uses Y', 'source_chunk_ids': list(question_sources)}
        create = Mock(side_effect=[SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]) for payload in (chain, {'questions': [question]})])
        self.ns['client'] = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        result = asyncio.run(self.ns['generate_batch'](1, 'reasoning'))
        return result, create

    def test_blueprint_and_question_receive_same_explicit_sources_beyond_first_fifteen(self):
        result, create = self.run_batch()
        self.assertEqual(len(result), 1)
        self.assertEqual(create.call_count, 2)
        for call in create.call_args_list:
            prompt = call.kwargs['messages'][0]['content']
            self.assertIn('Object A has component X', prompt)
            self.assertNotIn('Unrelated material', prompt)
        self.assertIn('Shared properties or analogous structures do NOT imply identical', create.call_args_list[0].kwargs['messages'][0]['content'])
        self.assertIn('Discard unsupported blueprint claims', create.call_args_list[1].kwargs['messages'][0]['content'])

    def test_question_cannot_cite_other_retrieved_chunks(self):
        result, _ = self.run_batch(question_sources=['1'])
        self.assertEqual(result, [])
        self.assertEqual(self.ns['refill_rejections'][-1]['codes'], ['invalid_question_sources'])

    def test_invalid_blueprint_never_reaches_question_generation(self):
        for blueprint in ({}, {'observable_change': '...', 'hidden_cause': '...', 'required_inference': '...'}):
            result, create = self.run_batch(blueprint=blueprint)
            self.assertEqual(result, [])
            self.assertEqual(create.call_count, 1)

    def test_neutral_opening_is_not_banned_but_direct_recall_still_is(self):
        check = self.ns['fails_basic_reasoning_check']
        self.assertFalse(check('Which of the following conclusions follows when condition A persists but condition B is removed from the described system?'))
        self.assertTrue(check('What is the primary mechanism by which T cells are activated in relation to antigen presentation?'))
        self.assertTrue(check('Which of the following is correct?'))
