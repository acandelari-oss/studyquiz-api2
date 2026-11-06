import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


class QuizRefillTests(unittest.TestCase):
    def setUp(self):
        tree = ast.parse(Path('main.py').read_text())
        self.quiz = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_quiz')
        self.ns = {'json': json, 'req': SimpleNamespace(question_style='balanced', num_questions=20)}
        helpers = [n for n in self.quiz.body if isinstance(n, ast.FunctionDef) and n.name in ('select_quiz_refill', 'quiz_refill_context')]
        exec(compile(ast.fix_missing_locations(ast.Module(body=helpers, type_ignores=[])), 'helpers', 'exec'), self.ns)

    def test_refill_requests_missing_style_only(self):
        exam = [{} for _ in range(10)]
        reasoning = [{} for _ in range(2)]
        styles = {id(q): 'exam' for q in exam}
        styles.update({id(q): 'reasoning' for q in reasoning})
        self.assertEqual(self.ns['select_quiz_refill'](exam + reasoning, styles), ('reasoning', 8))
        self.ns['req'].num_questions = 5
        self.assertEqual(self.ns['select_quiz_refill']([], {}), ('reasoning', 3))

    def test_nonbalanced_preserves_requested_style(self):
        self.ns['req'].question_style = 'exam'
        self.assertEqual(self.ns['select_quiz_refill']([{}] * 12, {}), ('exam', 8))

    def test_context_contains_accepted_targets_and_bounded_recent_reasons(self):
        context = self.ns['quiz_refill_context']([{'question': 'Accepted mechanism?', 'explanation': 'Mechanism answer'}],
            [{'codes': [f'reason-{i}']} for i in range(20)])
        self.assertIn('Accepted mechanism?', context)
        self.assertIn('Mechanism answer', context)
        self.assertIn('reason-19', context)
        self.assertNotIn('"reason-0"', context)
        self.assertIn('not merely their wording', context)
        self.assertEqual(self.ns['quiz_refill_context']([], []), '')

    def test_actual_refill_loop_passes_feedback_and_stops_at_exact_count(self):
        exam = [{'question': f'Existing distinct exam question number {i}'} for i in range(2)]
        styles = {id(q): 'exam' for q in exam}
        calls = []
        async def generate_batch(n, style, **kwargs):
            calls.append((n, style, kwargs))
            rows = [{'question': f'New reasoning question {len(calls)} number {i}'} for i in range(n)]
            for q in rows: styles[id(q)] = style
            return rows
        self.ns.update(req=SimpleNamespace(question_style='balanced', num_questions=5), all_questions=exam,
            hard_question_type_by_object_id=styles, retry_count=0, max_retries=10, generation_diagnostics={'refill_accepted': 0},
            cost=SimpleNamespace(metrics=lambda **kw: None), refill_rejections=[{'codes': ['heuristic_score_rejection']}],
            seen_questions=set(), generate_batch=generate_batch, print=lambda *args: None)
        loop = next(n for n in self.quiz.body if isinstance(n, ast.While) and 'retry_count' in ast.unparse(n.test))
        wrapper = ast.parse('async def run():\n    pass').body[0]
        wrapper.body = [ast.Global(names=['retry_count']), loop]
        exec(compile(ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[])), 'loop', 'exec'), self.ns)
        asyncio.run(self.ns['run']())
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], (3, 'reasoning'))
        self.assertEqual(len(calls[0][2]['accepted_questions']), 2)
        self.assertEqual(calls[0][2]['rejection_feedback'][0]['codes'], ['heuristic_score_rejection'])
        self.assertEqual(len(self.ns['all_questions']), 5)

    def test_context_reaches_scenario_and_question_requests(self):
        for name, assignment in [('generate_scenarios', 'scenario_prompt'), ('generate_batch', 'prompt')]:
            fn = next(n for n in self.quiz.body if isinstance(n, ast.AsyncFunctionDef) and n.name == name)
            self.assertTrue(any(isinstance(n, ast.AugAssign) and isinstance(n.target, ast.Name)
                and n.target.id == assignment and ast.unparse(n.value) == 'refill_context' for n in ast.walk(fn)))
