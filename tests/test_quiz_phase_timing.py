"""Offline checks for quiz instrumentation; no app import or external services."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class QuizPhaseTimingTests(unittest.TestCase):
    def test_phase_durations_use_monotonic_time_and_no_question_content(self):
        tree = ast.parse(Path('main.py').read_text())
        quiz = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_quiz')
        helper = next(n for n in quiz.body if isinstance(n, ast.FunctionDef) and n.name == 'log_quiz_phase')
        # Execute the actual helper in its closure, without the generation route.
        wrapper = ast.parse('def make():\n    quiz_checkpoint = 10.0\n    quiz_monotonic_start = 10.0\n').body[0]
        wrapper.body.extend([helper, ast.Return(value=ast.Name(id='log_quiz_phase', ctx=ast.Load()))])
        module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
        output = Mock()
        clock = Mock(side_effect=[12.5, 16.0])
        namespace = {'time': SimpleNamespace(monotonic=clock), 'json': json, 'print': output,
                     'project_id': 'project', 'req': SimpleNamespace(num_questions=3)}
        exec(compile(module, 'quiz_phase_test', 'exec'), namespace)
        log = namespace['make']()
        log('initial_generation')
        log('validation_and_refill')
        rows = [json.loads(call.args[1]) for call in output.call_args_list]
        self.assertEqual([r['seconds'] for r in rows], [2.5, 3.5])
        self.assertEqual(rows[-1]['elapsed_seconds'], 6.0)
        self.assertEqual(set(rows[0]), {'project_id', 'phase', 'seconds', 'elapsed_seconds', 'requested_questions'})
