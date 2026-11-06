"""Exercise both evaluation prompt paths offline, without importing the app."""
import ast
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock


class OralFeedbackInstructionsTests(unittest.TestCase):
    def test_stream_and_nonstream_use_subject_only_feedback(self):
        tree = ast.parse(Path('main.py').read_text())
        fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'active_recall_evaluate')
        fn.decorator_list = []
        fn.args.args[0].annotation = None
        for streaming in (False, True):
            with self.subTest(stream=streaming):
                create = Mock(return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"evaluation":"correct","hint":""}'))]))
                namespace = {'json': json, 'client': SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
                             'cost': SimpleNamespace(call=lambda call, **kw: call(**{k:v for k,v in kw.items() if not k.startswith('cost_')})),
                             'stream_chat': lambda factory, **kw: factory()}
                exec(compile(ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[])), 'evaluation_test', 'exec'), namespace)
                asyncio.run(namespace['active_recall_evaluate'](SimpleNamespace(stream=streaming, question='Explain the mechanism', student_answer='A spoken answer', history=[])))
                prompt = create.call_args.kwargs['messages'][1]['content']
                self.assertIn('Do not give spelling, grammar, punctuation, pronunciation, or writing-style advice', prompt)
                self.assertIn('Answers may be speech transcripts', prompt)
                self.assertIn('return an empty string for "hint"', prompt)
                self.assertIn('A spoken answer', prompt)
