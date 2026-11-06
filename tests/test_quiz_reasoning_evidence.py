import ast
import asyncio
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from quiz_reasoning import evaluate_reasoning_evidence, reasoning_evidence_instructions


class ReasoningEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.sources = [{'chunk_id': 'rule', 'chunk_text': 'Only systems containing B can perform A.'},
                        {'chunk_id': 'sample', 'chunk_text': 'Sample X lacks B.'}]
        self.question = {'question': 'What is possible for X?', 'options': ['X cannot perform A', 'X performs A', 'X contains B', 'All systems contain B', 'B is unnecessary'],
                         'correct': 0, 'explanation': 'A requires B, but X lacks B, so X cannot perform A.',
                         'source_chunk_ids': ['rule', 'sample'],
                         'reasoning': {'operation': 'application', 'premises': [
                             {'chunk_id': 'rule', 'quote': self.sources[0]['chunk_text']},
                             {'chunk_id': 'sample', 'quote': self.sources[1]['chunk_text']}],
                             'derivation': 'Applying the requirement for B to X rules out A.', 'conclusion': 'X cannot perform A'}}

    def test_actual_medium_validator_accepts_same_inference_with_any_opening(self):
        tree = ast.parse(Path('main.py').read_text())
        quiz = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'generate_quiz')
        nodes = [n for n in quiz.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in ('validate_question', 'score_reasoning_quality')]
        ns = {'req': SimpleNamespace(difficulty='medium'), 'print': lambda *args: None,
              'evaluate_reasoning_evidence': evaluate_reasoning_evidence, 'question_rejections': {}}
        exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])), 'validator_test', 'exec'), ns)
        for stem in ('What is possible for X?', 'During assessment, what is possible for X?', 'Which of the following conclusions follows about X?'):
            self.question['question'] = stem
            self.assertTrue(asyncio.run(ns['validate_question'](self.question, 'reasoning', self.sources)))
        self.question['reasoning']['operation'] = 'recall'
        self.assertFalse(asyncio.run(ns['validate_question'](self.question, 'reasoning', self.sources)))

    def test_supplied_recall_examples_do_not_qualify_as_reasoning(self):
        for stem in ('What is the specific role of CD 4+ T cells in MHC Class II presentation?',
                     'What is the primary mechanism by which T cells are activated in relation to antigen presentation?',
                     'Which of the following statements describes cytokine receptor activation?'):
            self.question['question'] = stem
            self.question['reasoning']['operation'] = 'recall'
            self.assertIn('not_a_reasoning_task', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])
        self.question.pop('reasoning')
        self.assertIn('missing_reasoning_evidence', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])

    def test_exact_quotes_required_and_foreign_sources_rejected(self):
        self.question['reasoning']['premises'][0]['quote'] = 'All systems can perform A.'
        self.assertIn('reasoning_quote_not_in_source', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])
        self.question['reasoning']['premises'][0]['chunk_id'] = 'other-document'
        self.assertIn('reasoning_source_outside_scope', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])

    def test_duplicate_premises_and_missing_derivation_rejected(self):
        self.question['reasoning']['premises'][1] = deepcopy(self.question['reasoning']['premises'][0])
        self.question['reasoning']['derivation'] = ''
        reasons = evaluate_reasoning_evidence(self.question, self.sources)['reasons']
        self.assertIn('insufficient_distinct_premises', reasons)
        self.assertIn('missing_reasoning_derivation', reasons)

    def test_answer_and_options_remain_validated(self):
        self.question['reasoning']['conclusion'] = 'X performs A'
        self.assertIn('reasoning_answer_mismatch', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])
        self.question['correct'] = 7
        self.assertIn('invalid_correct_answer', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])
        self.question['options'] = ['same'] * 5
        self.assertIn('invalid_answer_options', evaluate_reasoning_evidence(self.question, self.sources)['reasons'])

    def test_language_independent_evidence_and_input_preservation(self):
        self.sources[0]['chunk_text'] = 'Solo i sistemi con B possono eseguire A.'
        self.sources[1]['chunk_text'] = 'X è privo di B.'
        for p, c in zip(self.question['reasoning']['premises'], self.sources): p['quote'] = c['chunk_text']
        before = deepcopy(self.question)
        self.assertTrue(evaluate_reasoning_evidence(self.question, self.sources)['valid'])
        self.assertEqual(before, self.question)
        self.assertIn('do not mislabel it', reasoning_evidence_instructions())
