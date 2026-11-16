"""Offline diagnostics: preserve the existing fail-closed translation contract."""
import copy
import io
import json
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock
from taxonomy_organization import translate_final_taxonomy


class TranslationNotationDiagnosticsTests(unittest.TestCase):
    def run_translation(self, original, translated):
        data = {'categories': [{'name': 'Respiration', 'topics': [
            {'assignment_id': 'topic-1', 'title': 'Pressure', 'description': original,
             'evidence_ids': [42], 'candidate_ids': ['0']}]}]}
        before = copy.deepcopy(data)
        def response(prompt, stage):
            items = json.loads(prompt.split('\n', 1)[1])['items']
            return {'items': [{**item, 'text': translated if item['id'] == 'T:topic-1:description'
                              else item['text']} for item in items]}
        request = Mock(side_effect=response)
        output = io.StringIO()
        with redirect_stdout(output):
            with self.assertRaisesRegex(ValueError, 'Translation changed numeric or mathematical notation'):
                translate_final_taxonomy(data, 'Italian', request)
        self.assertEqual(data, before)
        self.assertEqual(request.call_count, 1)
        lines = [line for line in output.getvalue().splitlines()
                 if line.startswith('TAXONOMY_TRANSLATION_NOTATION_MISMATCH ')]
        self.assertEqual(len(lines), 1)
        result = json.loads(lines[0].split(' ', 1)[1])
        self.assertEqual(result['item_id'], 'T:topic-1:description')
        self.assertEqual(result['field'], 'description')
        self.assertEqual(result['batch'], 1)
        self.assertFalse(result['recovery'])
        self.assertEqual(result['original_text'], original)
        self.assertEqual(result['translated_text'], translated)
        return result

    def test_changed_value_reports_exact_difference(self):
        result = self.run_translation('Pressure = 12 mmHg.', 'Pressione = 21 mmHg.')
        self.assertEqual(result['removed_notation'], {'12': 1})
        self.assertEqual(result['added_notation'], {'21': 1})

    def test_changed_symbol_and_occurrence_count(self):
        result = self.run_translation('Δ = 2 + 2\nPressure', 'Delta = 2\nPressione')
        self.assertEqual(result['removed_notation'], {'Δ': 1, '+': 1, '2': 1})
        self.assertEqual(result['added_notation'], {})

    def test_unchanged_notation_does_not_log_mismatch(self):
        data = {'categories': [{'name': 'Area', 'topics': [
            {'assignment_id': 'id', 'title': 'Δ = 2', 'description': '12 mmHg'}]}]}
        def response(prompt, stage):
            return {'items': json.loads(prompt.split('\n', 1)[1])['items']}
        output = io.StringIO()
        with redirect_stdout(output):
            result = translate_final_taxonomy(data, 'Italian', response)
        self.assertEqual(result, data)
        self.assertNotIn('TAXONOMY_TRANSLATION_NOTATION_MISMATCH', output.getvalue())
