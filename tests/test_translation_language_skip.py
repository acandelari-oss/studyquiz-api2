import copy
import json
import unittest
from unittest.mock import patch
from taxonomy_organization import translate_final_taxonomy, _clearly_target_language

EN = 'This process occurs within the cell and involves several proteins which are required during replication.'
IT = 'Questo processo avviene nella cellula mentre queste proteine sono coinvolte nella replicazione.'

class LanguageSkipTests(unittest.TestCase):
    def test_matching_prose_and_mixed_text(self):
        self.assertTrue(_clearly_target_language(EN, 'English'))
        self.assertTrue(_clearly_target_language(IT, 'Italian'))
        for text in [IT, EN + ' ' + IT, EN + ' Ceci est une phrase française.', 'ATP', 'DNA replication']:
            self.assertFalse(_clearly_target_language(text, 'English'))

    def test_only_uncertain_fields_requested_preserve_identity_evidence_and_embeddings(self):
        data = {'categories': [{'name': 'Replication', 'macrocategory': 'Biology', 'topics': [
            {'assignment_id': 'id', 'title': 'DNA replication', 'description': EN,
             'evidence_ids': [1], 'embedding': [1, 2], 'candidate_ids': ['0']}]}]}
        before = copy.deepcopy(data)
        calls = []
        def request(prompt, stage):
            items = json.loads(prompt.split('\n', 1)[1])['items']
            calls.append(items)
            self.assertNotIn(EN, [i['text'] for i in items])
            return {'items': items}
        result = translate_final_taxonomy(data, 'English', request, skip_matching_prose=True)
        self.assertEqual(result, before)
        self.assertEqual(data, before)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 3)

    def test_no_request_when_every_required_field_is_clear(self):
        data = {'categories': [{'name': 'Student category', 'topics': [
            {'assignment_id': 'id', 'title': EN, 'description': EN}]}]}
        result = translate_final_taxonomy(data, 'English', lambda *a: self.fail('Unexpected call'),
                                         preserve_category_names=True, skip_matching_prose=True)
        self.assertEqual(result, data)

    def test_foreign_description_remains_in_translation_request(self):
        data = {'categories': [{'name': 'Area', 'topics': [
            {'assignment_id': 'id', 'title': 'Topic', 'description': IT}]}]}
        def request(prompt, stage):
            items = json.loads(prompt.split('\n', 1)[1])['items']
            self.assertIn(IT, [i['text'] for i in items])
            return {'items': [{'id': i['id'], 'text': EN if i['text'] == IT else i['text']} for i in items]}
        result = translate_final_taxonomy(data, 'English', request, skip_matching_prose=True)
        self.assertEqual(result['categories'][0]['topics'][0]['description'], EN)

    def test_foreign_sentences_are_not_hidden_by_english_majority(self):
        for foreign in [IT, 'Questo è italiano.',
                        'Les protéines sont nécessaires pour la réparation des cellules.',
                        'Las proteínas son necesarias para reparar las células.']:
            with self.subTest(foreign=foreign):
                self.assertFalse(_clearly_target_language((EN + ' ') * 10 + foreign, 'English'))

    def test_detector_unavailable_does_not_skip(self):
        with patch('taxonomy_organization._taxonomy_language_factory', return_value=None):
            self.assertFalse(_clearly_target_language(EN, 'English'))

    def test_repeated_detection_is_deterministic(self):
        self.assertEqual([_clearly_target_language(EN, 'English') for _ in range(5)], [True] * 5)
