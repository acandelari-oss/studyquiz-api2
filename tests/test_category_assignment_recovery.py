"""Bounded missing-assignment completion; mocked requests only."""
import copy
import json
import unittest
from taxonomy_organization import organize_categories, REVIEW_CATEGORY


class AssignmentRecoveryTests(unittest.TestCase):
    def run_case(self, mode='infer', count=239, missing=20, change_initial=None, change_recovery=None):
        self.data = {'categories': [{'name': 'Old', 'topics': [
            {'assignment_id': f'uuid-{i}', 'title': f'Subject {i}', 'description': 'Evidence-backed objective',
             'evidence_ids': [i], 'document_id': 'one-document'} for i in range(count)]}]}
        before = copy.deepcopy(self.data)
        self.calls = []
        self.initial = {'assignments': [{'topic_id': f'T{i+1:04d}', 'category_id': 'C0000'}
                                      for i in range(count-missing)], 'quality_issues': []}
        if change_initial:
            change_initial(self.initial)
        retained_before = copy.deepcopy(self.initial)
        categories = [{'name': 'Specific subject', 'macrocategory': 'Area', 'description': 'Distinct subject scope'}]
        def request(prompt, stage):
            self.calls.append(stage)
            self.assertLessEqual(len(prompt), 48000)
            if stage == 'category_proposal':
                return {'categories': categories}
            if stage == 'category_assignment':
                return self.initial
            self.assertEqual(stage, 'category_assignment_recovery')
            payload = json.loads(prompt.split('ORGANIZATION INPUT:\n')[1])
            self.assertEqual(payload['constraints'], {'syllabus': 'Keep supplied structure'})
            self.assertEqual([t['topic_id'] for t in payload['topics']],
                             [f'T{i+1:04d}' for i in range(count-missing, count)])
            self.assertEqual(payload['categories'][0]['category_id'], 'C0000')
            result = {'assignments': [{'topic_id': t['topic_id'], 'category_id': 'C0000'}
                                     for t in payload['topics']], 'quality_issues': []}
            if change_recovery:
                change_recovery(result)
            return result
        try:
            return organize_categories(self.data, mode, categories, request,
                                       constraints={'syllabus': 'Keep supplied structure'})
        finally:
            self.assertEqual(self.data, before)
            self.assertEqual(self.initial, retained_before)

    def test_actual_failure_shape_recovers_only_twenty_and_preserves_original_objects(self):
        result = self.run_case()
        self.assertEqual(self.calls, ['category_proposal', 'category_assignment', 'category_assignment_recovery'])
        topics = result['categories'][0]['topics']
        self.assertEqual(len(topics), 239)
        for original, actual in zip(self.data['categories'][0]['topics'], topics):
            self.assertIs(original, actual)

    def test_complete_response_does_not_call_recovery(self):
        self.run_case(count=3, missing=0)
        self.assertEqual(self.calls, ['category_proposal', 'category_assignment'])

    def test_invalid_initial_rows_are_not_repaired(self):
        for change in (
            lambda r: r['assignments'].__setitem__(1, r['assignments'][0]),
            lambda r: r['assignments'][0].update(topic_id='UNKNOWN'),
            lambda r: r['assignments'][0].update(category_id='UNKNOWN'),
            lambda r: r['assignments'][0].update(topic_id=1),
            lambda r: r.update(quality_issues=[{'code': 'incoherent'}]),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_case(count=4, missing=1, change_initial=change)
            self.assertNotIn('category_assignment_recovery', self.calls)

    def test_recovery_failure_stops_without_second_request(self):
        for change in (
            lambda r: r['assignments'].pop(),
            lambda r: r['assignments'].append(r['assignments'][0]),
            lambda r: r['assignments'][0].update(topic_id='T0001'),
            lambda r: r['assignments'][0].update(category_id='UNKNOWN'),
            lambda r: r['assignments'][0].update(category_id=None),
            lambda r: r.update(quality_issues=[{'code': 'unmatched'}]),
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.run_case(count=4, missing=2, change_recovery=change)
            self.assertEqual(self.calls.count('category_assignment_recovery'), 1)
            self.assertNotIn('category_correction', self.calls)

    def test_fixed_modes_preserve_unmatched_and_student_categories(self):
        for mode in ('manual', 'syllabus'):
            with self.subTest(mode=mode):
                result = self.run_case(mode=mode, count=3, missing=1,
                    change_recovery=lambda r: r['assignments'][0].update(category_id=None))
                self.assertEqual([c['name'] for c in result['categories']], ['Specific subject', REVIEW_CATEGORY])
                self.assertEqual(result['categories'][1]['topics'][0]['assignment_id'], 'uuid-2')
                self.assertEqual(self.calls, ['category_assignment', 'category_assignment_recovery'])

    def test_recovery_item_limit_fails_before_request(self):
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.run_case(count=65, missing=61)
        self.assertNotIn('category_assignment_recovery', self.calls)
