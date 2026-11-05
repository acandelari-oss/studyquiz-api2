import copy
import json
import unittest
from taxonomy_organization import organize_categories, OrganizationValidationError


class QualityTests(unittest.TestCase):
    def setUp(self):
        self.data = {'categories': [{'name': 'UNTRUSTED LEGACY SECTION', 'topics': [
            {'assignment_id': 'a', 'title': 'Contract formation', 'description': 'Offer and acceptance', 'document_id': 'doc', 'evidence_ids': [1]},
            {'assignment_id': 'b', 'title': 'Contract remedies', 'description': 'Remedies for breach', 'document_id': 'doc', 'evidence_ids': [2]}]}]}
        self.categories = [{'name': 'Contract formation', 'description': 'Offer, acceptance and agreement'},
                           {'name': 'Remedies for breach', 'description': 'Relief following nonperformance'}]
        self.response = {'assignments': [{'topic_id': 'T0001', 'category_id': 'C0000'},
                                         {'topic_id': 'T0002', 'category_id': 'C0001'}], 'quality_issues': []}
        self.calls = []

    def request(self, prompt, stage):
        self.calls.append((prompt, stage))
        if stage == 'category_proposal':
            return {'categories': self.categories}
        if stage == 'category_correction':
            return {'categories': self.categories, **self.response}
        return self.response

    def test_distribution_identity_and_context_without_legacy_authority(self):
        original = copy.deepcopy(self.data)
        result = organize_categories(self.data, 'infer', [], self.request,
                                     document_context={'document_title': 'Contract law.pdf'})
        self.assertEqual([len(c['topics']) for c in result['categories']], [1, 1])
        self.assertEqual(self.data, original)
        self.assertIs(result['categories'][0]['topics'][0], self.data['categories'][0]['topics'][0])
        for prompt, stage in self.calls:
            self.assertNotIn('UNTRUSTED LEGACY SECTION', prompt)
            self.assertIn('Contract law.pdf', prompt)
            self.assertLessEqual(len(prompt), 48000)
        self.assertIn('quality_issues', self.calls[1][0])

    def test_generic_missing_and_duplicate_scopes_rejected_after_one_correction(self):
        for categories in ([{'name': 'General', 'description': 'All content'}],
                           [{'name': 'Contract law'}],
                           [{'name': 'A', 'description': 'Same scope'}, {'name': 'B', 'description': 'Same scope!'}]):
            with self.subTest(categories=categories):
                self.calls.clear()
                self.categories = categories
                self.response["assignments"][1]["category_id"] = "C0000"
                with self.assertRaises(OrganizationValidationError):
                    organize_categories(self.data, 'infer', [], self.request)
                self.assertEqual(len(self.calls), 3)

    def test_semantic_failures_and_unmatched_never_force_assignment_or_retry(self):
        before = copy.deepcopy(self.data)
        for code in ('generic', 'overlap', 'incoherent', 'unmatched'):
            self.calls.clear()
            self.response['quality_issues'] = [{'code': code, 'topic_ids': ['T0002']}]
            with self.assertRaises(OrganizationValidationError) as error:
                organize_categories(self.data, 'infer', [], self.request)
            self.assertEqual(error.exception.diagnostics['code'], 'semantic_quality_failure')
            self.assertEqual(len(self.calls), 3)
            self.assertEqual(self.data, before)
        self.response['quality_issues'] = []
        self.response['assignments'][1]['category_id'] = None
        with self.assertRaises(OrganizationValidationError):
            organize_categories(self.data, 'infer', [], self.request)

    def test_missing_review_fails_closed(self):
        del self.response['quality_issues']
        with self.assertRaises(OrganizationValidationError) as error:
            organize_categories(self.data, 'infer', [], self.request)
        self.assertEqual(error.exception.diagnostics['code'], 'missing_quality_review')

    def test_document_isolation_before_ai(self):
        self.data['categories'][0]['topics'][1]['document_id'] = 'other'
        with self.assertRaises(ValueError):
            organize_categories(self.data, 'infer', [], self.request)
        self.assertEqual(self.calls, [])

    def test_chapter_like_labels_are_not_promoted_without_verified_context(self):
        # Source chapters are currently unavailable at this production boundary.
        # Even plausible labels must follow AI inference rather than be trusted.
        self.data['categories'][0]['name'] = '1. Chapter One > 1.1 Definitions'
        organize_categories(self.data, 'infer', [], self.request)
        self.assertEqual(len(self.calls), 2)
        self.assertNotIn('Chapter One', self.calls[0][0])

    def test_manual_categories_do_not_receive_inferred_quality_restrictions(self):
        self.response = {'assignments': [{'topic_id': 'T0001', 'category_id': 'C0000'},
                                         {'topic_id': 'T0002', 'category_id': None}]}
        result = organize_categories(self.data, 'manual', [{'name': 'General'}], self.request)
        self.assertEqual([c['name'] for c in result['categories']], ['General', 'TO ORGANIZE'])
        self.assertEqual(len(self.calls), 1)
