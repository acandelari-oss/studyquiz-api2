import unittest
from quiz_reasoning import evidence_catalog, resolve_evidence_references, evaluate_reasoning_evidence, reasoning_response_format

class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.material = [{'chunk_id': 7, 'chunk_text': 'A requires X. B lacks X. Rate is 1.2 and x² stays x².'}]
        self.catalog = evidence_catalog(self.material)
        self.q = {'question': 'Can B perform A?', 'options': ['No', 'Yes', 'Always', 'Sometimes', 'Unknown'], 'correct': 0,
                  'explanation': 'B lacks the required component.', 'reasoning': {'operation': 'application',
                  'premises': [{'evidence_id': 'E1'}, {'evidence_id': 'E2'}],
                  'derivation': 'The requirement excludes B.', 'conclusion': 'No'}}

    def test_server_expands_exact_sources_without_mutating_response(self):
        q = resolve_evidence_references(self.q, self.catalog)
        self.assertEqual(q['source_chunk_ids'], ['7'])
        self.assertTrue(evaluate_reasoning_evidence(q, self.material)['valid'])
        self.assertEqual(self.q['reasoning']['premises'][0], {'evidence_id': 'E1'})
        self.assertEqual(self.catalog['E3']['quote'], 'Rate is 1.2 and x² stays x².')

    def test_unknown_duplicate_and_injected_quotes_are_rejected(self):
        for premise in ({'evidence_id': 'unknown'}, {'evidence_id': 'E1'}, {'evidence_id': 'E2', 'quote': 'Invented'}):
            self.q['reasoning']['premises'][1] = premise
            q = resolve_evidence_references(self.q, self.catalog)
            self.assertFalse(evaluate_reasoning_evidence(q, self.material)['valid'])

    def test_missing_reasoning_remains_invalid(self):
        del self.q['reasoning']
        self.assertFalse(evaluate_reasoning_evidence(resolve_evidence_references(self.q, self.catalog), self.material)['valid'])

    def test_schema_requires_reasoning_and_forbids_generated_quotes(self):
        schema = reasoning_response_format()['json_schema']
        self.assertTrue(schema['strict'])
        q = schema['schema']['properties']['questions']['items']
        self.assertIn('reasoning', q['required'])
        premise = q['properties']['reasoning']['properties']['premises']['items']
        self.assertEqual(premise['required'], ['evidence_id'])
        self.assertFalse(premise['additionalProperties'])
