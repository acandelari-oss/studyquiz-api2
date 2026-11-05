"""Mocked scope mismatch handling, not a live semantic-quality benchmark."""
import copy
import json
import unittest
from taxonomy_organization import organize_categories


class CategoryScopeRevisionTests(unittest.TestCase):
    def test_scope_mismatch_can_replace_structure_without_changing_topics(self):
        topics = [dict(assignment_id=str(i), title=title, description=scope, evidence_ids=[i])
                  for i, (title, scope) in enumerate([
                      ('Galactose Metabolism', 'Carbohydrate metabolism'),
                      ('GLUT Transporters', 'Transport of glucose'),
                      ('Amino Acid Biosynthesis', 'Synthesis of amino acids'),
                      ('Urea Cycle', 'Nitrogen elimination')])]
        data = {'categories': [{'name': 'Old', 'topics': topics}]}
        before = copy.deepcopy(data)
        calls = []
        def request(prompt, stage):
            calls.append(stage)
            payload = json.loads(prompt.split('ORGANIZATION INPUT:\n')[1])
            self.assertEqual(len(payload['topics']), 4)
            if stage == 'category_proposal':
                self.assertIn('a name for one pathway, process or mechanism', prompt)
                return {'categories': [{'name': 'Fatty Acid Oxidation', 'description': 'Oxidation of fats'}]}
            if stage == 'category_assignment':
                self.assertIn('A category name is a constraint', prompt)
                return {'assignments': [{'topic_id': f'T{i+1:04d}', 'category_id': None} for i in range(4)],
                        'quality_issues': [{'code': 'unmatched', 'topic_ids': ['T0001'], 'category_ids': ['C0000']}]}
            self.assertEqual(stage, 'category_correction')
            return {'categories': [
                {'name': 'Carbohydrate Transport and Metabolism', 'description': 'Carbohydrate processing and transport'},
                {'name': 'Amino Acid Biosynthesis', 'description': 'Synthesis of amino acids'},
                {'name': 'Nitrogen Elimination', 'description': 'Disposal of nitrogen'}],
                'assignments': [{'topic_id': f'T{i+1:04d}', 'category_id': f'C{max(0,i-1):04d}'} for i in range(4)],
                'quality_issues': []}
        result = organize_categories(data, 'infer', [], request)
        self.assertEqual(calls, ['category_proposal', 'category_assignment', 'category_correction'])
        self.assertEqual(data, before)
        self.assertEqual([len(c['topics']) for c in result['categories']], [2, 1, 1])
        self.assertEqual([t for c in result['categories'] for t in c['topics']], topics)
