"""Production planning functions isolated from application startup and services."""
import ast
import copy
from pathlib import Path
import unittest
from typing import Optional, List
from test_syllabus_outline import ns, org

scope = dict(ns, OpenAI=object)
for node in ast.parse(Path('main.py').read_text()).body:
    if isinstance(node, ast.FunctionDef) and node.name in {'_generate_taxonomy_plan', '_build_taxonomy_plan_instruction'}:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), scope)

class IndexPlanningTests(unittest.TestCase):
    def test_complete_index_without_ai_or_category_target(self):
        data = org('\n'.join(f'### Section {i+1} — Area {i} (pp. {i+1}–{i+1})\n- Scope {i}' for i in range(35)))
        before = copy.deepcopy(data)
        diagnostics = {}
        plan = scope['_generate_taxonomy_plan'](None, [], '', '', '', '', data, diagnostics)
        self.assertEqual([c['name'] for c in plan['categories']], [f'Area {i}' for i in range(35)])
        self.assertTrue(all(c['target_topic_count'] is None for c in plan['categories']))
        self.assertEqual(diagnostics['planner_ai_calls'], 0)
        self.assertEqual(data, before)
        guide = scope['_build_taxonomy_plan_instruction'](plan)
        self.assertIn('Scope 34', guide)
        self.assertNotIn('target topics', guide)

    def test_actual_human_genetics_index(self):
        path = Path('../review-artifacts/human-genetics-index.md')
        if not path.exists():
            self.skipTest('Local review artifact unavailable')
        plan = scope['_generate_taxonomy_plan'](None, [], '', '', '', '', org(path.read_text()))
        self.assertEqual(len(plan['categories']), 16)
        self.assertEqual(plan['categories'][10]['name'], 'Array CGH')
