"""Offline tests of the production parser; no application startup or services."""
import ast
import copy
from pathlib import Path
import re
import unicodedata
import unittest
from typing import List, Optional
from taxonomy_organization import organize_categories

ns = dict(re=re, unicodedata=unicodedata, List=List, Optional=Optional)
tree = ast.parse(Path('main.py').read_text())
names = {'_taxonomy_normalized_phrase', '_syllabus_text', '_clean_syllabus_line',
         '_classify_syllabus_line', '_syllabus_display_title', '_is_structural_syllabus_label',
         '_parse_syllabus_outline', '_syllabus_likely_category_items',
         '_build_syllabus_structure_instruction', '_global_organization_categories', '_preflight_syllabus'}
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in names:
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'main.py', 'exec'), ns)

TITLES = ['Introduction to Immunology', 'Innate Immunity and Pathogen Recognition',
          'Inflammation and Immune Effector Mechanisms',
          'Adaptive Immunity and Antigen Presentation',
          'Lymphocyte Development, Receptor Diversity and Tolerance']
RANGES = [(1, 6), (7, 29), (30, 53), (54, 89), (90, 106)]

def org(text):
    return {'organization_mode': 'syllabus', 'organization_blueprint': {'syllabus_text': text}}

def fixture():
    return org('## Index — Inflammation & Immunity\n' + '\n'.join(
        f'### Lecture {i+1} — {title} (pp. {start}–{end})\n'
        '1. Learning subject\n   - Supporting detail\n   - Another detail\n2. Second subject'
        for i, (title, (start, end)) in enumerate(zip(TITLES, RANGES))))

class SyllabusTests(unittest.TestCase):
    def test_lectures_are_fixed_categories_with_scopes(self):
        categories = ns['_global_organization_categories'](fixture())
        self.assertEqual([c['name'] for c in categories], TITLES)
        for c in categories:
            self.assertIn('Supporting detail', c['description'])
            self.assertIn('Second subject', c['description'])
        outline = ns['_parse_syllabus_outline'](fixture())
        lectures = ns['_syllabus_likely_category_items'](outline)
        self.assertEqual([x['page_range'] for x in lectures], [list(x) for x in RANGES])
        self.assertEqual(outline[3]['parent_index'], 2)
        self.assertEqual(outline[4]['parent_index'], 2)

    def test_nested_sections_not_promoted(self):
        data = org('## Chapter 1: Foundations\n### Section 1: Detail\n- Evidence\n## Chapter 2: Applications')
        self.assertEqual([x['name'] for x in ns['_global_organization_categories'](data)], ['Foundations', 'Applications'])

    def test_part_numbered_legacy(self):
        data = org('PART I — Foundations\n1. First\n2. Second\n3. Third')
        self.assertEqual([x['name'] for x in ns['_global_organization_categories'](data)], ['First', 'Second', 'Third'])

    def test_no_silent_truncation(self):
        data = org('Lecture 1 — ' + 'X' * 200)
        self.assertEqual(len(ns['_parse_syllabus_outline'](data)[0]['display_title']), 200)
        self.assertEqual(ns['_preflight_syllabus'](data)[0]['name'], 'X' * 200)

    def test_assignment_uses_fixed_categories_preserves_topics(self):
        topics = [{'assignment_id': str(i), 'title': title, 'description': 'Full description',
                   'evidence_ids': [i], 'document_id': 'document'} for i, title in enumerate(TITLES)]
        data = {'categories': [{'name': 'Old', 'topics': topics}]}
        before = copy.deepcopy(data)
        calls = []
        def request(prompt, stage):
            calls.append(stage)
            return {'assignments': [{'topic_id': f'T{i+1:04d}', 'category_id': f'C{i:04d}'} for i in range(5)]}
        result = organize_categories(data, 'syllabus', ns['_global_organization_categories'](fixture()), request)
        self.assertEqual(calls, ['category_assignment'])
        self.assertEqual([c['name'] for c in result['categories']], TITLES)
        self.assertEqual([t for c in result['categories'] for t in c['topics']], topics)
        self.assertEqual(data, before)

    def test_syllabus_enforcement_does_not_depend_on_infer_flag(self):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_topics_task')
        assignment = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'global_organization_enabled' for t in n.targets))
        import os
        value = eval(compile(ast.Expression(assignment.value), 'main.py', 'eval'),
                     {'os': os, 'chapter_structure': None, 'module_organization': fixture()})
        self.assertTrue(value)

    def test_complete_supplied_index_preserved(self):
        text = Path('tests/fixtures/inflammation_immunity_index.md').read_text()
        self.assertGreater(len([line for line in text.splitlines() if line.strip()]), 120)
        data = org(text)
        before = copy.deepcopy(data)
        categories = ns['_preflight_syllabus'](data)
        self.assertEqual([c['name'] for c in categories], TITLES)
        for line in text.splitlines():
            if line.strip() and not line.startswith('## Index'):
                self.assertTrue(any(ns['_clean_syllabus_line'](line) in c['description'] for c in categories), line)
        self.assertEqual(data, before)

    def test_oversized_context_rejected_without_truncation(self):
        data = org('Lecture 1 — Area\n   - ' + 'Evidence ' * 6000)
        with self.assertRaisesRegex(ValueError, 'bounded context'):
            ns['_preflight_syllabus'](data)

    def test_duplicate_categories_rejected_early(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            ns['_preflight_syllabus'](org('Lecture 1 — Area\nLecture 2 — Area'))

    def test_preflight_precedes_discovery_and_language_detection(self):
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'process_topics_task')
        call = next(n for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_preflight_syllabus')
        source = Path('main.py').read_text().splitlines()
        for marker in ('pipeline_log.start("TAXONOMY LANGUAGE DETECTION")', 'for section_key, section_chunks'):
            line = next(i for i, text in enumerate(source, 1) if marker in text)
            self.assertLess(call.lineno, line)
