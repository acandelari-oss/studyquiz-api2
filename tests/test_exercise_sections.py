"""Offline section classification and eligibility tests; no app, AI or database."""
import ast
from pathlib import Path
import unittest
from chunk_roles import (ExerciseSectionFilter, chunk_document_block, classify_chunk_role,
                         normalize_chunk_role, is_assignment_eligible_chunk_role)
from test_source_chunk_alignment import production_functions


class ExerciseSectionsTests(unittest.TestCase):
    def test_mixed_page_and_following_exercise_pages(self):
        detector = ExerciseSectionFilter()
        text = ("Unità 2 - Moto in una dimensione\n"
                "7. Moto vario\nLa velocità è la derivata prima dello spazio.\n"
                "Esercizi Unità 2\n1. Quale risposta è corretta?\nA. Falso.\nB. Vero.\n")
        parts = detector.split(text)
        self.assertEqual("".join(value for value, _ in parts), text)
        self.assertEqual([role for _, role in parts], [None, 'exercise'])
        self.assertIn("derivata", parts[0][0])
        self.assertNotIn("A. Falso", parts[0][0])
        for page in ["Unità 2 - Moto in una dimensione 14\nC. Distrattore.\n",
                     "2. Calcola la velocità.\n"]:
            self.assertTrue(all(role == 'exercise' for _, role in detector.split(page)))
        resumed = detector.split("Unità 3 - Moto in due dimensioni\nSpiegazione.\n")
        self.assertEqual(resumed[0][1], None)

    def test_numbered_teaching_lists_and_mentions_are_preserved(self):
        text = "1. Posizione\n2. Tempo\nGli esercizi aiutano ad apprendere.\n"
        self.assertEqual(ExerciseSectionFilter().split(text), [(text, None)])

    def test_headingless_explicit_multiple_choice(self):
        text = "Spiegazione.\n1. Quale affermazione è corretta?\nA. Uno\nB. Due\nC. Tre\n"
        parts = ExerciseSectionFilter().split(text)
        self.assertEqual(parts[0], ("Spiegazione.\n", None))
        self.assertEqual(parts[1][1], 'exercise')

    def test_open_ended_and_english_sections(self):
        for heading in ['Exercises', 'Review questions', 'Esercizi proposti', 'Esercizi Unità 2']:
            parts = ExerciseSectionFilter().split(heading+"\n1. Explain the result.\n")
            self.assertEqual(parts[0][1], 'exercise')

    def test_state_is_document_local_and_ambiguous_end_stays_excluded(self):
        first, second = ExerciseSectionFilter(), ExerciseSectionFilter()
        first.split("Exercises\nQuestion\n")
        self.assertEqual(first.split("Continuation without a reliable boundary.")[0][1], 'exercise')
        self.assertEqual(second.split("Independent document.")[0][1], None)

    def test_chunk_overlap_never_crosses_exercise_boundary(self):
        clean, chunk = production_functions()
        text = ("Teaching explanation. "*70)+"\nEsercizi Unità 2\n"+("Wrong answer option. "*70)
        result, reusable = chunk_document_block(text, ExerciseSectionFilter(), clean, chunk)
        self.assertFalse(reusable)
        for value, role in result:
            if role == 'exercise':
                self.assertNotIn("Teaching explanation", value)
            else:
                self.assertNotIn("Wrong answer option", value)

    def test_ordinary_chunk_output_is_unchanged(self):
        clean, chunk = production_functions()
        text = "Chapter 1 - Motion\n"+("Teaching text. "*150)+"\nUnit 2 - Forces\nMore teaching."
        result, reusable = chunk_document_block(text, ExerciseSectionFilter(), clean, chunk)
        self.assertTrue(reusable)
        self.assertEqual([x for x, _ in result], [clean(c) for c in chunk(clean(text)) if clean(c)])

    def test_exercise_role_survives_reload_and_is_not_study_evidence(self):
        self.assertEqual(normalize_chunk_role('exercise', 'A. A false answer.'), 'exercise')
        self.assertFalse(is_assignment_eligible_chunk_role('exercise'))
        self.assertEqual(classify_chunk_role('Esercizi Unità 2\n1. Calcola.'), 'exercise')

    def test_ingestion_and_general_retrieval_are_wired(self):
        source = Path('main.py').read_text()
        ast.parse(source)
        self.assertEqual(source.count('exercise_filter = ExerciseSectionFilter()'), 2)
        self.assertIn('if alignment_unchanged and block_index', source)
        tree = ast.parse(source)
        search = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'search_project_chunks')
        sql = ast.get_source_segment(source, search)
        self.assertEqual(sql.count("chunk_role <> 'exercise'"), 2)

    def test_missing_migration_fails_before_processing(self):
        from unittest.mock import Mock
        from chunk_roles import require_exercise_role_support
        db = Mock()
        db.get_bind.return_value.dialect.name = 'postgresql'
        db.execute.return_value.scalars.return_value.all.return_value = [
            "CHECK (chunk_role IN ('teaching', 'intro'))"]
        with self.assertRaisesRegex(ValueError, '20261102_exercise_chunk_role'):
            require_exercise_role_support(db)
        db.execute.return_value.scalars.return_value.all.return_value = [
            "CHECK (chunk_role IN ('teaching', 'exercise'))"]
        require_exercise_role_support(db)
        db.commit.assert_not_called()

    def test_exercise_unit_header_does_not_resume_teaching_on_next_page(self):
        detector = ExerciseSectionFilter()
        detector.split("Esercizi Unità 2\nQuestion text.\n")
        continuation = detector.split("Unità 2 - Moto in una dimensione 14\nOther options.\n")
        self.assertTrue(all(role == 'exercise' for _, role in continuation))

    def test_exercise_sql_guard_is_not_applied_to_topic_category_queries(self):
        tree = ast.parse(Path('main.py').read_text())
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in (
                'update_topic_category', 'rename_topic_category', 'get_topics'):
                self.assertNotIn("chunk_role", ast.get_source_segment(Path('main.py').read_text(), node))
