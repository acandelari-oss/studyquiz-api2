import contextlib
import io
import json
from types import SimpleNamespace
import unittest
from ingestion_text_safety import sanitize_acquired_text, safe_ingestion_row
from test_source_chunk_alignment import CLEAN, CHUNK

class TextSafetyTests(unittest.TestCase):
    def test_extraction_and_ocr_preserve_all_other_characters(self):
        text = 'α β x² 1.2\nè\tCO₂ → ∞'
        for position in range(len(text)+1):
            dirty = text[:position] + '\x00' + text[position:]
            block = SimpleNamespace(text=dirty, raw_text=dirty, section='A\x00B', page=3, block_index=2, digital_text=dirty)
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                sanitize_acquired_text(SimpleNamespace(blocks=[block]))
            self.assertEqual(block.text, text)
            self.assertEqual(block.raw_text, text)
            self.assertEqual(block.section, 'AB')
            self.assertEqual(block.digital_text, dirty)
            self.assertNotIn(text, output.getvalue())
            self.assertIn('"page": 3', output.getvalue())
            self.assertEqual(CLEAN(dirty), CLEAN(text))
            self.assertEqual(CHUNK(CLEAN(dirty)), CHUNK(CLEAN(text)))

    def test_final_rows_and_json_preserve_types_and_literal_escape(self):
        row = {'title': 'A\x00B', 'text': 'x²\nα', 'page': 3, 'topic': None,
               'chapter_structure': json.dumps({'title': 'C\x00D', 'literal': r'\u0000', 'nodes': [None, 2]})}
        cleaned = safe_ingestion_row(row, ('chapter_structure',))
        self.assertEqual(cleaned['title'], 'AB')
        self.assertEqual(cleaned['text'], row['text'])
        self.assertEqual(cleaned['page'], 3)
        self.assertIsNone(cleaned['topic'])
        self.assertEqual(json.loads(cleaned['chapter_structure']), {'title': 'CD', 'literal': r'\u0000', 'nodes': [None, 2]})
        self.assertEqual(row['title'], 'A\x00B')
