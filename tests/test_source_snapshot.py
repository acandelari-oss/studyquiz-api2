"""Offline SQL shape, fingerprint and legacy-schema checks."""
import hashlib
import json
import unittest
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import Session
from taxonomy_assignment import source_snapshot


class SnapshotSession(Session):
    def execute(self, statement, *args, **kwargs):
        return super().execute(text(str(statement).replace('c.embedding::text', 'CAST(c.embedding AS text)')), *args, **kwargs)


class SourceSnapshotTests(unittest.TestCase):
    def setup_db(self, metadata=True):
        engine = create_engine('sqlite://')
        self.addCleanup(engine.dispose)
        db = SnapshotSession(engine)
        self.addCleanup(db.close)
        db.execute(text('CREATE TABLE documents(id text, project_id text, module_id text, text text' + (', chapter_structure text' if metadata else '') + ')'))
        db.execute(text('CREATE TABLE chunks(id integer, document_id text, project_id text, page integer, chunk_text text, section text, embedding text, chunk_role text' + (', structural_provenance text' if metadata else '') + ')'))
        db.execute(text("INSERT INTO documents(id,project_id,module_id,text) VALUES ('a','p','m','large unused document'), ('b','p','m','sibling')"))
        db.execute(text("INSERT INTO chunks(id,document_id,project_id,page,chunk_text,section,embedding,chunk_role) VALUES (1,'a','p',1,'body','section','[1,2]','teaching'), (2,'a','p',2,'more','section','[3,4]','teaching'), (3,'b','p',1,'sibling','other','[5,6]','teaching')"))
        db.commit()
        return db, engine

    def test_matches_old_fingerprint_and_reads_document_metadata_once(self):
        db, engine = self.setup_db()
        statements = []
        event.listen(engine, 'before_cursor_execute', lambda conn, cursor, sql, params, context, many: statements.append(sql))
        result = source_snapshot(db, 'p', 'm', 'a')
        expected = hashlib.sha256(json.dumps([1,'a',1,'body','section','[1,2]',None,None], ensure_ascii=False).encode()).hexdigest()
        self.assertEqual(result[1], expected)
        self.assertEqual(set(result), {1,2})
        reads = [s for s in statements if 'select d.id,' in s]
        self.assertEqual(len(reads), 1)
        self.assertFalse(any('to_jsonb' in s or 'd.text' in s or 'select *' in s.lower() for s in statements))

    def test_all_original_snapshot_fields_still_detect_changes(self):
        db, _ = self.setup_db()
        baseline = source_snapshot(db, 'p', 'm', 'a')
        for column, value in [('page',9),('chunk_text','changed'),('section','new'),('embedding','[9]'),('structural_provenance','{"changed":true}')]:
            with db.begin_nested() as savepoint:
                db.execute(text('UPDATE chunks SET '+column+'=:value WHERE id=1'), {'value': value})
                self.assertNotEqual(source_snapshot(db, 'p', 'm', 'a')[1], baseline[1])
                savepoint.rollback()
        db.execute(text("UPDATE documents SET chapter_structure='new revision' WHERE id='a'"))
        updated = source_snapshot(db, 'p', 'm', 'a')
        self.assertTrue(all(updated[k] != baseline[k] for k in baseline))

    def test_legacy_schema_and_module_wide_coverage(self):
        db, _ = self.setup_db(metadata=False)
        self.assertEqual(set(source_snapshot(db, 'p', 'm')), {1,2,3})
        self.assertEqual(set(source_snapshot(db, 'p', 'm', 'a')), {1,2})
        self.assertEqual(source_snapshot(db, 'wrong', 'm', 'a'), {})

class PostgreSQLSnapshotContractTests(unittest.TestCase):
    def snapshot(self, chunk_hash='chunkhash', document_hash='dochash', columns=True):
        from unittest.mock import Mock
        db = Mock()
        db.get_bind.return_value.dialect.name = 'postgresql'
        db.execute.side_effect = [
            [('chunks','structural_provenance'),('documents','chapter_structure')] if columns else [],
            [('a',document_hash)], [(1,'a',chunk_hash)]]
        result = source_snapshot(db, 'p', 'm', 'a')
        return result, [str(call.args[0]) for call in db.execute.call_args_list]

    def test_server_hash_covers_every_field_and_scope(self):
        result, statements = self.snapshot()
        self.assertEqual(set(result), {1})
        chunk_sql = statements[2]
        self.assertIn('jsonb_build_array(c.id,c.document_id,c.page,c.chunk_text,c.section,c.embedding::text,c.structural_provenance)', chunk_sql)
        self.assertIn("encode(sha256(convert_to(", chunk_sql)
        self.assertIn('d.module_id=:module_id', chunk_sql)
        self.assertIn('c.document_id=:document_id', chunk_sql)
        self.assertIn("c.chunk_role='teaching'", chunk_sql)
        self.assertIn('jsonb_build_array(d.chapter_structure)', statements[1])
        self.assertNotIn('to_jsonb', ' '.join(statements))

    def test_both_document_and_chunk_changes_invalidate_snapshot(self):
        baseline, _ = self.snapshot()
        self.assertNotEqual(baseline, self.snapshot(chunk_hash='changed')[0])
        self.assertNotEqual(baseline, self.snapshot(document_hash='changed')[0])
        self.assertEqual(baseline, self.snapshot()[0])

    def test_legacy_metadata_columns_are_optional(self):
        _, statements = self.snapshot(columns=False)
        self.assertIn('jsonb_build_array(NULL)', statements[1])
        self.assertIn('c.embedding::text,NULL)', statements[2])
