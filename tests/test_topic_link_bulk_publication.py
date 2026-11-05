"""Offline bulk publication and rollback; no application or external DB."""
import unittest
from unittest.mock import patch
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import Session
from taxonomy_assignment import publish_prepared_topic_links


class BulkPublicationTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine('sqlite:///:memory:')
        self.addCleanup(self.engine.dispose)
        with self.engine.begin() as db:
            for sql in [
                'create table documents(id text primary key, project_id text, module_id text)',
                'create table topics(id text primary key,project_id text,module_id text,document_id text)',
                'create table chunks(id integer primary key,project_id text,document_id text,chunk_role text)',
                'create table topic_chunks(topic_id text,chunk_id integer,primary key(topic_id,chunk_id))',
                "insert into documents values ('a','p','m'),('b','p','m')",
                "insert into topics values ('sibling','p','m','a'),('target','p','m','b')",
                "insert into chunks values (9999,'p','a','teaching')",
                "insert into topic_chunks values ('sibling',9999)",
            ]:
                db.execute(text(sql))
            db.execute(text("insert into chunks values (:id,'p','b','teaching')"),
                       [{'id': i} for i in range(1, 2655)])
        self.inserts = []
        @event.listens_for(self.engine, 'before_cursor_execute')
        def capture(conn, cursor, statement, parameters, context, executemany):
            if statement.lower().startswith('insert into topic_chunks'):
                self.inserts.append((statement, executemany))
        self.capture = capture

    def test_2654_links_use_seven_inserts_and_preserve_sibling(self):
        with Session(self.engine) as db:
            publish_prepared_topic_links(db, 'p', 'm', 'b', {'target': set(range(1, 2655))}, None)
            self.assertTrue(db.in_transaction())
            db.commit()
        self.assertEqual(len(self.inserts), 7)
        self.assertTrue(all(not many for _, many in self.inserts))
        with self.engine.connect() as db:
            self.assertEqual(db.execute(text('select count(*) from topic_chunks')).scalar(), 2655)
            self.assertEqual(list(db.execute(text("select chunk_id from topic_chunks where topic_id='sibling'"))), [(9999,)])

    def test_middle_batch_failure_rolls_back_all_target_links(self):
        @event.listens_for(self.engine, 'before_cursor_execute')
        def fail(conn, cursor, statement, parameters, context, executemany):
            if len(self.inserts) == 2 and statement.lower().startswith('insert into topic_chunks'):
                raise RuntimeError('Injected second batch failure')
        with Session(self.engine) as db:
            with self.assertRaises(RuntimeError):
                publish_prepared_topic_links(db, 'p', 'm', 'b', {'target': set(range(1, 2655))}, None)
            db.rollback()
        with self.engine.connect() as db:
            self.assertEqual(list(db.execute(text('select * from topic_chunks'))), [('sibling', 9999)])

    def test_cross_document_chunk_rejected_before_insert(self):
        with Session(self.engine) as db:
            with self.assertRaisesRegex(ValueError, 'out-of-scope'):
                publish_prepared_topic_links(db, 'p', 'm', 'b', {'target': {9999}}, None)
        self.assertFalse(self.inserts)

    def test_deadline_after_insert_rolls_back(self):
        with Session(self.engine) as db, patch('time.monotonic', side_effect=[0, 0, 2]):
            with self.assertRaises(TimeoutError):
                publish_prepared_topic_links(db, 'p', 'm', 'b', {'target': {1}}, 1)
            db.rollback()
        with self.engine.connect() as db:
            self.assertEqual(list(db.execute(text('select * from topic_chunks'))), [('sibling', 9999)])
