"""Exercise real scope filtering with SQLite; only vector scoring is substituted."""
import unittest
from unittest.mock import patch
from sqlalchemy import create_engine, text, bindparam
import main


class ApprovedLearningScopeTests(unittest.TestCase):
    def test_global_and_explicit_scope_exclude_draft_modules(self):
        engine = create_engine('sqlite://')
        db = engine.connect()
        for sql in [
            'create table study_modules (id text, project_id text, accepted_for_study boolean)',
            'create table topics (id text, project_id text, module_id text, topic text, source_section text, embedding text)',
            'create table chunks (id integer, project_id text, chunk_text text, doc_title text, page integer, chunk_role text, section text, embedding text)',
            'create table topic_chunks (topic_id text, chunk_id integer)',
            "insert into study_modules values ('approved', 'p', true), ('draft', 'p', false)",
            "insert into topics values ('a', 'p', 'approved', 'Approved topic', '', ''), ('b', 'p', 'draft', 'Draft topic', '', '')",
            "insert into topic_chunks values ('a', 1), ('b', 2)",
        ]:
            db.execute(text(sql))
        for chunk_id in (1, 2):
            db.execute(text("insert into chunks values (:id, 'p', :body, 'notes.pdf', 1, 'teaching', '', '')"), {'id': chunk_id, 'body': 'Study material. ' * 20})

        class ScopeSession:
            def execute(self, statement, params):
                sql = str(statement).replace('c.embedding <#> t.embedding', '0')
                query = text(sql)
                if 'topic_ids' in params:
                    query = query.bindparams(bindparam('topic_ids', expanding=True))
                return db.execute(query, params)

            def close(self):
                pass

        try:
            with patch.object(main, 'SessionLocal', return_value=ScopeSession()), patch.object(main, 'calculate_topic_chunk_score', return_value=1):
                for topic_ids in ([], ['a', 'b'], ['b']):
                    with self.subTest(topic_ids=topic_ids):
                        scope = main.resolve_learning_scope('p', topic_ids)
                        self.assertEqual([c['chunk_id'] for c in scope['chunks']], [] if topic_ids == ['b'] else ['1'])
        finally:
            db.close()
            engine.dispose()
