"""Reuse the disposable PostgreSQL harness; never reads production configuration."""
import unittest
from sqlalchemy import text
from category_reorganization import preview, apply, _CACHE
from tests import test_document_postgres_types as fixture


class CategoryOnlyPostgresTests(unittest.TestCase):
    setUpClass = classmethod(fixture.DocumentPostgresTypesTests.setUpClass.__func__)
    setUp = fixture.DocumentPostgresTypesTests.setUp

    def test_uuid_topics_text_document_and_category_only_apply(self):
        _CACHE.clear()
        self.conn.execute(text('CREATE TEMP TABLE planner_weeks(id text,project_id uuid)'))
        self.conn.execute(text('INSERT INTO topic_chunks VALUES (:t,1)'),{'t':fixture.T1})
        settings=lambda db,m: {'organization_mode':'infer'}
        calls=[]
        def request(prompt,stage):
            calls.append(stage)
            return ({'categories':[{'name':'Revised subject','description':'Specific learning scope'}]}
                    if stage=='category_proposal' else
                    {'assignments':[{'topic_id':'T0001','category_id':'C0000'}],'quality_issues':[]})
        with self.ns['SessionLocal']() as db:
            p=preview(db,fixture.U,fixture.P,fixture.A,self.ns['_require_owned_project'],
                      self.ns['_lock_taxonomy_scope'],settings,request)
            result=apply(db,fixture.U,fixture.P,fixture.A,p['preview_id'],True,
                         self.ns['_require_owned_project'],self.ns['_lock_taxonomy_scope'],settings)
        self.assertEqual(result['topic_count'],1)
        self.assertEqual(len(calls),2)
        rows=self.conn.execute(text('SELECT id,category FROM topics')).fetchall()
        self.assertEqual({str(r[0]):r[1] for r in rows},
                         {fixture.T1:'Revised subject',fixture.T2:'SAME',fixture.TL:'SAME'})
        self.assertEqual(self.conn.execute(text('SELECT count(*) FROM topic_chunks')).scalar(),1)
