import copy
import json
import os
import unittest
from unittest.mock import patch
from sqlalchemy import text
from fastapi import HTTPException
import taxonomy_assignment as assignment
from taxonomy_source_review import prepare_source_review, resolve_source_review
from test_document_taxonomy_isolation import DocumentIsolationTests, load

class SourceReviewTests(unittest.TestCase):
    def setUp(self):
        self.h = DocumentIsolationTests()
        self.h.setUp()
        self.addCleanup(self.h.doCleanups)
        self.env = patch.dict(os.environ, {'DOUNO_SOURCE_REVIEW':'1'})
        self.env.start(); self.addCleanup(self.env.stop)
        with self.h.engine.begin() as db:
            db.execute(text('ALTER TABLE documents ADD COLUMN source_review text'))
        self.h.prepare_partial_index()
        def plan(db, candidates, project, score, **kw):
            snapshot = assignment.source_snapshot(db, project, kw['module_id'], kw['document_id'])
            return {t['assignment_id']: ({2} if t['chapter_id']=='index-0' else set()) for t in candidates}, snapshot
        self.h.ns['plan_assignments'] = plan
        load({'get_document_source_review','update_document_source_review','_require_owned_project'}, self.h.ns)

    def publish(self):
        self.assertEqual(self.h.batch('b'), {'b':True}, self.h.output.getvalue())
        return json.loads(self.h.rows("select source_review from documents where id='b'")[0][0])

    def test_recovery_reason_survives_publication_and_read_api(self):
        with patch.dict(os.environ, {'DOUNO_SOURCE_RECOVERY_MAX_CALLS': '0'}):
            report = self.publish()
        pending = next(e for e in report['entries'] if e['status'] == 'pending')
        self.assertEqual(pending['reason'], 'call_budget')
        self.assertEqual(report['recovery']['calls'], 0)
        result = self.h.ns['get_document_source_review']('p', 'b', user={'id': 'u'})
        self.assertEqual(result['items'][0]['reason'], 'call_budget')

    def test_atomic_publication_preserves_sources_and_siblings(self):
        before = self.h.rows('select * from chunks')
        siblings = self.h.rows("select * from topics where id in ('ta','legacy')")
        report = self.publish()
        self.assertEqual([e['status'] for e in report['entries']], ['linked','pending'])
        self.assertEqual(len(report['unsupported_candidates']),1)
        self.assertEqual(len(self.h.rows("select * from topics where document_id='b'")),1)
        self.assertEqual(before, self.h.rows('select * from chunks'))
        self.assertEqual(siblings, self.h.rows("select * from topics where id in ('ta','legacy')"))
        self.assertIsNone(self.h.rows("select source_review from documents where id='a'")[0][0])
        result=self.h.ns['get_document_source_review']('p','b',user={'id':'u'})
        self.assertEqual(result['pending_count'],1)
        self.assertEqual(result['items'][0]['text'],'Boundary material')

    def test_review_write_failure_rolls_back_topics_and_links(self):
        before=self.h.rows('select * from topics'); links=self.h.rows('select * from topic_chunks')
        with patch('taxonomy_source_review.persist_source_review',side_effect=RuntimeError('disk/db failure')):
            self.assertEqual(self.h.batch('b'), {'b':False})
        self.assertEqual(before,self.h.rows('select * from topics'))
        self.assertEqual(links,self.h.rows('select * from topic_chunks'))
        self.assertIsNone(self.h.rows("select source_review from documents where id='b'")[0][0])

    def test_source_review_link_preserves_topic_identity_and_checks_stale_revision(self):
        self.publish()
        topic=self.h.rows("select id from topics where document_id='b'")[0][0]
        before=self.h.rows('select * from topics')
        result=self.h.ns['update_document_source_review']('p','b',4,
            {'revision':1,'action':'link','topic_id':topic},user={'id':'u'})
        self.assertEqual(result['pending_count'],0)
        self.assertEqual(before,self.h.rows('select * from topics'))
        self.assertIn((topic,4),self.h.rows('select * from topic_chunks'))
        with self.assertRaises(HTTPException):
            self.h.ns['update_document_source_review']('p','b',4,
                {'revision':1,'action':'outline','note':'outline'},user={'id':'u'})

    def test_ownership_study_lock_foreign_topic_and_source_change_rejected(self):
        self.publish()
        endpoint=self.h.ns['update_document_source_review']
        payload={'revision':1,'action':'link','topic_id':'ta'}
        for user in ({'id':'foreign'},{'id':'u'}):
            with self.assertRaises(HTTPException): endpoint('p','b',4,payload,user=user)
        with self.h.engine.begin() as db: db.execute(text('update study_modules set accepted_for_study=true'))
        with self.assertRaises(HTTPException): endpoint('p','b',4,{'revision':1,'action':'outline','note':'outline'},user={'id':'u'})
        with self.h.engine.begin() as db:
            db.execute(text('update study_modules set accepted_for_study=false'))
            db.execute(text("update chunks set chunk_text='changed' where id=4"))
        with self.assertRaises(HTTPException): endpoint('p','b',4,{'revision':1,'action':'outline','note':'outline'},user={'id':'u'})

    def test_outline_requires_explicit_confirmation_preserves_text(self):
        self.publish(); endpoint=self.h.ns['update_document_source_review']
        with self.assertRaises(HTTPException): endpoint('p','b',4,{'revision':1,'action':'outline'},user={'id':'u'})
        result=endpoint('p','b',4,{'revision':1,'action':'outline','note':'Only section headings, no explanation'},user={'id':'u'})
        self.assertEqual(result['pending_count'],0)
        self.assertEqual(self.h.rows('select chunk_text from chunks where id=4'), [('Boundary material',)])

    def test_flag_off_preserves_strict_failure(self):
        with patch.dict(os.environ,{'DOUNO_SOURCE_REVIEW':'0'}):
            self.assertEqual(self.h.batch('b'), {'b':False})

    def test_missing_migration_fails_before_ai(self):
        self.h.ns['_table_column_available']=lambda *a:False
        self.assertEqual(self.h.batch('b'), {'b':False})
        self.assertEqual(self.h.calls,[])

    def test_invalid_links_still_rejected_and_inputs_unchanged(self):
        candidates=[dict(assignment_id='x',topic='X',description='D',evidence_ids=[9],discovery_chunk_ids=[1],chapter_id='a')]
        before=copy.deepcopy(candidates)
        with self.assertRaises(ValueError): prepare_source_review(candidates,[{'chunk_id':1}],{1:'s'},{1:'a'})
        self.assertEqual(candidates,before)

    def test_verified_chapter_cannot_link_to_other_chapter(self):
        self.publish()
        with self.h.engine.begin() as db:
            raw=json.loads(db.execute(text("select source_review from documents where id='b'")).scalar())
            raw['entries'][1]['chapter_id']='other-chapter'
            db.execute(text('update documents set source_review=:r where id=\'b\''),{'r':json.dumps(raw)})
        topic=self.h.rows("select id from topics where document_id='b'")[0][0]
        with self.assertRaises(HTTPException):
            self.h.ns['update_document_source_review']('p','b',4,{'revision':1,'action':'link','topic_id':topic},user={'id':'u'})
        result=self.h.ns['get_document_source_review']('p','b',user={'id':'u'})
        self.assertEqual(result['items'][0]['topic_ids'],[])

    def test_review_records_deleted_with_document(self):
        self.publish()
        with self.assertRaises(HTTPException):
            self.h.ns['delete_document']('p','b',user={'id':'u'})
        # Remove only the fixture's intentional legacy cross-document link.
        with self.h.engine.begin() as db:
            db.execute(text("delete from topic_chunks where topic_id='legacy' and chunk_id=2"))
        self.h.ns['delete_document']('p','b',user={'id':'u'})
        self.assertEqual(self.h.rows("select source_review from documents where id='b'"),[])
        self.assertEqual(self.h.rows("select id from topics where id='ta'"),[('ta',)])

    def test_review_link_does_not_block_later_valid_chapter_link(self):
        from taxonomy_source_review import snapshot_token
        self.publish()
        topic=self.h.rows("select id from topics where document_id='b'")[0][0]
        # Extend only this disposable fixture with another pending chapter-local source.
        with self.h.engine.begin() as db:
            db.execute(text("INSERT INTO chunks(id,project_id,document_id,chunk_text,page,chunk_role,embedding) VALUES(5,'p','b','More evidence',1,'teaching','[1,0]')"))
        with self.h.factory() as db:
            report=json.loads(db.execute(text("select source_review from documents where id='b'")).scalar())
            report['entries'].append({'chunk_id':5,'status':'pending','reason':'no_validated_topic_link','chapter_id':'index-0'})
            report['source_token']=snapshot_token(assignment.source_snapshot(db,'p','m','b'))
            db.execute(text("update documents set source_review=:r where id='b'"),{'r':json.dumps(report)});db.commit()
        patch_review=self.h.ns['update_document_source_review']
        patch_review('p','b',4,{'revision':1,'action':'link','topic_id':topic},user={'id':'u'})
        response=self.h.ns['get_document_source_review']('p','b',user={'id':'u'})
        self.assertIn(topic,response['items'][0]['topic_ids'])
        result=patch_review('p','b',5,{'revision':2,'action':'link','topic_id':topic},user={'id':'u'})
        self.assertEqual(result['pending_count'],0)
        report=json.loads(self.h.rows("select source_review from documents where id='b'")[0][0])
        self.assertTrue(next(e for e in report['entries'] if e['chunk_id']==4)['chapter_id'].startswith('index-review-'))
        self.assertEqual({r[0] for r in self.h.rows("select chunk_id from topic_chunks where topic_id='"+topic+"'")},{2,4,5})

    def test_mixed_or_unknown_authoritative_scopes_still_fail(self):
        from taxonomy_source_review import chapter_accepts_source
        self.assertTrue(chapter_accepts_source('a',{'a','index-review-1'}))
        for scopes in ({'a','b','index-review-1'}, {None,'a'}, {'index-review-1'}, set()):
            self.assertFalse(chapter_accepts_source('a',scopes))
