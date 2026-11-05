"""Isolated PostgreSQL upload publication tests; AI/OCR never use the network."""
import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import BackgroundTasks
from sqlalchemy import event, text

import main
from document_extractors import ExtractedDocument, ExtractedBlock
from document_acquisition import finish_report, _unit
from tests import test_taxonomy_coverage_pipeline as taxonomy_fixture
TEST_URL = taxonomy_fixture.TEST_URL


@unittest.skipUnless(TEST_URL, 'requires isolated DOUNO_TEST_DATABASE_URL')
class UploadPublicationTests(unittest.TestCase):
    session = taxonomy_fixture.TaxonomyPipelineTests.session
    cleanup_database = taxonomy_fixture.TaxonomyPipelineTests.cleanup_database
    create_schema = taxonomy_fixture.TaxonomyPipelineTests.create_schema
    chat = taxonomy_fixture.TaxonomyPipelineTests.chat
    embedding = taxonomy_fixture.TaxonomyPipelineTests.embedding
    assert_no_ai_transaction = taxonomy_fixture.TaxonomyPipelineTests.assert_no_ai_transaction

    def setUp(self):
        taxonomy_fixture.TaxonomyPipelineTests.setUp(self)
        with self.engine.begin() as db:
            db.execute(text('ALTER TABLE study_modules ADD COLUMN name text DEFAULT \'Module\', ADD COLUMN order_index integer DEFAULT 1'))
            db.execute(text('ALTER TABLE documents ADD COLUMN text text'))
            db.execute(text('ALTER TABLE chunks ADD COLUMN topic text'))
            db.execute(text('CREATE SEQUENCE chunk_ids START 1000'))
            db.execute(text("ALTER TABLE chunks ALTER COLUMN id SET DEFAULT nextval('chunk_ids')"))
        self.stack.enter_context(patch.dict(main._UPLOAD_DIAGNOSTIC_SUMMARIES, {}, clear=True))
        self.stack.enter_context(patch.object(main,'UploadPipelineLogger'))
        self.stack.enter_context(patch.object(main,'classify_chunk_role',return_value='teaching'))
        self.stack.enter_context(patch('document_acquisition.acquire_document',side_effect=self.acquire))
        self.reject = False
        self.background = BackgroundTasks()

    def acquire(self, data, filename):
        self.assertFalse(any(db.in_transaction() for db in self.sessions))
        doc=ExtractedDocument(filename,'pdf',len(data),[ExtractedBlock('Short educational fact.', 'Short educational fact.',1)],1)
        report=finish_report(filename,'page',[_unit(1,'',visual=True)] if self.reject else [_unit(1,'Short educational fact.')])
        return doc, report

    def run_upload(self, count=2):
        async def run():
            req=main.IngestRequest(module_id='m',documents=[main.IngestDocument(title=f'{i}.pdf',file_bytes=base64.b64encode(b'fixture').decode()) for i in range(count)])
            response=await main.ingest_stream('p',req,self.background,user={'id':'u'})
            return ''.join([line async for line in response.body_iterator])
        return asyncio.run(run())

    def snapshot(self):
        with self.engine.connect() as db:
            return tuple(tuple(db.execute(text(query)).fetchall()) for query in (
                'select * from documents order by id', 'select * from chunks order by id',
                'select * from topics order by id', 'select * from topic_chunks order by topic_id',
                "select status,taxonomy_status,accepted_for_study from study_modules where id='m'"))

    def test_publishes_all_documents_together_and_keeps_short_chunks(self):
        seen=[]
        def before(conn,cursor,statement,params,context,many):
            if 'insert into chunks' in statement.lower():
                with self.engine.connect() as observer:
                    seen.append(observer.execute(text('select count(*) from documents')).scalar())
        event.listen(self.engine,'before_cursor_execute',before)
        self.addCleanup(event.remove,self.engine,'before_cursor_execute',before)
        output=self.run_upload()
        self.assertNotIn('UPLOAD_FAILED',output)
        self.assertEqual(seen,[3])
        with self.engine.connect() as db:
            self.assertEqual(db.execute(text('select count(*) from documents')).scalar(),5)
            self.assertEqual(db.execute(text('select count(*) from chunks')).scalar(),2)
            self.assertEqual(db.execute(text('select chunk_text from chunks limit 1')).scalar(),'Short educational fact.')
        self.assertEqual(len(self.background.tasks),1)
        self.assertIn('short_text_activity_limit',output)

    def test_rejected_acquisition_preserves_prior_state(self):
        self.reject=True; before=self.snapshot()
        output=self.run_upload()
        self.assertIn('UPLOAD_FAILED',output)
        self.assertEqual(before,self.snapshot())
        self.assertEqual(len(self.background.tasks),0)

    def test_embedding_failure_preserves_prior_state(self):
        before=self.snapshot()
        with patch.object(self.client.embeddings,'create',side_effect=RuntimeError('mock embedding error')):
            output=self.run_upload()
        self.assertIn('UPLOAD_FAILED',output)
        self.assertEqual(before,self.snapshot())

    def test_sql_failure_after_document_inserts_rolls_back_everything(self):
        before=self.snapshot()
        def fail(conn,cursor,statement,params,context,many):
            if 'insert into chunks' in statement.lower():
                raise RuntimeError('injected publication failure')
        event.listen(self.engine,'before_cursor_execute',fail)
        self.addCleanup(event.remove,self.engine,'before_cursor_execute',fail)
        output=self.run_upload()
        self.assertIn('UPLOAD_FAILED',output)
        self.assertEqual(before,self.snapshot())

    def test_study_mode_approval_during_staging_prevents_publication(self):
        original=self.client.embeddings.create
        def approve(**kwargs):
            result=original(**kwargs)
            with self.engine.begin() as db:
                main._lock_taxonomy_scope(db,'p','m')
                db.execute(text("update study_modules set accepted_for_study=true,status='studying',taxonomy_status='ready' where id='m'"))
            return result
        with patch.object(self.client.embeddings,'create',side_effect=approve):
            output=self.run_upload(1)
        self.assertIn('UPLOAD_FAILED',output)
        with self.engine.connect() as db:
            self.assertEqual(db.execute(text('select count(*) from documents')).scalar(),3)
            self.assertTrue(db.execute(text("select accepted_for_study from study_modules where id='m'")).scalar())

    def test_interrupted_building_module_can_retry(self):
        with self.engine.begin() as db:
            db.execute(text("update study_modules set taxonomy_status='building' where id='m'"))
        self.assertNotIn('UPLOAD_FAILED',self.run_upload())

    def test_concurrent_publication_is_not_overwritten_by_failed_upload(self):
        original=self.client.embeddings.create
        def concurrent(**kwargs):
            result=original(**kwargs)
            with self.engine.begin() as db:
                db.execute(text("insert into documents(id,project_id,module_id,module_order_index,title) values ('concurrent','p','m',3,'Concurrent upload')"))
                db.execute(text("update study_modules set taxonomy_status='ready' where id='m'"))
            return result
        with patch.object(self.client.embeddings,'create',side_effect=concurrent):
            output=self.run_upload(1)
        self.assertIn('Module documents changed',output)
        with self.engine.connect() as db:
            self.assertEqual(db.execute(text('select count(*) from documents')).scalar(),4)
            self.assertEqual(db.execute(text("select taxonomy_status from study_modules where id='m'")).scalar(),'ready')

    def test_module_status_is_independent_of_project_status(self):
        with patch.object(main,'engine',self.engine):
            self.assertEqual(main.get_topic_status('p',user={'id':'u'},module_id='m')['status'],'completed')
            with self.engine.begin() as db:
                db.execute(text("update study_modules set taxonomy_status='building' where id='m'"))
            self.assertEqual(main.get_topic_status('p',user={'id':'u'},module_id='m')['status'],'processing')

    def test_later_rejected_file_does_not_publish_earlier_accepted_file(self):
        before=self.snapshot()
        original=self.acquire
        calls=[]
        def acquire(data,filename):
            calls.append(filename)
            self.reject = len(calls) == 2
            return original(data,filename)
        with patch('document_acquisition.acquire_document',side_effect=acquire):
            output=self.run_upload(2)
        self.assertIn('UPLOAD_FAILED',output)
        self.assertEqual(len(calls),2)
        self.assertEqual(before,self.snapshot())

    def test_exactly_ten_percent_failure_can_publish_with_warnings(self):
        original=self.acquire
        def acquire(data,filename):
            document,_=original(data,filename)
            return document,finish_report(filename,'page',[_unit(i,'Readable') for i in range(1,10)]+[_unit(10,'',visual=True)])
        with patch('document_acquisition.acquire_document',side_effect=acquire):
            output=self.run_upload(1)
        self.assertNotIn('UPLOAD_FAILED',output)
        self.assertIn('completed_with_warnings',output)
        self.assertEqual(len(self.background.tasks),1)

    def test_multiple_insert_batches_publish_in_one_transaction(self):
        original=self.acquire
        def acquire(data,filename):
            document,report=original(data,filename)
            document.blocks=[ExtractedBlock('Educational fact '+str(i),'Educational fact '+str(i),i+1) for i in range(205)]
            return document,report
        with patch('document_acquisition.acquire_document',side_effect=acquire):
            output=self.run_upload(1)
        self.assertNotIn('UPLOAD_FAILED',output)
        with self.engine.connect() as db:
            self.assertEqual(db.execute(text('select count(*) from chunks')).scalar(),205)
