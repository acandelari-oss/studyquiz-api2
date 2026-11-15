"""Real PostgreSQL regression, isolated ephemeral cluster; never reads DATABASE_URL.
Requires initdb/pg_ctl on PATH. All fixtures are synthetic temporary tables.
Run: venv/bin/python -B -m unittest tests.test_document_postgres_types
"""
import asyncio
import json
import shutil
import subprocess
import tempfile
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
from sqlalchemy.exc import ProgrammingError

from category_identity import category_key
from taxonomy_assignment import source_scope, publish_prepared_topic_links
from taxonomy_document_state import replace_document_taxonomy, refresh_taxonomy_status
from planner.context_builder import _load_display_topics, _build_topic_context
from tests.test_document_taxonomy_isolation import load

P, M, U, A, B, T1, T2, TL = [str(uuid.UUID(int=i)) for i in range(1,9)]

class DocumentPostgresTypesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initdb, cls.pg_ctl = shutil.which('initdb'), shutil.which('pg_ctl')
        if not initdb or not cls.pg_ctl:
            raise unittest.SkipTest('PostgreSQL initdb/pg_ctl required; SQLite is not a substitute')
        cls.temp = tempfile.TemporaryDirectory(prefix='douno-pg-', dir='/tmp')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.data = str(Path(cls.temp.name) / 'data')
        subprocess.run([initdb,'-D',cls.data,'-U','douno_test','--auth=trust','--no-locale','--encoding=UTF8'],check=True,capture_output=True)
        subprocess.run([cls.pg_ctl,'-D',cls.data,'-l',str(Path(cls.temp.name)/'server.log'),'-o',
            f"-F -k {cls.temp.name} -c listen_addresses='' -p 55439",'-w','start'],check=True,capture_output=True)
        cls.addClassCleanup(lambda: subprocess.run([cls.pg_ctl,'-D',cls.data,'-m','immediate','-w','stop'],check=True,capture_output=True))
        cls.engine=create_engine(URL.create('postgresql+psycopg2',username='douno_test',database='postgres',query={'host':cls.temp.name,'port':'55439'}))
        cls.addClassCleanup(cls.engine.dispose)

    def setUp(self):
        self.conn=self.engine.connect(); self.addCleanup(self.conn.close)
        self.transaction=self.conn.begin();self.addCleanup(self.transaction.rollback)
        # Match catalog-inspected production ID types, including the absent topic/document FK.
        ddls=[
            'projects(id uuid primary key,user_id uuid,study_mode text,topic_status text,study_priority_categories jsonb)',
            'study_modules(id uuid primary key,project_id uuid,accepted_for_study boolean,status text,taxonomy_status text,name text,order_index integer,organization_mode text)',
            'documents(id text primary key,project_id uuid,module_id uuid,module_order_index integer,title text,taxonomy_status text,taxonomy_run_id uuid)',
            'chunks(id bigint primary key,project_id uuid,document_id text references documents(id),chunk_role text)',
            'topics(id uuid primary key,project_id uuid,document_id uuid,module_id uuid,category text,topic text,description text,is_display_topic boolean,source_section text,category_order_index integer,topic_order_index integer)',
            'topic_chunks(topic_id uuid references topics(id),chunk_id bigint references chunks(id))']
        for ddl in ddls:self.conn.execute(text('CREATE TEMP TABLE '+ddl))
        self.conn.execute(text("insert into projects values (:p,:u,'building','completed','[]')"),{'p':P,'u':U})
        self.conn.execute(text("insert into study_modules values (:m,:p,false,'pending_study','ready','Module',1,'infer')"),{'p':P,'m':M})
        for doc,title,order in [(A,'A.pdf',1),(B,'B.pdf',2),('legacy-file-name','Legacy.pdf',3)]:
            self.conn.execute(text("insert into documents values (:d,:p,:m,:i,:title,'ready',null)"),{'d':doc,'p':P,'m':M,'i':order,'title':title})
        for tid,doc,name in [(T1,A,'A objective'),(T2,B,'B objective'),(TL,None,'Legacy objective')]:
            self.conn.execute(text("insert into topics values (:t,:p,:d,:m,'SAME',:name,'description',true,null,1,1)"),{'t':tid,'p':P,'d':doc,'m':M,'name':name})
        self.conn.execute(text("insert into chunks values (1,:p,:a,'teaching'),(2,:p,:b,'teaching')"),{'p':P,'a':A,'b':B})
        self.ns={'Depends':lambda x:None,'verify_user':None,'text':text,'sql_text':text,'HTTPException':HTTPException,'json':json,'category_key':category_key,
            'SessionLocal':lambda:Session(bind=self.conn,join_transaction_mode='create_savepoint'),
            '_table_column_available':lambda *a:True,'_study_module_organization_columns_available':lambda *a:True,
            'REVIEW_MARKER':'__review__','MAX_STUDY_PRIORITY_CATEGORIES':3,
            'replace_document_taxonomy':replace_document_taxonomy,'refresh_taxonomy_status':refresh_taxonomy_status}
        load({'get_topics','_lock_taxonomy_scope','update_topic_category','_require_owned_project',
              'update_project_study_priorities','delete_document'},self.ns)

    def test_original_join_fails_on_real_postgres(self):
        with self.conn.begin_nested() as savepoint:
            with self.assertRaises(ProgrammingError) as error:
                self.conn.execute(text('select t.id from topics t left join documents d on d.id=t.document_id'))
            self.assertEqual(error.exception.orig.pgcode,'42883')
            savepoint.rollback()

    def test_http_topics_returns_owned_documents_and_null_legacy(self):
        app=FastAPI();app.get('/projects/{project_id}/topics')(self.ns['get_topics'])
        with TestClient(app) as client:
            response=client.get(f'/projects/{P}/topics')
        self.assertEqual(response.status_code,200,response.text)
        topics=response.json()['topics']
        self.assertEqual([t['id'] for t in topics],[T1,T2,TL])
        self.assertEqual([t['document_id'] for t in topics],[A,B,None])
        self.assertEqual([t['document_name'] for t in topics],['A.pdf','B.pdf',None])
        self.assertEqual(len({t['category_key'] for t in topics}),3)

    def test_planner_and_priorities_use_same_qualified_ownership(self):
        self.conn.execute(text('update study_modules set accepted_for_study=true'))
        with self.ns['SessionLocal']() as db:
            categories,groups,_=_build_topic_context(_load_display_topics(db,P))
        self.assertEqual(len(categories),3)
        self.assertEqual({t.id for values in groups.values() for t in values},{T1,T2,TL})
        selected=category_key('SAME',M,A,'A.pdf')
        req=SimpleNamespace(priority_categories=[selected],priorityCategories=None)
        result=self.ns['update_project_study_priorities'](P,req,{'id':U})
        self.assertEqual(result['priorityCategories'],[selected])

    def test_uuid_from_topic_row_can_lock_text_document_and_edit(self):
        native_id=self.conn.execute(text('select document_id from topics where id=:t'),{'t':T1}).scalar()
        self.assertIsInstance(native_id,uuid.UUID)
        with self.ns['SessionLocal']() as db:
            self.ns['_lock_taxonomy_scope'](db,P,M,native_id,require_editable=True)
        request=SimpleNamespace(category=None,topic='Renamed A')
        asyncio.run(self.ns['update_topic_category'](P,T1,request,{'id':U}))
        self.assertEqual(self.conn.execute(text('select topic from topics where id=:t'),{'t':T2}).scalar(),'B objective')

    def test_source_scope_assignment_and_deletion_use_compatible_types(self):
        join,clause,params=source_scope(P,M,uuid.UUID(A))
        self.assertEqual(self.conn.execute(text(f'select c.id from chunks c {join} where c.project_id=:project_id {clause}'),params).scalars().all(),[1])
        with self.ns['SessionLocal']() as db:
            publish_prepared_topic_links(db,P,M,uuid.UUID(A),{T1:{1}},None)
            db.commit()
        self.ns['delete_document'](P,A,{'id':U})
        self.assertEqual(set(self.conn.execute(text('select id from topics')).scalars()),{uuid.UUID(T2),uuid.UUID(TL)})
        self.assertEqual(self.conn.execute(text('select count(*) from chunks where document_id=:d'),{'d':B}).scalar(),1)

    def test_publication_positions_with_text_bound_document_id(self):
        import ast
        tree = ast.parse(Path('main.py').read_text())
        task = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'process_topics_task')
        env = {'topic_alias_filter': 'and t.module_id = :module_id',
               'chunk_alias_filter': 'and d.module_id = :module_id'}
        for node in ast.walk(task):
            if (isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name)
                    and node.target.id in env):
                exec(compile(ast.Module(body=[node], type_ignores=[]), '<scope>', 'exec'), env)
        query_node = next(n for n in ast.walk(task) if isinstance(n, ast.JoinedStr)
                          and any(isinstance(v, ast.Constant) and isinstance(v.value, str)
                                  and 'with ordered_chunks as' in v.value for v in n.values))
        query = eval(compile(ast.Expression(query_node), '<publication>', 'eval'), env)
        self.conn.execute(text('alter table chunks add column page integer'))
        self.conn.execute(text('alter table topics add column first_source_page integer'))
        self.conn.execute(text('alter table topics add column first_source_block_index integer'))
        self.conn.execute(text('update chunks set page=id'))
        self.conn.execute(text('insert into topic_chunks values (:a,1),(:b,2)'), {'a':T1,'b':T2})
        # PostgreSQL must accept one TEXT parameter in both TEXT and UUID filters,
        # as in the production psycopg 3 request. No live database is used.
        prepared = query.replace(':project_id','$1').replace(':module_id','$2').replace(':document_id','$3')
        self.conn.exec_driver_sql('PREPARE publication_positions(uuid, uuid, text) AS '+prepared)
        try:
            self.conn.exec_driver_sql(f"EXECUTE publication_positions('{P}', '{M}', '{A}')")
            rows = self.conn.execute(text('select id, first_source_page, first_source_block_index from topics')).all()
            positions = {str(row[0]): tuple(row[1:]) for row in rows}
            self.assertEqual(positions[T1], (1,1))
            self.assertEqual(positions[T2], (None,None))
            self.assertEqual(positions[TL], (None,None))
        finally:
            self.conn.exec_driver_sql('DEALLOCATE publication_positions')

    def test_chapter_metadata_snapshot_is_document_scoped_and_revision_sensitive(self):
        from taxonomy_assignment import source_snapshot
        for column in ('page integer','chunk_text text','section text','embedding text'):
            self.conn.execute(text('alter table chunks add column '+column))
        before=source_snapshot(self.conn,P,M,A)
        self.conn.execute(text('alter table documents add column chapter_structure jsonb'))
        self.conn.execute(text('alter table chunks add column structural_provenance jsonb'))
        self.assertEqual(before,source_snapshot(self.conn,P,M,A))
        self.conn.execute(text("update documents set chapter_structure='{}' where id=:b"),{'b':B})
        self.assertEqual(before,source_snapshot(self.conn,P,M,A))
        self.conn.execute(text("update documents set chapter_structure='{}' where id=:a"),{'a':A})
        changed=source_snapshot(self.conn,P,M,A)
        self.assertNotEqual(before,changed)
        self.conn.execute(text("update chunks set structural_provenance='{}' where document_id=:a"),{'a':A})
        self.assertNotEqual(changed,source_snapshot(self.conn,P,M,A))
        self.assertEqual(set(before),{1})
