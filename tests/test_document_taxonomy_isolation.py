"""Execute production functions against disposable in-memory SQLite, with AI mocked.
No application startup, external database, ingestion or network requests.
"""
import ast
import asyncio
import copy
import io
import json
import os
import re
import time
import traceback
import unittest
import uuid
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Optional
from unittest.mock import Mock, patch

from fastapi import HTTPException
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker
from category_identity import category_key
import taxonomy_assignment as assignment
import taxonomy_coverage as coverage
import taxonomy_document_state as state
from taxonomy_organization import organize_categories, category_distribution, parse_organization_response, REVIEW_MARKER

TREE=ast.parse(Path('main.py').read_text())

def load(names, ns):
    for node in TREE.body:
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)) and node.name in names:
            node=copy.deepcopy(node)
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),'main.py','exec',flags=__import__('__future__').annotations.compiler_flag),ns)

class SQLiteSession(Session):
    def execute(self, statement, *args, **kwargs):
        sql=str(statement).replace('c.embedding <#> t.embedding','-1.0').replace('c.embedding::text','CAST(c.embedding AS text)').replace('update topics t\n','update topics AS t\n')
        sql = sql.replace("to_jsonb(chunks)->'structural_provenance'", "chunks.structural_provenance")
        sql = sql.replace("to_jsonb(c)->'structural_provenance'", "c.structural_provenance")
        sql = sql.replace("to_jsonb(d)->'chapter_structure'", "d.chapter_structure")
        return super().execute(text(sql),*args,**kwargs)

class DocumentIsolationTests(unittest.TestCase):
    def test_automatic_macrocategory_persisted_without_changing_sibling(self):
        before = self.rows("select * from topics where document_id='a'")
        self.assertEqual(self.batch('b'), {'b':True})
        self.assertEqual(self.rows("select distinct source_section from topics where document_id='b'"), [('Parent',)])
        self.assertEqual(self.rows("select * from topics where document_id='a'"), before)

    def setUp(self):
        self.stack=ExitStack(); self.addCleanup(self.stack.close)
        self.stack.enter_context(patch('httpx.Client.send',side_effect=AssertionError('Network forbidden')))
        self.stack.enter_context(patch.dict(os.environ,{'DOUNO_GLOBAL_CATEGORY_ORGANIZATION':'1'}))
        self.output=io.StringIO(); self.stack.enter_context(redirect_stdout(self.output))
        self.engine=create_engine('sqlite:///:memory:')
        self.addCleanup(self.engine.dispose)
        self.factory=sessionmaker(bind=self.engine,class_=SQLiteSession)
        with self.engine.begin() as c:
            for ddl in [
                'CREATE TABLE projects(id text primary key,user_id text,study_mode text,topic_status text,taxonomy_language text)',
                'CREATE TABLE study_modules(id text primary key,project_id text,accepted_for_study boolean,status text,taxonomy_status text,name text,order_index integer)',
                'CREATE TABLE documents(id text primary key,project_id text,module_id text,module_order_index integer,title text,taxonomy_status text,taxonomy_run_id text)',
                'CREATE TABLE chunks(id integer primary key,project_id text,document_id text,doc_title text,chunk_text text,section text,page integer,chunk_role text,embedding text)',
                'CREATE TABLE topics(id text primary key,project_id text,document_id text,module_id text,category text,topic text,description text,embedding text,is_display_topic boolean,source_section text,category_order_index integer,topic_order_index integer,first_source_page integer,first_source_block_index integer)',
                'CREATE TABLE topic_chunks(topic_id text,chunk_id integer)']:
                c.execute(text(ddl))
            c.execute(text("INSERT INTO projects VALUES ('p','u','building','completed',null)"))
            c.execute(text("INSERT INTO study_modules VALUES ('m','p',false,'pending_study','ready','Module',1)"))
            for i,doc in enumerate(('a','b','c'),1):
                c.execute(text("INSERT INTO documents VALUES (:doc,'p','m',:i,:doc,:status,null)"),{'doc':doc,'i':i,'status':'ready' if doc=='a' else None})
                c.execute(text("INSERT INTO chunks VALUES (:i,'p',:doc,:doc,:body,'SECTION',1,'teaching','[1,0]')"),{'doc':doc,'i':i,'body':f'PRIVATE_{doc} instructional source'})
            c.execute(text("INSERT INTO topics(id,project_id,document_id,module_id,category,topic,description,embedding,is_display_topic) VALUES ('ta','p','a','m','SAME','A','A evidence','[1,0]',true),('legacy','p',null,'m','SAME','Legacy','Mixed evidence','[1,0]',true)"))
            c.execute(text("INSERT INTO topic_chunks VALUES ('ta',1),('legacy',1),('legacy',2)"))
            c.execute(text("ALTER TABLE documents ADD COLUMN chapter_structure text"))
            c.execute(text("ALTER TABLE chunks ADD COLUMN structural_provenance text"))
        self.calls=[]; self.fail_doc=None
        client=SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=self.chat)))
        client.with_options=lambda **kw:client
        self.ns=dict(vars(coverage)); self.ns.update(vars(state))
        self.ns.update({'__name__':'isolated_test','time':time,'uuid':uuid,'json':json,'os':os,'Optional':Optional,
            'category_key':category_key,'text':text,'sql_text':text,'traceback':traceback,'HTTPException':HTTPException,
            'Depends':lambda x:None,'verify_user':None,'SessionLocal':self.factory,'client':client,'cost':SimpleNamespace(call=lambda fn,**kw:fn(**{k:v for k,v in kw.items() if not k.startswith('cost_')}),enabled=lambda:False,metrics=lambda **kw:None),
            'source_snapshot':assignment.source_snapshot,'plan_assignments':self.plan,'validate_evidence':assignment.validate_evidence,
            'publish_prepared_topic_links':assignment.publish_prepared_topic_links,
            'organize_categories':organize_categories,'category_distribution':category_distribution,'parse_organization_response':parse_organization_response,'REVIEW_MARKER':REVIEW_MARKER,
            'MIN_TAXONOMY_LANGUAGE_CONFIDENCE':1,'TAXONOMY_PLANNER_MODEL':'mock','TAXONOMY_PLANNER_REQUEST_TIMEOUT_SECONDS':240,'TAXONOMY_PLANNER_MAX_OUTPUT_TOKENS':3000,
            'TAXONOMY_PUBLICATION_TIMEOUT_SECONDS':120,'PRIMARY_ASSIGNMENT_THRESHOLD':.54,'MAX_TOPICS_PER_CHUNK':3,'TOPIC_RESCUE_THRESHOLD':.46,'MAX_ASSIGNMENT_MATCHES':30000,
            '_UPLOAD_DIAGNOSTIC_SUMMARIES':{},'calculate_topic_chunk_score':None,'build_immutable_taxonomy_ledger':self.ledger,
            'detect_project_taxonomy_language':lambda chunks:{'language':{'code':'en','name':'English'},'confidence':0},
            '_get_study_module_organization_metadata':lambda *a:{'organization_mode':'infer'},
            '_extract_blueprint_category_names':lambda *a:[], '_uses_authoritative_student_organization':lambda *a:False,
            '_generate_taxonomy_plan':lambda *a,**kw:None,'_global_organization_categories':lambda *a:[],
            '_apply_module_organization_category_guard':lambda data,*a,**kw:data,
            '_apply_syllabus_category_guard':lambda data,*a,**kw:data,
            '_table_column_available':lambda *a:True,'_study_module_organization_columns_available':lambda *a:False,
            '_require_owned_project':lambda db,p,u:None,
            'normalize_chunk_role':lambda role,*a,**kw:role,'log_chunk_role_counts':lambda *a:None})
        for name in ('_build_module_organization_instruction','_build_student_taxonomy_granularity_instruction','_build_student_taxonomy_category_first_instruction','_build_taxonomy_plan_instruction'):
            self.ns[name]=lambda *a:''
        for name in ('_print_taxonomy_state_snapshot','_print_name_list'):
            self.ns[name]=lambda *a:None
        load({'_preflight_syllabus','UploadPipelineLogger','_unique_sorted','_get_taxonomy_diagnostic_snapshot','_merge_upload_diagnostic_snapshots',
              '_lock_taxonomy_scope','ensure_project_chunk_roles','process_topics_task','process_uploaded_documents_topics_task','assign_topics_to_chunks',
              'get_topics','list_documents','delete_document','retry_document_taxonomy','_require_owned_project','rename_topic_category','update_topic_category','merge_topics'},self.ns)

    def test_invalid_syllabus_stops_before_any_ai_and_preserves_taxonomy(self):
        from test_syllabus_outline import ns, org
        self.ns['_get_study_module_organization_metadata'] = lambda *a: org('Lecture 1 — Area\n' + 'x' * 50000)
        self.ns['_preflight_syllabus'] = ns['_preflight_syllabus']
        before = self.rows('select * from topics')
        links = self.rows('select * from topic_chunks')
        self.assertEqual(self.batch('b'), {'b': False})
        self.assertEqual(self.calls, [])
        self.assertEqual(self.rows('select * from topics'), before)
        self.assertEqual(self.rows('select * from topic_chunks'), links)
        self.assertIn('SYLLABUS VALIDATION', self.output.getvalue())

    def prepare_partial_index(self):
        from test_syllabus_outline import ns, org
        from source_chunk_alignment import metadata_revision, text_digest
        organization = org('### Section 1 — Area A (pp. 1–2)\n### Section 2 — Area B (pp. 2–3)')
        for name in ('_parse_syllabus_outline', '_syllabus_likely_category_items', '_global_organization_categories', '_preflight_syllabus'):
            self.ns[name] = ns[name]
        self.ns['_get_study_module_organization_metadata'] = lambda *a: organization
        structure = dict(version=1, document_id='b', status='unavailable', chapters=[],
                         index_page_provenance_version=1, index_boundaries=[],
                         source_end=dict(unit_index=4, block_index=1, start_offset=0))
        structure['revision'] = metadata_revision(structure)
        with self.engine.begin() as c:
            c.execute(text("UPDATE documents SET chapter_structure=:s WHERE id='b'"), {'s':json.dumps(structure)})
            c.execute(text("INSERT INTO chunks(id,project_id,document_id,doc_title,chunk_text,section,page,chunk_role,embedding) VALUES(4,'p','b','b','Boundary material','SECTION',2,'teaching','[1,0]')"))
            for cid,body,page in [(2,'PRIVATE_b instructional source',1),(4,'Boundary material',2)]:
                provenance = dict(version=1,document_id='b',revision=structure['revision'],
                    chunk_sha256=text_digest(body),source_page=page,
                    intervals=[dict(start=0,end=len(body),status='unresolved',source_spans=[])])
                c.execute(text('UPDATE chunks SET structural_provenance=:p WHERE id=:id'),
                          {'p':json.dumps(provenance),'id':cid})
        def plan(db,candidates,project_id,score,**kw):
            snapshot=assignment.source_snapshot(db,project_id,kw['module_id'],kw['document_id'])
            return {t['assignment_id']:{cid for cid,owner in kw['chapter_ownership'].items()
                                       if owner==t['chapter_id']} for t in candidates},snapshot
        self.ns['plan_assignments'] = plan
        self.ns['_apply_syllabus_category_guard'] = lambda *a,**kw:self.fail('Verified source scopes must not be reinterpreted')

    def test_partial_index_publication_review_reassignment_and_sibling_preservation(self):
        self.prepare_partial_index()
        before=self.rows("select * from topics where id in ('ta','legacy')")
        self.assertEqual(self.batch('b'),{'b':True},self.output.getvalue())
        topics=self.rows("select id,category,source_section from topics where document_id='b'")
        self.assertEqual({r[1] for r in topics},{'Area A','TO ORGANIZE'})
        review=next(r for r in topics if r[1]=='TO ORGANIZE')
        self.assertEqual(review[2],REVIEW_MARKER)
        self.assertEqual(len(self.calls),3) # Two discovery calls and one bounded boundary assignment.
        links=self.rows('select * from topic_chunks')
        displayed=asyncio.run(self.ns['get_topics']('p'))['topics']
        displayed_review=next(t for t in displayed if t['id']==review[0])
        self.assertEqual(displayed_review['system_category'],'to_organize')
        self.assertEqual(displayed_review['review_categories'],['Area A','Area B'])
        with self.engine.begin() as c:
            c.execute(text("UPDATE study_modules SET accepted_for_study=true WHERE id='m'"))
        with self.assertRaises(HTTPException):
            asyncio.run(self.ns['update_topic_category']('p',review[0],SimpleNamespace(category='Area B',topic=None),{'id':'u'}))
        with self.engine.begin() as c:
            c.execute(text("UPDATE study_modules SET accepted_for_study=false WHERE id='m'"))
        asyncio.run(self.ns['update_topic_category']('p',review[0],SimpleNamespace(category='Area B',topic=None),{'id':'u'}))
        self.assertFalse(any(t['system_category'] for t in asyncio.run(self.ns['get_topics']('p'))['topics']))
        self.assertEqual(self.rows('select * from topic_chunks'),links)
        self.assertEqual(self.rows("select * from topics where id in ('ta','legacy')"),before)
        self.assertEqual(self.rows("select category,source_section from topics where id='"+review[0]+"'"),[('Area B',None)])

    def test_partial_index_automatically_resolves_without_rewriting_evidence(self):
        self.prepare_partial_index()
        self.boundary_target = 'C0001'
        sibling = self.rows("select * from topics where document_id='a'")
        self.assertEqual(self.batch('b'), {'b': True}, self.output.getvalue())
        rows = self.rows("select category,source_section from topics where document_id='b'")
        self.assertEqual({r[0] for r in rows}, {'Area A', 'Area B'})
        self.assertFalse(any(r[1] == REVIEW_MARKER for r in rows))
        self.assertEqual(self.rows("select * from topics where document_id='a'"), sibling)
        self.assertEqual({r[0] for r in self.rows("select tc.chunk_id from topic_chunks tc join topics t on t.id=tc.topic_id where t.document_id='b'")}, {2, 4})

    def test_selected_language_precedes_source_detection(self):
        self.ns['_get_study_module_organization_metadata'] = lambda *a: {
            'organization_mode': 'infer', 'organization_blueprint': {'output_language': 'Italian'},
            'organization_source_title': None}
        self.ns['detect_project_taxonomy_language'] = Mock(side_effect=AssertionError('Must not detect'))
        self.assertEqual(self.batch('b'), {'b': True}, self.output.getvalue())
        self.ns['detect_project_taxonomy_language'].assert_not_called()
        self.assertIn('SOURCE-LANGUAGE PROCESSING', next(p for p in self.calls if 'CONTENT TO ANALYZE:' in p))
        self.assertIn('Italian', next(p for p in self.calls if 'FINAL TAXONOMY TRANSLATION' in p))
        self.assertEqual(self.rows("select topic from topics where document_id='b'"), [('Argomento sintetico',)])
        self.assertEqual(self.rows('select taxonomy_language from projects'), [('it',)])

    def test_translation_failure_preserves_existing_taxonomy_and_source_links(self):
        self.ns['_get_study_module_organization_metadata'] = lambda *a: {
            'organization_mode':'infer', 'organization_blueprint':{'output_language':'Italian'}}
        self.fail_translation=True
        before=self.rows('select * from topics')
        links=self.rows('select * from topic_chunks')
        chunks=self.rows('select * from chunks')
        self.assertEqual(self.batch('b'), {'b':False})
        self.assertEqual(before,self.rows('select * from topics'))
        self.assertEqual(links,self.rows('select * from topic_chunks'))
        self.assertEqual(chunks,self.rows('select * from chunks'))
        self.assertEqual(sum('FINAL TAXONOMY TRANSLATION' in p for p in self.calls),2)

    def test_partial_index_publication_failure_rolls_back(self):
        self.prepare_partial_index()
        before=self.rows('select * from topics'); links=self.rows('select * from topic_chunks')
        reached=[]
        def fail(*a,**kw):
            reached.append(True)
            raise RuntimeError('Publication failure')
        self.ns['assign_topics_to_chunks']=fail
        self.assertEqual(self.batch('b'),{'b':False})
        self.assertEqual(reached,[True],self.output.getvalue())
        self.assertEqual(self.rows('select * from topics'),before)
        self.assertEqual(self.rows('select * from topic_chunks'),links)

    def rows(self,sql):
        with self.engine.connect() as c:return c.execute(text(sql)).fetchall()

    def chat(self,**kw):
        prompt=kw['messages'][0]['content']; self.calls.append(prompt)
        if self.fail_doc and f'PRIVATE_{self.fail_doc}' in prompt: raise RuntimeError('mock discovery failure')
        if 'FINAL TAXONOMY TRANSLATION' in prompt:
            data=json.loads(prompt.split('\n',1)[1])
            translations={'SAME':'UGUALE','Synthetic topic':'Argomento sintetico',
                          'Synthetic learning objective':'Obiettivo didattico sintetico'}
            value={'items':[{'id':i['id'],'text':translations.get(i['text'],i['text'])} for i in data['items']]}
            if getattr(self, 'fail_translation', False):
                value={'items':[]}
        elif 'ORGANIZATION INPUT:\n' in prompt:
            data=json.loads(prompt.split('ORGANIZATION INPUT:\n')[1])
            if 'Resolve ambiguous index assignments' in prompt:
                value = {'categories': data['categories'], 'assignments': [
                    {'topic_id': t['topic_id'], 'category_id': getattr(self, 'boundary_target', None)} for t in data['topics']]}
            else:
                value=({'categories':[{'name':'SAME','description':'Scope','macrocategory':'Parent'}]} if 'categories' not in data else
                {'assignments':[{'topic_id':t['topic_id'],'category_id':'C0000'} for t in data['topics']], 'quality_issues':[]})
        elif 'CONTENT TO ANALYZE:' in prompt:
            value={'categories':[{'name':'SAME','topics':[{'title':'Synthetic topic','description':'Synthetic learning objective','importance':5}]}]}
        else: raise AssertionError('Unexpected AI stage')
        return SimpleNamespace(model='mock',usage=None,choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=json.dumps(value)))])

    def ledger(self,data):
        return [SimpleNamespace(original_category=c['name'],topic=t.get('title') or t.get('topic'),description=t['description'],embedding=[1.,0.],source_position=(i,j))
                for i,c in enumerate(data['categories']) for j,t in enumerate(c['topics'])]

    def plan(self,db,candidates,project_id,score,**kw):
        snap=assignment.source_snapshot(db,project_id,kw['module_id'],kw['document_id'])
        return {t['assignment_id']:set(snap) for t in candidates},snap

    def batch(self,*docs):
        return self.ns['process_uploaded_documents_topics_task']('p',[{'document_id':d,'title':d} for d in docs],module_id='m')

    def test_new_file_only_sources_and_publication(self):
        before=self.rows("select * from topics where id in ('ta','legacy')")
        self.assertEqual(self.batch('b'),{'b':True},self.output.getvalue())
        self.assertEqual(before,self.rows("select * from topics where id in ('ta','legacy')"))
        self.assertEqual(self.rows("select document_id,module_id from topics where document_id='b'"),[('b','m')])
        discovery=next(p for p in self.calls if 'CONTENT TO ANALYZE:' in p)
        self.assertIn('PRIVATE_b',discovery);self.assertNotIn('PRIVATE_a',discovery);self.assertNotIn('PRIVATE_c',discovery)
        self.assertEqual(self.rows("select tc.chunk_id from topic_chunks tc join topics t on t.id=tc.topic_id where t.document_id='b'"),[(2,)])
        count=len(self.calls);self.assertEqual(self.batch('b'),{'b':True});self.assertEqual(len(self.calls),count)

    def test_publication_failure_preserves_other_files_and_legacy(self):
        before=self.rows('select * from topics'); links=self.rows('select * from topic_chunks')
        original=self.ns['assign_topics_to_chunks']
        reached=[]
        def fail(*a,**kw):
            original(*a,**kw)
            reached.append(True)
            raise RuntimeError('fail after links inserted')
        self.ns['assign_topics_to_chunks']=fail
        self.assertEqual(self.batch('b'),{'b':False},self.output.getvalue())
        self.assertEqual(reached,[True])
        self.assertEqual(self.rows('select * from topics'),before)
        self.assertEqual(self.rows('select * from topic_chunks'),links)
        self.assertEqual(self.rows("select taxonomy_status from documents where id='b'"),[('failed',)])

    def test_mixed_batch_continues_and_does_not_finish_early(self):
        self.fail_doc='b'; observed=[]
        original=self.ns['process_topics_task']
        def observed_run(*a,**kw):
            observed.append(self.rows("select topic_status from projects")[0][0])
            return original(*a,**kw)
        self.ns['process_topics_task']=observed_run
        self.assertEqual(self.batch('b','c'),{'b':False,'c':True},self.output.getvalue())
        self.assertEqual(observed,['processing','processing'])
        self.assertEqual(self.rows('select topic_status from projects'),[('error',)])
        self.assertEqual(self.rows("select document_id from topics where document_id='c'"),[('c',)])
        count=len(self.calls);self.batch('b','c');self.assertEqual(len(self.calls),count)

    def test_cross_document_evidence_is_rejected(self):
        with self.factory() as db:
            with self.assertRaisesRegex(ValueError,'out-of-scope'):
                assignment.publish_prepared_topic_links(db,'p','m','a',{'ta':{2}},None)
        self.assertEqual(self.rows("select chunk_id from topic_chunks where topic_id='ta'"),[(1,)])

    def test_study_lock_and_document_ownership(self):
        with self.factory() as db:
            with self.assertRaises(ValueError):self.ns['_lock_taxonomy_scope'](db,'p','m','missing',True)
        with self.engine.begin() as c:c.execute(text("update study_modules set accepted_for_study=true"))
        with self.assertRaisesRegex(ValueError,'locked'):self.batch('b')
        self.assertFalse(self.calls)

    def test_repeated_category_rename_is_file_scoped(self):
        self.assertTrue(self.batch('b')['b'],self.output.getvalue())
        result=asyncio.run(self.ns['rename_topic_category']('p',SimpleNamespace(current_category='SAME',new_category='NEW',module_id='m',document_id='b'),{'id':'u'}))
        self.assertEqual(result['updated_topics'],1)
        self.assertEqual(self.rows("select category from topics where id in ('ta','legacy')"),[('SAME',),('SAME',)])

    def test_cross_file_merge_rejected_and_module_retrieval_keeps_all(self):
        self.assertTrue(self.batch('b')['b'],self.output.getvalue())
        b=self.rows("select id from topics where document_id='b'")[0][0]
        with self.assertRaises(HTTPException):
            asyncio.run(self.ns['merge_topics']('p',SimpleNamespace(source_topic_id='ta',target_topic_id=b,new_topic_name='Merged'),{'id':'u'}))
        result=asyncio.run(self.ns['get_topics']('p'))
        self.assertEqual({t['document_id'] for t in result['topics']},{'a','b',None})
        self.assertEqual({t['module_id'] for t in result['topics']},{'m'})

    def test_real_assignment_planner_never_scores_sibling_chunks(self):
        candidate={'assignment_id':'new','topic':'Objective','source_section':'SECTION','embedding':[1.,0.]}
        seen=[]
        def score(body,*args):
            seen.append(body)
            return .9
        with self.factory() as db:
            links,snapshot=assignment.plan_assignments(db,[candidate],'p',score,module_id='m',document_id='b')
        self.assertEqual(links,{'new':{2}})
        self.assertEqual(set(snapshot),{2})
        self.assertEqual(seen,['PRIVATE_b instructional source'])

    def test_reassignment_cannot_borrow_sibling_category(self):
        self.assertTrue(self.batch('b')['b'],self.output.getvalue())
        with self.engine.begin() as c:
            c.execute(text("update topics set category='ONLY_A' where id='ta'"))
        b=self.rows("select id from topics where document_id='b'")[0][0]
        with self.assertRaises(HTTPException):
            asyncio.run(self.ns['update_topic_category']('p',b,SimpleNamespace(category='ONLY_A',topic=None),{'id':'u'}))
        self.assertEqual(self.rows("select category from topics where document_id='b'"),[('SAME',)])

    def test_inflight_document_is_not_processed_twice(self):
        with self.engine.begin() as c:
            c.execute(text("update documents set taxonomy_status='building' where id='b'"))
        self.assertEqual(self.batch('b'),{'b':False})
        self.assertFalse(self.calls)
        self.assertEqual(self.rows("select taxonomy_status from documents where id='b'"),[('building',)])

    def test_study_lock_rechecked_before_publication(self):
        original=self.ns['_lock_taxonomy_scope']; calls=[]
        def lock(db,p,module_id=None,document_id=None,require_editable=False):
            if document_id=='b' and require_editable:
                calls.append(True)
                if len(calls)==2: raise ValueError('Study Mode taxonomy is locked')
            return original(db,p,module_id,document_id,require_editable)
        self.ns['_lock_taxonomy_scope']=lock
        before=self.rows('select * from topics')
        self.assertEqual(self.batch('b'),{'b':False})
        self.assertEqual(len(calls),2)
        self.assertEqual(self.rows('select * from topics'),before)

    def test_later_failure_keeps_successful_publication_and_waits_for_terminal_state(self):
        self.fail_doc='c'; states=[]
        original=self.ns['process_topics_task']
        def run(*a,**kw):
            result=original(*a,**kw)
            states.append(self.rows('select topic_status from projects')[0][0])
            return result
        self.ns['process_topics_task']=run
        self.assertEqual(self.batch('b','c'),{'b':True,'c':False},self.output.getvalue())
        self.assertEqual(states,['processing','error'])
        self.assertEqual(self.rows("select count(*) from topics where document_id='b'"),[(1,)])

    def test_old_rename_request_cannot_accidentally_change_legacy_category(self):
        self.assertTrue(self.batch('b')['b'],self.output.getvalue())
        with self.assertRaises(HTTPException) as error:
            asyncio.run(self.ns['rename_topic_category']('p',SimpleNamespace(current_category='SAME',new_category='NEW',module_id='m',document_id=None),{'id':'u'}))
        self.assertEqual(error.exception.status_code,409)
        self.assertEqual(self.rows('select distinct category from topics'),[('SAME',)])

    def test_expired_source_verification_stops_before_replacement(self):
        original = self.ns['source_snapshot']
        calls = []
        before = self.rows('select * from topics')
        replace = Mock(wraps=self.ns['replace_document_taxonomy'])
        self.ns['replace_document_taxonomy'] = replace
        with patch('time.monotonic', return_value=0.0) as clock:
            def snapshot(*args, **kwargs):
                result = original(*args, **kwargs)
                calls.append(True)
                if len(calls) == 2:
                    clock.return_value = 121.0
                return result
            self.ns['source_snapshot'] = snapshot
            self.assertEqual(self.batch('b'), {'b': False})
        replace.assert_not_called()
        self.assertEqual(self.rows('select * from topics'), before)
        self.assertIn('deadline exceeded at source_snapshot', self.output.getvalue())

    def test_topic_publication_uses_batch_and_reports_checkpoints(self):
        from sqlalchemy import event
        inserts = []
        def capture(conn, cursor, statement, parameters, context, executemany):
            if 'insert into topics' in statement.lower():
                inserts.append((statement, parameters))
        event.listen(self.engine, 'before_cursor_execute', capture)
        self.addCleanup(event.remove, self.engine, 'before_cursor_execute', capture)
        self.assertEqual(self.batch('b'), {'b': True}, self.output.getvalue())
        self.assertEqual(len(inserts), 1)
        for phase in ('scope_lock', 'source_snapshot', 'organization_settings', 'before_replacement', 'topic_batch', 'topic_links'):
            self.assertIn("'phase': '" + phase + "'", self.output.getvalue())

    def test_initial_snapshot_disconnect_retries_before_discovery(self):
        from sqlalchemy.exc import OperationalError
        original = self.ns['source_snapshot']
        attempts = []
        before = self.rows("select * from topics where id in ('ta','legacy')")
        def snapshot(*args, **kwargs):
            attempts.append(True)
            if len(attempts) == 1:
                self.assertFalse(any('CONTENT TO ANALYZE:' in p for p in self.calls))
                raise OperationalError('select', {}, Exception('closed'), connection_invalidated=True)
            return original(*args, **kwargs)
        self.ns['source_snapshot'] = snapshot
        self.assertEqual(self.batch('b'), {'b': True}, self.output.getvalue())
        self.assertEqual(before, self.rows("select * from topics where id in ('ta','legacy')"))
        self.assertEqual(sum('CONTENT TO ANALYZE:' in p for p in self.calls), 1)
        self.assertIn("'retry': True", self.output.getvalue())

    def test_source_change_before_publication_rejects_replacement(self):
        original=self.ns['source_snapshot']; calls=[]
        def snapshot(*a,**kw):
            result=original(*a,**kw);calls.append(True)
            if len(calls)==2: result={**result,999:'changed'}
            return result
        self.ns['source_snapshot']=snapshot
        before=self.rows('select * from topics')
        self.assertEqual(self.batch('b'),{'b':False})
        self.assertEqual(len(calls),2)
        self.assertEqual(self.rows('select * from topics'),before)

    def test_aggregate_status_prioritizes_inflight_and_preserves_legacy(self):
        self.assertIsNone(state.aggregate_status([None]))
        self.assertEqual(state.aggregate_status(['ready','failed','queued',None]),'building')
        self.assertEqual(state.aggregate_status(['ready','failed',None]),'failed')
        self.assertEqual(state.aggregate_status(['ready',None]),'ready')


    def test_document_api_order_and_qualified_categories(self):
        docs=self.ns['list_documents']('p',user={'id':'u'})['documents']
        self.assertEqual([d['document_order_index'] for d in docs],[1,2,3])
        self.batch('b')
        topics=asyncio.run(self.ns['get_topics']('p'))['topics']
        file_topics=[t for t in topics if t['document_id']]
        self.assertEqual(len({t['category_key'] for t in file_topics}),2)
        self.assertEqual({t['document_name'] for t in file_topics},{'a','b'})

    def test_delete_only_owned_document_and_protect_legacy_evidence(self):
        self.batch('b')
        before=self.rows('select * from topics')
        with self.assertRaises(HTTPException) as caught:
            self.ns['delete_document']('p','b',user={'id':'u'})
        self.assertEqual(caught.exception.status_code,409)
        self.assertEqual(before,self.rows('select * from topics'))
        with self.engine.begin() as c:c.execute(text("delete from topic_chunks where topic_id='legacy' and chunk_id=2"))
        siblings=self.rows("select * from topics where document_id='a' or document_id is null")
        self.ns['delete_document']('p','b',user={'id':'u'})
        self.assertEqual(siblings,self.rows('select * from topics'))
        self.assertFalse(self.rows("select * from chunks where document_id='b'"))
        self.assertTrue(self.rows("select * from documents where id='a'"))

    def test_delete_and_retry_ownership_and_study_mode(self):
        from fastapi import BackgroundTasks
        for user in ({'id':'other'},{'id':'u'}):
            if user['id']=='u':
                with self.engine.begin() as c:c.execute(text("update study_modules set accepted_for_study=true where id='m'"))
            for action in (lambda:self.ns['delete_document']('p','c',user=user),
                           lambda:self.ns['retry_document_taxonomy']('p','c',BackgroundTasks(),user=user)):
                with self.assertRaises(HTTPException):action()
        self.assertEqual(len(self.rows('select * from documents')),3)

    def test_recovery_is_explicit_fenced_and_file_scoped(self):
        from fastapi import BackgroundTasks
        tasks=BackgroundTasks()
        with self.engine.begin() as c:c.execute(text("update documents set taxonomy_status='building',taxonomy_run_id='old' where id='b'"))
        before=self.rows("select * from topics where document_id='a'")
        result=self.ns['retry_document_taxonomy']('p','b',tasks,user={'id':'u'})
        self.assertEqual(result['taxonomy_status'],'queued');self.assertEqual(len(tasks.tasks),1)
        self.assertFalse(self.ns['process_topics_task']('p',document_id='b',module_id='m',expected_run_id='old'))
        self.assertFalse(self.calls)
        task=tasks.tasks[0];self.assertTrue(task.func(*task.args,**task.kwargs),self.output.getvalue())
        self.assertEqual(before,self.rows("select * from topics where document_id='a'"))
        with self.assertRaises(HTTPException):self.ns['retry_document_taxonomy']('p','b',BackgroundTasks(),user={'id':'u'})

    def test_superseded_worker_cannot_publish_or_mark_new_attempt_failed(self):
        original=self.ns['build_immutable_taxonomy_ledger']
        calls=[]
        def ledger(data):
            result=original(data);calls.append(1)
            if len(calls)==2:
                with self.engine.begin() as c:c.execute(text("update documents set taxonomy_status='queued',taxonomy_run_id='replacement' where id='b'"))
            return result
        self.ns['build_immutable_taxonomy_ledger']=ledger
        before=self.rows('select * from topics')
        self.assertEqual(self.batch('b'),{'b':False})
        self.assertEqual(before,self.rows('select * from topics'))
        self.assertEqual(self.rows("select taxonomy_status,taxonomy_run_id from documents where id='b'"),[('queued','replacement')])

    def test_repeated_filename_requires_id(self):
        with self.engine.begin() as c:c.execute(text("update documents set title='same.pdf',taxonomy_status='failed'"))
        with self.assertRaises(HTTPException) as caught:self.ns['delete_document']('p','same.pdf',user={'id':'u'})
        self.assertEqual(caught.exception.status_code,409)
        self.ns['delete_document']('p','c',user={'id':'u'})
        self.assertEqual(len(self.rows('select * from documents')),2)

    def test_planner_all_documents_remain_separate_and_legacy_stays_compatible(self):
        from planner.context_builder import _load_display_topics, _build_topic_context
        self.batch('b')
        with self.engine.begin() as c:c.execute(text("update study_modules set accepted_for_study=true"))
        with self.factory() as db: rows=_load_display_topics(db,'p')
        categories,groups,lookup=_build_topic_context(rows)
        self.assertEqual(len(categories),3)
        self.assertEqual({t.id for group in groups.values() for t in group},{'ta','legacy',str(next(r[0] for r in rows if r[4]=='b'))})
        self.assertIn('SAME',categories)
        _,_,ambiguous=_build_topic_context([('1','Same','Repeated','m','a','a.pdf'),('2','Same','Repeated','m','b','b.pdf')])
        self.assertIsNone(ambiguous['repeated'])


    def test_delete_failure_rolls_back_taxonomy_and_source_deletion(self):
        with self.engine.begin() as c:c.execute(text("update documents set taxonomy_status='failed' where id='c'"))
        before=self.rows('select * from documents'); chunks=self.rows('select * from chunks')
        original=SQLiteSession.execute
        def fail(session, statement, *args, **kwargs):
            if str(statement).startswith('delete from documents'):
                raise RuntimeError('Simulated deletion failure')
            return original(session,statement,*args,**kwargs)
        with patch.object(SQLiteSession,'execute',fail):
            with self.assertRaises(RuntimeError):self.ns['delete_document']('p','c',user={'id':'u'})
        self.assertEqual(before,self.rows('select * from documents'))
        self.assertEqual(chunks,self.rows('select * from chunks'))

    def seed_verified_chapters(self):
        from test_chapter_provenance import fixture
        from source_chunk_alignment import metadata_revision
        structure, chunks = fixture()
        structure['document_id']='b';structure['revision']=metadata_revision(structure)
        for chunk in chunks:
            chunk['document_id']='b'
            chunk['structural_provenance']['document_id']='b'
            chunk['structural_provenance']['revision']=structure['revision']
        with self.engine.begin() as c:
            c.execute(text("UPDATE documents SET chapter_structure=:s WHERE id='b'"),{'s':json.dumps(structure)})
            c.execute(text("UPDATE chunks SET chunk_text=:body,structural_provenance=:p WHERE id=2"),
                      {'body':chunks[0]['chunk_text'],'p':json.dumps(chunks[0]['structural_provenance'])})
            c.execute(text("INSERT INTO chunks VALUES (4,'p','b','b',:body,'SECTION',2,'teaching','[1,0]',:p)"),
                      {'body':chunks[1]['chunk_text'],'p':json.dumps(chunks[1]['structural_provenance'])})
        self.ns['plan_assignments']=assignment.plan_assignments
        self.ns['calculate_topic_chunk_score']=lambda *a:1.0

    def test_verified_index_scopes_discovery_consolidation_and_publication(self):
        from test_syllabus_outline import ns, org
        self.seed_verified_chapters()
        organization = org('### Lecture 1 — 1 Chapter Alpha (pp. 1–1)\n### Lecture 2 — 2 Chapter Beta (pp. 2–2)')
        self.ns['_get_study_module_organization_metadata'] = lambda *a: organization
        for name in ('_preflight_syllabus', '_parse_syllabus_outline', '_syllabus_likely_category_items'):
            self.ns[name] = ns[name]
        before = self.rows("select * from topics where id in ('ta','legacy')")
        self.assertEqual(self.batch('b'), {'b': True}, self.output.getvalue())
        result = self.rows("select t.category,t.topic,tc.chunk_id from topics t join topic_chunks tc on tc.topic_id=t.id where t.document_id='b' order by t.category_order_index")
        self.assertEqual(result, [('1 Chapter Alpha','Synthetic topic',2), ('2 Chapter Beta','Synthetic topic',4)])
        self.assertFalse(any('ORGANIZATION INPUT:' in p for p in self.calls))
        self.assertFalse(any('DUPLICATE PAIR REVIEW' in p for p in self.calls))
        self.assertEqual(before, self.rows("select * from topics where id in ('ta','legacy')"))
        self.assertIn('"status": "mapped"', self.output.getvalue())

    def test_chapter_publication_and_recovery_keep_siblings_and_exact_links(self):
        self.seed_verified_chapters()
        before=self.rows("select * from topics where id in ('ta','legacy')")
        # Explicit recovery queues the persisted sources; failed uploads are not retried implicitly.
        with self.engine.begin() as c:c.execute(text("update documents set taxonomy_status='queued' where id='b'"))
        self.assertEqual(self.batch('b'),{'b':True},self.output.getvalue())
        result=self.rows("select t.category,t.topic,tc.chunk_id from topics t join topic_chunks tc on tc.topic_id=t.id where t.document_id='b' order by t.category_order_index")
        self.assertEqual(result,[('1 Chapter Alpha','Synthetic topic',2),('2 Chapter Beta','Synthetic topic',4)])
        self.assertFalse(any('ORGANIZATION INPUT:' in p for p in self.calls))
        self.assertEqual(before,self.rows("select * from topics where id in ('ta','legacy')"))
        count=len(self.calls); self.batch('b');self.assertEqual(count,len(self.calls))

    def test_chapter_publication_failure_rolls_back(self):
        self.seed_verified_chapters()
        before=self.rows("select * from topics");links=self.rows("select * from topic_chunks")
        def fail(*a,**kw): raise RuntimeError('publication failure')
        self.ns['assign_topics_to_chunks']=fail
        self.assertEqual(self.batch('b'),{'b':False})
        self.assertEqual(before,self.rows("select * from topics"))
        self.assertEqual(links,self.rows("select * from topic_chunks"))

    def test_unavailable_chapters_use_existing_document_wide_ai(self):
        with self.engine.begin() as c:
            c.execute(text("update documents set chapter_structure=:s where id='b'"),
                      {'s':json.dumps({'version':1,'status':'unavailable'})})
        self.assertEqual(self.batch('b'),{'b':True},self.output.getvalue())
        self.assertEqual(sum('ORGANIZATION INPUT:' in p for p in self.calls),2)

