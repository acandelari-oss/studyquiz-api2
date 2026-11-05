"""Disposable SQLite, mocked organizer calls, no application startup or network."""
import json
import unittest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session
from category_reorganization import preview, apply, _CACHE
from tests.test_document_taxonomy_isolation import load


class CategoryOnlyTests(unittest.TestCase):
    def setUp(self):
        _CACHE.clear()
        self.engine=create_engine('sqlite://')
        self.addCleanup(self.engine.dispose)
        self.db=Session(self.engine); self.addCleanup(self.db.close)
        for ddl in [
            'projects(id text primary key,user_id text,study_mode text,study_priority_categories text)',
            'study_modules(id text primary key,project_id text,accepted_for_study boolean)',
            'documents(id text primary key,project_id text,module_id text,title text,taxonomy_status text)',
            'topics(id text primary key,project_id text,document_id text,module_id text,topic text,description text,category text,is_display_topic boolean,category_order_index int,topic_order_index int)',
            'topic_chunks(topic_id text,chunk_id int)', 'planner_weeks(id text,project_id text)',
            'study_records(id text,topic_id text,kind text,value text)']:
            self.db.execute(text('CREATE TABLE '+ddl))
        self.db.execute(text("INSERT INTO projects VALUES ('p','u','building','[]')"))
        self.db.execute(text("INSERT INTO study_modules VALUES ('m','p',false)"))
        self.db.execute(text("INSERT INTO documents VALUES ('d','p','m','Source.pdf','ready'),('sibling','p','m','Other.pdf','ready')"))
        for tid,doc in [('t1','d'),('t2','d'),('s','sibling'),('legacy',None)]:
            self.db.execute(text("INSERT INTO topics VALUES (:t,'p',:d,'m',:t,'description','Old',true,1,1)"),{'t':tid,'d':doc})
            self.db.execute(text('INSERT INTO topic_chunks VALUES (:t,1)'),{'t':tid})
            for kind in ['quiz','flashcard','progress']:
                self.db.execute(text('INSERT INTO study_records VALUES (:id,:t,:k,\'untouched\')'),{'id':tid+kind,'t':tid,'k':kind})
        self.db.execute(text("ALTER TABLE topics ADD COLUMN source_section text"))
        self.db.commit()
        ns={'text':text}
        load(['_lock_taxonomy_scope'],ns)
        self.lock=ns['_lock_taxonomy_scope']
        self.calls=[]

    def authorize(self,db,p,u):
        if db.execute(text('SELECT user_id FROM projects WHERE id=:p'),{'p':p}).scalar()!=u:
            raise ValueError('Ownership rejected')

    def settings(self,db,m): return {'organization_mode':'infer'}

    def request(self,prompt,stage):
        self.assertFalse(self.db.in_transaction())
        self.calls.append(stage)
        if stage=='category_proposal':
            return {'categories':[{'name':'Formation','description':'Establishing agreements','macrocategory':'Contracts'}, {'name':'Remedies','description':'Relief after breach','macrocategory':'Contracts'}]}
        return {'assignments':[{'topic_id':'T0001','category_id':'C0000'}, {'topic_id':'T0002','category_id':'C0001'}], 'quality_issues':[]}

    def make(self):
        return preview(self.db,'u','p','d',self.authorize,self.lock,self.settings,self.request)

    def use(self,p,owner='u',doc='d',approved=True):
        return apply(self.db,owner,'p',doc,p['preview_id'],approved,self.authorize,self.lock,self.settings)

    def rows(self,table): return [tuple(r) for r in self.db.execute(text('SELECT * FROM '+table+' ORDER BY 1'))]

    def test_preview_read_only_apply_preserves_all_other_data_and_does_not_call_ai(self):
        before={t:self.rows(t) for t in ['topics','topic_chunks','study_records','documents']}
        self.db.rollback()
        p=self.make()
        self.assertEqual(p['topic_count'],2)
        for t,v in before.items(): self.assertEqual(self.rows(t),v)
        self.db.rollback()
        # Caller cannot alter the saved server-side preview.
        p['assignments'][0]['category']='tampered'
        self.use(p)
        self.assertEqual(self.calls,['category_proposal','category_assignment'])
        after=self.rows('topics')
        for a,b in zip(after,before['topics']):
            if a[2]=='d': self.assertEqual(a[:6]+a[7:8],b[:6]+b[7:8])
            else: self.assertEqual(a,b)
        self.assertEqual({a[6] for a in after if a[2]=='d'},{'Formation','Remedies'})
        for t in ['topic_chunks','study_records','documents']: self.assertEqual(self.rows(t),before[t])
        with self.assertRaises(ValueError): self.use(p)

    def test_stale_topic_and_link_previews(self):
        for sql in ["UPDATE topics SET description='edited' WHERE id='t1'", "INSERT INTO topic_chunks VALUES ('t1',99)"]:
            p=self.make(); self.db.execute(text(sql));self.db.commit()
            with self.assertRaisesRegex(ValueError,'stale'): self.use(p)
            self.assertEqual(self.db.execute(text("SELECT category FROM topics WHERE id='t1'")).scalar(),'Old')
            self.db.rollback()

    def test_dependency_blocks(self):
        for sql in ["INSERT INTO planner_weeks VALUES ('w','p')", "UPDATE projects SET study_priority_categories='[\"Old\"]'"]:
            p=self.make();self.db.execute(text(sql));self.db.commit()
            with self.assertRaisesRegex(ValueError,'Planner'): self.use(p)
            self.assertEqual(self.db.execute(text("SELECT category FROM topics WHERE id='t1'")).scalar(),'Old')
            self.db.execute(text('DELETE FROM planner_weeks'));self.db.commit()

    def test_study_mode_lock_after_preview(self):
        p=self.make()
        self.db.execute(text('UPDATE study_modules SET accepted_for_study=true'));self.db.commit()
        with self.assertRaisesRegex(ValueError,'locked'): self.use(p)

    def test_scope_approval_expiry(self):
        p=self.make()
        for kwargs in [{'owner':'intruder'},{'doc':'sibling'},{'approved':False}]:
            with self.assertRaises(ValueError): self.use(p,**kwargs)
        _CACHE[p['preview_id']]['expires']=0
        with self.assertRaises(ValueError): self.use(p)

    def test_atomic_rollback(self):
        p=self.make()
        self.db.execute(text("CREATE TRIGGER fail_update BEFORE UPDATE ON topics WHEN NEW.id='t2' BEGIN SELECT RAISE(ABORT,'injected failure'); END"));self.db.commit()
        with self.assertRaises(Exception): self.use(p)
        self.assertEqual([r[0] for r in self.db.execute(text("SELECT category FROM topics WHERE document_id='d'"))],['Old','Old'])
        self.assertIn(p['preview_id'],_CACHE)

    def test_incomplete_ai_output_does_not_retain_preview(self):
        original=self.request
        def bad(prompt,stage):
            r=original(prompt,stage)
            if stage=='category_assignment': r['assignments'].pop()
            return r
        with self.assertRaises(ValueError):
            preview(self.db,'u','p','d',self.authorize,self.lock,self.settings,bad)
        self.assertFalse(_CACHE)

    def test_preview_rejects_wrong_owner_and_locked_module_before_ai(self):
        with self.assertRaises(ValueError):
            preview(self.db,'other','p','d',self.authorize,self.lock,self.settings,self.request)
        self.db.rollback()
        self.db.execute(text('UPDATE study_modules SET accepted_for_study=true'));self.db.commit()
        with self.assertRaises(ValueError): self.make()
        self.assertEqual(self.calls,[])
