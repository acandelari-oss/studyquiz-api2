"""Real PostgreSQL pipeline tests, with deterministic AI and no network requests.

Set DOUNO_TEST_DATABASE_URL to an isolated disposable PostgreSQL instance.
Every test uses its own schema; no existing application tables are accessed.
"""
import hashlib
import io
import json
import os
import re
import threading
import time
import unittest
import uuid
from contextlib import ExitStack, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker

import main
from taxonomy_coverage import consolidate_candidates
from taxonomy_assignment import plan_assignments

TEST_URL = os.environ.get("DOUNO_TEST_DATABASE_URL")


def response(data):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
        content=data if isinstance(data, str) else json.dumps(data)))])


@unittest.skipUnless(TEST_URL, "requires isolated DOUNO_TEST_DATABASE_URL")
class TaxonomyPipelineTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch("httpx.Client.send", side_effect=AssertionError("Network forbidden")))
        self.schema = "coverage_test_" + uuid.uuid4().hex
        self.admin = create_engine(TEST_URL)
        with self.admin.begin() as db:
            db.execute(text(f'CREATE SCHEMA "{self.schema}"'))
        self.engine = create_engine(TEST_URL, connect_args={"options": f"-csearch_path={self.schema},public"})
        self.addCleanup(self.cleanup_database)
        self.sessions = []
        self.factory = sessionmaker(bind=self.engine)
        self.stack.enter_context(patch.object(main, "SessionLocal", self.session))
        self.output = io.StringIO()
        self.stack.enter_context(redirect_stdout(self.output))
        self.create_schema()
        self.bad_discovery = 0
        self.bad_consolidation = False
        self.merge = False
        self.discovery_calls = 0
        self.seen = []
        self.ai_calls = 0
        self.client = SimpleNamespace()
        self.client.with_options = lambda **kwargs: self.client
        self.client.chat = SimpleNamespace(completions=SimpleNamespace(create=self.chat))
        self.client.embeddings = SimpleNamespace(create=self.embedding)
        self.stack.enter_context(patch.object(main, "client", self.client))
        # Audit-only call does not influence the taxonomy; avoid needing another mock contract.
        self.stack.enter_context(patch.object(main, "audit_taxonomy_reorganization", return_value={}))

    def session(self):
        db = self.factory()
        self.sessions.append(db)
        return db

    def cleanup_database(self):
        for db in self.sessions:
            db.close()
        self.engine.dispose()
        with self.admin.begin() as db:
            db.execute(text(f'DROP SCHEMA "{self.schema}" CASCADE'))
        self.admin.dispose()

    def create_schema(self):
        with self.engine.begin() as db:
            # Deterministic two-dimensional negative-inner-product operator. The
            # production SQL and scoring run unchanged; this is NOT a pgvector benchmark.
            db.execute(text("CREATE DOMAIN vector AS text"))
            db.execute(text("""CREATE FUNCTION negative_product(vector, vector) RETURNS double precision
                LANGUAGE SQL IMMUTABLE AS $$ SELECT -((($1::jsonb->>0)::float * ($2::jsonb->>0)::float)
                + (($1::jsonb->>1)::float * ($2::jsonb->>1)::float)) $$"""))
            db.execute(text("CREATE OPERATOR <#> (LEFTARG=vector, RIGHTARG=vector, FUNCTION=negative_product)"))
            db.execute(text("""CREATE TABLE projects(id text primary key, user_id text, name text,
                study_mode text default 'building', professor_mode text default 'coverage',
                topic_status text default 'completed', taxonomy_language text)"""))
            db.execute(text("""CREATE TABLE study_modules(id text primary key, project_id text,
                status text default 'pending_study', taxonomy_status text default 'ready',
                accepted_for_study boolean default false, accepted_at timestamp)"""))
            db.execute(text("""CREATE TABLE documents(id text primary key, project_id text,
                module_id text, module_order_index integer, title text)"""))
            db.execute(text("""CREATE TABLE chunks(id integer primary key, project_id text,
                document_id text, doc_title text, chunk_text text, section text, page integer,
                chunk_role text, embedding vector)"""))
            db.execute(text("""CREATE TABLE topics(id text primary key default gen_random_uuid()::text,
                project_id text, document_id text, module_id text, category text, topic text,
                description text, embedding vector, is_display_topic boolean, source_section text,
                category_order_index integer, topic_order_index integer,
                first_source_page integer, first_source_block_index integer)"""))
            db.execute(text("CREATE TABLE topic_chunks(topic_id text references topics(id), chunk_id integer)"))
            db.execute(text("INSERT INTO projects(id,user_id,name) VALUES ('p','u','Test')"))
            db.execute(text("INSERT INTO study_modules(id,project_id) VALUES ('m','p'),('other','p')"))
            db.execute(text("INSERT INTO documents VALUES ('d','p','m',1,'First'),('d2','p','m',2,'Second'),('other-d','p','other',1,'Other')"))
            db.execute(text("""INSERT INTO topics(id,project_id,module_id,document_id,category,topic,description,embedding,is_display_topic,source_section)
                VALUES ('old','p','m','d','CATEGORY','Previous topic','Previous evidence','[1,0]',true,'CATEGORY'),
                       ('untouched','p','other','other-d','OTHER','Other topic','Other evidence','[1,0]',true,'OTHER')"""))

    def seed(self, count=5):
        with self.engine.begin() as db:
            db.execute(text("""INSERT INTO chunks VALUES (:id,'p',:doc,:title,:body,'SECTION',:page,'teaching','[1,0]')"""),
                       [{"id": i, "doc": "d" if i <= 400 else "d2", "title": "First" if i <= 400 else "Second",
                         "page": i if i <= 400 else i-400, "body": "Educational source " + str(i) + " " + "Detailed instructional evidence. " * 5}
                        for i in range(1, count + 1)])
            db.execute(text("INSERT INTO chunks VALUES (999999,'p','other-d','Other','Other module content','OTHER',1,'teaching','[1,0]')"))
            db.execute(text("INSERT INTO topic_chunks VALUES ('old',1),('untouched',999999)"))

    def assert_no_ai_transaction(self):
        self.ai_calls += 1
        self.assertFalse(any(db.in_transaction() for db in self.sessions), "AI call inside database transaction")

    def embedding(self, **kwargs):
        self.assert_no_ai_transaction()
        inputs = kwargs['input']
        count = len(inputs) if isinstance(inputs, list) else 1
        return SimpleNamespace(data=[SimpleNamespace(index=i, embedding=[1.0, 0.0])
                                     for i in range(count)])

    def chat(self, **kwargs):
        self.assert_no_ai_transaction()
        prompt = kwargs["messages"][-1]["content"]
        if "dominant natural language" in prompt:
            return response({"language_code": "en", "confidence": .99})
        if "SOURCE SAMPLES:" in prompt:
            return response({"strategy": "Source faithful", "categories": [{"name": "CATEGORY", "target_topic_count": 5}]})
        if "DUPLICATE PAIR REVIEW" in prompt:
            if self.bad_consolidation:
                return response({"categories": []})
            payload = json.loads(prompt.rsplit("REVIEW INPUT:\n", 1)[1])
            inputs = {c["input_id"]: c for c in payload["candidates"]}
            return response({"decisions": [{
                "input_ids": pair, "decision": "duplicate" if self.merge else "distinct",
                "reason": "Deterministic fixture decision", "category": inputs[pair[0]]["category"],
                "evidence_checks": {i: [inputs[i]["validated_evidence_samples"][0]["chunk_id"]] for i in pair}
            } for pair in payload["relationships"]]})
        if "CONTENT TO ANALYZE:" in prompt:
            self.discovery_calls += 1
            if self.bad_discovery:
                self.bad_discovery -= 1
                return response("invalid json")
            ids = [int(value) for value in re.findall(r"SOURCE_BLOCK: (\d+)", prompt)]
            self.seen.extend(ids)
            return response({"categories": [{"name": "CATEGORY", "topics": [
                {"title": hashlib.sha256(str(i).encode()).hexdigest(),
                 "description": ("Same instructional objective" if self.merge else f"Distinct concept from source {i}"), "importance": 5} for i in ids]}]})
        raise AssertionError("Unexpected AI prompt")

    def run_pipeline(self, **kwargs):
        return main.process_topics_task("p", **({"module_id": "m"} if not kwargs else kwargs))

    def rows(self, query):
        with self.engine.connect() as db:
            return db.execute(text(query)).fetchall()

    def test_planner_matches_legacy_scoring_ranking_and_rescue(self):
        self.seed(225)
        # Identical embeddings create ties; varying scores exercise boundaries,
        # section/keyword bonuses, null vectors, and orphan rescue.
        with self.engine.begin() as db:
            db.execute(text("delete from topic_chunks where topic_id='old'"))
            db.execute(text("delete from topics where id='old'"))
            db.execute(text("update chunks set embedding='[0.5,0.0]' where id % 3=0 and id<999999"))
            db.execute(text("update chunks set embedding=NULL where id=225"))
            candidates = []
            for i, value in enumerate([1., 1., 1., 1., .54, .5399, .46, .4599, .1, -.2]):
                title = "Educational evidence" if i == 8 else "Concept " + str(i)
                section = "SECTION" if i == 9 else "CATEGORY"
                c = {"assignment_id": "fixed-%02d" % i, "topic": title,
                     "source_section": section, "embedding": [value, 0.]}
                candidates.append(c)
                db.execute(text("""insert into topics(id,project_id,module_id,category,topic,embedding,source_section)
                    values (:id,'p','m','CATEGORY',:title,:embedding,:section)"""),
                    {"id": c["assignment_id"], "title": title, "embedding": json.dumps(c["embedding"]), "section": section})
        with self.factory() as db:
            planned, snapshot = plan_assignments(db, candidates, "p", main.calculate_topic_chunk_score,
                                                module_id="m", pair_budget=600)
            self.assertEqual(len(snapshot), 225)
            self.assertEqual(self.rows("select count(*) from topic_chunks where topic_id like 'fixed-%'"), [(0,)])
        main.assign_topics_to_chunks("p", module_id="m")
        actual = {c["assignment_id"]: set() for c in candidates}
        for topic_id, chunk_id in self.rows("select topic_id,chunk_id from topic_chunks where topic_id like 'fixed-%'"):
            actual[topic_id].add(chunk_id)
        self.assertEqual(planned, actual)
        self.assertEqual(planned["fixed-07"], set())  # below rescue
        self.assertEqual(planned["fixed-06"], {1})  # exact rescue threshold, lowest-ID tie
        self.assertNotIn(225, set().union(*planned.values()))  # null embedding
        self.assertEqual(self.rows("select chunk_id from topic_chunks where topic_id='untouched'"), [(999999,)])

    def test_unmerged_publication_matches_legacy_for_same_topic_ids(self):
        self.seed(20)
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-5000:])
        before = set(self.rows("select tc.topic_id,tc.chunk_id from topic_chunks tc join topics t on t.id=tc.topic_id where t.module_id='m'"))
        main.assign_topics_to_chunks("p", module_id="m")
        after = set(self.rows("select tc.topic_id,tc.chunk_id from topic_chunks tc join topics t on t.id=tc.topic_id where t.module_id='m'"))
        self.assertEqual(before, after)

    def test_merged_publication_preserves_exact_planned_union(self):
        self.seed(20)
        self.merge = True
        captured = {}
        original = main.consolidate_candidates
        def capture(candidates, *args, **kwargs):
            data = original(candidates, *args, **kwargs)
            captured.update({t["assignment_id"]: set(t["evidence_ids"])
                             for c in data["categories"] for t in c["topics"]})
            return data
        with patch.object(main, "consolidate_candidates", side_effect=capture):
            self.assertTrue(self.run_pipeline(), self.output.getvalue()[-5000:])
        actual = {key: set() for key in captured}
        for tid, cid in self.rows("select tc.topic_id,tc.chunk_id from topic_chunks tc join topics t on t.id=tc.topic_id where t.module_id='m'"):
            actual[tid].add(cid)
        self.assertEqual(captured, actual)
        self.assertEqual(set().union(*actual.values()), set(range(1,21)))

    def test_twelve_duplicate_objectives_publish_one_topic_with_all_evidence(self):
        self.seed(12)
        self.merge = True
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-5000:])
        self.assertEqual(self.rows("select count(*) from topics where module_id='m'"), [(1,)])
        links = self.rows("select tc.chunk_id from topic_chunks tc join topics t on t.id=tc.topic_id where t.module_id='m'")
        self.assertEqual({row[0] for row in links}, set(range(1,13)))
        self.assertEqual(len(links), 12)
        self.assertEqual(self.rows("select id from topics where module_id='other'"), [("untouched",)])

    def test_evidence_validation_failure_preserves_old_taxonomy(self):
        self.seed()
        original = main.consolidate_candidates
        def corrupt(*args, **kwargs):
            data = original(*args, **kwargs)
            data["categories"][0]["topics"][0]["evidence_ids"] = [999999]
            return data
        with patch.object(main, "consolidate_candidates", side_effect=corrupt):
            self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_publication_rechecks_in_place_source_changes(self):
        self.seed()
        original = main.consolidate_candidates
        def mutate(*args, **kwargs):
            result = original(*args, **kwargs)
            with self.engine.begin() as db:
                db.execute(text("update chunks set chunk_text='changed in place' where id=1"))
            return result
        with patch.object(main, "consolidate_candidates", side_effect=mutate):
            self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_manual_category_guard_does_not_lexically_merge_generated_candidates(self):
        organization = {"organization_mode": "manual", "organization_blueprint": {
            "categories": [{"name": "ALGEBRA", "description": "Linear algebra"}]}}
        data = {"categories": [{"name": "ALGEBRA", "topics": [
            {"title": "Matrix determinant", "description": "Definition", "candidate_ids": ["0"]},
            {"title": "Matrix determinant", "description": "Proof", "candidate_ids": ["1"]}]}]}
        guarded = main._apply_module_organization_category_guard(data, organization, deduplicate=False)
        self.assertEqual(len(guarded["categories"][0]["topics"]), 2)
        # Existing editing helper default remains available and unchanged.
        legacy = main._apply_module_organization_category_guard(data, organization)
        self.assertEqual(len(legacy["categories"][0]["topics"]), 1)

    def test_more_than_400_and_40_with_late_distinct_concept(self):
        self.seed(425)
        started = time.monotonic()
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-5000:])
        self.assertEqual(self.seen, list(range(1,426)))
        titles = {r[0] for r in self.rows("select topic from topics where module_id='m'")}
        self.assertEqual(len(titles), 425)
        self.assertIn(hashlib.sha256(b"425").hexdigest(), titles)
        self.assertEqual(self.rows("select id from topics where module_id='other'"), [("untouched",)])
        self.assertEqual(self.rows("select topic_status from projects"), [("completed",)])
        self.assertIn("'retained_or_merged': 425", self.output.getvalue())
        print("LOCAL PIPELINE ELAPSED", time.monotonic()-started)

    def test_small_document_and_document_isolation(self):
        self.seed(5)
        self.assertTrue(self.run_pipeline(document_id="d"), self.output.getvalue()[-4000:])
        self.assertEqual(set(self.seen), set(range(1,6)))
        self.assertEqual(len(self.rows("select * from topics where document_id='d'")), 5)
        self.assertEqual(self.rows("select id from topics where module_id='other'"), [("untouched",)])

    def test_discovery_json_retries(self):
        self.seed()
        self.bad_discovery = 2
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(self.discovery_calls, 3)

    def test_exhausted_retries_preserve_old_taxonomy(self):
        self.seed()
        self.bad_discovery = 100
        self.assertFalse(self.run_pipeline())
        self.assertEqual(self.discovery_calls, 3)
        self.assert_old()
        self.assertEqual(self.rows("select taxonomy_status from study_modules where id='m'"), [("failed",)])
        self.assertEqual(self.rows("select topic_status from projects"), [("error",)])

    def assert_old(self):
        self.assertEqual(self.rows("select id from topics where module_id='m'"), [("old",)])
        self.assertEqual(self.rows("select chunk_id from topic_chunks where topic_id='old'"), [(1,)])

    def test_consolidation_failure_retains_all_candidates(self):
        self.seed(65)
        self.bad_consolidation = True
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(len(self.rows("select * from topics where module_id='m'")), 65)

    def test_bounded_nontransitive_merging_accounts_for_all_candidates(self):
        self.seed(65)
        self.merge = True
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        rows = self.rows("select description from topics where module_id='m'")
        self.assertGreater(len(rows), 1)  # explicit budget, not unsafe transitive collapse
        self.assertLess(len(rows), 65)
        self.assertTrue(all(row[0] == "Same instructional objective" for row in rows))
        self.assertIn("'retained_or_merged': 65", self.output.getvalue())
        self.assertIn('"deferred":', self.output.getvalue())

    def test_failure_after_actual_assignment_rolls_back_everything(self):
        self.seed()
        original = main.assign_topics_to_chunks
        def fail(*args, **kwargs):
            original(*args, **kwargs)
            self.assert_old()  # separate connection still sees previous committed state
            raise RuntimeError("Injected failure after assignment")
        with patch.object(main, "assign_topics_to_chunks", side_effect=fail):
            self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_locked_module_never_calls_ai_or_changes_taxonomy(self):
        self.seed()
        with self.engine.begin() as db:
            db.execute(text("update study_modules set accepted_for_study=true where id='m'"))
        self.assertFalse(self.run_pipeline())
        self.assertEqual(self.ai_calls, 0)
        self.assert_old()

    def test_assignment_deadline_rolls_back(self):
        self.seed()
        original = main.assign_topics_to_chunks
        def expired(*args, **kwargs):
            kwargs["deadline"] = time.monotonic()-1
            return original(*args, **kwargs)
        with patch.object(main, "assign_topics_to_chunks", side_effect=expired):
            self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_background_failure_does_not_regenerate_or_delete_documents(self):
        self.seed()
        self.bad_discovery = 100
        main.process_uploaded_documents_topics_task("p", [{"document_id":"d", "title":"First"}], "m")
        self.assert_old()
        self.assertEqual(self.discovery_calls, 3)
        self.assertEqual(self.rows("select count(*) from documents"), [(3,)])

    def test_approval_waits_for_atomic_publication(self):
        self.seed()
        entered, release, approved = threading.Event(), threading.Event(), threading.Event()
        original = main.assign_topics_to_chunks
        outcomes, errors = [], []
        def pause(*args, **kwargs):
            original(*args, **kwargs)
            entered.set()
            if not release.wait(4):
                raise RuntimeError("test synchronization timeout")
        def generate():
            try:
                outcomes.append(self.run_pipeline())
            except Exception as error:
                errors.append(error)
        def approve():
            try:
                main.begin_project_study("p", module_id="m", user={"id":"u"})
                approved.set()
            except Exception as error:
                errors.append(error)
        with patch.object(main, "assign_topics_to_chunks", side_effect=pause):
            worker = threading.Thread(target=generate)
            worker.start()
            try:
                self.assertTrue(entered.wait(10), self.output.getvalue()[-4000:])
                approver = threading.Thread(target=approve)
                approver.start()
                self.assertFalse(approved.wait(.2))
                self.assert_old()
            finally:
                release.set()
                worker.join(10)
                if 'approver' in locals():
                    approver.join(10)
        self.assertFalse(errors, errors)
        self.assertEqual(outcomes, [True])
        self.assertTrue(approved.is_set())
        self.assertEqual(self.rows("select accepted_for_study from study_modules where id='m'"), [(True,)])
        self.assertEqual(len(self.rows("select * from topics where module_id='m'")), 5)

    def test_document_scope_includes_only_late_document(self):
        self.seed(425)
        self.assertTrue(self.run_pipeline(document_id="d2"), self.output.getvalue()[-4000:])
        self.assertEqual(set(self.seen), set(range(401,426)))
        self.assert_old()
        self.assertEqual(len(self.rows("select * from topics where document_id='d2'")), 25)

    def test_unassigned_chunk_still_has_discovery_coverage(self):
        self.seed()
        with self.engine.begin() as db:
            db.execute(text("update chunks set embedding='[0,1]' where id=5"))
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(set(self.seen), set(range(1,6)))
        self.assertEqual(self.rows("select count(*) from topic_chunks where chunk_id=5"), [(0,)])

    def test_existing_retrieval_contract_after_publication(self):
        self.seed()
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        main.begin_project_study("p", module_id="m", user={"id":"u"})
        scope = main.resolve_learning_scope("p", limit=80)
        self.assertEqual(scope["chunk_count"], 5)
        self.assertTrue(all({"chunk_id", "chunk_text", "topic_id", "topic", "doc_title", "page"} <= set(c) for c in scope["chunks"]))

    def test_failure_before_assignment_commit_preserves_old_topics_on_sql_error(self):
        self.seed()
        original = main.assign_topics_to_chunks
        def sql_failure(*args, **kwargs):
            original(*args, **kwargs)
            kwargs["db_session"].execute(text("select 1/0"))
        with patch.object(main, "assign_topics_to_chunks", side_effect=sql_failure):
            self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_changed_source_rejected_before_publication(self):
        self.seed()
        original = self.embedding
        changed = False
        def change_source(**kwargs):
            nonlocal changed
            result = original(**kwargs)
            if not changed:
                changed = True
                with self.engine.begin() as db:
                    db.execute(text("insert into chunks select 500,project_id,document_id,doc_title,chunk_text,section,page,chunk_role,embedding from chunks where id=1"))
            return result
        self.client.embeddings.create = change_source
        self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_publication_rechecks_study_mode_after_ai(self):
        self.seed()
        original = self.embedding
        changed = False
        def accept_after_generation_started(**kwargs):
            nonlocal changed
            result = original(**kwargs)
            if not changed:
                changed = True
                # Simulate an independently completed state transition while no
                # publication lock is held. Publication must re-read this state.
                with self.engine.begin() as db:
                    db.execute(text("update study_modules set accepted_for_study=true, taxonomy_status='ready', status='accepted_for_study' where id='m'"))
            return result
        self.client.embeddings.create = accept_after_generation_started
        self.assertFalse(self.run_pipeline())
        self.assert_old()
        self.assertEqual(self.rows("select taxonomy_status,status from study_modules where id='m'"), [("ready","accepted_for_study")])

    def test_empty_teaching_chunk_is_explicit_failure(self):
        self.seed()
        with self.engine.begin() as db:
            db.execute(text("update chunks set chunk_text='' where id=5"))
        self.assertFalse(self.run_pipeline())
        self.assertEqual(self.ai_calls, 0)
        self.assert_old()

    def test_valid_json_with_omitted_candidate_uses_lossless_fallback(self):
        self.seed(45)
        original = self.chat
        def omit(**kwargs):
            result = original(**kwargs)
            if "DUPLICATE PAIR REVIEW" in kwargs["messages"][-1]["content"]:
                data = json.loads(result.choices[0].message.content)
                data["decisions"].pop()
                return response(data)
            return result
        self.client.chat.completions.create = omit
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(len(self.rows("select * from topics where module_id='m'")), 45)

    def test_publication_lock_timeout_retains_old_taxonomy(self):
        self.seed()
        original = main._lock_taxonomy_scope
        blocker = self.factory()
        def blocked(db, *args, **kwargs):
            # The publication lock is the second call; initial marking is allowed.
            if self.ai_calls:
                blocker.execute(text("select id from projects where id='p' for update"))
                try:
                    original(db, *args, **kwargs)
                finally:
                    blocker.rollback()
                    blocker.close()
            else:
                original(db, *args, **kwargs)
        with patch.object(main, "_lock_taxonomy_scope", side_effect=blocked):
            started = time.monotonic()
            self.assertFalse(self.run_pipeline())
            self.assertLess(time.monotonic()-started, 8)
        self.assert_old()

    def test_publication_lock_does_not_block_unrelated_project_study_writes(self):
        self.seed()
        with self.engine.begin() as db:
            db.execute(text("create table study_results(id integer, project_id text references projects(id))"))
        publisher = self.factory()
        try:
            main._lock_taxonomy_scope(publisher, "p", "m", require_editable=True)
            with self.engine.begin() as db:
                db.execute(text("set local lock_timeout='250ms'"))
                db.execute(text("insert into study_results values (1,'p')"))
        finally:
            publisher.rollback()
            publisher.close()
        self.assertEqual(self.rows("select count(*) from study_results"), [(1,)])
