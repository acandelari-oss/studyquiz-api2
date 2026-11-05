"""Isolated PostgreSQL integration; external network blocked by test runner."""
import asyncio
import os
import json
import unittest
from unittest.mock import patch
from sqlalchemy import text
from fastapi import HTTPException
import main
from taxonomy_organization import REVIEW_MARKER
from tests.test_taxonomy_coverage_pipeline import TaxonomyPipelineTests, TEST_URL, response


@unittest.skipUnless(TEST_URL, "requires isolated DOUNO_TEST_DATABASE_URL")
class CategoryPublicationTests(unittest.TestCase):
    session = TaxonomyPipelineTests.session
    cleanup_database = TaxonomyPipelineTests.cleanup_database
    create_schema = TaxonomyPipelineTests.create_schema
    embedding = TaxonomyPipelineTests.embedding
    assert_no_ai_transaction = TaxonomyPipelineTests.assert_no_ai_transaction
    seed = TaxonomyPipelineTests.seed
    rows = TaxonomyPipelineTests.rows
    run_pipeline = TaxonomyPipelineTests.run_pipeline
    assert_old = TaxonomyPipelineTests.assert_old

    def setUp(self):
        self.organization_bad = False
        self.unmatched = False
        self.stages = []
        TaxonomyPipelineTests.setUp(self)
        self.stack.enter_context(patch.dict(os.environ, DOUNO_GLOBAL_CATEGORY_ORGANIZATION="1"))
        self.stack.enter_context(patch.object(main, "_study_module_organization_columns_available", return_value=True))
        self.stack.enter_context(patch.object(main, "_table_column_available", return_value=True))
        with self.engine.begin() as db:
            db.execute(text("""ALTER TABLE study_modules ADD COLUMN organization_mode text DEFAULT 'infer',
                ADD COLUMN organization_blueprint jsonb, ADD COLUMN organization_source_title text,
                ADD COLUMN name text DEFAULT 'Module', ADD COLUMN order_index integer DEFAULT 1"""))

    def chat(self, **kwargs):
        prompt = kwargs["messages"][-1]["content"]
        if "ORGANIZATION INPUT:\n" not in prompt:
            return TaxonomyPipelineTests.chat(self, **kwargs)
        self.assert_no_ai_transaction()
        payload = json.loads(prompt.split("ORGANIZATION INPUT:\n")[1])
        self.assertNotIn("current_category", prompt)
        if "categories" not in payload:
            self.stages.append("proposal")
            return response({"categories": [{"name": "Reorganized area", "description": "Study scope", "macrocategory": "Parent"}]})
        self.stages.append("assignment")
        rows = [{"topic_id": t["topic_id"], "category_id": None if self.unmatched else "C0000"}
                for t in payload["topics"]]
        if self.organization_bad:
            rows = rows[:-1]
        return response({"assignments": rows, "quality_issues": []})

    def manual(self):
        with self.engine.begin() as db:
            db.execute(text("""UPDATE study_modules SET organization_mode='manual',
                organization_blueprint='{"categories":[{"name":"Student Area","description":""},{"name":"Empty Area","description":""}]}'
                WHERE id='m'"""))

    def test_infer_publishes_and_preserves_all_evidence(self):
        self.seed(5)
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(self.stages, ["proposal", "assignment"])
        self.assertEqual(self.rows("select distinct category from topics where module_id='m'"), [("Reorganized area",)])
        self.assertEqual(self.rows("select count(distinct chunk_id) from topic_chunks where chunk_id<999999"), [(5,)])

    def test_incomplete_organization_preserves_old_taxonomy(self):
        self.seed(5)
        self.organization_bad = True
        self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_publication_failure_rolls_back_organized_topics(self):
        self.seed(5)
        with patch.object(main, "assign_topics_to_chunks", side_effect=RuntimeError("Injected publication error")):
            self.assertFalse(self.run_pipeline())
        self.assert_old()

    def test_flag_off_keeps_existing_pipeline(self):
        self.seed(5)
        with patch.dict(os.environ, DOUNO_GLOBAL_CATEGORY_ORGANIZATION="0"):
            self.assertTrue(self.run_pipeline())
        self.assertEqual(self.stages, [])

    def test_manual_unmatched_ready_and_reassignment_to_empty_category(self):
        self.seed(3)
        self.manual()
        self.unmatched = True
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(self.stages, ["assignment"])
        self.assertEqual(self.rows("select taxonomy_status from study_modules where id='m'"), [("ready",)])
        topics = asyncio.run(main.get_topics("p"))["topics"]
        pending = [t for t in topics if t["system_category"] == "to_organize"]
        self.assertEqual(len(pending), 3)
        self.assertIn("Empty Area", pending[0]["review_categories"])
        topic_id = pending[0]["id"]
        asyncio.run(main.update_topic_category("p", topic_id,
            main.TopicCategoryUpdateRequest(category="Empty Area"), user={"id": "u"}))
        topics = asyncio.run(main.get_topics("p"))["topics"]
        self.assertEqual(sum(t["system_category"] == "to_organize" for t in topics), 2)
        moved = next(t for t in topics if t["id"] == topic_id)
        self.assertIsNone(moved["system_category"])
        self.assertEqual(moved["category"], "Empty Area")
        for topic in topics:
            if topic["system_category"] == "to_organize":
                asyncio.run(main.update_topic_category("p", topic["id"],
                    main.TopicCategoryUpdateRequest(category="Student Area"), user={"id": "u"}))
        self.assertEqual(self.rows("select count(*) from topics where source_section='__DOUNO_CATEGORY_REVIEW__'"), [(0,)])

    def test_review_reassignment_enforces_owner_target_and_study_lock(self):
        self.seed(2)
        self.manual()
        self.unmatched = True
        self.assertTrue(self.run_pipeline())
        tid = self.rows("select id from topics where module_id='m' limit 1")[0][0]
        with self.assertRaises(HTTPException):
            asyncio.run(main.rename_topic_category("p",
                main.TopicCategoryRenameRequest(current_category="TO ORGANIZE", new_category="Renamed", module_id="m"),
                user={"id": "u"}))
        for user, target in (("intruder", "Student Area"), ("u", "Invented")):
            with self.assertRaises(HTTPException):
                asyncio.run(main.update_topic_category("p", tid,
                    main.TopicCategoryUpdateRequest(category=target), user={"id": user}))
        with self.engine.begin() as db:
            db.execute(text("update study_modules set accepted_for_study=true where id='m'"))
        with self.assertRaises(HTTPException):
            asyncio.run(main.update_topic_category("p", tid,
                main.TopicCategoryUpdateRequest(category="Student Area"), user={"id": "u"}))

    def test_syllabus_structure_is_fixed_and_unmatched_is_preserved(self):
        self.seed(3)
        blueprint = {"syllabus_text": "Part I: Foundations\nChapter 1: Principles\n1.1 Applications"}
        organization = {"organization_mode": "syllabus", "organization_blueprint": blueprint}
        names = [c["name"] for c in main._global_organization_categories(organization)]
        self.assertEqual(names, ["Foundations", "Foundations / Principles",
                                "Foundations / Principles / Applications"])
        with self.engine.begin() as db:
            db.execute(text("""UPDATE study_modules SET organization_mode='syllabus',
                organization_blueprint=CAST(:blueprint AS jsonb) WHERE id='m'"""),
                {"blueprint": json.dumps(blueprint)})
        self.unmatched = True
        self.assertTrue(self.run_pipeline(), self.output.getvalue()[-4000:])
        self.assertEqual(self.stages, ["assignment"])
        pending = [t for t in asyncio.run(main.get_topics("p"))["topics"]
                   if t["system_category"] == "to_organize"]
        self.assertEqual(len(pending), 3)
        self.assertEqual(pending[0]["review_categories"], names)

    def test_organization_constraint_change_prevents_publication(self):
        self.seed(3)
        original = main.organize_categories
        def change(*args, **kwargs):
            result = original(*args, **kwargs)
            self.manual()
            return result
        with patch.object(main, "organize_categories", side_effect=change):
            self.assertFalse(self.run_pipeline())
        self.assert_old()
