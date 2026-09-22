import unittest
import io
from contextlib import redirect_stdout
from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool
from unittest.mock import patch

import main
from main import app


class StudyModulesApiTests(unittest.TestCase):
    def setUp(self):
        self.engine = create_engine(
            "sqlite://",
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
        )
        self._create_schema()
        app.dependency_overrides[main.verify_user] = lambda: {"id": "user-a"}

    def tearDown(self):
        app.dependency_overrides.clear()
        self.engine.dispose()

    def _session(self):
        connection = self.engine.connect()
        connection.execute(text("PRAGMA foreign_keys = ON"))
        return connection

    def _create_schema(self):
        with self.engine.begin() as db:
            db.execute(text("PRAGMA foreign_keys = ON"))
            db.execute(text("""
                create table projects (
                    id text primary key,
                    name text not null,
                    study_mode text default 'building',
                    professor_mode text default 'coverage',
                    user_id text
                )
            """))
            db.execute(text("""
                create table study_modules (
                    id text primary key,
                    project_id text not null references projects(id) on delete cascade,
                    name text not null,
                    order_index integer not null,
                    created_at text default CURRENT_TIMESTAMP not null,
                    status text default 'building' not null,
                    accepted_for_study boolean default false not null,
                    accepted_at text,
                    taxonomy_status text default 'not_started' not null,
                    unique(project_id, order_index)
                )
            """))
            db.execute(text("""
                create table documents (
                    id text primary key,
                    project_id text,
                    module_id text references study_modules(id) on delete set null,
                    module_order_index integer,
                    title text not null,
                    text text not null,
                    created_at text default CURRENT_TIMESTAMP
                )
            """))
            db.execute(text("""
                create table topics (
                    id text primary key,
                    project_id text,
                    document_id text,
                    module_id text references study_modules(id) on delete set null,
                    category text,
                    topic text,
                    description text,
                    is_display_topic boolean default true,
                    category_order_index integer,
                    topic_order_index integer,
                    first_source_page integer,
                    first_source_block_index integer
                )
            """))
            db.execute(text("""
                create table topic_chunks (
                    id text primary key,
                    topic_id text,
                    chunk_id integer
                )
            """))
            db.execute(text("""
                insert into projects (id, name, user_id)
                values
                ('project-a', 'Project A', 'user-a'),
                ('project-b', 'Project B', 'user-b')
            """))
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values
                ('module-b', 'project-b', 'Private Module', 1, 'building', 'not_started', false)
            """))

    def test_create_module_in_owned_project(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-a/modules",
                json={"name": "Slides 1-3"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["project_id"], "project-a")
        self.assertEqual(data["name"], "Slides 1-3")
        self.assertEqual(data["order_index"], 1)
        self.assertEqual(data["status"], "building")
        self.assertEqual(data["taxonomy_status"], "not_started")
        self.assertFalse(data["accepted_for_study"])
        self.assertIsNone(data["accepted_at"])

    def test_list_modules_ordered_by_order_index(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values
                ('module-2', 'project-a', 'Second', 2, 'building', 'not_started', false),
                ('module-1', 'project-a', 'First', 1, 'building', 'not_started', false)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-a/modules")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [module["name"] for module in response.json()["modules"]],
            ["First", "Second"],
        )

    def test_cannot_access_another_users_module(self):
        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-b/modules/module-b")

        self.assertEqual(response.status_code, 404)

    def test_module_project_foreign_key_is_enforced(self):
        with self.assertRaises(IntegrityError):
            with self.engine.begin() as db:
                db.execute(text("PRAGMA foreign_keys = ON"))
                db.execute(text("""
                    insert into study_modules (
                        id,
                        project_id,
                        name,
                        order_index,
                        status,
                        taxonomy_status,
                        accepted_for_study
                    )
                    values (
                        'orphan-module',
                        'missing-project',
                        'Orphan',
                        1,
                        'building',
                        'not_started',
                        false
                    )
                """))

    def test_project_deletion_removes_module_rows(self):
        with self.engine.begin() as db:
            db.execute(text("PRAGMA foreign_keys = ON"))
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values ('module-a', 'project-a', 'Module A', 1, 'building', 'not_started', false)
            """))
            db.execute(text("delete from projects where id = 'project-a'"))
            remaining = db.execute(
                text("select count(*) from study_modules where project_id = 'project-a'")
            ).scalar()

        self.assertEqual(remaining, 0)

    def test_legacy_documents_and_topics_with_null_module_id_still_work(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into documents (id, project_id, module_id, title, text)
                values ('document-1', 'project-a', null, 'Legacy.pdf', 'Legacy text')
            """))
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    document_id,
                    module_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values (
                    'topic-1',
                    'project-a',
                    'document-1',
                    null,
                    'Legacy',
                    'Legacy Topic',
                    'Still visible',
                    true
                )
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.get("/projects/project-a/topics")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["topics"][0]["topic"], "Legacy Topic")
        self.assertEqual(response.json()["topics"][0]["category"], "Legacy")

    def test_update_topic_category_moves_topic_within_owned_project(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values
                ('topic-a', 'project-a', 'Unassigned', 'Topic A', 'A', true),
                ('topic-b', 'project-a', 'Target Category', 'Topic B', 'B', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/topics/topic-a",
                json={"category": "Target Category"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["category"], "Target Category")

        with self.engine.begin() as db:
            category = db.execute(
                text("select category from topics where id = 'topic-a'")
            ).scalar()

        self.assertEqual(category, "Target Category")

    def test_update_topic_category_rejects_missing_target_category(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values ('topic-a', 'project-a', 'Unassigned', 'Topic A', 'A', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/topics/topic-a",
                json={"category": "Missing Category"},
            )

        self.assertEqual(response.status_code, 400)

    def test_update_topic_category_rejects_accepted_module(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values ('module-a', 'project-a', 'Module A', 1, 'accepted_for_study', 'ready', true)
            """))
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    module_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values
                ('topic-a', 'project-a', 'module-a', 'Unassigned', 'Topic A', 'A', true),
                ('topic-b', 'project-a', 'module-a', 'Target Category', 'Topic B', 'B', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/topics/topic-a",
                json={"category": "Target Category"},
            )

        self.assertEqual(response.status_code, 409)

    def test_update_topic_name_before_study_acceptance(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values ('topic-a', 'project-a', 'Category A', 'Old Topic', 'A', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/topics/topic-a",
                json={"topic": "New Topic"},
            )

        self.assertEqual(response.status_code, 200)

        with self.engine.begin() as db:
            topic_name = db.execute(
                text("select topic from topics where id = 'topic-a'")
            ).scalar()

        self.assertEqual(topic_name, "New Topic")

    def test_rename_topic_category_before_study_acceptance(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values ('module-a', 'project-a', 'Module A', 1, 'pending_study', 'ready', false)
            """))
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    module_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values
                ('topic-a', 'project-a', 'module-a', 'Old Category', 'Topic A', 'A', true),
                ('topic-b', 'project-a', 'module-a', 'Old Category', 'Topic B', 'B', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/topic-categories",
                json={
                    "current_category": "Old Category",
                    "new_category": "New Category",
                    "module_id": "module-a",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["updated_topics"], 2)

        with self.engine.begin() as db:
            categories = db.execute(
                text("select distinct category from topics order by category")
            ).fetchall()

        self.assertEqual([row[0] for row in categories], ["New Category"])

    def test_rename_topic_category_rejects_existing_category_name(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values
                ('topic-a', 'project-a', 'Category A', 'Topic A', 'A', true),
                ('topic-b', 'project-a', 'Category B', 'Topic B', 'B', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/topic-categories",
                json={
                    "current_category": "Category A",
                    "new_category": "Category B",
                    "module_id": None,
                },
            )

        self.assertEqual(response.status_code, 400)

    def test_merge_topics_moves_chunk_links_and_deletes_source_topic(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values
                ('topic-a', 'project-a', 'Category A', 'Source Topic', 'Source description', true),
                ('topic-b', 'project-a', 'Category A', 'Target Topic', 'Target description', true)
            """))
            db.execute(text("""
                insert into topic_chunks (id, topic_id, chunk_id)
                values
                ('link-a', 'topic-a', 101),
                ('link-b', 'topic-b', 102)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-a/topics/merge",
                json={
                    "source_topic_id": "topic-a",
                    "target_topic_id": "topic-b",
                    "new_topic_name": "Merged Topic",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["topic"], "Merged Topic")

        with self.engine.begin() as db:
            topic_names = db.execute(
                text("select id, topic, description from topics order by id")
            ).fetchall()
            chunk_links = db.execute(
                text("select topic_id, chunk_id from topic_chunks order by chunk_id")
            ).fetchall()

        self.assertEqual(len(topic_names), 1)
        self.assertEqual(topic_names[0][0], "topic-b")
        self.assertEqual(topic_names[0][1], "Merged Topic")
        self.assertIn("Target description", topic_names[0][2])
        self.assertIn("Source description", topic_names[0][2])
        self.assertEqual(chunk_links, [("topic-b", 101), ("topic-b", 102)])

    def test_merge_topics_rejects_cross_category_merge(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into topics (
                    id,
                    project_id,
                    category,
                    topic,
                    description,
                    is_display_topic
                )
                values
                ('topic-a', 'project-a', 'Category A', 'Topic A', 'A', true),
                ('topic-b', 'project-a', 'Category B', 'Topic B', 'B', true)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.post(
                "/projects/project-a/topics/merge",
                json={
                    "source_topic_id": "topic-a",
                    "target_topic_id": "topic-b",
                    "new_topic_name": "Merged Topic",
                },
            )

        self.assertEqual(response.status_code, 400)

    def test_begin_study_freezes_modules(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values ('module-a', 'project-a', 'Module A', 1, 'pending_study', 'ready', false)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.post("/projects/project-a/begin_study")

        self.assertEqual(response.status_code, 200)

        with self.engine.begin() as db:
            row = db.execute(
                text("""
                    select status, accepted_for_study, accepted_at
                    from study_modules
                    where id = 'module-a'
                """)
            ).fetchone()

        self.assertEqual(row[0], "accepted_for_study")
        self.assertTrue(row[1])
        self.assertIsNotNone(row[2])

    def test_patch_module_updates_lifecycle_fields(self):
        with self.engine.begin() as db:
            db.execute(text("""
                insert into study_modules (
                    id,
                    project_id,
                    name,
                    order_index,
                    status,
                    taxonomy_status,
                    accepted_for_study
                )
                values ('module-a', 'project-a', 'Module A', 1, 'building', 'not_started', false)
            """))

        client = TestClient(app)

        with patch.object(main, "SessionLocal", self._session):
            response = client.patch(
                "/projects/project-a/modules/module-a",
                json={
                    "name": "Updated Module",
                    "status": "accepted_for_study",
                    "taxonomy_status": "ready",
                    "accepted_for_study": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["name"], "Updated Module")
        self.assertEqual(data["status"], "accepted_for_study")
        self.assertEqual(data["taxonomy_status"], "ready")
        self.assertTrue(data["accepted_for_study"])
        self.assertIsNotNone(data["accepted_at"])

    def test_module_upload_schedules_one_module_level_taxonomy_task(self):
        uploaded_documents = [
            {"document_id": "doc-1", "title": "Lecture 1.pdf"},
            {"document_id": "doc-2", "title": "Lecture 2.pdf"},
        ]

        with patch.object(main, "process_topics_task") as process_topics_task:
            main.process_uploaded_documents_topics_task(
                "project-a",
                uploaded_documents,
                module_id="module-a",
            )

        process_topics_task.assert_called_once_with(
            "project-a",
            document_id=None,
            document_title="Lecture 1.pdf, Lecture 2.pdf",
            mark_project_completed=True,
            module_id="module-a",
        )

    def test_legacy_upload_still_schedules_document_level_taxonomy_tasks(self):
        uploaded_documents = [
            {"document_id": "doc-1", "title": "Lecture 1.pdf"},
            {"document_id": "doc-2", "title": "Lecture 2.pdf"},
        ]

        with patch.object(main, "process_topics_task") as process_topics_task:
            main.process_uploaded_documents_topics_task(
                "project-a",
                uploaded_documents,
            )

        self.assertEqual(process_topics_task.call_count, 2)
        process_topics_task.assert_any_call(
            "project-a",
            "doc-1",
            "Lecture 1.pdf",
            mark_project_completed=False,
        )
        process_topics_task.assert_any_call(
            "project-a",
            "doc-2",
            "Lecture 2.pdf",
            mark_project_completed=True,
        )

    def test_taxonomy_planner_request_has_timeout_and_output_bound(self):
        class FakeCompletions:
            def __init__(self):
                self.kwargs = None

            def create(self, **kwargs):
                self.kwargs = kwargs
                return SimpleNamespace(
                    choices=[
                        SimpleNamespace(
                            message=SimpleNamespace(
                                content=(
                                    '{"strategy":"compact","categories":['
                                    '{"name":"PHYSIOLOGY","role":"core",'
                                    '"source_focus":"systems",'
                                    '"topic_granularity":"broad",'
                                    '"target_topic_count":4}],'
                                    '"merge_guidance":"merge local details"}'
                                )
                            )
                        )
                    ]
                )

        completions = FakeCompletions()
        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=completions
            )
        )
        diagnostics = {}

        plan = main._generate_taxonomy_plan(
            fake_client,
            [
                {
                    "text": "Cardiac physiology and vascular control.",
                    "section": "Cardiovascular",
                    "document": "physiology.docx",
                    "page": None,
                    "chunk_id": 1,
                }
            ],
            "",
            "",
            "",
            "",
            {"organization_mode": "infer"},
            diagnostics=diagnostics,
        )

        self.assertEqual(plan["categories"][0]["name"], "PHYSIOLOGY")
        self.assertEqual(
            completions.kwargs["timeout"],
            main.TAXONOMY_PLANNER_REQUEST_TIMEOUT_SECONDS,
        )
        self.assertEqual(
            completions.kwargs["max_tokens"],
            main.TAXONOMY_PLANNER_MAX_OUTPUT_TOKENS,
        )
        self.assertEqual(completions.kwargs["model"], main.TAXONOMY_PLANNER_MODEL)
        self.assertEqual(diagnostics["planner_status"], "success")
        self.assertEqual(diagnostics["planner_sampled_chunks"], 1)
        self.assertGreater(diagnostics["planner_prompt_chars"], 0)
        self.assertEqual(diagnostics["planner_category_count"], 1)
        self.assertEqual(diagnostics["planner_target_topic_total"], 4)

    def test_taxonomy_planner_error_diagnostics_are_recorded(self):
        class FakeCompletions:
            def create(self, **_kwargs):
                raise TimeoutError("planner timed out")

        fake_client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=FakeCompletions()
            )
        )
        diagnostics = {}

        with self.assertRaises(TimeoutError):
            main._generate_taxonomy_plan(
                fake_client,
                [
                    {
                        "text": "Respiratory physiology.",
                        "section": "Respiratory",
                        "document": "physiology.docx",
                        "page": None,
                        "chunk_id": 1,
                    }
                ],
                "",
                "",
                "",
                "",
                {"organization_mode": "infer"},
                diagnostics=diagnostics,
            )

        self.assertEqual(diagnostics["planner_status"], "error")
        self.assertEqual(diagnostics["planner_error_type"], "TimeoutError")
        self.assertIn("planner timed out", diagnostics["planner_error_message"])

    def test_syllabus_part_numbered_units_become_likely_categories(self):
        organization = {
            "organization_mode": "syllabus",
            "organization_blueprint": {
                "syllabus_text": "\n".join([
                    "PART I — BONE AND MUSCLE PHYSIOLOGY",
                    "",
                    "1. General Muscle Physiology",
                    "Muscle Types and Functional Specialization",
                    "Modes of Activation and Energetic Basis of Contraction",
                    "",
                    "2. Skeletal Muscle",
                    "Skeletal Muscle Architecture and Structural Organization",
                    "Organization of Skeletal Muscle",
                    "",
                    "3. Skeletal Muscle Contraction",
                    "Sliding Filament Mechanism",
                    "",
                    "PART II — CARDIOVASCULAR PHYSIOLOGY",
                    "",
                    "9. Fundamentals of Hemodynamics",
                    "Blood Flow, Pressure and Resistance",
                    "10. Cardiac Electrophysiology",
                    "SA Node",
                ])
            },
        }

        outline = main._parse_syllabus_outline(organization)
        likely_categories = main._syllabus_likely_category_items(outline)
        likely_titles = [
            item["display_title"].upper()
            for item in likely_categories
        ]
        instruction = main._build_syllabus_structure_instruction(organization)

        self.assertIn("GENERAL MUSCLE PHYSIOLOGY", likely_titles)
        self.assertIn("SKELETAL MUSCLE", likely_titles)
        self.assertIn("FUNDAMENTALS OF HEMODYNAMICS", likely_titles)
        self.assertNotIn("BONE AND MUSCLE PHYSIOLOGY", likely_titles)
        self.assertIn(
            "Use the numbered items as the visible category-level study blocks",
            instruction,
        )
        self.assertIn(
            "the index is a guide to the correct",
            instruction,
        )
        self.assertIn(
            "Generic index labels such as \"Premessa\"",
            instruction,
        )
        self.assertIn(
            "not text to copy literally",
            instruction,
        )

    def test_syllabus_guard_replaces_structural_chapter_labels(self):
        organization = {
            "organization_mode": "syllabus",
            "organization_blueprint": {
                "syllabus_text": "\n".join([
                    "CAPITOLO I",
                    "L'origine delle aziende di erogazione e di produzione",
                    "Classificazione delle aziende",
                    "CAPITOLO II",
                    "I caratteri istituzionali delle aziende",
                    "Unità e autonomia delle aziende",
                ])
            },
        }
        final_data = {
            "categories": [
                {
                    "name": "CAPITOLO I",
                    "topics": [
                        {
                            "title": "Classificazione delle aziende",
                            "description": "Classificazione aziendale.",
                        }
                    ],
                },
                {
                    "name": "CAPITOLO II",
                    "topics": [
                        {
                            "title": "Unità e autonomia delle aziende",
                            "description": "Caratteri istituzionali.",
                        }
                    ],
                },
            ]
        }

        guarded = main._apply_syllabus_category_guard(final_data, organization)
        category_names = [
            category["name"]
            for category in guarded["categories"]
        ]

        self.assertEqual(
            category_names,
            [
                "L'ORIGINE DELLE AZIENDE DI EROGAZIONE E DI PRODUZIONE",
                "I CARATTERI ISTITUZIONALI DELLE AZIENDE",
            ],
        )

    def test_manual_category_descriptions_become_topic_anchors(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Cardiac Function",
                        "description": (
                            "Cardiac Conduction System; Cardiac Cycle; "
                            "Stroke Volume; Frank–Starling Mechanism"
                        ),
                    }
                ]
            },
        }

        categories = main._extract_blueprint_categories(organization)
        anchors = categories[0]["topic_anchors"]
        organization_instruction = main._build_module_organization_instruction(
            organization
        )
        granularity_instruction = (
            main._build_student_taxonomy_granularity_instruction(organization)
        )
        strategy_instruction = (
            main._build_student_taxonomy_category_first_instruction(organization)
        )

        self.assertEqual(categories[0]["name"], "CARDIAC FUNCTION")
        self.assertIn("Cardiac Cycle", anchors)
        self.assertIn("Frank–Starling Mechanism", anchors)
        self.assertIn("Preferred topic anchors", organization_instruction)
        self.assertIn("Cardiac Conduction System", organization_instruction)
        self.assertIn("PREFERRED TOPIC ANCHORS BY CATEGORY", granularity_instruction)
        self.assertIn("CARDIAC FUNCTION: Cardiac Conduction System", granularity_instruction)
        self.assertIn("preferred topic anchors", strategy_instruction)

    def test_manual_topic_anchors_survive_taxonomy_plan_normalization(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Hemodynamics",
                        "description": (
                            "Blood Flow, Pressure and Resistance; "
                            "Driving Pressure; Vascular Resistance"
                        ),
                    }
                ]
            },
        }

        normalized = main._normalize_taxonomy_planning_result(
            {
                "strategy": "Use student categories.",
                "categories": [
                    {
                        "name": "HEMODYNAMICS",
                        "role": "Cardiovascular flow principles.",
                        "source_focus": "Relevant source content.",
                        "topic_granularity": "Broad topics.",
                        "target_topic_count": 4,
                    }
                ],
                "merge_guidance": "Merge details.",
            },
            organization,
        )

        category = normalized["categories"][0]
        self.assertEqual(category["name"], "HEMODYNAMICS")
        self.assertIn("Preferred topic anchors", category["topic_granularity"])
        self.assertIn("Driving Pressure", category["topic_granularity"])

    def test_manual_guard_rescues_unassigned_topics_matching_anchors(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Vascular Physiology",
                        "description": (
                            "Functional Vessel Types; Arterial Resistance; "
                            "Vascular Compliance; Arterial Elasticity; "
                            "Venous Compliance and Blood Storage"
                        ),
                    },
                    {
                        "name": "Cardiac Function",
                        "description": (
                            "Cardiac Cycle; Stroke Volume; Cardiac Output; "
                            "Autonomic Regulation of the Heart"
                        ),
                    },
                ]
            },
        }
        final_data = {
            "categories": [
                {
                    "name": "UNASSIGNED",
                    "topics": [
                        {
                            "title": "Role of Arterial Compartment and Elasticity",
                            "description": "Arterial elasticity and compliance in vessels.",
                        },
                        {
                            "title": "Distribution of Cardiac Output",
                            "description": "How cardiac output is distributed across circulation.",
                        },
                        {
                            "title": "Unrelated Topic",
                            "description": "A topic with no overlap.",
                        },
                    ],
                }
            ]
        }

        guarded = main._apply_module_organization_category_guard(
            final_data,
            organization,
        )
        grouped = {
            category["name"]: [
                topic["title"]
                for topic in category.get("topics", [])
            ]
            for category in guarded["categories"]
        }

        self.assertIn(
            "Role of Arterial Compartment and Elasticity",
            grouped["VASCULAR PHYSIOLOGY"],
        )
        self.assertIn(
            "Distribution of Cardiac Output",
            grouped["CARDIAC FUNCTION"],
        )
        self.assertIn("Unrelated Topic", grouped["UNASSIGNED"])

    def test_manual_guard_does_not_rescue_weak_anchor_overlap(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Hemodynamics",
                        "description": (
                            "Blood Flow, Pressure and Resistance; "
                            "Driving Pressure"
                        ),
                    }
                ]
            },
        }
        final_data = {
            "categories": [
                {
                    "name": "UNASSIGNED",
                    "topics": [
                        {
                            "title": "Current Flow and Membrane Properties",
                            "description": "Electrical membrane current in excitable cells.",
                        },
                    ],
                }
            ]
        }

        guarded = main._apply_module_organization_category_guard(
            final_data,
            organization,
        )

        self.assertEqual(guarded["categories"][0]["name"], "UNASSIGNED")
        self.assertEqual(
            guarded["categories"][0]["topics"][0]["title"],
            "Current Flow and Membrane Properties",
        )

    def test_manual_guard_restores_shortened_student_category_name(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Fondamenti della programmazione",
                        "description": (
                            "Concetto di programma e programmazione; "
                            "linguaggio macchina; linguaggi di alto livello"
                        ),
                    },
                    {
                        "name": "Algoritmi e logica di esecuzione",
                        "description": "Algoritmi; sequenzialità; selezione; cicli",
                    },
                ]
            },
        }
        final_data = {
            "categories": [
                {
                    "name": "Fondamenti",
                    "topics": [
                        {
                            "title": "Concetto di programma",
                            "description": "Introduzione al programma e alla programmazione.",
                        }
                    ],
                }
            ]
        }

        guarded = main._apply_module_organization_category_guard(
            final_data,
            organization,
        )

        self.assertEqual(
            guarded["categories"][0]["name"],
            "FONDAMENTI DELLA PROGRAMMAZIONE",
        )

    def test_manual_guard_rescues_unassigned_topics_matching_category_name(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Class B Push-Pull Output Stages",
                        "description": (
                            "BJT and MOSFET Class B push-pull stages, "
                            "crossover distortion, efficiency and power dissipation"
                        ),
                    },
                    {
                        "name": "Class AB Output Stages",
                        "description": "Class AB operating principle and quiescent current",
                    },
                ]
            },
        }
        final_data = {
            "categories": [
                {
                    "name": "UNASSIGNED",
                    "topics": [
                        {
                            "title": "MOSFET Class B Amplifier Characteristics",
                            "description": "MOSFET behavior in a Class B output stage.",
                        },
                        {
                            "title": "Efficiency and Power Dissipation in Class B",
                            "description": "Efficiency and dissipation limits for Class B.",
                        },
                    ],
                }
            ]
        }

        guarded = main._apply_module_organization_category_guard(
            final_data,
            organization,
        )
        grouped = {
            category["name"]: [
                topic["title"]
                for topic in category.get("topics", [])
            ]
            for category in guarded["categories"]
        }

        self.assertEqual(
            grouped["CLASS B PUSH-PULL OUTPUT STAGES"],
            [
                "MOSFET Class B Amplifier Characteristics",
                "Efficiency and Power Dissipation in Class B",
            ],
        )
        self.assertNotIn("UNASSIGNED", grouped)

    def test_manual_guard_deduplicates_strongly_overlapping_topics(self):
        organization = {
            "organization_mode": "manual",
            "organization_blueprint": {
                "categories": [
                    {
                        "name": "Fondamenti e principi delle indagini preliminari",
                        "description": (
                            "Funzione ed evoluzione delle indagini preliminari; "
                            "principi costituzionali collegati"
                        ),
                    }
                ]
            },
        }
        final_data = {
            "categories": [
                {
                    "name": "Fondamenti e principi delle indagini preliminari",
                    "topics": [
                        {
                            "title": "Definizione e funzione delle indagini preliminari",
                            "description": "Definizione e funzione della fase investigativa.",
                        },
                        {
                            "title": "Funzione e evoluzione delle indagini preliminari",
                            "description": "Evoluzione della funzione investigativa.",
                        },
                        {
                            "title": "Principi costituzionali collegati",
                            "description": "Principi costituzionali della fase.",
                        },
                    ],
                }
            ]
        }

        guarded = main._apply_module_organization_category_guard(
            final_data,
            organization,
        )
        topics = guarded["categories"][0]["topics"]

        self.assertEqual(len(topics), 2)
        self.assertEqual(
            topics[0]["title"],
            "Definizione e funzione delle indagini preliminari",
        )
        self.assertIn(
            "Evoluzione della funzione investigativa.",
            topics[0]["description"],
        )

    def test_shared_taxonomy_cleanup_deduplicates_automatic_topics(self):
        final_data = {
            "categories": [
                {
                    "name": "SVILUPPO E PROCESSO DI PROGRAMMAZIONE",
                    "topics": [
                        {
                            "title": "Fasi del Processo di Programmazione",
                            "description": "Fasi principali del processo.",
                        },
                        {
                            "title": "Processo di programmazione e sue fasi",
                            "description": "Sviluppo progressivo del programma.",
                        },
                        {
                            "title": "Individuazione del Problema e dei Dati",
                            "description": "Dati di ingresso, risultati e problema.",
                        },
                    ],
                }
            ]
        }

        cleaned = main._deduplicate_taxonomy_topics(final_data)
        topics = cleaned["categories"][0]["topics"]

        self.assertEqual(len(topics), 2)
        self.assertEqual(
            topics[0]["title"],
            "Fasi del Processo di Programmazione",
        )
        self.assertIn(
            "Sviluppo progressivo del programma.",
            topics[0]["description"],
        )

    def test_shared_taxonomy_cleanup_merges_anatomy_title_variants(self):
        final_data = {
            "categories": [
                {
                    "name": "GENERAL STRUCTURE OF A VERTEBRA",
                    "topics": [
                        {
                            "title": "Anatomy of a Vertebra",
                            "description": "Vertebral body, arch and processes.",
                        },
                        {
                            "title": "Anatomical Features of Vertebrae",
                            "description": "Pedicles, laminae and vertebral foramen.",
                        },
                    ],
                },
                {
                    "name": "SACRUM AND COCCYX",
                    "topics": [
                        {
                            "title": "Sacrum Anatomy and Structure",
                            "description": "General sacrum anatomy.",
                        },
                        {
                            "title": "Dorsal Surface of the Sacrum",
                            "description": "Specific dorsal surface landmarks.",
                        },
                    ],
                },
            ]
        }

        cleaned = main._deduplicate_taxonomy_topics(final_data)

        vertebra_topics = cleaned["categories"][0]["topics"]
        sacrum_topics = cleaned["categories"][1]["topics"]

        self.assertEqual(len(vertebra_topics), 1)
        self.assertIn(
            "Pedicles, laminae and vertebral foramen.",
            vertebra_topics[0]["description"],
        )
        self.assertEqual(len(sacrum_topics), 2)

    def test_taxonomy_planning_failure_summary_uses_planning_phase(self):
        logger = main.UploadPipelineLogger("TEST")
        logger.start("TAXONOMY PLANNING")
        output = io.StringIO()

        with redirect_stdout(output):
            logger.failure_summary(TimeoutError("planner timed out"))

        self.assertIn("Current phase: TAXONOMY PLANNING", output.getvalue())

    def test_background_summary_labels_taxonomy_chunks_separately(self):
        output = io.StringIO()

        with redirect_stdout(output):
            main.UploadPipelineLogger.print_summary(
                "TEST SUMMARY",
                {
                    "project_id": "project-a",
                    "phase_totals": {},
                    "counters": {
                        "taxonomy_chunks_loaded": 400,
                        "chunks_created": 477,
                    },
                    "total_elapsed": 1.0,
                },
            )

        rendered = output.getvalue()
        self.assertIn("Taxonomy chunks loaded", rendered)
        self.assertIn("Document chunks generated", rendered)
        self.assertNotIn("Pages processed", rendered)

    def test_topic_generation_timeout_uses_topic_generation_start(self):
        with patch.object(main.time, "time", return_value=1000):
            topic_generation_start = main.time.time()

        with patch.object(
            main.time,
            "time",
            return_value=topic_generation_start + main.MAX_TOPIC_PROCESSING_SECONDS - 1,
        ):
            self.assertFalse(
                main._topic_generation_timeout_reached(topic_generation_start)
            )

        with patch.object(
            main.time,
            "time",
            return_value=topic_generation_start + main.MAX_TOPIC_PROCESSING_SECONDS + 1,
        ):
            self.assertTrue(
                main._topic_generation_timeout_reached(topic_generation_start)
            )


if __name__ == "__main__":
    unittest.main()
