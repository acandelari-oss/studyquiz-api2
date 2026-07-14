print("🚨 MAIN.PY LOADED 🚨")
import os
import uuid
import base64
import io
import math
import unicodedata
from dataclasses import dataclass, replace
from difflib import SequenceMatcher
from typing import List, Optional
import json
import requests
import random
from fastapi import FastAPI, Depends, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text as sql_text
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import asyncio
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from urllib.parse import unquote
from fastapi import HTTPException
from fastapi import Body
from typing import Optional, List
from language_registry import (
    get_enabled_language,
    get_enabled_languages,
)
from chunk_roles import (
    classify_chunk_role,
    is_assignment_eligible_chunk_role,
    log_chunk_role_counts,
    normalize_chunk_role,
)
from planner.context_builder import build_real_planner_context
from planner.planner_engine import PlannerEngine
from planner.planner_models import PlannerContext, PlannerPreferences
from planner.planner_repository import PlannerRepository, build_planning_parameters
from planner.planner_serializers import serialize_planner_domain
from planner.planner_state import PlannerState
from planner.planner_state_evaluator import (
    PlannerStateEvaluator,
    serialize_planner_state_evaluation,
)
from planner.professor_knowledge import ProfessorKnowledgeBuilder
from planner.professor_voice import ProfessorVoiceService
from planner.survey_bootstrap import (
    apply_survey_bootstrap_bias,
    should_apply_survey_bootstrap,
)
import time
import re

MAX_RECOMMENDED_PAGES = 150
DEFAULT_MAX_VISIBLE_PLANNER_MODULES = 12
MAX_VISIBLE_PLANNER_MODULES = int(
    os.getenv(
        "MAX_VISIBLE_PLANNER_MODULES",
        str(DEFAULT_MAX_VISIBLE_PLANNER_MODULES),
    )
)
MAX_WARNING_PAGES = 100
MAX_STRONG_WARNING_PAGES = 180

MAX_TOPIC_PROCESSING_SECONDS = 600
MAX_ASSIGNMENT_MATCHES = 30000
HARD_ACCEPTED_DIAGNOSTIC_SAMPLE_SIZE = 10
PRIMARY_ASSIGNMENT_THRESHOLD = 0.54
MAX_TOPICS_PER_CHUNK = 3
TOPIC_RESCUE_THRESHOLD = 0.46
MIN_TAXONOMY_LANGUAGE_CONFIDENCE = 0.80

CATEGORY_CONSOLIDATION_VERSION = "v1.1"
CATEGORY_RECIPROCAL_NEIGHBORS = 2
CATEGORY_MERGE_SCORE_THRESHOLD = 0.78
CATEGORY_PAIR_CENTROID_THRESHOLD = 0.75
CATEGORY_PAIR_AFFINITY_THRESHOLD = 0.60
CATEGORY_MIN_COHESION = 0.80
CATEGORY_MAX_COHESION_LOSS = 0.06
CATEGORY_MAX_SEMANTIC_SPREAD = 0.58
CATEGORY_MAX_PROJECT_FRACTION = 0.25
CATEGORY_ABSOLUTE_TOPIC_LIMIT = 40
CATEGORY_LEXICAL_ALIAS_THRESHOLD = 0.60
CATEGORY_ALIAS_CENTROID_THRESHOLD = 0.82
CATEGORY_ALIAS_AFFINITY_THRESHOLD = 0.65
CATEGORY_QUALITY_ALIAS_LEXICAL_THRESHOLD = 0.40
CATEGORY_QUALITY_ALIAS_CENTROID_THRESHOLD = 0.72
CATEGORY_QUALITY_ALIAS_AFFINITY_THRESHOLD = 0.62
CATEGORY_QUALITY_ALIAS_MAX_SIZE_RATIO = 2.0
CATEGORY_QUALITY_BALANCED_ALIAS_MAX_SIZE_RATIO = 1.5
CATEGORY_QUALITY_ALIAS_MAX_TOPICS = 12
CATEGORY_QUALITY_ALIAS_MAX_COHESION_LOSS = 0.065

def normalize_string(s: str) -> str:
    if not s: return ""
    # Sostituisce \xa0 (non-breaking space) con spazio normale
    s = str(s).replace('\xa0', ' ')
    # Riduce spazi multipli a uno solo
    return re.sub(r'\s+', ' ', s).strip()


def calculate_topic_chunk_score(
    chunk_text,
    chunk_section,
    topic_name,
    topic_section,
    negative_inner_product,
):
    if negative_inner_product is None:
        return None

    chunk_text = (chunk_text or "").lower()
    chunk_section = (chunk_section or "").lower()
    topic_name = (topic_name or "").lower()
    topic_section = (topic_section or "").lower()

    similarity = -negative_inner_product
    same_section = chunk_section == topic_section
    keyword_overlap = sum(
        1
        for word in topic_name.split()
        if len(word) > 4 and word in chunk_text
    )
    section_bonus = 0.20 if same_section else 0

    return (
        similarity
        + section_bonus
        + (keyword_overlap * 0.05)
    )


def ensure_project_chunk_roles(db, project_id):
    rows = db.execute(
        text("""
            select id, chunk_text, page, doc_title, chunk_role
            from chunks
            where project_id = :project_id
            order by page, id
        """),
        {"project_id": project_id}
    ).fetchall()

    updates = []
    roles = []

    for row in rows:
        role = normalize_chunk_role(
            row[4],
            row[1],
            page_number=row[2],
            doc_title=row[3],
        )
        roles.append(role)

        if row[4] != role:
            updates.append({
                "chunk_id": row[0],
                "chunk_role": role,
            })

    if updates:
        db.execute(
            text("""
                update chunks
                set chunk_role = :chunk_role
                where id = :chunk_id
            """),
            updates
        )

    log_chunk_role_counts(roles)
    return roles

class ActiveRecallRequest(BaseModel):
    topics: Optional[List[str]] = None
    index: int = 0
    language: str = "English"

print("✅ ActiveRecallRequest model loaded with topics")
print("🚨 MAIN.PY LOADED 🚨")



import os

if os.path.exists("/opt/homebrew/bin/tesseract"):
    pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"




# ======================
# LOAD ENV
# ======================

load_dotenv()

topic_index = 0

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not DATABASE_URL:
    raise Exception("DATABASE_URL missing")

if not SUPABASE_URL or not SUPABASE_ANON_KEY:
    raise Exception("Supabase env missing")


# ======================
# APP INIT
# ======================

app = FastAPI()
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=OPENAI_API_KEY)
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# ======================
# AUTH
# ======================

def verify_user(authorization: str = Header(None)):
    
    print("AUTH HEADER:", authorization)

    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    response = requests.get(
        f"{SUPABASE_URL}/auth/v1/user",
        headers={
            "Authorization": authorization,
            "apikey": SUPABASE_ANON_KEY
        }
    )

    if response.status_code != 200:
        raise HTTPException(status_code=401, detail="Invalid token")

    return response.json()


# ======================
# MODELS
# ======================

class ProjectCreate(BaseModel):
    name: str


class QuizRequest(BaseModel):
    num_questions: int
    difficulty: str
    language: str

    question_style: str = "balanced"

    # 🔥 NUOVO SISTEMA
    topic_ids: Optional[List[str]] = []

    # 🔥 LEGACY TEMPORANEO
    topics: Optional[List[str]] = []

    # 🔥 LEGACY TEMPORANEO
    topic: Optional[str] = None

class IngestDocument(BaseModel):
    title: str
    file_bytes: str  # PDF base64


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class PlannerGenerationPreferences(BaseModel):
    studyDurationMinutes: int
    questionPaceSeconds: int
    sessionsPerWeek: Optional[int] = None
    questionStyle: str


class PlannerGenerationConfiguration(BaseModel):
    survey: Optional[dict] = None
    preferences: PlannerGenerationPreferences
    study_language: Optional[str] = "English"


class PlannerProfessorActivityDebriefRequest(BaseModel):
    module_index: int
    activity_result: dict
    study_language: Optional[str] = None


class PlannerProfessorModuleDebriefRequest(BaseModel):
    module_index: int
    module_results: dict
    study_language: Optional[str] = None


class PlannerProfessorHomeworkRecommendationRequest(BaseModel):
    module_index: int
    module_results: dict
    study_language: Optional[str] = None


class PlannerProfessorModuleQuestionMessage(BaseModel):
    role: str
    content: str


class PlannerProfessorModuleQuestionRequest(BaseModel):
    module_index: int
    question: str
    module_results: dict
    conversation: Optional[List[PlannerProfessorModuleQuestionMessage]] = None
    study_language: Optional[str] = None


class PlannerProfessorStudyPlanDebriefRequest(BaseModel):
    study_plan_results: dict
    study_language: Optional[str] = None


# ======================
# HEALTH
# ======================

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/planner/week")
def get_planner_week_without_project():
    raise HTTPException(
        status_code=400,
        detail="project_id is required for Study Planner.",
    )


@app.get("/projects/{project_id}/planner/week")
def get_project_planner_week(project_id: str):
    db = SessionLocal()
    try:
        context = build_real_planner_context(db, project_id=project_id)

        if not context.project:
            raise HTTPException(
                status_code=404,
                detail="Project not found for Study Planner.",
            )

        resolved_project_id = str(context.project["id"])
        repository = PlannerRepository(db)
        evaluation = PlannerStateEvaluator(
            db,
            repository=repository,
        ).evaluate(project_id=resolved_project_id)

        response = serialize_planner_state_evaluation(evaluation)

        if evaluation.state == PlannerState.ACTIVE_WEEK:
            response["week"] = serialize_planner_domain(evaluation.active_week)

        return response
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/week/generate")
def generate_project_planner_week(
    project_id: str,
    req: PlannerGenerationConfiguration,
):
    db = SessionLocal()
    try:
        context = build_real_planner_context(db, project_id=project_id)

        if not context.project:
            raise HTTPException(
                status_code=404,
                detail="Project not found for Study Planner.",
            )

        resolved_project_id = str(context.project["id"])
        repository = PlannerRepository(db)
        evaluation = PlannerStateEvaluator(
            db,
            repository=repository,
        ).evaluate(project_id=resolved_project_id)

        if evaluation.state == PlannerState.ACTIVE_WEEK:
            response = serialize_planner_state_evaluation(evaluation)
            response["week"] = serialize_planner_domain(evaluation.active_week)
            return response

        _validate_planner_generation_configuration(
            req=req,
            project_categories=set(context.categories),
            planner_state=evaluation.state,
        )

        analytics = context.analytics
        if should_apply_survey_bootstrap(
            survey=req.survey,
            is_first_study_plan=not _project_has_planner_week_history(
                db,
                resolved_project_id,
            ),
            has_objective_learning_evidence=_project_has_objective_learning_evidence(
                db,
                resolved_project_id,
            ),
        ):
            analytics = apply_survey_bootstrap_bias(
                analytics=context.analytics,
                survey=req.survey,
            )

        generation_context = PlannerContext(
            project=context.project,
            categories=context.categories,
            topics_by_category=context.topics_by_category,
            analytics=analytics,
            preferences=PlannerPreferences(
                question_pace_seconds=req.preferences.questionPaceSeconds,
                question_style=req.preferences.questionStyle,
            ),
            study_language=req.study_language or context.study_language,
            number_of_sessions=MAX_VISIBLE_PLANNER_MODULES,
            planning_budget_minutes=req.preferences.studyDurationMinutes,
            week_start_date=context.week_start_date,
            week_id=context.week_id,
        )
        week = PlannerEngine().generate_week(generation_context)
        additional_modules_remain = _planner_additional_modules_remain(
            context=context,
            week=week,
        )
        week = replace(
            week,
            weekly_statistics=replace(
                week.weekly_statistics,
                metadata={
                    **dict(week.weekly_statistics.metadata or {}),
                    "max_visible_modules": MAX_VISIBLE_PLANNER_MODULES,
                    "additional_modules_remain": additional_modules_remain,
                },
            ),
        )
        planning_parameters = {
            **build_planning_parameters(generation_context),
            "plan_type": "study_plan",
            "onboarding_mode": (
                "self_assessment"
                if req.survey
                else "planner_configuration"
            ),
            "studyDurationMinutes": req.preferences.studyDurationMinutes,
            "questionPaceSeconds": req.preferences.questionPaceSeconds,
            "maxVisibleModules": MAX_VISIBLE_PLANNER_MODULES,
            "additionalModulesRemain": additional_modules_remain,
            "questionStyle": req.preferences.questionStyle,
            "studyLanguage": req.study_language or context.study_language,
            "survey": req.survey,
        }
        week = repository.save_active_week(
            project_id=resolved_project_id,
            week=week,
            planning_parameters=planning_parameters,
        )

        response = serialize_planner_state_evaluation(
            PlannerStateEvaluator(
                db,
                repository=repository,
            ).evaluate(project_id=resolved_project_id)
        )
        response["week"] = serialize_planner_domain(week)
        return response
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/assessment/generate")
def generate_project_planner_assessment(
    project_id: str,
    req: PlannerGenerationConfiguration,
):
    db = SessionLocal()
    try:
        context = build_real_planner_context(db, project_id=project_id)

        if not context.project:
            raise HTTPException(
                status_code=404,
                detail="Project not found for Study Planner.",
            )

        resolved_project_id = str(context.project["id"])
        repository = PlannerRepository(db)
        evaluation = PlannerStateEvaluator(
            db,
            repository=repository,
        ).evaluate(project_id=resolved_project_id)

        if evaluation.state == PlannerState.ACTIVE_WEEK:
            response = serialize_planner_state_evaluation(evaluation)
            response["week"] = serialize_planner_domain(evaluation.active_week)
            return response

        validation_req = req.model_copy(update={"survey": None})
        _validate_planner_generation_configuration(
            req=validation_req,
            project_categories=set(context.categories),
            planner_state=PlannerState.READY_FOR_FIRST_PLAN,
        )

        generation_context = PlannerContext(
            project=context.project,
            categories=context.categories,
            topics_by_category=context.topics_by_category,
            analytics=context.analytics,
            preferences=PlannerPreferences(
                question_pace_seconds=req.preferences.questionPaceSeconds,
                question_style=req.preferences.questionStyle,
            ),
            study_language=req.study_language or context.study_language,
            number_of_sessions=0,
            planning_budget_minutes=req.preferences.studyDurationMinutes,
            week_start_date=context.week_start_date,
            week_id=f"{context.project['id']}-assessment-{context.week_start_date.isoformat()}",
        )
        week = PlannerEngine().generate_assessment_week(generation_context)
        planning_parameters = {
            **build_planning_parameters(generation_context),
            "plan_type": "assessment",
            "onboarding_mode": "professor_assessment",
            "studyDurationMinutes": req.preferences.studyDurationMinutes,
            "questionPaceSeconds": req.preferences.questionPaceSeconds,
            "questionStyle": req.preferences.questionStyle,
            "studyLanguage": req.study_language or context.study_language,
            "survey": None,
        }
        week = repository.save_active_week(
            project_id=resolved_project_id,
            week=week,
            planning_parameters=planning_parameters,
        )

        response = serialize_planner_state_evaluation(
            PlannerStateEvaluator(
                db,
                repository=repository,
            ).evaluate(project_id=resolved_project_id)
        )
        response["week"] = serialize_planner_domain(week)
        return response
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/assessment/complete")
def complete_project_planner_assessment(project_id: str):
    db = SessionLocal()
    try:
        context = build_real_planner_context(db, project_id=project_id)

        if not context.project:
            raise HTTPException(
                status_code=404,
                detail="Project not found for Study Planner.",
            )

        repository = PlannerRepository(db)
        active_week = repository.load_active_week(project_id=str(context.project["id"]))

        if not active_week:
            return serialize_planner_state_evaluation(
                PlannerStateEvaluator(
                    db,
                    repository=repository,
                ).evaluate(project_id=str(context.project["id"]))
            )

        if active_week.plan_type != "assessment":
            raise HTTPException(
                status_code=400,
                detail="Active Planner plan is not an assessment.",
            )

        repository.complete_active_week(project_id=str(context.project["id"]))
        return serialize_planner_state_evaluation(
            PlannerStateEvaluator(
                db,
                repository=repository,
            ).evaluate(project_id=str(context.project["id"]))
        )
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/professor/activity-debrief")
def generate_project_planner_activity_debrief(
    project_id: str,
    req: PlannerProfessorActivityDebriefRequest,
):
    db = SessionLocal()
    try:
        knowledge = _build_active_planner_professor_knowledge(
            db,
            project_id,
            study_language=req.study_language,
        )
        debrief = ProfessorVoiceService().generate_activity_debrief(
            knowledge,
            req.module_index,
            req.activity_result,
        )
        return {"debrief": debrief}
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/professor/module-debrief")
def generate_project_planner_module_debrief(
    project_id: str,
    req: PlannerProfessorModuleDebriefRequest,
):
    db = SessionLocal()
    try:
        knowledge = _build_active_planner_professor_knowledge(
            db,
            project_id,
            study_language=req.study_language,
        )
        debrief = ProfessorVoiceService().generate_module_debrief(
            knowledge,
            req.module_index,
            req.module_results,
        )
        return {"debrief": debrief}
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/professor/homework-recommendation")
def generate_project_planner_homework_recommendation(
    project_id: str,
    req: PlannerProfessorHomeworkRecommendationRequest,
):
    db = SessionLocal()
    try:
        knowledge = _build_active_planner_professor_knowledge(
            db,
            project_id,
            study_language=req.study_language,
        )
        homework = ProfessorVoiceService().generate_homework_recommendation(
            knowledge,
            req.module_index,
            req.module_results,
        )
        return {
            "homework_recommendation": {
                "text": homework,
                "rationale": "",
                "related_categories": [],
                "estimated_effort": 10,
            }
        }
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/professor/module-question")
def answer_project_planner_module_question(
    project_id: str,
    req: PlannerProfessorModuleQuestionRequest,
):
    db = SessionLocal()
    try:
        knowledge = _build_active_planner_professor_knowledge(
            db,
            project_id,
            study_language=req.study_language,
        )
        answer = ProfessorVoiceService().generate_module_question_answer(
            knowledge,
            req.module_index,
            req.module_results,
            req.question,
            [
                {
                    "role": message.role,
                    "content": message.content,
                }
                for message in (req.conversation or [])
            ],
        )
        return {"answer": answer}
    finally:
        db.close()


@app.post("/projects/{project_id}/planner/professor/study-plan-debrief")
def generate_project_planner_study_plan_debrief(
    project_id: str,
    req: PlannerProfessorStudyPlanDebriefRequest,
):
    db = SessionLocal()
    try:
        knowledge = _build_active_planner_professor_knowledge(
            db,
            project_id,
            study_language=req.study_language,
        )
        debrief = ProfessorVoiceService().generate_study_plan_debrief(
            knowledge,
            req.study_plan_results,
        )
        return {"debrief": debrief}
    finally:
        db.close()


def _build_active_planner_professor_knowledge(
    db,
    project_id: str,
    study_language: Optional[str] = None,
):
    context = build_real_planner_context(db, project_id=project_id)

    if not context.project:
        raise HTTPException(
            status_code=404,
            detail="Project not found for Study Planner.",
        )

    resolved_project_id = str(context.project["id"])
    week = PlannerRepository(db).load_active_week(project_id=resolved_project_id)

    if not week:
        raise HTTPException(
            status_code=404,
            detail="No active Study Plan found for this project.",
        )

    metadata = dict(week.weekly_statistics.metadata or {})
    max_visible_modules = metadata.get("max_visible_modules")
    resolved_study_language = (
        study_language
        or week.study_language
        or _json_value(metadata, "study_language")
        or _json_value(metadata, "studyLanguage")
        or context.study_language
    )
    context = replace(context, study_language=resolved_study_language)
    week = replace(week, study_language=resolved_study_language)

    return ProfessorKnowledgeBuilder().build(
        context=context,
        week=week,
        max_visible_modules=(
            int(max_visible_modules)
            if isinstance(max_visible_modules, int)
            else None
        ),
        additional_modules_remain=metadata.get("additional_modules_remain"),
    )


def _json_value(value, key: str):
    if isinstance(value, dict):
        return value.get(key)
    return None


def _validate_planner_generation_configuration(
    req: PlannerGenerationConfiguration,
    project_categories: set,
    planner_state: PlannerState,
):
    if req.preferences.studyDurationMinutes not in {30, 45, 60}:
        raise HTTPException(
            status_code=400,
            detail="Invalid study duration.",
        )

    if req.preferences.questionPaceSeconds not in {30, 60, 90, 120}:
        raise HTTPException(
            status_code=400,
            detail="Invalid question pace.",
        )

    if req.preferences.questionStyle not in {"exam", "balanced", "reasoning"}:
        raise HTTPException(
            status_code=400,
            detail="Invalid quiz style.",
        )

    survey = req.survey or {}
    survey_categories = set(survey.keys())
    invalid_categories = survey_categories - project_categories

    if invalid_categories:
        raise HTTPException(
            status_code=400,
            detail="Invalid survey category.",
        )

    if planner_state == PlannerState.NEW_PROJECT:
        missing_categories = project_categories - survey_categories

        if missing_categories:
            raise HTTPException(
                status_code=400,
                detail="Survey is required for every project category.",
            )

    valid_survey_values = {"confident", "practice", "unsure"}

    if any(value not in valid_survey_values for value in survey.values()):
        raise HTTPException(
            status_code=400,
            detail="Invalid survey answer.",
        )


def _project_has_planner_week_history(db, project_id: str) -> bool:
    return bool(
        db.execute(
            text("""
                select 1
                from planner_weeks
                where project_id = :project_id
                limit 1
            """),
            {"project_id": project_id},
        ).fetchone()
    )


def _project_has_objective_learning_evidence(db, project_id: str) -> bool:
    quiz_evidence = db.execute(
        text("""
            select 1
            from quiz_answers qa
            join quiz_questions qq
                on qq.id = qa.question_id
            join quizzes q
                on q.id = qq.quiz_id
            where q.project_id = :project_id
            limit 1
        """),
        {"project_id": project_id},
    ).fetchone()

    if quiz_evidence:
        return True

    flashcard_evidence = db.execute(
        text("""
            select 1
            from flashcard_reviews
            where project_id = :project_id
            limit 1
        """),
        {"project_id": project_id},
    ).fetchone()

    return bool(flashcard_evidence)


def _planner_additional_modules_remain(context: PlannerContext, week) -> bool:
    """Return whether project topics remain outside the visible Study Plan."""

    all_topic_ids = {
        str(topic.id)
        for topics in context.topics_by_category.values()
        for topic in topics
        if getattr(topic, "id", None)
    }
    planned_topic_ids = {
        str(topic.id)
        for daily_plan in week.daily_plans
        for allocation in daily_plan.planned_allocations
        for topic in allocation.selected_topics
        if getattr(topic, "id", None)
    }

    return bool(all_topic_ids - planned_topic_ids)


# ======================
# CREATE PROJECT
# ======================

@app.post("/projects")
def create_project(
    data: ProjectCreate,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    project_id = str(uuid.uuid4())

    db.execute(
        text("""
            insert into projects (id, name, user_id)
            values (:id, :name, :user_id)
        """),
        {
            "id": project_id,
            "name": data.name,
            "user_id": user_id
        }
    )

    db.commit()
    db.close()

    return {
        "project_id": project_id,
        "name": data.name
    }


# ======================
# LIST PROJECTS
# ======================

@app.get("/projects")
def list_projects(
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    rows = db.execute(
        text("""
            select id, name
            from projects
            where user_id = :user_id
            order by name
        """),
        {"user_id": user_id}
    ).fetchall()

    db.close()

    return {
        "projects": [
            {"id": r[0], "name": r[1]}
            for r in rows
        ]
    }

# ======================
# DELETE PROJECT
# ======================

@app.delete("/projects/{project_id}")
def delete_project(
    project_id: str,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    # verifica che il progetto appartenga all'utente
    project = db.execute(
        text("""
            select id
            from projects
            where id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    if not project:
        db.close()
        raise HTTPException(status_code=404, detail="Project not found")

    # cancella i dati collegati
    db.execute(
        text("delete from chunks where project_id = :project_id"),
        {"project_id": project_id}
    )

    db.execute(
        text("delete from quizzes where project_id = :project_id"),
        {"project_id": project_id}
    )

    db.execute(
        text("delete from flashcards where project_id = :project_id"),
        {"project_id": project_id}
    )

    # cancella il progetto
    db.execute(
        text("delete from projects where id = :project_id"),
        {"project_id": project_id}
    )

    db.commit()
    db.close()

    return {"status": "deleted"}
# ======================
# LIST DOCUMENTS
# ======================

@app.get("/projects/{project_id}/documents")
def list_documents(
    project_id: str,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    project = db.execute(
        text("""
            select id from projects
            where id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    if not project:
        db.close()
        raise HTTPException(status_code=403, detail="Access denied")

    rows = db.execute(
        text("""
            select distinct doc_title
            from chunks
            where project_id = :project_id
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    return {
    "documents": [
        {"id": r[0], "title": r[0]}
        for r in rows
    ]
}
# ======================
# OCR FALLBACK
# ======================

def ocr_pdf_page(pdf_bytes, page_index):

    print(
        f"🖼️ OCR START PAGE {page_index+1}"
    )

    try:

        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_index + 1,
            last_page=page_index + 1
        )

        if not images:
            return ""

        img = images[0]

        text = pytesseract.image_to_string(img)

        return text.strip()

    except Exception as e:

        print("OCR ERROR:", e)

        return ""

# ======================
# TEXT CHUNKING
# ======================

def chunk_text(text, max_chars=1000, overlap=200):

    import re

    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for p in paragraphs:

        p = p.strip()

        if not p:
            continue

        if len(p) > max_chars:

            start = 0

            while start < len(p):
                end = start + max_chars
                sub = p[start:end]

                chunks.append(sub.strip())

                start += max_chars - overlap

            continue

        if len(current_chunk) + len(p) < max_chars:

            current_chunk += "\n\n" + p

        else:

            chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + p

    if current_chunk:
        chunks.append(current_chunk.strip())
    print("📄 TOTAL CHUNKS:", len(chunks))
    return chunks

# ======================
# INGEST (WITH PAGE SUPPORT)
# ======================

@app.post("/projects/{project_id}/ingest")
def ingest(
    project_id: str,
    data: IngestRequest,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    project = db.execute(
        text("""
            select id from projects
            where id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    if not project:
        db.close()
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        total_chunks_document = 0
        project_chunk_roles = []
        for doc in data.documents:

            pdf_bytes = base64.b64decode(doc.file_bytes)
            pdf_stream = io.BytesIO(pdf_bytes)

            reader = PdfReader(pdf_stream)

            for page_index, page in enumerate(reader.pages):

                #if page_index > 30:
                #    print("PAGE LIMIT REACHED → STOP")
                #    break
                print(f"PAGE {page_index+1}")
                page_text = page.extract_text()

                if not page_text or not page_text.strip():

                    print(
                        f"⚠️ OCR REQUIRED PAGE {page_index+1}"
                    )

                    page_text = ocr_pdf_page(
                        pdf_bytes,
                        page_index
                    )

                    if not page_text:
                        continue

                chunks = chunk_text(page_text)
                chunks = [c for c in chunks if len(c) > 100]

                total_chunks_document += len(chunks)

                print("CHUNKS CREATED:", len(chunks))

                print(
                    f"📄 PAGE {page_index+1} -> {len(chunks)} CHUNKS"
                )

                for chunk in chunks:
                    clean_topic = normalize_string(doc.title) if doc.title else "General"
                    chunk_role = classify_chunk_role(
                        chunk,
                        page_number=page_index + 1,
                        doc_title=doc.title,
                    )
                    project_chunk_roles.append(chunk_role)

                    emb = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    print("EMBEDDING CHUNK...")
                    embedding = emb.data[0].embedding
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                    db.execute(
                        text("""
                            insert into chunks
                            (
                                project_id,
                                doc_title,
                                chunk_text,
                                embedding,
                                page,
                                chunk_role
                            )
                            values
                            (
                                :project_id,
                                :doc_title,
                                :chunk_text,
                                CAST(:embedding AS vector),
                                :page,
                                :chunk_role
                            )
                        """),
                        {
                            "project_id": project_id,
                            "doc_title": doc.title,
                            "chunk_text": chunk,
                            "embedding": embedding_str,
                            "page": page_index + 1,
                            "topic": clean_topic,
                            "chunk_role": chunk_role,
                        }
                    )
                    
                    
                
                if (page_index + 1) % 10 == 0:
                    db.commit()
                    print(
                        f"💾 PARTIAL COMMIT PAGE {page_index+1}"
                    )   
                         
        print("START DOCUMENT:", doc.title)

        print(
            "📚 DOCUMENT TOTAL CHUNKS:",
            total_chunks_document
        )
        log_chunk_role_counts(project_chunk_roles)
        db.commit()

    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=str(e))

    db.close()
    return {"status": "ok"}

def should_hide_topic(topic_title: str):

    t = topic_title.lower().strip()

    weak_patterns = [
        "deep muscles",
        "superficial muscles",
        "compartments of",
        "muscles of the posterior compartment",
        "muscles of the anterior compartment",
        "pronators and supinators",
        "muscle origins and insertions",
        "actions and innervation",
    ]

    return any(p in t for p in weak_patterns)

def rebalance_taxonomy(final_data):

    MIN_TOPICS_PER_CATEGORY = 2

    categories = final_data.get("categories", [])

    strong_categories = []
    weak_topics = []

    # separa categorie forti/deboli
    for cat in categories:

        topics = cat.get("topics", [])

        if len(topics) >= MIN_TOPICS_PER_CATEGORY:
            strong_categories.append(cat)
        else:
            weak_topics.extend(topics)

    # fallback
    if not strong_categories:
        return final_data

    # category più grande
    main_category = max(
        strong_categories,
        key=lambda c: len(c.get("topics", []))
    )

    # sposta topic deboli
    main_category["topics"].extend(weak_topics)

    return {
        "categories": strong_categories
    }


@dataclass(frozen=True)
class TaxonomyTopicLedgerEntry:
    ledger_id: str
    original_category: str
    topic: str
    description: str
    importance: object
    source_position: tuple
    embedding: tuple


def _taxonomy_dot(vector_a, vector_b):
    return sum(
        value_a * value_b
        for value_a, value_b in zip(vector_a, vector_b)
    )


def _taxonomy_unit_vector(vector):
    magnitude = math.sqrt(_taxonomy_dot(vector, vector))

    if magnitude <= 0:
        raise ValueError("Taxonomy consolidation received a zero embedding")

    return tuple(value / magnitude for value in vector)


def _taxonomy_centroid(vectors):
    if not vectors:
        raise ValueError("Cannot calculate a centroid without topic embeddings")

    dimensions = len(vectors[0])
    centroid = [
        sum(vector[index] for vector in vectors) / len(vectors)
        for index in range(dimensions)
    ]

    return _taxonomy_unit_vector(centroid)


def _taxonomy_vector_metrics(vectors):
    centroid = _taxonomy_centroid(vectors)
    cohesion = sum(
        _taxonomy_dot(vector, centroid)
        for vector in vectors
    ) / len(vectors)

    if len(vectors) < 2:
        minimum_pair_similarity = 1.0
    else:
        minimum_pair_similarity = min(
            _taxonomy_dot(vectors[left], vectors[right])
            for left in range(len(vectors))
            for right in range(left + 1, len(vectors))
        )

    return {
        "centroid": centroid,
        "cohesion": cohesion,
        "maximum_spread": 1.0 - minimum_pair_similarity,
    }


def _normalize_category_name(category_name):
    normalized = "".join(
        character
        for character in unicodedata.normalize(
            "NFKD",
            (category_name or "").lower()
        )
        if not unicodedata.combining(character)
    )
    normalized = re.sub(r"[^a-z0-9 ]+", " ", normalized)

    normalized_tokens = []
    suffixes = (
        "zioni",
        "zione",
        "iche",
        "ici",
        "ale",
        "ali",
        "ica",
        "ico",
        "i",
        "e",
        "s",
    )

    for token in normalized.split():
        normalized_token = token

        for suffix in suffixes:
            if (
                len(normalized_token) > len(suffix) + 3
                and normalized_token.endswith(suffix)
            ):
                normalized_token = normalized_token[:-len(suffix)]
                break

        normalized_tokens.append(normalized_token)

    return " ".join(normalized_tokens), set(normalized_tokens)


def _category_lexical_similarity(category_a, category_b):
    normalized_a, tokens_a = _normalize_category_name(category_a)
    normalized_b, tokens_b = _normalize_category_name(category_b)
    token_union = tokens_a | tokens_b
    token_jaccard = (
        len(tokens_a & tokens_b) / len(token_union)
        if token_union
        else 0.0
    )
    character_similarity = SequenceMatcher(
        None,
        normalized_a,
        normalized_b
    ).ratio()

    return (
        (0.60 * token_jaccard)
        + (0.40 * character_similarity)
    )


_CATEGORY_NAME_NOISE_TOKENS = {
    "a",
    "an",
    "and",
    "case",
    "cases",
    "casi",
    "dei",
    "del",
    "della",
    "delle",
    "dell",
    "di",
    "e",
    "example",
    "examples",
    "general",
    "generale",
    "in",
    "of",
    "overview",
    "panoramica",
    "the",
    "tipi",
    "tipologie",
    "type",
    "types",
}


def _category_core_tokens(category_name):
    _, normalized_tokens = _normalize_category_name(category_name)

    return {
        token
        for token in normalized_tokens
        if token not in _CATEGORY_NAME_NOISE_TOKENS
    }


def _category_name_quality(category_name):
    normalized_name, normalized_tokens = _normalize_category_name(
        category_name
    )
    core_tokens = _category_core_tokens(category_name)
    noise_count = len(normalized_tokens - core_tokens)
    token_count = len(normalized_tokens)
    generic_penalty = 1 if any(
        token in {
            "general",
            "generale",
            "overview",
            "panoramica",
            "case",
            "cases",
            "casi",
            "type",
            "types",
            "tipi",
            "tipologie",
        }
        for token in normalized_tokens
    ) else 0

    return (
        generic_penalty,
        noise_count,
        token_count,
        len(normalized_name),
        category_name,
    )


def _category_alias_token_match_count(tokens_a, tokens_b):
    matched_tokens_b = set()
    match_count = 0

    for token_a in sorted(tokens_a):
        best_token_b = None

        for token_b in sorted(tokens_b - matched_tokens_b):
            common_prefix_length = len(
                os.path.commonprefix((token_a, token_b))
            )

            if (
                token_a == token_b
                or common_prefix_length >= 5
            ):
                best_token_b = token_b
                break

        if best_token_b is not None:
            matched_tokens_b.add(best_token_b)
            match_count += 1

    return match_count


def _is_category_alias(metrics, profiles):
    category_a = metrics["category_a"]
    category_b = metrics["category_b"]
    profile_a = profiles[category_a]
    profile_b = profiles[category_b]
    topic_count_a = profile_a["topic_count"]
    topic_count_b = profile_b["topic_count"]
    combined_topic_count = topic_count_a + topic_count_b
    size_ratio = (
        max(topic_count_a, topic_count_b)
        / min(topic_count_a, topic_count_b)
    )
    core_tokens_a = _category_core_tokens(category_a)
    core_tokens_b = _category_core_tokens(category_b)
    exact_core_alias = (
        bool(core_tokens_a)
        and core_tokens_a == core_tokens_b
    )
    core_token_matches = _category_alias_token_match_count(
        core_tokens_a,
        core_tokens_b,
    )
    normalized_core_alias = (
        bool(core_tokens_a)
        and len(core_tokens_a) == len(core_tokens_b)
        and core_token_matches == len(core_tokens_a)
    )
    meaningful_containment = (
        bool(core_tokens_a)
        and bool(core_tokens_b)
        and core_tokens_a != core_tokens_b
        and (
            core_tokens_a < core_tokens_b
            or core_tokens_b < core_tokens_a
        )
    )

    strict_alias = (
        not meaningful_containment
        and core_token_matches >= 2
        and size_ratio
        <= CATEGORY_QUALITY_BALANCED_ALIAS_MAX_SIZE_RATIO
        and
        metrics["lexical_similarity"]
        >= CATEGORY_LEXICAL_ALIAS_THRESHOLD
        and metrics["centroid_similarity"]
        >= CATEGORY_ALIAS_CENTROID_THRESHOLD
        and metrics["cross_topic_affinity"]
        >= CATEGORY_ALIAS_AFFINITY_THRESHOLD
    )
    quality_alias = (
        not meaningful_containment
        and core_token_matches >= 2
        and size_ratio
        <= CATEGORY_QUALITY_BALANCED_ALIAS_MAX_SIZE_RATIO
        and combined_topic_count
        <= CATEGORY_QUALITY_ALIAS_MAX_TOPICS
        and metrics["lexical_similarity"]
        >= CATEGORY_QUALITY_ALIAS_LEXICAL_THRESHOLD
        and metrics["centroid_similarity"]
        >= CATEGORY_QUALITY_ALIAS_CENTROID_THRESHOLD
        and metrics["cross_topic_affinity"]
        >= CATEGORY_QUALITY_ALIAS_AFFINITY_THRESHOLD
    )
    core_alias = (
        (exact_core_alias or normalized_core_alias)
        and size_ratio <= CATEGORY_QUALITY_ALIAS_MAX_SIZE_RATIO
        and combined_topic_count
        <= CATEGORY_QUALITY_ALIAS_MAX_TOPICS
        and metrics["centroid_similarity"] >= 0.68
        and metrics["cross_topic_affinity"] >= 0.58
    )

    if strict_alias:
        return "strict_lexical_semantic_alias"

    if core_alias:
        return "normalized_name_alias"

    if quality_alias:
        return "balanced_lexical_semantic_alias"

    return None


def _category_cross_topic_affinity(vectors_a, vectors_b):
    forward_affinity = sum(
        max(
            _taxonomy_dot(vector_a, vector_b)
            for vector_b in vectors_b
        )
        for vector_a in vectors_a
    ) / len(vectors_a)

    reverse_affinity = sum(
        max(
            _taxonomy_dot(vector_b, vector_a)
            for vector_a in vectors_a
        )
        for vector_b in vectors_b
    ) / len(vectors_b)

    return 0.5 * (forward_affinity + reverse_affinity)


def build_immutable_taxonomy_ledger(final_data):
    ledger_entries = []

    for category_index, category in enumerate(
        final_data.get("categories", [])
    ):
        original_category = (
            category.get("name") or "GENERAL"
        ).strip().upper()

        for topic_index, topic_object in enumerate(
            category.get("topics", [])
        ):
            topic_title = (
                topic_object.get("title")
                or topic_object.get("topic")
                or ""
            ).strip()

            if not topic_title:
                continue

            description = (
                topic_object.get("description") or ""
            ).strip()
            importance = topic_object.get("importance", 5)
            embedding_input = (
                f"{original_category} - "
                f"{topic_title}: "
                f"{description}"
            )
            embedding_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=embedding_input
            )
            embedding = tuple(
                embedding_response.data[0].embedding
            )

            ledger_entries.append(
                TaxonomyTopicLedgerEntry(
                    ledger_id=f"T{len(ledger_entries) + 1:04d}",
                    original_category=original_category,
                    topic=topic_title,
                    description=description,
                    importance=importance,
                    source_position=(category_index, topic_index),
                    embedding=embedding,
                )
            )

    if not ledger_entries:
        raise ValueError("Cannot consolidate an empty taxonomy")

    ledger_ids = [entry.ledger_id for entry in ledger_entries]

    if len(ledger_ids) != len(set(ledger_ids)):
        raise ValueError("Immutable topic ledger contains duplicate IDs")

    return tuple(ledger_entries)


def _build_category_profiles(topic_ledger):
    category_entries = {}

    for entry in topic_ledger:
        category_entries.setdefault(
            entry.original_category,
            []
        ).append(entry)

    profiles = {}

    for category_name, entries in category_entries.items():
        vectors = [
            _taxonomy_unit_vector(entry.embedding)
            for entry in entries
        ]
        metrics = _taxonomy_vector_metrics(vectors)
        profiles[category_name] = {
            "name": category_name,
            "entries": tuple(entries),
            "vectors": tuple(vectors),
            "topic_count": len(entries),
            **metrics,
        }

    return profiles


def _build_category_pair_metrics(profiles):
    category_names = sorted(profiles)
    pair_metrics = {}

    for left_index, category_a in enumerate(category_names):
        profile_a = profiles[category_a]

        for category_b in category_names[left_index + 1:]:
            profile_b = profiles[category_b]
            centroid_similarity = _taxonomy_dot(
                profile_a["centroid"],
                profile_b["centroid"]
            )
            cross_topic_affinity = _category_cross_topic_affinity(
                profile_a["vectors"],
                profile_b["vectors"]
            )
            lexical_similarity = _category_lexical_similarity(
                category_a,
                category_b
            )
            combined_vectors = (
                profile_a["vectors"]
                + profile_b["vectors"]
            )
            merged_metrics = _taxonomy_vector_metrics(
                combined_vectors
            )
            combined_topic_count = (
                profile_a["topic_count"]
                + profile_b["topic_count"]
            )
            weighted_source_cohesion = (
                (
                    profile_a["cohesion"]
                    * profile_a["topic_count"]
                )
                + (
                    profile_b["cohesion"]
                    * profile_b["topic_count"]
                )
            ) / combined_topic_count
            cohesion_loss = max(
                0.0,
                weighted_source_cohesion
                - merged_metrics["cohesion"]
            )
            cohesion_retention = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - (
                        cohesion_loss
                        / CATEGORY_MAX_COHESION_LOSS
                    )
                )
            )
            merge_score = (
                (0.45 * centroid_similarity)
                + (0.30 * cross_topic_affinity)
                + (0.15 * lexical_similarity)
                + (0.10 * cohesion_retention)
            )

            pair_metrics[(category_a, category_b)] = {
                "category_a": category_a,
                "category_b": category_b,
                "centroid_similarity": centroid_similarity,
                "cross_topic_affinity": cross_topic_affinity,
                "lexical_similarity": lexical_similarity,
                "cohesion_retention": cohesion_retention,
                "cohesion_loss": cohesion_loss,
                "merged_cohesion": merged_metrics["cohesion"],
                "maximum_spread": merged_metrics["maximum_spread"],
                "merge_score": merge_score,
            }

    return pair_metrics


def _category_pair_key(category_a, category_b):
    return tuple(sorted((category_a, category_b)))


def _calculate_category_size_limit(profiles):
    total_topics = sum(
        profile["topic_count"]
        for profile in profiles.values()
    )
    largest_original_category = max(
        profile["topic_count"]
        for profile in profiles.values()
    )
    relative_limit = math.ceil(
        total_topics * CATEGORY_MAX_PROJECT_FRACTION
    )

    return min(
        max(largest_original_category, relative_limit),
        CATEGORY_ABSOLUTE_TOPIC_LIMIT
    )


def _choose_canonical_category_name(group, profiles):
    ordered_group = sorted(group)

    if len(ordered_group) == 1:
        return ordered_group[0]

    ranked_names = []

    for category_name in ordered_group:
        other_names = [
            other_name
            for other_name in ordered_group
            if other_name != category_name
        ]
        centrality = sum(
            _taxonomy_dot(
                profiles[category_name]["centroid"],
                profiles[other_name]["centroid"]
            )
            for other_name in other_names
        ) / len(other_names)
        ranked_names.append(
            (
                _category_name_quality(category_name),
                -centrality,
                -profiles[category_name]["topic_count"],
                category_name,
            )
        )

    return min(ranked_names)[3]


def consolidate_taxonomy_categories_v1(topic_ledger):
    profiles = _build_category_profiles(topic_ledger)
    category_names = sorted(profiles)

    if len(category_names) < 2:
        identity_mapping = {
            category_name: category_name
            for category_name in category_names
        }
        final_groups = [
            {
                "canonical_category": category_name,
                "source_categories": [category_name],
                "topic_count": profiles[category_name]["topic_count"],
            }
            for category_name in category_names
        ]
        diagnostics = build_taxonomy_quality_diagnostics(
            topic_ledger=topic_ledger,
            category_mapping=identity_mapping,
            final_groups=final_groups,
            accepted_merges=[],
        )

        return {
            "mapping": identity_mapping,
            "groups": final_groups,
            "accepted_merges": [],
            "rejected_merges": [],
            "category_size_limit": (
                profiles[category_names[0]]["topic_count"]
                if category_names
                else 0
            ),
            "diagnostics": diagnostics,
        }

    pair_metrics = _build_category_pair_metrics(profiles)
    neighbor_count = min(
        CATEGORY_RECIPROCAL_NEIGHBORS,
        len(category_names) - 1
    )
    nearest_neighbors = {}

    for category_name in category_names:
        ranked_neighbors = sorted(
            (
                (
                    -_taxonomy_dot(
                        profiles[category_name]["centroid"],
                        profiles[other_name]["centroid"]
                    ),
                    other_name,
                )
                for other_name in category_names
                if other_name != category_name
            )
        )
        nearest_neighbors[category_name] = {
            other_name
            for _, other_name in ranked_neighbors[:neighbor_count]
        }

    candidate_edges = []
    rejected_merges = []

    for pair_key, metrics in pair_metrics.items():
        category_a, category_b = pair_key
        reciprocal_neighbors = (
            category_b in nearest_neighbors[category_a]
            and category_a in nearest_neighbors[category_b]
        )
        alias_reason = _is_category_alias(
            metrics,
            profiles,
        )
        score_valid = (
            metrics["merge_score"]
            >= CATEGORY_MERGE_SCORE_THRESHOLD
        )

        if reciprocal_neighbors and (score_valid or alias_reason):
            candidate_edges.append({
                **metrics,
                "candidate_reason": (
                    alias_reason
                    or "merge_score_above_threshold"
                ),
            })
        elif reciprocal_neighbors:
            rejected_merges.append({
                **metrics,
                "reason": "merge_score_below_threshold",
            })

    candidate_edges.sort(
        key=lambda edge: (
            -edge["merge_score"],
            edge["category_a"],
            edge["category_b"],
        )
    )

    groups = {
        category_name: {category_name}
        for category_name in category_names
    }
    group_owner = {
        category_name: category_name
        for category_name in category_names
    }
    accepted_merges = []
    category_size_limit = _calculate_category_size_limit(
        profiles
    )

    for edge in candidate_edges:
        category_a = edge["category_a"]
        category_b = edge["category_b"]
        owner_a = group_owner[category_a]
        owner_b = group_owner[category_b]

        if owner_a == owner_b:
            continue

        proposed_group = groups[owner_a] | groups[owner_b]
        proposed_topic_count = sum(
            profiles[category_name]["topic_count"]
            for category_name in proposed_group
        )

        if proposed_topic_count > category_size_limit:
            rejected_merges.append({
                **edge,
                "reason": "category_size_limit",
                "proposed_topic_count": proposed_topic_count,
            })
            continue

        incompatible_pair = None

        for left_index, group_category_a in enumerate(
            sorted(proposed_group)
        ):
            for group_category_b in sorted(proposed_group)[
                left_index + 1:
            ]:
                group_pair_metrics = pair_metrics[
                    _category_pair_key(
                        group_category_a,
                        group_category_b
                    )
                ]
                group_pair_alias = _is_category_alias(
                    group_pair_metrics,
                    profiles,
                )

                if (
                    not group_pair_alias
                    and (
                        group_pair_metrics["centroid_similarity"]
                        < CATEGORY_PAIR_CENTROID_THRESHOLD
                        or group_pair_metrics["cross_topic_affinity"]
                        < CATEGORY_PAIR_AFFINITY_THRESHOLD
                    )
                ):
                    incompatible_pair = (
                        group_category_a,
                        group_category_b,
                    )
                    break

            if incompatible_pair:
                break

        if incompatible_pair:
            rejected_merges.append({
                **edge,
                "reason": "complete_group_pair_incompatible",
                "incompatible_pair": incompatible_pair,
            })
            continue

        proposed_vectors = tuple(
            vector
            for category_name in sorted(proposed_group)
            for vector in profiles[category_name]["vectors"]
        )
        proposed_metrics = _taxonomy_vector_metrics(
            proposed_vectors
        )
        weighted_source_cohesion = sum(
            (
                profiles[category_name]["cohesion"]
                * profiles[category_name]["topic_count"]
            )
            for category_name in proposed_group
        ) / proposed_topic_count
        proposed_cohesion_loss = max(
            0.0,
            weighted_source_cohesion
            - proposed_metrics["cohesion"]
        )

        if proposed_metrics["cohesion"] < CATEGORY_MIN_COHESION:
            rejected_merges.append({
                **edge,
                "reason": "merged_cohesion_below_minimum",
                "proposed_cohesion": proposed_metrics["cohesion"],
            })
            continue

        alias_candidate = (
            edge["candidate_reason"]
            != "merge_score_above_threshold"
        )
        maximum_cohesion_loss = (
            CATEGORY_QUALITY_ALIAS_MAX_COHESION_LOSS
            if alias_candidate
            else CATEGORY_MAX_COHESION_LOSS
        )

        if proposed_cohesion_loss > maximum_cohesion_loss:
            rejected_merges.append({
                **edge,
                "reason": "cohesion_loss_above_maximum",
                "proposed_cohesion_loss": proposed_cohesion_loss,
                "maximum_cohesion_loss": maximum_cohesion_loss,
            })
            continue

        if (
            proposed_metrics["maximum_spread"]
            > CATEGORY_MAX_SEMANTIC_SPREAD
        ):
            rejected_merges.append({
                **edge,
                "reason": "semantic_spread_above_maximum",
                "proposed_maximum_spread": (
                    proposed_metrics["maximum_spread"]
                ),
            })
            continue

        new_owner = min(proposed_group)
        del groups[owner_a]
        del groups[owner_b]
        groups[new_owner] = proposed_group

        for category_name in proposed_group:
            group_owner[category_name] = new_owner

        accepted_merges.append({
            **edge,
            "source_categories": sorted(proposed_group),
            "topic_count": proposed_topic_count,
            "merged_cohesion": proposed_metrics["cohesion"],
            "maximum_spread": proposed_metrics["maximum_spread"],
            "cohesion_loss": proposed_cohesion_loss,
            "rationale": (
                "Categories are reciprocal semantic neighbors and passed "
                f"the {edge['candidate_reason']} gate while preserving "
                "merged size, cohesion, semantic spread, and complete-group "
                "compatibility."
            ),
        })

    category_mapping = {}
    final_groups = []

    for group in sorted(
        groups.values(),
        key=lambda item: sorted(item)
    ):
        canonical_name = _choose_canonical_category_name(
            group,
            profiles
        )
        topic_count = sum(
            profiles[category_name]["topic_count"]
            for category_name in group
        )

        for category_name in group:
            category_mapping[category_name] = canonical_name

        final_groups.append({
            "canonical_category": canonical_name,
            "source_categories": sorted(group),
            "topic_count": topic_count,
        })

    validate_taxonomy_consolidation(
        topic_ledger=topic_ledger,
        category_mapping=category_mapping,
        final_groups=final_groups,
        category_size_limit=category_size_limit,
    )
    diagnostics = build_taxonomy_quality_diagnostics(
        topic_ledger=topic_ledger,
        category_mapping=category_mapping,
        final_groups=final_groups,
        accepted_merges=accepted_merges,
    )

    return {
        "mapping": category_mapping,
        "groups": final_groups,
        "accepted_merges": accepted_merges,
        "rejected_merges": rejected_merges,
        "category_size_limit": category_size_limit,
        "diagnostics": diagnostics,
    }


def build_taxonomy_quality_diagnostics(
    topic_ledger,
    category_mapping,
    final_groups,
    accepted_merges,
):
    before_counts = {}

    for entry in topic_ledger:
        before_counts[entry.original_category] = (
            before_counts.get(entry.original_category, 0) + 1
        )

    after_counts = {}

    for entry in topic_ledger:
        final_category = category_mapping[entry.original_category]
        after_counts[final_category] = (
            after_counts.get(final_category, 0) + 1
        )

    rationale_by_group = {}

    for merge in accepted_merges:
        group_key = tuple(merge["source_categories"])
        rationale_by_group[group_key] = merge["rationale"]

    category_changes = []

    for group in final_groups:
        source_categories = group["source_categories"]
        canonical_category = group["canonical_category"]

        if len(source_categories) <= 1:
            continue

        group_key = tuple(source_categories)
        renamed_categories = [
            category_name
            for category_name in source_categories
            if category_name != canonical_category
        ]
        category_changes.append({
            "source_categories": source_categories,
            "canonical_category": canonical_category,
            "renamed_categories": renamed_categories,
            "topic_count": group["topic_count"],
            "rationale": rationale_by_group.get(
                group_key,
                (
                    "Categories were connected through validated reciprocal "
                    "alias merges and passed complete-group integrity checks."
                ),
            ),
        })

    return {
        "categories_before_review": [
            {
                "category": category_name,
                "topic_count": before_counts[category_name],
            }
            for category_name in sorted(before_counts)
        ],
        "categories_after_review": [
            {
                "category": category_name,
                "topic_count": after_counts[category_name],
            }
            for category_name in sorted(after_counts)
        ],
        "category_count_before": len(before_counts),
        "category_count_after": len(after_counts),
        "topic_count_before": len(topic_ledger),
        "topic_count_after": sum(after_counts.values()),
        "categories_merged_or_renamed": category_changes,
    }


def validate_taxonomy_consolidation(
    topic_ledger,
    category_mapping,
    final_groups,
    category_size_limit,
):
    original_categories = {
        entry.original_category
        for entry in topic_ledger
    }

    if set(category_mapping) != original_categories:
        raise ValueError(
            "Category mapping does not cover every original category"
        )

    ledger_ids = [entry.ledger_id for entry in topic_ledger]

    if len(ledger_ids) != len(set(ledger_ids)):
        raise ValueError(
            "Topic integrity failed: duplicate ledger topic IDs"
        )

    mapped_topic_ids = []
    final_category_counts = {}

    for entry in topic_ledger:
        final_category = category_mapping.get(
            entry.original_category
        )

        if not final_category:
            raise ValueError(
                f"Topic {entry.ledger_id} has no final category"
            )

        mapped_topic_ids.append(entry.ledger_id)
        final_category_counts[final_category] = (
            final_category_counts.get(final_category, 0) + 1
        )

    if mapped_topic_ids != ledger_ids:
        raise ValueError(
            "Topic integrity failed: ledger ordering changed"
        )

    if len(mapped_topic_ids) != len(set(mapped_topic_ids)):
        raise ValueError(
            "Topic integrity failed: a topic was assigned more than once"
        )

    if sum(final_category_counts.values()) != len(topic_ledger):
        raise ValueError(
            "Topic integrity failed: topic count changed"
        )

    if any(
        topic_count <= 0
        for topic_count in final_category_counts.values()
    ):
        raise ValueError(
            "Taxonomy consolidation produced an empty category"
        )

    if any(
        topic_count > category_size_limit
        for topic_count in final_category_counts.values()
    ):
        raise ValueError(
            "Taxonomy consolidation exceeded the category size limit"
        )

    canonical_names = [
        group["canonical_category"]
        for group in final_groups
    ]

    if len(canonical_names) != len(set(canonical_names)):
        raise ValueError(
            "Taxonomy consolidation produced duplicate canonical names"
        )

    if len(final_category_counts) > len(original_categories):
        raise ValueError(
            "Taxonomy consolidation increased the category count"
        )

def sample_project_language_text(
    all_chunks,
    max_samples=24,
    max_chars_per_sample=700
):
    usable_chunks = [
        chunk
        for chunk in all_chunks
        if len((chunk.get("text") or "").strip()) >= 120
    ]

    if not usable_chunks:
        return ""

    sample_count = min(max_samples, len(usable_chunks))

    if sample_count == 1:
        selected_indexes = [0]
    else:
        selected_indexes = [
            round(i * (len(usable_chunks) - 1) / (sample_count - 1))
            for i in range(sample_count)
        ]

    selected_chunks = []
    seen_indexes = set()

    for index in selected_indexes:
        if index in seen_indexes:
            continue

        seen_indexes.add(index)
        chunk = usable_chunks[index]
        text_value = (chunk.get("text") or "").strip()

        selected_chunks.append(
            text_value[:max_chars_per_sample]
        )

    return "\n\n---\n\n".join(selected_chunks)


def detect_project_taxonomy_language(all_chunks):
    sample_text = sample_project_language_text(all_chunks)

    if not sample_text:
        raise ValueError("No suitable source text for language detection")

    enabled_languages = get_enabled_languages()
    enabled_language_list = [
        {
            "code": language["code"],
            "name": language["name"],
            "native_name": language["native_name"],
        }
        for language in enabled_languages
    ]

    detection_prompt = f"""
    Determine the dominant natural language of the explanatory prose in the
    supplied project excerpts.

    Select exactly one language from this enabled language registry:
    {json.dumps(enabled_language_list, ensure_ascii=False)}

    RULES:
    - Judge the language of the explanatory prose.
    - Ignore formulas, source code, citations, bibliography entries, proper
      names, acronyms, and isolated foreign technical terminology.
    - Do not infer language from the academic domain.
    - If no enabled language clearly dominates, return the closest candidate
      with a low confidence score rather than inventing another code.
    - Return ONLY valid JSON.

    JSON FORMAT:
    {{
      "language_code": "BCP47_CODE",
      "confidence": 0.0
    }}

    PROJECT EXCERPTS:
    {sample_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You classify the dominant language of educational source "
                    "material."
                )
            },
            {"role": "user", "content": detection_prompt}
        ],
        temperature=0,
        response_format={"type": "json_object"}
    )

    detection_data = json.loads(
        response.choices[0].message.content
    )

    language = get_enabled_language(
        detection_data.get("language_code")
    )
    confidence = detection_data.get("confidence")

    if not language:
        raise ValueError(
            "Language detector returned an unsupported or disabled code"
        )

    if not isinstance(confidence, (int, float)):
        raise ValueError(
            "Language detector returned an invalid confidence score"
        )

    confidence = float(confidence)

    if confidence < 0 or confidence > 1:
        raise ValueError(
            "Language detector confidence must be between 0 and 1"
        )

    return {
        "language": language,
        "confidence": confidence
    }


def audit_taxonomy_reorganization(final_data, project_id: str):
    """
    Audit-only taxonomy reorganization.

    This function never mutates final_data and its result is not used by
    persistence, embeddings, or topic-chunk assignment.
    """
    audit_topics = []
    current_category_counts = {}

    for category_index, category in enumerate(final_data.get("categories", [])):
        category_name = (
            category.get("name") or "GENERAL"
        ).strip().upper()

        for topic_index, topic_obj in enumerate(category.get("topics", [])):
            topic_title = (
                topic_obj.get("title")
                or topic_obj.get("topic")
                or ""
            ).strip()

            if not topic_title:
                continue

            current_category_counts[category_name] = (
                current_category_counts.get(category_name, 0) + 1
            )

            audit_topics.append({
                "audit_id": f"T{len(audit_topics) + 1:04d}",
                "current_category": category_name,
                "topic": topic_title,
                "description": (
                    topic_obj.get("description") or ""
                ).strip(),
                "importance": topic_obj.get("importance", 5),
                "source_position": {
                    "category_index": category_index,
                    "topic_index": topic_index
                }
            })

    current_stats = {
        "category_count": len(current_category_counts),
        "topic_count": len(audit_topics),
        "topics_per_category": current_category_counts
    }

    print("🔎 TAXONOMY AUDIT START:", project_id)
    print(
        "🔎 TAXONOMY AUDIT CURRENT STATS:",
        json.dumps(current_stats, sort_keys=True)
    )

    if not audit_topics:
        raise ValueError("Taxonomy audit received no topics")

    immutable_payload = [
        {
            "audit_id": topic["audit_id"],
            "current_category": topic["current_category"],
            "topic": topic["topic"],
            "description": topic["description"],
            "importance": topic["importance"]
        }
        for topic in audit_topics
    ]

    audit_prompt = f"""
    Act as a senior domain-agnostic educational taxonomy auditor.

    You are reviewing a completed taxonomy. The study topics are immutable.
    Your only task is to propose a cleaner category organization.

    ALLOWED:
    - Move an existing topic to another category.
    - Merge overlapping categories.
    - Rename categories.

    FORBIDDEN:
    - Do not modify topic titles.
    - Do not modify descriptions.
    - Do not modify importance.
    - Do not delete topics.
    - Do not create topics.
    - Do not merge or split topics.

    CATEGORY QUALITY RULES:
    - Work across any academic domain, including medicine, law, engineering,
      humanities, economics, and multidisciplinary material.
    - Categories must be stable educational families that can contain multiple
      meaningful study topics.
    - Merge lexical variants and categories with the same educational scope.
    - Reorganize topics out of generic umbrella categories when a more precise
      educational family is supported by the supplied topics.
    - Do not merge categories merely because they are related.
    - Preserve meaningful disciplinary boundaries.
    - Prefer fewer, stronger categories, but never target a fixed count.
    - Base decisions on topic meaning and description, not word overlap alone.

    OUTPUT RULES:
    - Return ONLY valid JSON.
    - Return each audit_id exactly once.
    - Do not return topic titles or descriptions.
    - Category names must be concise, professional, and uppercase.
    - confidence_score must be a number from 0 to 1 representing confidence in
      the overall category organization.

    JSON FORMAT:
    {{
      "proposed_categories": [
        {{
          "name": "CANONICAL CATEGORY",
          "topic_ids": ["T0001", "T0002"]
        }}
      ],
      "confidence_score": 0.0
    }}

    IMMUTABLE TOPICS:
    {json.dumps(immutable_payload, ensure_ascii=False)}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You audit educational category structure without "
                    "changing any study topic."
                )
            },
            {"role": "user", "content": audit_prompt}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )

    audit_data = json.loads(
        response.choices[0].message.content
    )

    proposed_categories = audit_data.get("proposed_categories")
    if not isinstance(proposed_categories, list) or not proposed_categories:
        raise ValueError("Taxonomy audit returned no proposed categories")

    expected_ids = {topic["audit_id"] for topic in audit_topics}
    assigned_ids = []
    proposed_category_counts = {}
    proposed_category_by_topic = {}

    for category in proposed_categories:
        proposed_name = str(category.get("name") or "").strip().upper()
        topic_ids = category.get("topic_ids")

        if not proposed_name:
            raise ValueError("Taxonomy audit returned an empty category name")

        if not isinstance(topic_ids, list) or not topic_ids:
            raise ValueError(
                f"Taxonomy audit category has no topics: {proposed_name}"
            )

        if proposed_name in proposed_category_counts:
            raise ValueError(
                f"Taxonomy audit returned duplicate category: {proposed_name}"
            )

        proposed_category_counts[proposed_name] = len(topic_ids)

        for audit_id in topic_ids:
            if not isinstance(audit_id, str):
                raise ValueError("Taxonomy audit returned a non-string topic ID")

            assigned_ids.append(audit_id)
            proposed_category_by_topic[audit_id] = proposed_name

    assigned_id_set = set(assigned_ids)

    if len(assigned_ids) != len(assigned_id_set):
        raise ValueError("Taxonomy audit assigned at least one topic more than once")

    missing_ids = expected_ids - assigned_id_set
    unknown_ids = assigned_id_set - expected_ids

    if missing_ids or unknown_ids:
        raise ValueError(
            "Taxonomy audit topic coverage mismatch: "
            f"missing={sorted(missing_ids)}, unknown={sorted(unknown_ids)}"
        )

    confidence_score = audit_data.get("confidence_score")
    if not isinstance(confidence_score, (int, float)):
        raise ValueError("Taxonomy audit returned an invalid confidence score")

    confidence_score = max(0.0, min(1.0, float(confidence_score)))

    source_to_targets = {}
    target_to_sources = {}

    for topic in audit_topics:
        source = topic["current_category"]
        target = proposed_category_by_topic[topic["audit_id"]]

        source_to_targets.setdefault(source, {})
        source_to_targets[source][target] = (
            source_to_targets[source].get(target, 0) + 1
        )

        target_to_sources.setdefault(target, set()).add(source)

    merged_categories = [
        {
            "proposed_category": target,
            "source_categories": sorted(sources),
            "source_category_count": len(sources),
            "topic_count": proposed_category_counts[target]
        }
        for target, sources in sorted(target_to_sources.items())
        if len(sources) > 1
    ]

    category_merge_map = {
        "source_to_proposed_categories": source_to_targets,
        "merged_categories": merged_categories
    }

    current_category_count = len(current_category_counts)
    proposed_category_count = len(proposed_category_counts)
    category_reduction = current_category_count - proposed_category_count
    category_reduction_pct = (
        round((category_reduction / current_category_count) * 100, 2)
        if current_category_count
        else 0.0
    )

    proposed_stats = {
        "category_count": proposed_category_count,
        "topic_count": len(assigned_ids),
        "topics_per_category": proposed_category_counts
    }

    estimated_reduction = {
        "current_category_count": current_category_count,
        "proposed_category_count": proposed_category_count,
        "category_reduction": category_reduction,
        "category_reduction_percent": category_reduction_pct
    }

    print(
        "🔎 TAXONOMY AUDIT PROPOSED STATS:",
        json.dumps(proposed_stats, sort_keys=True)
    )
    print(
        "🔎 TAXONOMY AUDIT CATEGORY MERGE MAP:",
        json.dumps(category_merge_map, sort_keys=True)
    )
    print(
        "🔎 TAXONOMY AUDIT ESTIMATED REDUCTION:",
        json.dumps(estimated_reduction, sort_keys=True)
    )
    print(
        "🔎 TAXONOMY AUDIT CONFIDENCE:",
        confidence_score
    )
    print(
        "✅ TAXONOMY AUDIT COMPLETE — "
        "NO TAXONOMY, EMBEDDING, DATABASE, OR ASSIGNMENT CHANGES APPLIED"
    )

    return {
        "current_statistics": current_stats,
        "proposed_statistics": proposed_stats,
        "category_merge_map": category_merge_map,
        "estimated_category_reduction": estimated_reduction,
        "confidence_score": confidence_score
    }

def process_topics_task(project_id: str):
    db = SessionLocal()
    try:
        print("BACKGROUND TOPICS START:", project_id)
        
       
        topic_phase_timer = time.time()
        topic_start_time = time.time()
        # 1. Update status and clear old topics
        db.execute(text("update projects set topic_status = 'processing' where id = :project_id"), {"project_id": project_id})
        db.execute(text("delete from topics where project_id = :project_id"), {"project_id": project_id})
        ensure_project_chunk_roles(db, project_id)
        db.commit()

        # Fetch chunks to analyze (up to 120 chunks)
        rows = db.execute(
            text("""
                select chunk_text, section, chunk_role
                from chunks
                where project_id = :project_id
                and chunk_role = 'teaching'
                order by page asc
                limit 400
            """),
            {"project_id": project_id}
        ).fetchall()

        all_chunks = [
            {
                "text": r[0],
                "section": r[1] or "GENERAL",
                "chunk_role": r[2],
            }
            for r in rows
            if r[0]
        ]
        chunk_eligibility_counts = db.execute(
            text("""
                select
                    count(*) filter (
                        where chunk_role = 'teaching'
                    ) as eligible_chunks,
                    count(*) filter (
                        where chunk_role <> 'teaching'
                    ) as excluded_chunks
                from chunks
                where project_id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()
        print("📦 TOTAL CHUNKS:", len(all_chunks))
        print(
            "ELIGIBLE TEACHING CHUNKS:",
            chunk_eligibility_counts[0]
        )
        print("EXCLUDED CHUNKS:", chunk_eligibility_counts[1])
        print(
            "⏱️ CHUNKS LOADED:",
            round(time.time() - topic_phase_timer, 1),
            "seconds"
        )

        taxonomy_language = None
        taxonomy_language_confidence = None
        language_enforcement_applied = False
        detected_language_code = None

        try:
            detection_result = detect_project_taxonomy_language(
                all_chunks
            )

            detected_language = detection_result["language"]
            detected_language_code = detected_language["code"]
            taxonomy_language_confidence = detection_result["confidence"]

            if (
                taxonomy_language_confidence
                >= MIN_TAXONOMY_LANGUAGE_CONFIDENCE
            ):
                db.execute(
                    text("""
                        UPDATE projects
                        SET taxonomy_language = :taxonomy_language
                        WHERE id = :project_id
                    """),
                    {
                        "taxonomy_language": detected_language["code"],
                        "project_id": project_id
                    }
                )
                db.commit()

                taxonomy_language = detected_language
                language_enforcement_applied = True
            else:
                print(
                    "⚠️ LANGUAGE DETECTION CONFIDENCE TOO LOW — "
                    "CONTINUING EXISTING PIPELINE UNCHANGED"
                )

        except Exception as language_error:
            db.rollback()
            print(
                "⚠️ LANGUAGE DETECTION FAILED — "
                "CONTINUING EXISTING PIPELINE UNCHANGED:",
                repr(language_error)
            )

        print(
            "DETECTED LANGUAGE:",
            detected_language_code
        )
        print(
            "DETECTION CONFIDENCE:",
            taxonomy_language_confidence
        )
        print(
            "TAXONOMY LANGUAGE:",
            taxonomy_language["code"] if taxonomy_language else None
        )
        print(
            "LANGUAGE ENFORCEMENT APPLIED:",
            language_enforcement_applied
        )

        language_instruction = ""

        if taxonomy_language:
            language_instruction = f"""
                LANGUAGE REQUIREMENT:
                - The authoritative language for this project is
                  {taxonomy_language["name"]}
                  ({taxonomy_language["code"]}).
                - Generate ALL category names, topic titles, and topic
                  descriptions exclusively in
                  {taxonomy_language["name"]}.
                - Do not switch to the language used by these instructions or
                  by model defaults.
                - Preserve proper names, formulas, symbols, abbreviations,
                  citations, standards identifiers, and established technical
                  terms when translating them would be academically incorrect
                  or unnatural.
                - Language consistency must not change topic selection,
                  educational scope, granularity, or factual meaning.
            """

        topic_phase_timer = time.time()
        # Group chunks into batches of 20
        from collections import defaultdict

        section_groups = defaultdict(list)

        for chunk in all_chunks:
            section_groups[chunk["section"]].append(chunk["text"])
            print("📚 TOTAL SECTIONS:", len(section_groups))
            print("📦 TOTAL CHUNKS:", len(all_chunks))
        
        seen_titles = set()
        all_candidate_topics = []

        for section_name, section_chunks in section_groups.items():
            total_mini_groups = 0

            for chunks in section_groups.values():
                total_mini_groups += len([
                    chunks[i:i+20]
                    for i in range(0, len(chunks), 20)
                ])

            print("📦 TOTAL MINI GROUPS:", total_mini_groups)
            
            print(
                f"📂 SECTION: {section_name}"
            )

            print(
                f"📦 CHUNKS IN SECTION: {len(section_chunks)}"
            )

            if time.time() - topic_start_time > MAX_TOPIC_PROCESSING_SECONDS:

                print("⏰ TOPIC PROCESSING TIMEOUT")

                db.execute(
                    text("""
                        UPDATE projects
                        SET topic_status = 'error'
                        WHERE id = :project_id
                    """),
                    {"project_id": project_id}
                )

                db.commit()

                raise Exception(
                    "Topic generation timeout. "
                    "The uploaded file could not be fully processed."
                )

            mini_groups = [
                section_chunks[i:i+20]
                for i in range(0, len(section_chunks), 20)
            ]
            for mini_group in mini_groups:
                if time.time() - topic_start_time > MAX_TOPIC_PROCESSING_SECONDS:
                    print("⏰ TOPIC MINI-GROUP TIMEOUT - stopping safely")

                    db.execute(
                        text("""
                            UPDATE projects
                            SET topic_status = 'error'
                            WHERE id = :project_id
                        """),
                        {"project_id": project_id}
                    )

                    db.commit()

                    raise Exception(
                        "Topic generation timeout. "
                        "The uploaded file could not be fully processed."
                    )

                group_text = "\n\n".join(mini_group)
            
                # UNIVERSAL PROMPT: Works for any discipline (Medicine, Law, Engineering, etc.)
                # OPTIMIZED PROMPT: Focused on Pedagogical Hierarchy and Topic Consolidation
                prompt = f"""
                Act as a specialist in Instructional Design and Knowledge Organization. 
                Analyze the provided text to extract its fundamental conceptual hierarchy.
                
                GOAL:
                Organize the information into a logical structure of Macro-Categories and robust Study Topics.

                {language_instruction}
                
                STRICT RULES:
                1. CATEGORY RULES:

                The provided DOCUMENT SECTION is contextual information only.

                Your task is to identify the most appropriate educational Category.

                CATEGORY RULES:

                - Categories should represent broad learning domains.
                - Categories should group multiple related study topics.
                - Categories should be stable across the entire document.
                - Categories should not be derived directly from paragraph titles.
                - Categories should not represent individual lessons, examples, or subtopics.
                - Multiple document sections may belong to the same Category.
                - Prefer fewer, stronger Categories over many fragmented Categories.

                Good Categories:
                - Broad educational domains
                - Major areas of study
                - Conceptual learning groups

                Bad Categories:
                - Individual definitions
                - Single examples
                - Paragraph titles
                - Very narrow concepts
                - Temporary document headings

                - Categories should preserve the educational structure of the source material while grouping concepts into semantically coherent learning domains.

                - Avoid creating categories unrelated to the document structure.

                - DO NOT invent unrelated or alternative high-level Categories.

                - Keep the hierarchy aligned with the actual structure of the source document.

                - Categories must preserve the educational organization already present in the material.

                - Use UPPERCASE for Category names.
                -Each topic must semantically belong to its parent category.

                -Topics and categories must describe the same conceptual domain.
                -Topics must remain narrow enough to represent a focused retrievable study unit.

                -Avoid placing unrelated concepts inside the same category.

                - Categories should describe a broad educational area, not a specific Study Topic.

                - ALWAYS USE UPPERCASE.
                
                2. TOPIC CREATION RULES:

                - Create ONLY pedagogically meaningful Study Topics.
                - A Study Topic must represent a coherent unit suitable for an actual study session.
                - A Topic title must not duplicate or trivially restate its parent Category name.
                - DO NOT create Topics for:
                - isolated terms
                - single definitions
                - individual list items
                - tiny subcomponents
                - concepts explained with minimal context
                unless they are universally recognized as major standalone concepts in the discipline.

                - Consolidate strongly related concepts into broader educational Topics.

                - Prefer broader conceptual Topics over highly granular fragmentation.

                GOOD TOPIC CHARACTERISTICS:
                - Represents a coherent study unit
                - Covers a meaningful conceptual area
                - Supports multiple quiz questions
                - Supports multiple flashcards
                - Can reasonably correspond to a focused study session
                - Contains concepts that are commonly studied together
                - Reflects the actual educational structure of the source material

                BAD TOPIC CHARACTERISTICS:
                - Isolated keywords or vocabulary terms
                - Extremely narrow details with little standalone relevance
                - Topics containing only one trivial fact
                - Fragmented micro-topics that do not support meaningful study
                - Artificially broad topics combining unrelated concepts

                - A Topic should:
                - support a complete study session
                - contain multiple internal concepts
                - support multiple quiz questions
                - support multiple flashcards
                - represent meaningful educational scope

                - Prefer FEWER but STRONGER Topics.

                - Topics should normally aggregate multiple related concepts internally, even if those concepts are not explicitly listed in the title.
                
                3. DESCRIPTION: Provide a dense, academic definition. If the Topic is a consolidation of multiple items, the description must briefly summarize all of them.
                
                4. IMPORTANCE: Rate 1-10 based on how fundamental the concept is to the subject.

                FORMAT RULES:
                - Return ONLY valid JSON.
                - Ensure names are professional and academically accurate.

                JSON STRUCTURE:
                {{
                "categories": [
                    {{
                    "name": "CATEGORY NAME",
                    "topics": [
                        {{ 
                        "title": "Consolidated Topic Name", 
                        "description": "Comprehensive academic definition covering the grouped concepts", 
                        "importance": 8 
                        }}
                    ]
                    }}
                ]
                }}

                DOCUMENT SECTION:
                {section_name}

                CONTENT TO ANALYZE:
                {group_text}
                """

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    response_format={ "type": "json_object" } # Ensures stable JSON output[cite: 2]
                )

                try:
                    data = json.loads(response.choices[0].message.content.replace("```json", "").replace("```", "").strip())

                    for cat in data.get("categories", []):
                        # KEEP ORIGINAL DOCUMENT CATEGORY
                        raw_category = (
                            cat.get("name") or "GENERAL"
                        ).strip().upper()

                        category_name = raw_category
                        
                        
                        
                        for t_obj in cat.get("topics", []):
                            topic_title = (t_obj.get("title") or "").strip()
                            description = (t_obj.get("description") or "").strip()

                            # Recuperiamo l'importanza se presente, altrimenti default a 5
                            importance = t_obj.get("importance", 5)

                            # 🔥 DISPLAY FILTER LOGIC

                            is_display_topic = True

                            #if should_hide_topic(topic_title):
                                #is_display_topic = False

                        

                            #if importance <= 4:
                                #is_display_topic = False

                            if not topic_title:
                                continue

                            # Manteniamo il titolo pulito senza regex aggressive che 
                            # possono rovinare acronimi medici o tecnici
                            topic_title = topic_title.strip()

                            key = topic_title.lower().strip()
                            if key in seen_titles:
                                continue
                            seen_titles.add(key)

                            print(f"➡️ TOPIC: {topic_title} [{category_name}]")
                            all_candidate_topics.append({
                                "category": category_name,
                                "topic": topic_title,
                                "description": description,
                                "importance": importance
    })

                           
                            

                except Exception as e:
                    print(f"Error parsing JSON in chunk: {e}")
                    continue 
        print(
            "📚 TOTAL CANDIDATE TOPICS:",
            len(all_candidate_topics)
        )
        print(
            "⏱️ EXTRACTION COMPLETE:",
            round(time.time() - topic_phase_timer, 1),
            "seconds"
        )

        topic_phase_timer = time.time()
        print("🌍 STARTING GLOBAL TOPIC CONSOLIDATION")

        print(
                "⏱️ BEFORE FINAL CONSOLIDATION:",
                round(time.time() - topic_start_time, 1),
                "seconds"
             )

        print(
            "📦 TOTAL CANDIDATE TOPICS:",
            len(all_candidate_topics)
        )
        from collections import defaultdict

        section_map = defaultdict(list)

        for t in all_candidate_topics:

            category = t["category"]

            section_map[category].append(t)

        topics_text = ""

        MAX_TOPICS_FOR_CONSOLIDATION = 40
        print("📊 SECTIONS:", len(section_map))


        for category, topics in section_map.items():

            print("🧠 CONSOLIDATION CATEGORY:", category)
            print("🧠 CONSOLIDATION TOPICS:", len(topics))

            topics_text += f"\n\n=== CATEGORY: {category} ===\n"

            for t in topics[:MAX_TOPICS_FOR_CONSOLIDATION]:

                

               
                topics_text += f"""
        TOPIC: {t['topic']}
        DESCRIPTION: {t['description']}
        IMPORTANCE: {t['importance']}
        """

        global_prompt = f"""
        Act as a senior Instructional Designer and Knowledge Architect.

        You are given MANY candidate Study Topics extracted from different portions of the same document.

        Your task is to CONSOLIDATE them into a SINGLE coherent educational taxonomy.

        {language_instruction}

        GOALS:
        - Merge ONLY genuinely redundant or overlapping Topics.
        - NEVER merge Topics belonging to different Categories.
        - Categories represent hard educational boundaries.
        - Topics from different Categories must always remain separated even if semantically related.
        - Preserve important educational subdomains as separate Topics.
        - Do NOT merge Topics that belong to different conceptual depths.
        - Only merge Topics representing the same pedagogical level.
        - Do NOT merge Topics that represent distinct study areas, even if related.
        - Remove redundant Topics
        - Eliminate overly granular Topics
        - Create pedagogically balanced Study Topics with moderate granularity
        - Ensure complete document coverage
        - Preserve ALL major educational concepts

        IMPORTANT:
        - Categories must be broad academic domains.
        - Topics must represent meaningful study units.
        - Avoid fragmentation.
        - Avoid duplicate or overlapping Topics.
        - Prefer pedagogically balanced Topics.
        Examples of Topics that SHOULD remain separate:
        - Concepts that are independently studied
        - Topics with distinct educational objectives
        - Topics that support separate quiz generation
        - Topics with substantially different conceptual scope
        - Topics that are commonly taught as separate units
        - Topics that require different reasoning processes or learning goals
        - Avoid excessive fragmentation.
        - Avoid overly broad mega-topics.
        - A Topic should represent a focused but complete study unit.
        - Topics should normally correspond to 15-30 minutes of focused study.
        - Preserve important subdomains when they have substantial educational relevance.

        Return ONLY valid JSON.

        FORMAT:

        {{
        "categories": [
            {{
            "name": "CATEGORY",
            "topics": [
                {{
                "title": "TOPIC",
                "description": "DESCRIPTION",
                "importance": 8
                }}
            ]
            }}
        ]
        }}

        CANDIDATE TOPICS:

        {topics_text}
        """
        print(
            "📏 CONSOLIDATION TEXT LENGTH:",
            len(topics_text)
        )
        print("🚀 STARTING FINAL CONSOLIDATION CALL")
        print("🧠 CONSOLIDATION INPUT LENGTH:", len(topics_text))
        print("🧠 TOTAL CATEGORIES:", len(section_map))
        total_candidate_topics = sum(
            len(v)
            for v in section_map.values()
        )

        print(
            "📊 CANDIDATE TOPICS BEFORE CONSOLIDATION:",
            total_candidate_topics
        )
        print(
            "🚀 STARTING FINAL CONSOLIDATION GPT CALL"
        )

        print(
            "⏱️ ELAPSED:",
            round(time.time() - topic_start_time, 1),
            "seconds"
        )

        try:

            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": global_prompt}
                ],
                response_format={"type": "json_object"}
            )

            final_data = json.loads(
                final_response.choices[0].message.content
            )
            print(
                "🏁 AFTER GPT CONSOLIDATION:",
                round(time.time() - topic_start_time, 1),
                "seconds"
            )
            print("✅ FINAL CONSOLIDATION RESPONSE RECEIVED")
            print("✅ GLOBAL CONSOLIDATION COMPLETE")
            print(
                "⏱️ CONSOLIDATION COMPLETE:",
                round(time.time() - topic_phase_timer, 1),
                "seconds"
            )

            topic_phase_timer = time.time()
            print(json.dumps(final_data, indent=2))

        except Exception as e:

            print("❌ FINAL CONSOLIDATION FAILED:", e)

            # FALLBACK:
            # usa direttamente i candidate topics
            final_data = {
                "categories": []
            }

            for category, topics in section_map.items():

                final_data["categories"].append({
                    "name": category,
                    "topics": topics[:40]
                })

            print("⚠️ USING FALLBACK TOPIC STRUCTURE")
        final_data = rebalance_taxonomy(final_data)        
        final_topics = sum(
            1
            for cat in final_data.get("categories", [])
            for topic_object in cat.get("topics", [])
            if (
                topic_object.get("title")
                or topic_object.get("topic")
                or ""
            ).strip()
        )

        print(
            "📊 TOPICS AFTER CONSOLIDATION:",
            final_topics
        )

        print("🧬 PRECOMPUTING IMMUTABLE TOPIC LEDGER")
        topic_ledger = build_immutable_taxonomy_ledger(
            final_data
        )

        if len(topic_ledger) != final_topics:
            raise Exception(
                "Immutable topic ledger count mismatch: "
                f"expected {final_topics}, got {len(topic_ledger)}"
            )

        print(
            "✅ IMMUTABLE TOPIC LEDGER READY:",
            len(topic_ledger),
            "topics"
        )

        try:
            audit_taxonomy_reorganization(
                final_data=final_data,
                project_id=project_id
            )
        except Exception as audit_error:
            print(
                "❌ TAXONOMY AUDIT FAILED — "
                "CONTINUING EXISTING PIPELINE UNCHANGED:",
                repr(audit_error)
            )

        original_category_mapping = {
            entry.original_category: entry.original_category
            for entry in topic_ledger
        }
        category_mapping = original_category_mapping

        try:
            consolidation_result = (
                consolidate_taxonomy_categories_v1(
                    topic_ledger
                )
            )
            category_mapping = consolidation_result["mapping"]
            quality_diagnostics = consolidation_result[
                "diagnostics"
            ]

            print(
                "🧭 TAXONOMY QUALITY PASS VERSION:",
                CATEGORY_CONSOLIDATION_VERSION
            )
            print(
                "🧭 CATEGORIES BEFORE REVIEW:",
                json.dumps(
                    quality_diagnostics[
                        "categories_before_review"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(
                "🧭 CATEGORIES AFTER REVIEW:",
                json.dumps(
                    quality_diagnostics[
                        "categories_after_review"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(
                "🧭 CATEGORY COUNT BEFORE REVIEW:",
                quality_diagnostics["category_count_before"]
            )
            print(
                "🧭 CATEGORY COUNT AFTER REVIEW:",
                quality_diagnostics["category_count_after"]
            )
            print(
                "🧭 CATEGORY SIZE LIMIT:",
                consolidation_result["category_size_limit"]
            )
            print(
                "🧭 ACCEPTED CATEGORY MERGES:",
                len(consolidation_result["accepted_merges"])
            )

            for merge in consolidation_result[
                "accepted_merges"
            ]:
                print(
                    "✅ CATEGORY MERGE:",
                    " + ".join(merge["source_categories"]),
                    "->",
                    category_mapping[
                        merge["source_categories"][0]
                    ],
                    "| score:",
                    round(merge["merge_score"], 4),
                    "| topics:",
                    merge["topic_count"],
                    "| cohesion:",
                    round(merge["merged_cohesion"], 4),
                    "| spread:",
                    round(merge["maximum_spread"], 4),
                    "| rationale:",
                    merge["rationale"]
                )

            print(
                "🧭 CATEGORIES MERGED OR RENAMED:",
                json.dumps(
                    quality_diagnostics[
                        "categories_merged_or_renamed"
                    ],
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            print(
                "🧭 REJECTED CATEGORY MERGES:",
                len(consolidation_result["rejected_merges"])
            )
            print(
                "✅ PRE-PERSISTENCE TAXONOMY QUALITY PASS COMPLETE"
            )

        except Exception as consolidation_error:
            category_mapping = original_category_mapping
            print(
                "❌ DETERMINISTIC CATEGORY CONSOLIDATION FAILED:",
                repr(consolidation_error)
            )
            print(
                "✅ ORIGINAL TAXONOMY RESTORED — "
                "NO PARTIAL CATEGORY MERGES APPLIED"
            )

        # 🔥 DELETE OLD TOPIC-CHUNK LINKS

        db.execute(
            text("""
                DELETE FROM topic_chunks
                WHERE topic_id IN (
                    SELECT id
                    FROM topics
                    WHERE project_id = :project_id
                )
            """),
            {"project_id": project_id}
        )

        # 🔥 DELETE OLD TOPICS

        db.execute(
            text("""
                DELETE FROM topics
                WHERE project_id = :project_id
            """),
            {"project_id": project_id}
        )

        db.commit()

        print("🧹 OLD TOPICS + TOPIC_CHUNKS REMOVED")
        for ledger_entry in topic_ledger:
            category_name = category_mapping[
                ledger_entry.original_category
            ]
            embedding_str = "[" + ",".join(
                map(str, ledger_entry.embedding)
            ) + "]"

            db.execute(
                text("""
                    insert into topics
                    (
                        project_id,
                        category,
                        topic,
                        description,
                        embedding,
                        is_display_topic,
                        source_section
                    )
                    values
                    (
                        :project_id,
                        :category,
                        :topic,
                        :description,
                        CAST(:embedding AS vector),
                        :is_display_topic,
                        :source_section
                    )
                """),
                {
                    "project_id": project_id,
                    "category": category_name,
                    "topic": ledger_entry.topic,
                    "description": ledger_entry.description,
                    "embedding": embedding_str,
                    "is_display_topic": True,
                    "source_section": (
                        ledger_entry.original_category
                    )
                }
            )

        db.commit()
        print(
            "⏱️ TOPICS SAVED:",
            round(time.time() - topic_phase_timer, 1),
            "seconds"
        )

        topic_phase_timer = time.time()
        print("🧠 STARTING TOPIC-CHUNK ASSIGNMENT")

        try:
            assign_topics_to_chunks(project_id)
            print("✅ TOPIC-CHUNK ASSIGNMENT COMPLETED")
            print(
            "⏱️ ASSIGNMENT COMPLETE:",
            round(time.time() - topic_phase_timer, 1),
            "seconds"
        )
        except Exception as e:
            print("❌ TOPIC-CHUNK ASSIGNMENT FAILED:", e)
            raise

        output_valid = db.execute(
            text("""
                SELECT
                    EXISTS (
                        SELECT 1
                        FROM topics
                        WHERE project_id = :project_id
                    )
                    AND EXISTS (
                        SELECT 1
                        FROM topic_chunks tc
                        JOIN topics t ON t.id = tc.topic_id
                        WHERE t.project_id = :project_id
                    )
            """),
            {"project_id": project_id}
        ).scalar()

        if not output_valid:
            raise Exception("Topic processing completed without topics or topic-chunk assignments")

        
        # ======================
        # FINAL STATUS UPDATE
        # ======================

        

        final_db = SessionLocal()

        final_db.execute(
            text("""
                UPDATE projects
                SET topic_status = 'completed'
                WHERE id = :project_id
            """),
            {"project_id": project_id}
        )

        final_db.commit()

        check = final_db.execute(
            text("""
                SELECT topic_status
                FROM projects
                WHERE id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()

        print("✅ PROJECT TOPIC STATUS = COMPLETED")
        print("🔥 STATUS READBACK:", check[0])

        final_db.close()
        

    except Exception as e:
        db.rollback()
        print("BACKGROUND TOPICS ERROR:", e)
        db.execute(text("update projects set topic_status = 'error' where id = :project_id"), {"project_id": project_id})
        db.commit()
    finally:
        db.close()

def assign_topics_to_chunks(project_id: str):

    db = SessionLocal()

    def build_in_clause(prefix, values):
        placeholders = []
        params = {}

        for index, value in enumerate(values):
            key = f"{prefix}_{index}"
            placeholders.append(f":{key}")
            params[key] = value

        return ", ".join(placeholders), params

    def calculate_final_score(row):
        return calculate_topic_chunk_score(
            row[1],
            row[2],
            row[4],
            row[5],
            row[6],
        )

    def get_coverage_stats():
        return db.execute(
            text("""
                select
                    (
                        select count(*)
                        from topic_chunks tc
                        join topics t on t.id = tc.topic_id
                        where t.project_id = :project_id
                    ) as total_links,
                    (
                        select count(distinct tc.topic_id)
                        from topic_chunks tc
                        join topics t on t.id = tc.topic_id
                        where t.project_id = :project_id
                    ) as covered_topics,
                    (
                        select count(distinct tc.chunk_id)
                        from topic_chunks tc
                        join topics t on t.id = tc.topic_id
                        where t.project_id = :project_id
                    ) as covered_chunks,
                    (
                        select coalesce(max(topic_count), 0)
                        from (
                            select tc.chunk_id, count(*) as topic_count
                            from topic_chunks tc
                            join topics t on t.id = tc.topic_id
                            where t.project_id = :project_id
                            group by tc.chunk_id
                        ) chunk_counts
                    ) as max_topics_per_chunk
            """),
            {"project_id": project_id}
        ).fetchone()

    try:

        print("🔗 START TOPIC ASSIGNMENT")
        print("🧪 ENTER assign_topics_to_chunks")

        project_chunk_roles = ensure_project_chunk_roles(
            db,
            project_id,
        )
        db.commit()

        topics_count = db.execute(
            text("""
                select count(*)
                from topics
                where project_id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()[0]

        chunks_count = db.execute(
            text("""
                select count(*)
                from chunks
                where project_id = :project_id
                and chunk_role = 'teaching'
            """),
            {"project_id": project_id}
        ).fetchone()[0]

        total_chunks_count = len(project_chunk_roles)
        print("ELIGIBLE TEACHING CHUNKS:", chunks_count)
        print(
            "EXCLUDED CHUNKS:",
            total_chunks_count - chunks_count
        )

        if topics_count <= 0 or chunks_count <= 0:
            raise Exception(
                f"Topic assignment requires topics and chunks "
                f"(topics={topics_count}, chunks={chunks_count})"
            )

        if topics_count > MAX_ASSIGNMENT_MATCHES:
            raise Exception(
                "Topic count exceeds the complete-ranking candidate budget "
                f"({topics_count} > {MAX_ASSIGNMENT_MATCHES})"
            )

        total_candidate_pairs = chunks_count * topics_count
        phase_a_batch_size = max(
            1,
            min(
                chunks_count,
                MAX_ASSIGNMENT_MATCHES // topics_count
            )
        )
        phase_a_total_batches = (
            chunks_count + phase_a_batch_size - 1
        ) // phase_a_batch_size

        print("📊 ASSIGNMENT CHUNKS:", chunks_count)
        print("📊 ASSIGNMENT TOPICS:", topics_count)
        print("🔢 TOTAL CANDIDATE PAIRS:", total_candidate_pairs)
        print("📦 CANDIDATE BUDGET PER BATCH:", MAX_ASSIGNMENT_MATCHES)
        print("📦 PHASE A CHUNK BATCH SIZE:", phase_a_batch_size)
        print("📦 PHASE A EXPECTED BATCHES:", phase_a_total_batches)

        # Phase A is atomic with deletion so a failure preserves old links.
        db.execute(
            text("""
                delete from topic_chunks
                where topic_id in (
                    select id
                    from topics
                    where project_id = :project_id
                )
            """),
            {"project_id": project_id}
        )

        phase_a_links = 0
        phase_a_pairs_processed = 0
        last_chunk_id = 0

        for batch_number in range(1, phase_a_total_batches + 1):
            chunk_rows = db.execute(
                text("""
                    select id
                    from chunks
                    where project_id = :project_id
                    and chunk_role = 'teaching'
                    and id > :last_chunk_id
                    order by id
                    limit :batch_size
                """),
                {
                    "project_id": project_id,
                    "last_chunk_id": last_chunk_id,
                    "batch_size": phase_a_batch_size,
                }
            ).fetchall()

            chunk_ids = [row[0] for row in chunk_rows]
            if not chunk_ids:
                raise Exception(
                    f"Phase A batch {batch_number} returned no chunks"
                )

            chunk_clause, chunk_params = build_in_clause(
                "phase_a_chunk",
                chunk_ids
            )
            candidate_params = {
                "project_id": project_id,
                **chunk_params,
            }
            matches = db.execute(
                text(f"""
                    select
                        c.id as chunk_id,
                        c.chunk_text,
                        c.section,
                        t.id as topic_id,
                        t.topic,
                        t.source_section,
                        c.embedding <#> t.embedding as negative_inner_product
                    from chunks c
                    join topics t on t.project_id = c.project_id
                    where c.project_id = :project_id
                    and c.id in ({chunk_clause})
                    order by c.id, t.id
                """),
                candidate_params
            ).fetchall()

            expected_pairs = len(chunk_ids) * topics_count
            actual_pairs = len(matches)

            print(
                f"📦 PHASE A BATCH: {batch_number}/{phase_a_total_batches}"
            )
            print("📦 PHASE A BATCH CHUNKS:", len(chunk_ids))
            print("🔢 PHASE A EXPECTED PAIRS:", expected_pairs)
            print("🔢 PHASE A ACTUAL PAIRS:", actual_pairs)

            if actual_pairs != expected_pairs:
                raise Exception(
                    "Incomplete Phase A ranking: "
                    f"expected {expected_pairs} pairs, got {actual_pairs}"
                )

            candidates_by_chunk = {
                chunk_id: []
                for chunk_id in chunk_ids
            }

            for row in matches:
                final_score = calculate_final_score(row)
                if final_score is None:
                    continue

                candidates_by_chunk[row[0]].append(
                    {
                        "topic_id": row[3],
                        "chunk_id": row[0],
                        "score": final_score,
                    }
                )

            batch_assignments = []
            for chunk_id in chunk_ids:
                eligible = [
                    candidate
                    for candidate in candidates_by_chunk[chunk_id]
                    if candidate["score"] >= PRIMARY_ASSIGNMENT_THRESHOLD
                ]
                eligible.sort(
                    key=lambda candidate: (
                        -candidate["score"],
                        str(candidate["topic_id"]),
                    )
                )
                batch_assignments.extend(
                    {
                        "topic_id": candidate["topic_id"],
                        "chunk_id": candidate["chunk_id"],
                    }
                    for candidate in eligible[:MAX_TOPICS_PER_CHUNK]
                )

            if batch_assignments:
                db.execute(
                    text("""
                        insert into topic_chunks
                        (topic_id, chunk_id)
                        values
                        (:topic_id, :chunk_id)
                    """),
                    batch_assignments
                )

            batch_links = len(batch_assignments)
            phase_a_links += batch_links
            phase_a_pairs_processed += actual_pairs
            last_chunk_id = chunk_ids[-1]

            print("🔗 PHASE A BATCH LINKS:", batch_links)

        if phase_a_pairs_processed != total_candidate_pairs:
            raise Exception(
                "Incomplete Phase A candidate processing: "
                f"expected {total_candidate_pairs}, "
                f"processed {phase_a_pairs_processed}"
            )

        phase_a_max = db.execute(
            text("""
                select coalesce(max(topic_count), 0)
                from (
                    select tc.chunk_id, count(*) as topic_count
                    from topic_chunks tc
                    join topics t on t.id = tc.topic_id
                    where t.project_id = :project_id
                    group by tc.chunk_id
                ) chunk_counts
            """),
            {"project_id": project_id}
        ).scalar()

        if phase_a_max > MAX_TOPICS_PER_CHUNK:
            raise Exception(
                "Phase A validation failed: "
                f"max topics per chunk is {phase_a_max}"
            )

        phase_a_stats = get_coverage_stats()
        phase_a_total_links = phase_a_stats[0]
        phase_a_covered_topics = phase_a_stats[1]
        phase_a_covered_chunks = phase_a_stats[2]

        if phase_a_total_links != phase_a_links:
            raise Exception(
                "Phase A link-count validation failed: "
                f"expected {phase_a_links}, found {phase_a_total_links}"
            )

        db.commit()
        print("✅ PHASE A COMMIT COMPLETE")
        print("🔗 PHASE A LINKS CREATED:", phase_a_links)
        print(
            "📊 PHASE A TOPIC COVERAGE:",
            f"{phase_a_covered_topics}/{topics_count}"
        )
        print(
            "📊 PHASE A CHUNK COVERAGE:",
            f"{phase_a_covered_chunks}/{chunks_count}"
        )

        rescue_links = 0
        lowest_rescue_score = None

        try:
            orphan_rows = db.execute(
                text("""
                    select t.id, t.topic
                    from topics t
                    where t.project_id = :project_id
                    and not exists (
                        select 1
                        from topic_chunks tc
                        where tc.topic_id = t.id
                    )
                    order by t.id
                """),
                {"project_id": project_id}
            ).fetchall()

            orphan_count = len(orphan_rows)
            print("🛟 ORPHAN TOPICS DETECTED:", orphan_count)

            if orphan_count:
                if chunks_count > MAX_ASSIGNMENT_MATCHES:
                    raise Exception(
                        "Chunk count exceeds the complete-rescue candidate "
                        f"budget ({chunks_count} > {MAX_ASSIGNMENT_MATCHES})"
                    )

                phase_b_batch_size = max(
                    1,
                    min(
                        orphan_count,
                        MAX_ASSIGNMENT_MATCHES // chunks_count
                    )
                )
                phase_b_total_batches = (
                    orphan_count + phase_b_batch_size - 1
                ) // phase_b_batch_size

                print(
                    "📦 PHASE B TOPIC BATCH SIZE:",
                    phase_b_batch_size
                )
                print(
                    "📦 PHASE B EXPECTED BATCHES:",
                    phase_b_total_batches
                )

                phase_b_pairs_processed = 0

                for batch_index in range(phase_b_total_batches):
                    start = batch_index * phase_b_batch_size
                    end = start + phase_b_batch_size
                    orphan_batch = orphan_rows[start:end]
                    orphan_ids = [row[0] for row in orphan_batch]

                    topic_clause, topic_params = build_in_clause(
                        "phase_b_topic",
                        orphan_ids
                    )
                    matches = db.execute(
                        text(f"""
                            select
                                c.id as chunk_id,
                                c.chunk_text,
                                c.section,
                                t.id as topic_id,
                                t.topic,
                                t.source_section,
                                c.embedding <#> t.embedding
                                    as negative_inner_product
                            from topics t
                            join chunks c on c.project_id = t.project_id
                            where t.project_id = :project_id
                            and t.id in ({topic_clause})
                            and c.chunk_role = 'teaching'
                            order by t.id, c.id
                        """),
                        {
                            "project_id": project_id,
                            **topic_params,
                        }
                    ).fetchall()

                    expected_pairs = len(orphan_ids) * chunks_count
                    actual_pairs = len(matches)
                    batch_number = batch_index + 1

                    print(
                        f"📦 PHASE B BATCH: "
                        f"{batch_number}/{phase_b_total_batches}"
                    )
                    print(
                        "📦 PHASE B BATCH TOPICS:",
                        len(orphan_ids)
                    )
                    print(
                        "🔢 PHASE B EXPECTED PAIRS:",
                        expected_pairs
                    )
                    print(
                        "🔢 PHASE B ACTUAL PAIRS:",
                        actual_pairs
                    )

                    if actual_pairs != expected_pairs:
                        raise Exception(
                            "Incomplete Phase B ranking: "
                            f"expected {expected_pairs} pairs, "
                            f"got {actual_pairs}"
                        )

                    best_by_topic = {}
                    for row in matches:
                        final_score = calculate_final_score(row)
                        if final_score is None:
                            continue

                        topic_id = row[3]
                        current_best = best_by_topic.get(topic_id)
                        candidate_key = (
                            final_score,
                            -int(row[0]),
                        )

                        if (
                            current_best is None
                            or candidate_key > current_best["key"]
                        ):
                            best_by_topic[topic_id] = {
                                "key": candidate_key,
                                "topic_id": topic_id,
                                "chunk_id": row[0],
                                "score": final_score,
                            }

                    rescue_assignments = []
                    batch_rescue_scores = []
                    for topic_id in orphan_ids:
                        best = best_by_topic.get(topic_id)
                        if (
                            best
                            and best["score"] >= TOPIC_RESCUE_THRESHOLD
                        ):
                            rescue_assignments.append(
                                {
                                    "topic_id": best["topic_id"],
                                    "chunk_id": best["chunk_id"],
                                }
                            )
                            batch_rescue_scores.append(best["score"])

                    if rescue_assignments:
                        db.execute(
                            text("""
                                insert into topic_chunks
                                (topic_id, chunk_id)
                                values
                                (:topic_id, :chunk_id)
                            """),
                            rescue_assignments
                        )

                    rescue_links += len(rescue_assignments)
                    phase_b_pairs_processed += actual_pairs

                    if batch_rescue_scores:
                        batch_lowest = min(batch_rescue_scores)
                        lowest_rescue_score = (
                            batch_lowest
                            if lowest_rescue_score is None
                            else min(lowest_rescue_score, batch_lowest)
                        )

                    print(
                        "🛟 PHASE B BATCH RESCUES:",
                        len(rescue_assignments)
                    )

                expected_phase_b_pairs = orphan_count * chunks_count
                if phase_b_pairs_processed != expected_phase_b_pairs:
                    raise Exception(
                        "Incomplete Phase B candidate processing: "
                        f"expected {expected_phase_b_pairs}, "
                        f"processed {phase_b_pairs_processed}"
                    )

                rescued_topic_count = db.execute(
                    text("""
                        select count(*)
                        from topics t
                        where t.project_id = :project_id
                        and exists (
                            select 1
                            from topic_chunks tc
                            where tc.topic_id = t.id
                        )
                    """),
                    {"project_id": project_id}
                ).scalar() - phase_a_covered_topics

                if rescued_topic_count != rescue_links:
                    raise Exception(
                        "Phase B rescue-count validation failed: "
                        f"expected {rescue_links}, "
                        f"found {rescued_topic_count}"
                    )

                db.commit()
                print("✅ PHASE B RESCUE COMMIT COMPLETE")

            else:
                db.rollback()

        except Exception as rescue_error:
            db.rollback()
            rescue_links = 0
            lowest_rescue_score = None
            print("❌ TOPIC RESCUE FAILED:", rescue_error)
            print("✅ PHASE A RESULTS PRESERVED")

        final_stats = get_coverage_stats()
        final_links = final_stats[0]
        final_covered_topics = final_stats[1]
        final_covered_chunks = final_stats[2]
        final_max_topics_per_chunk = final_stats[3]
        final_orphans = topics_count - final_covered_topics
        topic_coverage = (
            (final_covered_topics / topics_count) * 100
        )
        chunk_coverage = (
            (final_covered_chunks / chunks_count) * 100
        )

        print("🛟 RESCUE LINKS CREATED:", rescue_links)
        print(
            "🛟 ORPHANS BELOW RESCUE THRESHOLD:",
            final_orphans
        )
        print(
            "🛟 LOWEST RESCUE SCORE USED:",
            (
                round(lowest_rescue_score, 4)
                if lowest_rescue_score is not None
                else "none"
            )
        )
        print("🔗 FINAL TOPIC-CHUNK LINKS:", final_links)
        print(
            "📊 FINAL TOPIC COVERAGE:",
            f"{final_covered_topics}/{topics_count} "
            f"({topic_coverage:.1f}%)"
        )
        print(
            "📊 FINAL CHUNK COVERAGE:",
            f"{final_covered_chunks}/{chunks_count} "
            f"({chunk_coverage:.1f}%)"
        )
        print("🛟 FINAL ORPHAN TOPICS:", final_orphans)
        print(
            "📊 FINAL MAX TOPICS PER CHUNK:",
            final_max_topics_per_chunk
        )

    except Exception as e:

        db.rollback()
        print("❌ TOPIC ASSIGNMENT ERROR:", e)
        raise

    finally:
        db.close()

def detect_section_title(text: str, current_section="GENERAL"):



    if not text:
        return current_section

    lines = [
        l.strip()
        for l in text.split("\n")
        if l.strip()
    ]

    print()
    print("📑 FIRST LINES")
    for x in lines[:5]:
        print(">", x)
    print()

    strong_candidates = []

    for line in lines[:15]:

        clean = line.strip()

        upper_ratio = (
            sum(1 for c in clean if c.isupper())
            / max(len(clean), 1)
        )

        if (
            upper_ratio > 0.7
            and 2 <= len(clean.split()) <= 8
        ):
            strong_candidates.append(clean.upper())

        anatomical_keywords = [
            "upper limb",
            "lower limb",
            "shoulder",
            "arm",
            "forearm",
            "hand",
            "hip",
            "thigh",
            "leg",
            "foot",
            "pelvis"
        ]

        if any(
            kw in clean.lower()
            for kw in anatomical_keywords
        ):
            strong_candidates.append(clean.upper())

        

        if len(clean) < 5 or len(clean) > 120:
            continue

        

        

        if (
            any(k in clean.lower() for k in anatomical_keywords)
            and upper_ratio > 0.5
        ):

            if upper_ratio > 0.45:
                strong_candidates.append(clean)

    if strong_candidates:

        print("📚 CANDIDATES:", strong_candidates)

        return strong_candidates[0].upper()

    return current_section

@app.post("/projects/{project_id}/ingest_stream")
async def ingest_stream(
    project_id: str,
    data: IngestRequest,
    background_tasks: BackgroundTasks,
    user = Depends(verify_user)
):
    docs = data.documents

    async def generate():
        db = SessionLocal()

        try:
            db.execute(
                text("""
                    update projects
                    set topic_status = 'processing'
                    where id = :project_id
                """),
                {"project_id": project_id}
            )
            db.commit()

            yield "Starting upload...\n"
            
            import re

            def clean_text(text):
                if not text:
                    return ""

                # separa parole attaccate tipo "TEAMBefore"
                text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

                # separa numeri e lettere
                text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
                text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

                # aggiunge spazio dopo punto
                text = re.sub(r'\.(\w)', r'. \1', text)

                # normalizza spazi
                text = re.sub(r'\s+', ' ', text)

                return text.strip()
            # ======================
            # SAVE CHUNKS
            # ======================
            project_chunk_roles = []
            for doc in docs:
                yield f"Processing document: {doc.title}\n"

                pdf_bytes = base64.b64decode(doc.file_bytes)
                pdf_stream = io.BytesIO(pdf_bytes)

                reader = PdfReader(pdf_stream)

                total_pages = len(reader.pages)

                yield f"FILE_ANALYSIS|pages={total_pages}\n"

                if total_pages > MAX_WARNING_PAGES:
                    yield (
                        "LARGE_FILE_WARNING|"
                        f"pages={total_pages}|"
                        "message=Large academic file detected. "
                        "We can process it, but topic generation may take longer. "
                        "For best results, consider splitting very large files by chapter.\n"
                    )

                current_section = (
                    clean_text(doc.title)
                    .upper()
                )
                for page_index, page in enumerate(reader.pages):
                    
                    print(f"📄 PROCESSING PAGE {page_index+1}/{len(reader.pages)}")
                    yield f"Page {page_index+1}\n"
                    db.execute(
                        text("""
                            UPDATE projects
                            SET last_processed_page = :page
                            WHERE id = :project_id
                        """),
                        {
                            "page": page_index + 1,
                            "project_id": project_id
                        }
                    )

                    db.commit()

                    raw_page_text = page.extract_text()

                    current_section = detect_section_title(
                        raw_page_text,
                        current_section
                    )

                    section_title = current_section
                    print(f"📚 DETECTED SECTION: {section_title}")

                    page_text = clean_text(raw_page_text)

                    if not page_text or not page_text.strip():
                        yield f"OCR page {page_index+1}\n"
                        page_text = ocr_pdf_page(pdf_bytes, page_index)

                        if not page_text:
                            continue

                    chunks = chunk_text(page_text)
                    chunks = [clean_text(c) for c in chunks]
                    chunks = [c for c in chunks if len(c) > 100]

                    print(
                        f"📦 PAGE {page_index+1} -> {len(chunks)} CHUNKS"
                    )

                    yield f"{len(chunks)} chunks created\n"

                    for i, chunk in enumerate(chunks):
                        yield f"Embedding chunk {i+1}/{len(chunks)}\n"
                        await asyncio.sleep(0)
                        chunk_role = classify_chunk_role(
                            chunk,
                            page_number=page_index + 1,
                            doc_title=doc.title,
                        )
                        project_chunk_roles.append(chunk_role)

                        emb = await asyncio.to_thread(
                            client.embeddings.create,
                            model="text-embedding-3-small",
                            input=chunk
                        )

                        embedding = emb.data[0].embedding
                        embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                        # --- AGGIUNGI QUI IL LOG DI DEBUG ---
                        print(f"DEBUG: Processando chunk {i} per il documento: {doc.title}")
                        if not doc.title:
                            print("ATTENZIONE: doc.title è vuoto o None!")
    # ------------------------------------

                        db.execute(
                            text("""
                                insert into chunks
                                (
                                    project_id,
                                    doc_title,
                                    chunk_text,
                                    embedding,
                                    page,
                                    topic,
                                    section,
                                    chunk_role
                                )
                                values
                                (
                                    :project_id,
                                    :doc_title,
                                    :chunk_text,
                                    CAST(:embedding AS vector),
                                    :page,
                                    :topic,
                                    :section,
                                    :chunk_role
                                )
                            """),
                            {
                                "project_id": project_id,
                                "doc_title": doc.title,
                                "chunk_text": chunk,
                                "embedding": embedding_str,
                                "page": page_index + 1,      
                                "topic": None,
                                "section": section_title,
                                "chunk_role": chunk_role,
                            }
                        )
                        db.commit()
                        print(f"CHUNK {i} SALVATO CON TOPIC: {doc.title}")

            
            log_chunk_role_counts(project_chunk_roles)

            background_tasks.add_task(process_topics_task, project_id)
            print("✅ BACKGROUND TOPICS TASK SCHEDULED:", project_id)

            yield "Upload complete ✅\n"

        except Exception as e:
            print("UPLOAD EXCEPTION:", repr(e))
            db.rollback()
            db.execute(text("update projects set topic_status = 'error' where id = :project_id"), {"project_id": project_id})
            db.commit()
            yield f"Upload failed: {str(e)}\n"
        finally:
            db.close()

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

def expand_topics_with_db(project_id, topics, db):
    if not topics:
        return []

    expanded = []

    try:
        rows = db.execute(
            text("""
                select topic
                from chunks
                where project_id = :project_id
                and topic is not null
                group by topic
                limit 50
            """),
            {
                "project_id": project_id
            }
        ).fetchall()

        all_topics = [normalize_string(row[0]) for row in rows]

        for t in topics:
            t_norm = normalize_string(t)

            matches = [
                topic for topic in all_topics
                if t_norm.lower().replace("actions", "action") == topic.lower()
            ]

            print(f"🔎 AUTO-MATCH for '{t}' (norm: '{t_norm}'):", matches)

            expanded.extend(matches)
    except Exception as e:
        print("❌ ERROR IN expand_topics_with_db:", e)
        return topics

    if not expanded:
        return topics

    return list(set(expanded))

# ======================
# LEARNING SCOPE
# ======================

def resolve_learning_scope(project_id: str, topic_ids=None, limit: int = 80):
    db = SessionLocal()

    try:
        topic_ids = topic_ids or []
        print("🚨 RESOLVE LEARNING SCOPE")
        print("🚨 TOPIC IDS:", topic_ids)
        print("🚨 LEN:", len(topic_ids))

        if topic_ids:
            print("🚨 ENTERING TOPIC BRANCH")
            scope_type = "single_topic" if len(topic_ids) == 1 else "multi_topic"

            rows = db.execute(
                text("""
                    SELECT
                        c.id,
                        c.chunk_text,
                        c.doc_title,
                        c.page,
                        t.topic,
                        t.id as topic_id,
                        c.chunk_role,
                        c.section,
                        t.source_section,
                        c.embedding <#> t.embedding
                            as negative_inner_product
                    FROM topic_chunks tc
                    JOIN chunks c
                        ON c.id = tc.chunk_id
                    JOIN topics t
                        ON t.id = tc.topic_id
                    WHERE t.project_id = :project_id
                    AND tc.topic_id IN :topic_ids
                    AND c.chunk_text IS NOT NULL
                    AND length(c.chunk_text) > 100
                """),
                {
                    "project_id": project_id,
                    "topic_ids": tuple(topic_ids),
                }
            ).fetchall()

        else:
            print("🚨 ENTERING GLOBAL BRANCH")
            scope_type = "global"

            rows = db.execute(
                text("""
                    SELECT
                        c.id,
                        c.chunk_text,
                        c.doc_title,
                        c.page,
                        COALESCE(t.topic, 'General') as topic,
                        t.id as topic_id,
                        c.chunk_role,
                        c.section,
                        t.source_section,
                        c.embedding <#> t.embedding
                            as negative_inner_product
                    FROM chunks c
                    LEFT JOIN topic_chunks tc
                        ON tc.chunk_id = c.id
                    LEFT JOIN topics t
                        ON t.id = tc.topic_id
                    WHERE c.project_id = :project_id
                    AND c.chunk_text IS NOT NULL
                    AND length(c.chunk_text) > 100
                """),
                {
                    "project_id": project_id,
                }
            ).fetchall()

        eligible_candidates = []
        excluded_chunk_ids = set()
        for r in rows:
            chunk_role = normalize_chunk_role(
                r[6],
                r[1],
                page_number=r[3],
                doc_title=r[2],
            )

            if not is_assignment_eligible_chunk_role(chunk_role):
                excluded_chunk_ids.add(r[0])
                continue

            score = calculate_topic_chunk_score(
                r[1],
                r[7],
                r[4],
                r[8],
                r[9],
            )
            eligible_candidates.append({
                "chunk_id": str(r[0]),
                "chunk_text": r[1],
                "doc_title": r[2],
                "page": r[3],
                "topic": r[4],
                "topic_id": str(r[5]) if r[5] else None,
                "chunk_role": chunk_role,
                "_assignment_score": score,
            })

        print(
            "RETRIEVAL CANDIDATES BEFORE DEDUP:",
            len(eligible_candidates)
        )
        print(
            "ELIGIBLE TEACHING CHUNKS:",
            len({
                candidate["chunk_id"]
                for candidate in eligible_candidates
            })
        )
        print(
            "RETRIEVAL EXCLUDED NON-TEACHING CHUNKS:",
            len(excluded_chunk_ids)
        )
        print("EXCLUDED CHUNKS:", len(excluded_chunk_ids))

        best_by_chunk = {}
        for candidate in eligible_candidates:
            chunk_id = candidate["chunk_id"]
            current = best_by_chunk.get(chunk_id)
            candidate_key = (
                candidate["_assignment_score"]
                if candidate["_assignment_score"] is not None
                else float("-inf"),
                candidate["topic_id"] or "",
            )
            current_key = (
                current["_assignment_score"]
                if (
                    current
                    and current["_assignment_score"] is not None
                )
                else float("-inf"),
                current["topic_id"] or "" if current else "",
            )

            if current is None or candidate_key > current_key:
                best_by_chunk[chunk_id] = candidate

        chunks = list(best_by_chunk.values())
        print(
            "RETRIEVAL CANDIDATES AFTER DEDUP:",
            len(chunks)
        )

        if topic_ids:
            chunks.sort(
                key=lambda candidate: (
                    -(
                        candidate["_assignment_score"]
                        if candidate["_assignment_score"] is not None
                        else float("-inf")
                    ),
                    candidate["chunk_id"],
                )
            )
        else:
            random.shuffle(chunks)

        chunks = chunks[:limit]
        for candidate in chunks:
            candidate.pop("_assignment_score", None)

        if topic_ids:
            print("🎯 TOPIC FILTER ACTIVE")

            for candidate in chunks[:20]:
                print()
                print("TOPIC:", candidate["topic"])
                print("PAGE:", candidate["page"])
                print((candidate["chunk_text"] or "")[:250])
                print("-" * 80)

        topic_map = {
            c["chunk_id"]: c["topic"]
            for c in chunks
            if c.get("topic")
        }

        topics = {}

        for c in chunks:
            if c["topic_id"]:
                topics[c["topic_id"]] = {
                    "id": c["topic_id"],
                    "topic": c["topic"]
                }

        return {
            "scope_type": scope_type,
            "topic_ids": topic_ids,
            "topics": list(topics.values()),
            "chunks": chunks,
            "topic_map": topic_map,
            "chunk_count": len(chunks)
        }

    finally:
        db.close()
# ======================
# GENERATE QUIZ
# ======================
LOW_QUALITY_PATTERNS = [
                "what is the primary",
                "what is the main",
                "what role",
                "what function",
                "which of the following",
                "which process is",
                "which pathway is",
            ]

HARD_QUIZ_SPEC = {
    "minimum_supported_propositions": 2,
    "minimum_stem_words": 14,
    "minimum_explanation_words": 16,
    "reasoning_modes": (
        "comparison between closely related concepts",
        "distinction between mechanisms with shared features",
        "identification of a subtle source-supported exception",
        "synthesis of facts from multiple supplied statements",
        (
            "analysis of an implication explicitly established by the "
            "supplied evidence"
        ),
    ),
    "forbidden_question_forms": (
        "simple definitions",
        "identification of one named fact",
        "direct mechanism lookup",
        "keyword matching",
        '"Which statement is true/false?"',
        '"Which mutation/mechanism/process causes X?"',
        "questions answerable from one isolated sentence",
    ),
    "forbidden_added_content": (
        "invented cases",
        "invented actors",
        "invented events",
        "invented processes",
        "invented consequences",
        "unstated premises",
    ),
    "direct_recall_openings": (
        "what is ",
        "what does ",
        "which statement is true",
        "which statement is false",
        "which of the following is true",
        "which of the following is false",
        "which mutation ",
        "which mechanism causes",
        "which mechanism is",
        "which process ",
        "which enzyme ",
        "which molecule ",
        "che cos'è ",
        "che cosa è ",
        "quale affermazione è vera",
        "quale affermazione è falsa",
        "quale affermazione è corretta",
        "quale affermazione non è corretta",
        "quale mutazione ",
        "quale meccanismo causa",
        "quale meccanismo è",
        "quale processo ",
        "quale enzima ",
        "quale molecola ",
    ),
    "relational_markers": (
        "although",
        "both ",
        "by combining",
        "compare",
        "compared to",
        "compared with",
        "comparison",
        "considering",
        "contrast",
        "contrasted",
        "contrasting",
        "distinct",
        "distinction",
        "difference",
        "different",
        "differ ",
        "differ from",
        "differentiate",
        "distinguish",
        "except",
        "exception",
        "given that",
        "however",
        "if both",
        "in contrast",
        "integrate",
        "integrating",
        "integration",
        "relate",
        "relation between",
        "relationship",
        "relationship between",
        "synthesize",
        "synthesizes",
        "synthesized",
        "synthesizing",
        "synthesis",
        "unless",
        "unlike",
        "whereas",
        "while",
        "a differenza",
        "combinando",
        "comparando",
        "confrontando",
        "considerando",
        "contrasto",
        "distinto",
        "distinzione",
        "differire",
        "differisce",
        "differiscono",
        "differenza",
        "distinguere",
        "eccezione",
        "eccetto",
        "entrambi",
        "integrare",
        "integrando",
        "integrazione",
        "mentre",
        "mettendo in relazione",
        "rapporto tra",
        "relazionare",
        "relazione tra",
        "rispetto a",
        "se entrambi",
        "sintetizzare",
        "sintetizza",
        "sintetizzando",
        "sintesi",
        "salvo che",
        "a meno che",
        "tuttavia",
    ),
    "explanation_connectors": (
        "because",
        "since",
        "therefore",
        "thus",
        "consequently",
        "hence",
        "thereby",
        "whereas",
        "while",
        "which means",
        "which explains",
        "as a result",
        "in contrast",
        "unlike",
        "poiché",
        "perché",
        "dato che",
        "dal momento che",
        "quindi",
        "pertanto",
        "dunque",
        "di conseguenza",
        "ne consegue",
        "in tal modo",
        "mentre",
        "ciò significa",
        "ciò spiega",
        "a differenza",
        "invece",
    ),
}


def render_hard_quiz_specification(spec=HARD_QUIZ_SPEC):
    reasoning_modes = "\n".join(
        f"- {mode}"
        for mode in spec["reasoning_modes"]
    )
    forbidden_question_forms = "\n".join(
        f"- {question_form}"
        for question_form in spec["forbidden_question_forms"]
    )
    forbidden_added_content = ", ".join(
        spec["forbidden_added_content"]
    )

    return f"""
HARD COGNITIVE-DIFFICULTY SPECIFICATION:

- Every question must require at least
  {spec["minimum_supported_propositions"]} explicitly supported propositions.
- The stem must contain at least {spec["minimum_stem_words"]} words.
- The explanation must contain at least
  {spec["minimum_explanation_words"]} words.
- The stem must explicitly express comparison, distinction, exception
  handling, synthesis, or another listed relational reasoning mode.
- Comparison and distinction are intermediate reasoning operations, not
  sufficient HARD outcomes by themselves. The student must use them to
  determine an explicitly supported implication, consequence, outcome,
  exception, applicable rule, or conclusion. If the question can be answered
  by independently recalling fact A and fact B, it is not HARD.
- The explanation must explicitly connect the supporting propositions and
  show the reasoning needed to reach the answer.
- Direct-recall openings are invalid unless the stem itself clearly expresses
  one of the allowed relational reasoning modes.
- Use explicit relational wording in the stem and an explicit logical
  connector in the explanation so compliance is unambiguous.

ALLOWED REASONING MODES:
{reasoning_modes}

HARD EXAMPLE:
- Given the supported differences between A and B, which implication follows
  when both are considered together?

FORBIDDEN QUESTION FORMS:
{forbidden_question_forms}

FORBIDDEN ADDED CONTENT:
{forbidden_added_content}.

Difficulty must come from reasoning over retrieved evidence. It must not come
from invented context or external knowledge.
"""


def render_hard_context_specification(spec=HARD_QUIZ_SPEC):
    reasoning_modes = ", ".join(spec["reasoning_modes"])

    return f"""
HARD CONTEXT REQUIREMENT:

Each context must place at least
{spec["minimum_supported_propositions"]} explicitly supported source
propositions into one of these reasoning relationships:
{reasoning_modes}.

Do not return a single definition, isolated mechanism, or standalone fact.
The context may remain conceptual and must not add invented content.
"""


def render_hard_exam_formats(spec=HARD_QUIZ_SPEC):
    minimum_propositions = spec["minimum_supported_propositions"]

    return f"""
- Considering {minimum_propositions} supported principles, which conclusion
  follows only when they are applied together?
- Which distinction between the supported mechanisms explains their different
  outcomes?
- Which source-supported exception changes the otherwise applicable rule?
- Which interpretation synthesizes the supplied propositions?
"""


def evaluate_hard_question_reasoning(
    question,
    spec=HARD_QUIZ_SPEC,
):
    question_text = normalize_string(
        question.get("question", "")
    ).lower()
    explanation_candidates = [
        normalize_string(question.get(field_name) or "").lower()
        for field_name in ("explanation", "explanation_long")
    ]
    stem_relational_reasoning = any(
        marker in question_text
        for marker in spec["relational_markers"]
    )
    direct_recall_opening = any(
        question_text.startswith(pattern)
        for pattern in spec["direct_recall_openings"]
    )
    explanation_evaluations = [
        {
            "text": explanation,
            "word_count": len(explanation.split()),
            "connector_present": any(
                connector in explanation
                for connector in spec["explanation_connectors"]
            ),
        }
        for explanation in explanation_candidates
        if explanation
    ]
    compliant_explanations = [
        evaluation
        for evaluation in explanation_evaluations
        if (
            evaluation["word_count"]
            >= spec["minimum_explanation_words"]
        )
    ]
    best_explanation = max(
        explanation_evaluations,
        key=lambda evaluation: (
            evaluation["word_count"],
            evaluation["connector_present"],
        ),
        default={
            "text": "",
            "word_count": 0,
            "connector_present": False,
        },
    )
    rejection_reasons = []

    if direct_recall_opening and not stem_relational_reasoning:
        rejection_reasons.append("direct_recall_opening")

    if (
        len(question_text.split())
        < spec["minimum_stem_words"]
    ):
        rejection_reasons.append("stem_too_short_for_hard_reasoning")

    if not compliant_explanations:
        rejection_reasons.append(
            "explanation_does_not_show_two_step_reasoning"
        )

    return {
        "valid": not rejection_reasons,
        "reasons": rejection_reasons,
        "relational_reasoning": stem_relational_reasoning,
        "explanation_word_count": best_explanation["word_count"],
        "explanation_connector_present": any(
            evaluation["connector_present"]
            for evaluation in explanation_evaluations
        ),
        "source_chunk_count": len(
            set(question.get("source_chunk_ids") or [])
        ),
    }


def build_hard_generation_metrics(
    requested_questions,
    generated_questions,
    rejected_questions,
    rejection_reasons,
):
    rejected_questions = min(
        max(rejected_questions, 0),
        generated_questions,
    )
    accepted_questions = max(
        generated_questions - rejected_questions,
        0,
    )
    rejection_reasons_breakdown = {}

    for reason in rejection_reasons:
        rejection_reasons_breakdown[reason] = (
            rejection_reasons_breakdown.get(reason, 0) + 1
        )

    acceptance_rate = (
        round(
            accepted_questions / generated_questions,
            4,
        )
        if generated_questions
        else 0.0
    )

    return {
        "requested_questions": requested_questions,
        "generated_questions": generated_questions,
        "accepted_questions": accepted_questions,
        "rejected_questions": rejected_questions,
        "acceptance_rate": acceptance_rate,
        "rejection_reasons_breakdown": dict(
            sorted(rejection_reasons_breakdown.items())
        ),
    }


def resolve_quiz_question_topic(
    question,
    chunk_topic_map,
    canonical_project_topics,
):
    source_chunk_ids = question.get("source_chunk_ids") or []
    if not isinstance(source_chunk_ids, list):
        source_chunk_ids = [source_chunk_ids]

    resolved_topics = []
    seen_chunk_ids = set()
    normalized_source_chunk_ids = []

    for source_chunk_id in source_chunk_ids:
        normalized_chunk_id = str(source_chunk_id).strip()
        if (
            not normalized_chunk_id
            or normalized_chunk_id in seen_chunk_ids
        ):
            continue

        seen_chunk_ids.add(normalized_chunk_id)
        normalized_source_chunk_ids.append(normalized_chunk_id)
        resolved_topic = chunk_topic_map.get(normalized_chunk_id)
        if resolved_topic:
            resolved_topics.append(resolved_topic)

    resolution = "unattributed"
    resolved_topic = None

    if resolved_topics:
        topic_counts = {}
        for topic in resolved_topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

        if len(topic_counts) == 1:
            resolved_topic = resolved_topics[0]
            resolution = "source_same_topic"
        else:
            majority_topic = next(
                (
                    topic
                    for topic, count in topic_counts.items()
                    if count > len(resolved_topics) / 2
                ),
                None,
            )

            if majority_topic:
                resolved_topic = majority_topic
                resolution = "source_majority_topic"
            else:
                resolved_topic = resolved_topics[0]
                resolution = "source_first_topic_tiebreak"
    else:
        model_topic = question.get("topic")
        if isinstance(model_topic, str):
            model_topic = model_topic.strip()
            if model_topic in canonical_project_topics:
                resolved_topic = model_topic
                resolution = "canonical_model_fallback"

    return {
        "topic": resolved_topic,
        "resolution": resolution,
        "source_chunk_ids": normalized_source_chunk_ids,
        "resolved_source_topics": resolved_topics,
    }


def build_hard_question_diagnostic_sample(
    question,
    outcome,
    question_type,
    rejection_reasons=None,
    topic_category_by_name=None,
):
    topic = question.get("topic")
    topic_text = str(topic).strip() if topic else None
    category = question.get("category")

    if not category and topic_text and topic_category_by_name:
        category = topic_category_by_name.get(
            normalize_string(topic_text).lower()
        )

    return {
        "outcome": outcome,
        "question_stem": str(
            question.get("question") or ""
        ).strip(),
        "rejection_reasons": list(rejection_reasons or []),
        "question_type": question_type,
        "topic": topic_text,
        "category": str(category).strip() if category else None,
    }


def persist_hard_generation_diagnostics(metrics, samples):
    diagnostics_db = SessionLocal()

    try:
        run_id = diagnostics_db.execute(
            text("""
                insert into hard_quiz_generation_runs (
                    project_id,
                    project_name,
                    quiz_id,
                    difficulty,
                    question_style,
                    requested_questions,
                    generated_questions,
                    accepted_questions,
                    rejected_questions,
                    acceptance_rate,
                    rejection_reasons_breakdown
                )
                values (
                    :project_id,
                    :project_name,
                    :quiz_id,
                    :difficulty,
                    :question_style,
                    :requested_questions,
                    :generated_questions,
                    :accepted_questions,
                    :rejected_questions,
                    :acceptance_rate,
                    CAST(:rejection_reasons_breakdown AS jsonb)
                )
                returning id
            """),
            {
                **metrics,
                "rejection_reasons_breakdown": json.dumps(
                    metrics["rejection_reasons_breakdown"]
                ),
            },
        ).scalar()

        if samples:
            diagnostics_db.execute(
                text("""
                    insert into hard_quiz_generation_samples (
                        run_id,
                        outcome,
                        question_stem,
                        rejection_reasons,
                        question_type,
                        topic,
                        category
                    )
                    values (
                        :run_id,
                        :outcome,
                        :question_stem,
                        CAST(:rejection_reasons AS jsonb),
                        :question_type,
                        :topic,
                        :category
                    )
                """),
                [
                    {
                        **sample,
                        "run_id": run_id,
                        "rejection_reasons": json.dumps(
                            sample["rejection_reasons"]
                        ),
                    }
                    for sample in samples
                ],
            )

        diagnostics_db.commit()
        print("HARD GENERATION DIAGNOSTICS STORED:", run_id)
        return str(run_id)

    except Exception as diagnostic_error:
        diagnostics_db.rollback()
        print(
            "HARD GENERATION DIAGNOSTICS STORAGE FAILED:",
            repr(diagnostic_error),
        )
        return None

    finally:
        diagnostics_db.close()


def validate_quiz_requested_count(requested_count, questions):
    accepted_count = len(questions)

    if accepted_count != requested_count:
        raise ValueError(
            "Quiz assembly incomplete: "
            f"requested {requested_count}, accepted {accepted_count}"
        )


@app.post("/projects/{project_id}/generate_quiz")
async def generate_quiz(
    

    
    project_id: str,
    req: QuizRequest,
    user = Depends(verify_user)
):
    quiz_start = time.time()

    print("🚀 QUIZ START")
    # 🔥 PRINT 1 — INIZIO FUNZIONE
    print("🔥 ENTER generate_quiz")
    print("🔥 FULL REQUEST:", req.dict())
    print("🎨 QUESTION STYLE:", req.question_style)

    user_id = user["id"]
    db = SessionLocal()

    existing_questions = db.execute(
        text("""
            select question
            from quiz_questions q
            join quizzes z on q.quiz_id = z.id
            where z.project_id = :project_id
        """),
        {"project_id": project_id}
    ).fetchall()

    existing_texts = set(q[0].strip().lower() for q in existing_questions)

    project = db.execute(
        text("""
            select id, name from projects
            where id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    if not project:
        db.close()
        raise HTTPException(status_code=403, detail="Access denied")

    project_name = project[1]
    quiz_id = str(uuid.uuid4())

    # ======================
    # RETRIEVAL (UNA SOLA VOLTA)
    # ======================

    print(f"DEBUG: Avvio generazione quiz per progetto {project_id}")

    # 🔥 NUOVO SISTEMA
    topic_ids = req.topic_ids or []

    # 🔥 LEGACY
    legacy_topics = req.topics or []

    if not topic_ids and legacy_topics:

        topic_rows = db.execute(
            text("""
                select id
                from topics
                where project_id = :project_id
                and topic in :topics
            """),
            {
                "project_id": project_id,
                "topics": tuple(legacy_topics)
            }
        ).fetchall()

        topic_ids = [
            str(r[0])
            for r in topic_rows
        ]

        print(
            "🔄 CONVERTED LEGACY TOPICS TO IDS:",
            topic_ids
        )
    print("🚨 FINAL TOPIC IDS USED:", topic_ids)

    scope = resolve_learning_scope(
        project_id,
        topic_ids,
        limit=80
    )

    print("🧠 LEARNING SCOPE:", scope["scope_type"])
    print("📦 SCOPE CHUNKS:", scope["chunk_count"])
    print("🎯 SCOPE TOPICS:", scope["topics"][:3])

    print("📥 TOPIC IDS:", topic_ids)
    print("📥 LEGACY TOPICS:", legacy_topics)
    print("🔥 QUIZ REQUEST RECEIVED")
    print("topic_ids:", topic_ids)
    print("legacy_topics:", legacy_topics)
    print("num_questions:", req.num_questions)
    
    # 1. RETRIEVAL POTENZIATO (120 chunk)
    query_text = " ".join(req.topics) if req.topics else "General overview of the provided documents"
    emb_res = client.embeddings.create(model="text-embedding-3-small", input=query_text)
    query_embedding = emb_res.data[0].embedding

    rows = []

    rows = []

    # =====================================
    # NEW LEARNING GRAPH RETRIEVAL
    # =====================================

    scope = resolve_learning_scope(
        project_id=project_id,
        topic_ids=topic_ids,
        limit=80
    )

    print("🧠 LEARNING SCOPE:", scope["scope_type"])

    retrieved_chunks = scope["chunks"]

    print("📦 RETRIEVED CHUNKS:", len(retrieved_chunks))
    from collections import Counter

    topic_counter = Counter(
        str(c.get("topic", "General"))
        for c in retrieved_chunks
    )

    print("📊 RETRIEVED TOPIC DISTRIBUTION:")
    print(topic_counter.most_common(20))
    for c in retrieved_chunks[:5]:

        print("📄 QUIZ CHUNK SAMPLE")
        print("TOPIC:", c["topic"])

        print(
            c["chunk_text"][:500]
        )

        print("=" * 80)

    topic_map = scope["topic_map"]


    for c in retrieved_chunks[:5]:
        print(
            f"📄 CHUNK DA TOPIC: {c['topic']} | TESTO: {c['chunk_text'][:100]}..."
        )

    chunk_topic_map = {
        c["chunk_id"]: " ".join(str(c["topic"]).split())
        for c in retrieved_chunks
        if c["topic"]
    }
    print("🧠 CHUNK TOPIC MAP SAMPLE:")
    print(list(chunk_topic_map.items())[:5])

    if req.topics:
        active_topics = [normalize_string(t) for t in req.topics]
    else:
        active_topics = list(set(
            normalize_string(c["topic"])
            for c in retrieved_chunks
            if c["topic"]
        )) or ["General"]

    # 4. Fallback se non ci sono topic
    if not active_topics:
        active_topics = ["General"]

    print("🎯 ACTIVE TOPICS (CLEANED):", active_topics)

    topic_category_rows = db.execute(
        text("""
            select topic, category
            from topics
            where project_id = :project_id
        """),
        {"project_id": project_id},
    ).fetchall()
    topic_category_by_name = {
        normalize_string(row[0]).lower(): row[1]
        for row in topic_category_rows
        if row[0]
    }
    

    if not retrieved_chunks:
        db.close()
        return {"quiz": []}

        

    def fails_basic_reasoning_check(question_text):

        normalized = (
            question_text
            .strip()
            .lower()
        )
        print("🔍 CHECKING:", normalized)
        print("🚨 ENTER LOOP")
        for pattern in LOW_QUALITY_PATTERNS:

            if normalized.startswith(pattern):

                print("🚫 MATCHED:", pattern)

                return True
            
            # NEW RULE 😄
            if "?" in normalized:

                first_part = normalized.split("?")[0]

                if len(first_part.split()) < 12:

                    return True
        print("✅ CHECK PASSED")
        return False

    def score_reasoning_quality(question_text):

        normalized = question_text.lower()

        score = 0

        scenario_indicators = [
            "during",
            "after",
            "when",
            "following",
            "in response",
            "as a result",
            "because",
            "while",
            "despite",
            "under",
            "increased",
            "decreased",
            "reduced",
            "elevated",
            "shift",
            "adaptation",
            "response"
        ]

        generic_patterns = [
            "what is",
            "which enzyme",
            "which molecule",
            "what role",
            "what is the main",
            "what is the primary",
            "which process",
            "which pathway",
            "what function"
        ]

        causal_words = [
            "because",
            "therefore",
            "leading to",
            "resulting in",
            "causing",
            "due to",
            "consequence"
        ]

        if any(x in normalized for x in scenario_indicators):
            score += 3

        if any(x in normalized for x in causal_words):
            score += 2

        if len(question_text.split()) > 18:
            score += 1

        if any(x in normalized for x in generic_patterns):
            score -= 5

        return score

    hard_rejection_reasons = []
    hard_rejection_samples = []
    hard_question_type_by_object_id = {}

    async def validate_question(
        question,
        style
    ):

        print("✅ VALIDATOR CALLED")

        question_text = question.get("question", "")

        normalized = question_text.lower()

        heuristic_score = score_reasoning_quality(
            question_text
        )

        print("📊 HEURISTIC SCORE:", heuristic_score)

        print("🔍 VALIDATING:", question_text)
        

        print("🎨 STYLE:", style)

        print(
            "✅ CORRECT:",
            question.get("correct")
        )

        print(
            "✅ CORRECT_ANSWER:",
            question.get("correct_answer")
        )

        print(
            "📋 OPTIONS:",
            question.get("options")
        )

        print(
            "📝 EXPLANATION:",
            question.get("explanation")
        )

        if (req.difficulty or "").strip().lower() == "hard":
            hard_evaluation = evaluate_hard_question_reasoning(
                question
            )
            print(
                "🧠 HARD REASONING VALIDATION:",
                hard_evaluation
            )

            if not hard_evaluation["valid"]:
                print(
                    "🚫 HARD QUESTION REJECTED:",
                    hard_evaluation["reasons"]
                )
                hard_rejection_reasons.extend(
                    hard_evaluation["reasons"]
                )
                hard_rejection_samples.append(
                    build_hard_question_diagnostic_sample(
                        question=question,
                        outcome="rejected",
                        question_type=style,
                        rejection_reasons=hard_evaluation["reasons"],
                        topic_category_by_name=(
                            topic_category_by_name
                        ),
                    )
                )
                return False

        scenario_indicators = [

            # EN
            "during",
            "after",
            "when",
            "following",
            "in response",
            "as a result",
            "because",
            "while",
            "as",
            "if",
            "upon",
            "given",
            "leads to",
            "results in",
            "increases",
            "decreases",

            # IT
            "durante",
            "dopo",
            "quando",
            "in seguito",
            "in risposta",
            "come conseguenza",
            "poiché",
            "mentre",

            # comuni
            "aumento",
            "diminuzione",
            "incremento",
            "riduzione",
            "crescita",
            "calo"
        ]

        if style != "exam":

            if not any(
                x in normalized
                for x in scenario_indicators
            ):

                print("⚠️ NO SCENARIO STRUCTURE")
                print("QUESTION:", question_text)

                heuristic_score -= 1

        if style != "exam":

            if heuristic_score <= -2:

                print("🚫 HEURISTIC REJECTION")
                if (req.difficulty or "").strip().lower() == "hard":
                    hard_rejection_reasons.append(
                        "heuristic_score_rejection"
                    )
                    hard_rejection_samples.append(
                        build_hard_question_diagnostic_sample(
                            question=question,
                            outcome="rejected",
                            question_type=style,
                            rejection_reasons=[
                                "heuristic_score_rejection"
                            ],
                            topic_category_by_name=(
                                topic_category_by_name
                            ),
                        )
                    )
                
                return False

        if style != "exam":

            if fails_basic_reasoning_check(question_text):

                print("🚫 BASIC FILTER REJECTED")
                print("❌ REJECT REASON: heuristic")
                if (req.difficulty or "").strip().lower() == "hard":
                    hard_rejection_reasons.append(
                        "basic_reasoning_filter_rejection"
                    )
                    hard_rejection_samples.append(
                        build_hard_question_diagnostic_sample(
                            question=question,
                            outcome="rejected",
                            question_type=style,
                            rejection_reasons=[
                                "basic_reasoning_filter_rejection"
                            ],
                            topic_category_by_name=(
                                topic_category_by_name
                            ),
                        )
                    )
                return False
        print("✅ QUESTION ACCEPTED")
        return True

    def rewrite_question_opening(question_text):

        replacements = [
            ("What is the role of", "During a system response,"),
            ("What is the primary role of", "In the context of a changing system,"),
            ("What is the main role of", "During a regulatory process,"),
            ("What is the function of", "Within an adaptive process,"),
            ("What is the primary function of", "Within a dynamic system,"),
            ("What is the significance of", "In a situation where"),
            ("What is the primary consequence of", "A prolonged alteration results in"),
            ("What is the main consequence of", "A system disruption leads to"),
            ("What happens to", "During a changing condition,"),
            ("What occurs", "During this transition,"),

            ("What is", "In this situation,"),
            ("Which of the following", "Among the following scenarios,"),
            ("Which process", "During a changing situation,"),
            ("Which pathway", "Within this sequence of events,"),
            ("Which mechanism", "The explanation that BEST fits this situation"),
            ("Which component", "An important element in this context"),
            ("Which factor", "A key factor in this situation"),
            ("Which condition", "Under these circumstances"),
            ("Which statement", "The most accurate interpretation"),
            ("Which enzyme", "A required element in this process"),
            ("Which molecule", "An important component in this situation"),
            ("Which amino acid", "In this context,"),
            ("Which metabolic process", "During this system change,"),
        ]

        for old, new in replacements:

            if question_text.startswith(old):

                question_text = question_text.replace(old, new, 1)

                break

        return question_text

    async def generate_reasoning_chain(
        scenario_text,
        reasoning_material,
        client
    ):

        reasoning_prompt = f"""
    Analyze the following proposed learning scenario against the retrieved
    evidence.

    SCENARIO:
    {scenario_text}

    The RETRIEVED MATERIAL is the authoritative factual source.
    The scenario is only a proposed restatement of that material.

    Use ONLY facts explicitly stated in the retrieved material.
    Do not use general knowledge to complete, enrich, or repair the scenario.
    If the scenario contains an actor, institution, object, event,
    relationship, process, condition, or consequence that is not supported
    by the retrieved material, exclude that detail from the reasoning
    blueprint.

    Extract an EVIDENCE-GROUNDED REASONING BLUEPRINT.

    

    Focus on:
    - what is explicitly established by the retrieved material
    - what comparison, distinction, or inference the material supports
    - what mechanism or rule the material explicitly describes
    - what incorrect interpretation a weak student might make
    - what consequence is explicitly supported by the material

    Focus on:
    - the condition
    - the supported mechanism or rule
    - the supported consequence
    - the key reasoning target

    A valid inference may connect multiple retrieved statements, but every
    premise and the resulting relationship must be present in the material.

    If the material is primarily definitional or conceptual, use comparison,
    classification, distinction, exception handling, or synthesis. Do not
    invent a fictional event to make it scenario-based.

    DO NOT generate a question yet.

    Return STRICT JSON:

    {{
        "observable_change": "supported evidence or condition",
        "hidden_cause": "supported rule, mechanism, or relationship",
        "required_inference": "...",
        "common_wrong_interpretation": "...",
        "downstream_effect": "explicitly supported consequence or empty string"
    }}

    SUPPORTING MATERIAL:
    {json.dumps(retrieved_chunks[:15], indent=2)}
    """

        try:
            reasoning_start = time.time()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": reasoning_prompt
                    }
                ],
                temperature=0.3
            )
            
            content = (
                response
                .choices[0]
                .message.content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            print(
                f"🧠 REASONING TIME: {time.time() - reasoning_start:.1f}s"
            )

            data = json.loads(content)

            print("🧠 REASONING CHAIN:", data)

            return data

        except Exception as e:

            print("❌ REASONING CHAIN ERROR:", e)

            return {
                "observable_change": "...",
                "hidden_cause": "...",
                "required_inference": "...",
                "common_wrong_interpretation": "...",
                "downstream_effect": "..."
            }

    async def generate_scenarios(n):
        import random

        scenario_chunks = random.sample(
            retrieved_chunks,
            min(30, len(retrieved_chunks))
        )
        hard_context_instruction = ""

        if (req.difficulty or "").strip().lower() == "hard":
            hard_context_instruction = (
                render_hard_context_specification()
            )

        print(
            "🎲 SCENARIO CHUNKS:",
            len(scenario_chunks)
        )

        scenario_prompt = f"""
        The provided material is the authoritative factual source.

        Create {n} evidence-grounded learning contexts.

        Every factual element in each context MUST be explicitly supported
        by one or more provided chunks.

        You may use only:

        - concepts stated in the material
        - entities stated in the material
        - objects stated in the material
        - events or processes stated in the material
        - relationships stated in the material
        - consequences stated in the material

        Do NOT introduce a new actor, institution, object, event, process,
        condition, relationship, consequence, proper name, location,
        motivation, measurement, or causal link.

        Do NOT make assumptions from general knowledge, even when they seem
        plausible in the academic domain.

        Preserve the source's level of specificity. Do not turn a general
        rule or definition into a detailed fictional case.

        If the source explicitly describes a concrete situation, you may
        restate that situation without adding details.

        If the source is primarily definitional or conceptual, create a
        conceptual context based on:

        - comparison
        - distinction
        - classification
        - an explicitly stated exception
        - synthesis of two or more provided statements

        In that case, do not create a fictional narrative.

        Difficulty must come from reasoning over the supplied evidence, not
        from adding facts.

        {hard_context_instruction}

        Before returning each context, verify that every noun, event,
        relationship, and consequence can be traced to the provided chunks.
        If it cannot, remove it.

        LANGUAGE REQUIREMENT:

            Generate ALL content exclusively in {req.language}.

            This includes:
            - question text
            - answer options
            - explanations
            - feedback

            Do NOT use any other language.
            Do NOT mix languages.

        Return STRICT JSON:

        {{
        "scenarios": [
            {{
            "scenario": "...",
            "source_chunk_ids": ["..."],
            "evidence_facts": ["..."]
            }}
        ]
        }}

        MATERIAL:
        {json.dumps(scenario_chunks, indent=2)}
        
        """

        try:
            scenario_start = time.time()
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": scenario_prompt
                    }
                ],
                temperature=0.7
            )

            content = response.choices[0].message.content.strip()
            print(
                f"🎲 SCENARIO TIME: {time.time() - scenario_start:.1f}s"
            )
            content = (
                content
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )

            data = json.loads(content)

            return data.get("scenarios", [])
            print(
                f"⏱ TOTAL QUIZ TIME: {time.time() - quiz_start:.1f}s"
            )
        except Exception as e:

            print("⚠️ SCENARIO GENERATION ERROR:", e)
            

            return []

    async def generate_missing_explanation(question):
        """Generate only a missing explanation for an already accepted question."""

        def resolve_correct_answer_text():
            answer_key = (
                question.get("correct_answer")
                if question.get("correct_answer") is not None
                else question.get("correct", question.get("answer"))
            )
            options = question.get("options") or []

            try:
                if isinstance(answer_key, int):
                    return options[answer_key]

                if isinstance(answer_key, str):
                    answer_key = answer_key.strip()

                    if answer_key.isdigit():
                        return options[int(answer_key)]

                    if len(answer_key) == 1 and answer_key.upper() in "ABCDE":
                        return options[ord(answer_key.upper()) - ord("A")]

                    return answer_key
            except Exception:
                return str(answer_key or "")

            return str(answer_key or "")

        source_chunk_ids = {
            str(chunk_id)
            for chunk_id in question.get("source_chunk_ids", [])
        }
        explanation_chunks = [
            chunk
            for chunk in retrieved_chunks
            if str(chunk.get("chunk_id")) in source_chunk_ids
        ] or retrieved_chunks[:10]

        explanation_prompt = f"""
        Generate ONLY the missing explanation for an existing multiple-choice
        quiz question.

        Do NOT rewrite the question.
        Do NOT rewrite answer options.
        Do NOT change the correct answer.
        Do NOT add unsupported facts.

        Use the supporting material as the authoritative source.

        LANGUAGE REQUIREMENT:
        Write the explanation exclusively in {req.language}.

        Return STRICT JSON:

        {{
            "explanation": "..."
        }}

        QUESTION:
        {question.get("question", "")}

        OPTIONS:
        {json.dumps(question.get("options", []), ensure_ascii=False)}

        CORRECT ANSWER:
        {resolve_correct_answer_text()}

        SUPPORTING MATERIAL:
        {json.dumps(explanation_chunks, ensure_ascii=False, indent=2)}
        """

        for attempt in range(2):
            try:
                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": explanation_prompt,
                        }
                    ],
                    temperature=0.2,
                )

                content = (
                    response
                    .choices[0]
                    .message.content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                data = json.loads(content)
                explanation = str(data.get("explanation", "")).strip()

                if explanation:
                    print(
                        "✅ GENERATED MISSING EXPLANATION "
                        f"ON ATTEMPT {attempt + 1}"
                    )
                    return explanation

            except Exception as explanation_error:
                print(
                    "⚠️ MISSING EXPLANATION GENERATION FAILED "
                    f"ATTEMPT {attempt + 1}:",
                    repr(explanation_error),
                )

        return ""

    generation_diagnostics = {
        "requested_question_count": req.num_questions,
        "initial_generated": 0,
        "initial_rejected": 0,
        "initial_accepted": 0,
        "refill_attempts": 0,
        "refill_generated": 0,
        "refill_rejected": 0,
        "refill_accepted": 0,
        "final_accepted": 0,
    }

    async def generate_batch(n, style, phase="initial"):
        batch_start = time.time()
        hard_question_contract = ""

        if (req.difficulty or "").strip().lower() == "hard":
            hard_question_contract = (
                render_hard_quiz_specification()
            )
            exam_preferred_formats = render_hard_exam_formats()
        else:
            exam_preferred_formats = """
                - Which conclusion follows from the supplied principles?
                - Which distinction best explains the difference?
                - Which exception applies under the stated source conditions?
                - Which statement is TRUE?
                - Which statement is FALSE?
            """

        print("📦 BATCH START")
        print("📦 BATCH PHASE:", phase)
        if style == "exam":
            scenarios = []
        else:
            scenarios = await generate_scenarios(n)

        print("📊 SCENARIOS GENERATED:", len(scenarios))
        print("🚀 GENERATE_BATCH START")
        print("STYLE:", style)
        print("N:", n)
        
        i = -1

        for i, s in enumerate(scenarios):
            print(
                f"📄 SCENARIO {i+1}:",
                s.get("scenario", "")[:120]
            )

        validated_questions = []

        if style == "exam":
            loop_items = [None]
        else:
            loop_items = scenarios[:n]

        all_validated_questions = []

        for i, scenario_obj in enumerate(loop_items):

            if style != "exam":

                print(
                    f"🎯 PROCESSING SCENARIO {i+1}/{len(scenarios)}"
                )
                print("🔢 N INSIDE LOOP:", n)

                scenario_text = scenario_obj.get(
                    "scenario",
                    ""
                )

                reasoning_chain = await generate_reasoning_chain(
                    scenario_text,
                    retrieved_chunks,
                    client
                )
            # 🎨 QUESTION STYLE

            if style == "exam":

                if (req.difficulty or "").strip().lower() == "hard":
                    style_instruction = """
                Follow the shared HARD COGNITIVE-DIFFICULTY SPECIFICATION.
                Keep the university-exam format concise and non-narrative.
                """
                else:
                    style_instruction = """
                Generate questions that resemble real university exams.

                DO NOT use:
                - Following...
                - During...
                - After...
                - A patient...
                - Scenario-based introductions

                DO NOT create narratives.

                DO NOT create stories.

                Questions should be concise.

                Questions should assess:

                - definitions
                - distinctions
                - rules and exceptions
                - mechanisms
                - direct applications

                The question stem should usually be one sentence.

                Exam style must look clearly different from reasoning style.
                """

            elif style == "reasoning":

                style_instruction = """
                Prioritize evidence-grounded reasoning.

                Use comparisons, distinctions, exceptions, mechanisms,
                relationships, and cause-effect links that are explicitly
                present in the retrieved material.

                Synthesize multiple retrieved statements when useful.

                Require logical inference.

                Require students to connect multiple concepts.

                Avoid pure recall questions whenever possible.

                If the material is definitional or conceptual, create a
                higher-order conceptual question. Do not invent a fictional
                scenario merely to make the question appear difficult.
                """

            

            

            if style == "exam":

                prompt = f"""
                The SUPPORTING MATERIAL is the authoritative factual source.

                Generate EXACTLY {n} university-style multiple choice questions.

                STRICT GROUNDING CONTRACT:

                - Every fact needed to understand or answer a question must
                  be explicitly supported by the supplied chunks.
                - Do not introduce new actors, institutions, objects, events,
                  processes, relationships, conditions, or consequences.
                - Do not use general knowledge to extend the source.
                - A plausible statement is not permitted unless the material
                  supports it.
                - Difficulty must come from comparison, distinction,
                  inference, mechanism analysis, exception handling, or
                  synthesis across supplied chunks.
                - If the material is definitional, ask a higher-order
                  conceptual question instead of inventing a scenario.

                {hard_question_contract}

                IMPORTANT:
                Each question MUST contain EXACTLY 5 answer options.

                Questions with 4 options are invalid.

                Return exactly 5 options for every question.

                Requirements:

                - concise question stems
                - no scenarios
                - no narratives
                - no patient stories
                - no "Following..."
                - no "During..."
                - no "After..."
                - no "When..."
                - no "If..."

                Questions should assess:
                - application of knowledge
                - interpretation of mechanisms
                - cause-effect relationships
                - distinctions and exceptions
                - synthesis across related concepts

                Use a mechanism, relationship, or consequence only when it is
                explicitly described in the material.

                Definitions and technical terms should only be tested when
                educationally relevant.

                Questions should resemble real university exams.

                Wrong answers must be plausible.

                Each question must focus on a different concept.

                Do not generate multiple questions
                about the same concept,
                the same rule,
                the same distinction,
                or the same mechanism.

                Max one question per major concept.

                


                Distractors should:

                - be conceptually related to the correct answer
                - appear realistic to a partially prepared student
                - reflect common misconceptions
                - avoid obviously incorrect answers

                Preferred question formats:

                {exam_preferred_formats}

                

                Difficulty: {req.difficulty}

                Difficulty Rules:

                EASY:
                - direct factual recall
                - definitions
                - terminology
                - single rules
                - single concept questions

                
                MEDIUM:
                MEDIUM:
                - At least 70% of questions must require application of knowledge.
                - No more than 25% of questions may be direct recall questions.
                - Avoid pure definition questions.
                - Require interpretation of mechanisms.
                - Require cause-effect reasoning.
                - If a question can be answered by recalling a single isolated fact, it should usually be classified as EASY.

                - Which statement is TRUE?
                - Which statement is FALSE?
                - Which conclusion follows from these statements?
                - Which distinction best explains...?
                - Which exception applies?

                HARD:
                - Follow the shared HARD COGNITIVE-DIFFICULTY SPECIFICATION
                  above.


                LANGUAGE REQUIREMENT:

                Generate ALL content exclusively in {req.language}.

                This includes:
                - question text
                - answer options
                - explanations
                - feedback

                Do NOT use any other language.
                Do NOT mix languages.

                Return STRICT JSON.

                Generate EXACTLY {n} questions.

                For each question, include "source_chunk_ids" containing the
                chunk IDs that support the stem and correct answer.

                Before returning, verify that "correct" or "correct_answer"
                points to the option justified by the explanation.

                SUPPORTING MATERIAL:
                {json.dumps(retrieved_chunks[:10], indent=2)}
                """

            else:

                prompt = f"""
                The SUPPORTING MATERIAL is the authoritative factual source.
                The proposed scenario and reasoning blueprint are subordinate
                aids and may not add facts.

                STRICT GROUNDING CONTRACT:

                - Every entity, actor, institution, object, event, process,
                  condition, relationship, and consequence in the question
                  must be explicitly supported by the supporting material.
                - Do not use general knowledge to enrich the question.
                - Do not introduce plausible but unstated facts.
                - If any scenario detail is unsupported, omit that detail and
                  build the question directly from the supporting material.
                - Every premise needed for the correct answer must be
                  traceable to the supporting material.
                - Difficulty must come from comparison, distinction,
                  inference, mechanism analysis, exception handling, or
                  synthesis—not from invented context.
                - If the material is primarily definitional or conceptual,
                  generate a higher-order conceptual question without a
                  fictional scenario.

                {hard_question_contract}

                IMPORTANT:
                Each question MUST contain EXACTLY 5 answer options.

                Questions with 4 options are invalid.

                Return exactly 5 options for every question.

                QUESTION STYLE:

                {style_instruction}

                Question Diversity Rules:

                - Each question must test a DIFFERENT inference.
                - Each question must focus on a DIFFERENT supported
                  distinction, mechanism, consequence, exception,
                  or relationship.
                - Do NOT generate multiple questions that can be answered
                using the same reasoning process.
                - Do NOT paraphrase the same question multiple times.
                - If multiple questions are generated, they should explore
                  different aspects of the retrieved evidence.

                You may connect:
                - explicitly stated mechanisms or rules
                - explicitly stated consequences
                - explicitly stated relationships
                - comparisons and distinctions supported by the material
                - statements from multiple supplied chunks

                Do not infer a new factual premise. A conclusion is valid only
                when every premise and relationship required for it appears
                in the supporting material.

                PROPOSED SCENARIO CONTEXT:

                {scenario_text}


                DIAGNOSTIC REASONING BLUEPRINT:

                Supported Evidence or Condition:
                {reasoning_chain.get("observable_change", "")}

                Supported Rule, Mechanism, or Relationship:
                {reasoning_chain.get("hidden_cause", "")}

                Required Inference:
                {reasoning_chain.get("required_inference", "")}

                Common Wrong Interpretation:
                {reasoning_chain.get("common_wrong_interpretation", "")}

                Explicitly Supported Consequence:
                {reasoning_chain.get("downstream_effect", "")}

                IMPORTANT:

                First verify the proposed scenario against the supporting
                material. Retain only supported details.

                If the verified scenario is fully supported and educationally
                useful, it may be used as the question context.

                If the scenario contains unsupported detail, discard that
                detail. If removing it leaves no useful scenario, ask a
                conceptual reasoning question directly from the material.

                Do not force a narrative opening. A precise comparison,
                distinction, exception, or synthesis question is preferable
                to an embellished scenario.

                The question MUST specifically test the REQUIRED INFERENCE
                from the diagnostic reasoning blueprint above.

                The correct answer must follow from the supplied evidence.

                The wrong answers should reflect:
                - common misconceptions
                - superficial interpretations
                - incomplete causal reasoning
                or
                - confusion between related mechanisms.

                evidence
                → comparison or interpretation
                → supported rule or mechanism
                → answer

                NOT:

                source concept
                → invented scenario facts
                → answer

                The question MUST require reasoning about:
                - a comparison
                - a distinction
                - an explicitly stated consequence
                - a supported mechanism or rule
                - an exception
                - a dependency explicitly described by the source
                - synthesis across multiple supplied statements

                Questions solvable through direct factual recall alone
                are LOW QUALITY.

                Avoid:
                - glossary-style questions
                - isolated factual recall
                - single-step recognition questions
                - fictional details absent from the source
                - consequences supported only by general knowledge

                IMPORTANT:
                - Each question MUST focus on a DIFFERENT concept
                - Do NOT repeat the same rule or mechanism
                - Cover different parts of the material
                - Exactly ONE answer must be correct
                - Do NOT use "All of the above"
                - Incorrect options must still be plausible

                Difficulty: {req.difficulty}

                EASY:
                - basic understanding
                - direct supported mechanisms or rules
                - limited reasoning

                MEDIUM:
                - supported cause-effect reasoning
                - comparison or interaction of supplied concepts
                - moderate interpretation

                HARD:
                - Follow the shared HARD COGNITIVE-DIFFICULTY SPECIFICATION
                  above.

                LANGUAGE REQUIREMENT:

                Generate ALL content exclusively in {req.language}.

                This includes:
                - question text
                - answer options
                - explanations
                - feedback

                Do NOT use any other language.
                Do NOT mix languages.
                

                Return STRICT JSON.

                Generate EXACTLY 1 question.

                The questions array MUST contain EXACTLY 1 question object.

                Returning fewer than 1 question is INVALID.

                Before returning, verify that "correct" or "correct_answer"
                points to the option justified by the explanation.

                Example structure:

                {{
                "questions": [
                    {{
                        "question": "...",
                        "options": ["...", "...", "...", "...", "..."],
                        "correct": 0,
                        "topic": "...",
                        "explanation": "Short explanation",
                        "explanation_long": "2-3 sentences maximum",
                        "source_document": "Exact file name",
                        "source_page": "Page number",
                        "source_chunk_ids": ["Supporting chunk ID"]
                    }},
                    {{
                        "question": "...",
                        "options": ["...", "...", "...", "...", "..."],
                        "correct": 1,
                        "topic": "...",
                        "explanation": "Short explanation",
                        "explanation_long": "2-3 sentences maximum",
                        "source_document": "Exact file name",
                        "source_page": "Page number",
                        "source_chunk_ids": ["Supporting chunk ID"]
                    }},
                    {{
                        "question": "...",
                        "options": ["...", "...", "...", "...", "..."],
                        "correct": 2,
                        "topic": "...",
                        "explanation": "Short explanation",
                        "explanation_long": "2-3 sentences maximum",
                        "source_document": "Exact file name",
                        "source_page": "Page number",
                        "source_chunk_ids": ["Supporting chunk ID"]
                    }}
                ]
            }}

            SUPPORTING MATERIAL ONLY:
            {json.dumps(retrieved_chunks[:10], indent=2)}
            """
            print("🎨 STYLE:", style)
            print("📝 PROMPT EXISTS:", "prompt" in locals())
            gpt_start = time.time()

            print("🤖 OPENAI START")
            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o" if req.difficulty == "hard" else "gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.7
            )
            if "response" not in locals():
                print("❌ RESPONSE NEVER CREATED CHECKPOINT 2879")
                return []
            content = response.choices[0].message.content.strip()
            print(
                f"🤖 OPENAI TIME: {time.time() - gpt_start:.1f}s"
            )
            print("RAW RESPONSE:", content[:500])
            #print(
             #   "📊 QUESTIONS FROM THIS SCENARIO:",
            #    len(normalized_questions)
            #)

            try:

                content = content.replace("```json","").replace("```","").strip()

                data = json.loads(content)

                questions = []

                if isinstance(data, list):

                    questions = data

                elif isinstance(data, dict):

                    if "questions" in data:

                        questions = data["questions"]
                        for q in questions:
                            print("🔥 BEFORE NORMALIZATION")
                        
                            print("🎨 STYLE:", style)
                            print(
                                "🔢 OPTIONS COUNT:",
                                len(q.get("options", []))
                            )

                    elif (
                        "question" in data
                        and "options" in data
                    ):

                        questions = [data]

                    # 🔥 GPT ha restituito:
                    # { "question": { ... } }

                    elif (
                        "question" in data
                        and isinstance(data["question"], dict)
                    ):

                        questions = [data["question"]]

                print("📊 GPT GENERATED:", len(questions))
                generated_key = (
                    "refill_generated"
                    if phase == "refill"
                    else "initial_generated"
                )
                generation_diagnostics[generated_key] += len(
                    questions
                )
                
                
                
                print("🚨 ABOUT TO ENTER VALIDATION LOOP")
                if isinstance(data, dict):
                    print("🔑 JSON KEYS:", data.keys())
                else:
                    print("🔑 JSON TYPE:", type(data))
                 
                
                validated_questions = []
                print(
                    f"📊 VALIDATION START: {len(questions)} QUESTIONS"
                )
                if style == "exam":

                    scenario_text = ""

                    reasoning_chain = {
                        "observable_change": "",
                        "hidden_cause": "",
                        "required_inference": "",
                        "common_wrong_interpretation": "",
                        "downstream_effect": ""
                    }

                    loop_items = range(n)

                else:

                    loop_items = scenarios[:n]
                print("🚨 LOOP START")
                
                for q in questions:

                    # Normalize GPT field variants
                    if "question" not in q and "stem" in q:
                        q["question"] = q["stem"]

                    if "question" not in q:
                        print("❌ QUESTION FIELD MISSING:", q)
                        if (
                            (req.difficulty or "").strip().lower()
                            == "hard"
                        ):
                            hard_rejection_reasons.append(
                                "missing_question_field"
                            )
                            hard_rejection_samples.append(
                                build_hard_question_diagnostic_sample(
                                    question=q,
                                    outcome="rejected",
                                    question_type=style,
                                    rejection_reasons=[
                                        "missing_question_field"
                                    ],
                                    topic_category_by_name=(
                                        topic_category_by_name
                                    ),
                                )
                            )
                        rejected_key = (
                            "refill_rejected"
                            if phase == "refill"
                            else "initial_rejected"
                        )
                        generation_diagnostics[rejected_key] += 1
                        continue

                    

                    reasoning_score = score_reasoning_quality(
                        q["question"]
                    )

                    

                    print("🧠 REASONING SCORE:", reasoning_score)

                    
                    
                    is_valid = await validate_question(
                        q,
                        style
                    )

                    if not is_valid:

                        print("❌ QUESTION REJECTED")
                        rejected_key = (
                            "refill_rejected"
                            if phase == "refill"
                            else "initial_rejected"
                        )
                        generation_diagnostics[rejected_key] += 1

                        continue

                    if style != "exam":

                        q["question"] = rewrite_question_opening(
                            q["question"]
                        )

                    hard_question_type_by_object_id[id(q)] = style

                    validated_questions.append(q)
                    print(
                        f"📊 AFTER VALIDATION: {len(validated_questions)}"
                    )                

                seen = set()
                unique_questions = []

                for q in validated_questions:

                    text_q = q.get("question","").strip().lower()

                    if text_q not in seen:
                        seen.add(text_q)
                        unique_questions.append(q)

                                    
                print(
                    f"📊 AFTER DEDUP: {len(unique_questions)}"
                )
                print("📋 UNIQUE QUESTIONS")

                for q_idx, q in enumerate(unique_questions):
                    print(
                        f"{i+1}.",
                        q.get("question", "")[:120]
                    )
                print(
                    f"📦 SCENARIO PRODUCED {len(unique_questions)} QUESTIONS"
                )
                all_validated_questions.extend(unique_questions)

                print(
                    f"✅ SCENARIO {i+1}/{len(loop_items)} COMPLETED | "
                    f"TOTAL COLLECTED: {len(all_validated_questions)}"
                )
           

            except Exception as e:

                print("QUIZ JSON ERROR:", e)
                print("RAW GPT OUTPUT:", content)
                print("📊 AFTER VALIDATION:", len(validated_questions))
                continue
                
        print("LAST SCENARIO INDEX:", i)    
        print("🏁 GENERATE_BATCH END") 
        print("TOTAL SCENARIOS:", len(loop_items))
        print(
            "TOTAL QUESTIONS RETURNED:",
            len(all_validated_questions)
        )

        for idx, q in enumerate(all_validated_questions):
            print(
                f"RETURN {idx+1}:",
                q.get("question", "")[:100]
            )
        return all_validated_questions  
    

    all_questions = []
    tasks = []

    if req.question_style == "balanced":

        exam_count = req.num_questions // 2

        reasoning_count = (
            req.num_questions - exam_count
        )

        tasks.append(
            generate_batch(
                exam_count,
                "exam"
            )
        )

        tasks.append(
            generate_batch(
                reasoning_count,
                "reasoning"
            )
        )

    else:

        batch_size = (
            4 if req.difficulty == "hard"
            else 8
        )

        num_batches = (
            req.num_questions + batch_size - 1
        ) // batch_size

        for i in range(num_batches):

            n = min(
                batch_size,
                req.num_questions - i * batch_size
            )

            tasks.append(
                generate_batch(
                    n,
                    req.question_style
                )
            )

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True,
    )
    print(
        "📊 BEFORE DEDUP:",
        sum(
            len(batch or [])
            for batch in results
            if not isinstance(batch, Exception)
        )
    )
    for idx, batch in enumerate(results):

        print(
            f"📦 RESULT {idx}:",
            type(batch)
        )

        if isinstance(batch, Exception):
            print(
                f"❌ INITIAL BATCH {idx} FAILED:",
                repr(batch)
            )
            continue

        if batch is None:
            print(
                f"❌ RESULT {idx} RETURNED NONE"
            )
            continue

        all_questions.extend(batch)
        print(
            f"📊 AFTER APPEND: {len(all_questions)}"
        )
    

    # REMOVE DUPLICATES

    seen_questions = set()
    unique_questions = []

    for q in all_questions:

        text_q = q.get("question", "").strip().lower()

        key = " ".join(text_q.split()[:8])

        if key not in seen_questions:

            seen_questions.add(key)

            unique_questions.append(q)

    all_questions = unique_questions[:req.num_questions]
    generation_diagnostics["initial_accepted"] = len(
        all_questions
    )
    print("📊 AFTER DEDUP:", len(all_questions))
    print(
        "📊 ACCEPTED AFTER INITIAL GENERATION:",
        generation_diagnostics["initial_accepted"]
    )

    print("📋 UNIQUE QUESTIONS")

    
    # REFILL MISSING QUESTIONS 😄
    retry_count = 0
    max_retries = 10

    while (
        len(all_questions) < req.num_questions
        and retry_count < max_retries
    ):

        retry_count += 1
        generation_diagnostics["refill_attempts"] = retry_count

        missing = req.num_questions - len(all_questions)

        print("🔁 GENERO DOMANDE MANCANTI:", missing)

        if req.question_style == "balanced":

            refill_style = random.choice(
                ["exam", "reasoning"]
            )

        else:

            refill_style = req.question_style

        try:
            extra_batch = await generate_batch(
                missing,
                refill_style,
                phase="refill",
            )
        except Exception as refill_error:
            print(
                "❌ REFILL BATCH FAILED — "
                "CONTINUING WITH REMAINING RETRIES:",
                repr(refill_error)
            )
            continue

        print(
            "🎲 REFILL STYLE:",
            refill_style
        )
        if not extra_batch:
            print(
                "⚠️ ZERO-YIELD REFILL — "
                "CONTINUING WITH REMAINING RETRIES"
            )
            continue

        for q in extra_batch:

            text_q = q.get("question", "").strip().lower()
            key = " ".join(text_q.split()[:8])

            if key not in seen_questions:

                print(
                    "➕ ADDING:",
                    q.get("question", "")[:120]
                )

                seen_questions.add(key)
                all_questions.append(q)
                generation_diagnostics["refill_accepted"] += 1

            else:

                print(
                    "🚫 DUPLICATE:",
                    q.get("question", "")[:120]
                )

            if len(all_questions) >= req.num_questions:
                break

        print(
            "📦 EXTRA BATCH SIZE:",
            len(extra_batch)
        )

        print(
            f"📊 REFILL RESULT: {len(all_questions)}/{req.num_questions}"
        )
        print("\n🧠 FINAL QUESTIONS BEFORE SAVE")
        print("=" * 80)

        for q in all_questions:

            print("QUESTION:")
            print(q.get("question"))

            print("OPTIONS:")
            print(q.get("options"))

            print("CORRECT:")
            print(q.get("correct_answer"))

            print("EXPLANATION:")
            print(q.get("explanation"))

            print("-" * 80)
            print("\n🧠 FINAL QUESTIONS BEFORE SAVE")
            print("=" * 80)

            for q in all_questions:

                print("QUESTION:")
                print(q.get("question"))

                print("OPTIONS:")
                print(q.get("options"))

                print("CORRECT:")
                print(
                    q.get("correct_answer", q.get("correct"))
                )

                print("EXPLANATION:")
                print(q.get("explanation"))

                print("-" * 80)

    generation_diagnostics["final_accepted"] = len(
        all_questions
    )
    generation_diagnostics["total_rejected"] = (
        generation_diagnostics["initial_rejected"]
        + generation_diagnostics["refill_rejected"]
    )

    canonical_project_topics = {
        row[0]
        for row in topic_category_rows
        if row[0]
    }
    for question in all_questions:
        attribution = resolve_quiz_question_topic(
            question,
            chunk_topic_map,
            canonical_project_topics,
        )
        question["topic"] = attribution["topic"]

        if len(set(attribution["resolved_source_topics"])) > 1:
            print(
                "MULTI-TOPIC QUIZ ATTRIBUTION:",
                json.dumps(
                    {
                        "source_chunk_ids": attribution[
                            "source_chunk_ids"
                        ],
                        "resolved_source_topics": attribution[
                            "resolved_source_topics"
                        ],
                        "selected_topic": attribution["topic"],
                        "resolution": attribution["resolution"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            )
        elif attribution["topic"] is None:
            print(
                "UNRESOLVED QUIZ TOPIC ATTRIBUTION:",
                question.get("source_chunk_ids") or [],
            )

    print(
        "📊 QUIZ ASSEMBLY DIAGNOSTICS:",
        json.dumps(
            generation_diagnostics,
            sort_keys=True,
        )
    )

    if (req.difficulty or "").strip().lower() == "hard":
        hard_generation_metrics = build_hard_generation_metrics(
            requested_questions=req.num_questions,
            generated_questions=(
                generation_diagnostics["initial_generated"]
                + generation_diagnostics["refill_generated"]
            ),
            rejected_questions=(
                generation_diagnostics["initial_rejected"]
                + generation_diagnostics["refill_rejected"]
            ),
            rejection_reasons=hard_rejection_reasons,
        )
        hard_generation_metrics.update({
            "project_id": project_id,
            "project_name": project_name,
            "quiz_id": quiz_id,
            "difficulty": (
                req.difficulty or "hard"
            ).strip().lower(),
            "question_style": req.question_style or "balanced",
        })
        accepted_sample_count = min(
            HARD_ACCEPTED_DIAGNOSTIC_SAMPLE_SIZE,
            len(all_questions),
        )
        accepted_questions_sample = (
            random.SystemRandom().sample(
                all_questions,
                accepted_sample_count,
            )
            if accepted_sample_count
            else []
        )
        hard_accepted_samples = [
            build_hard_question_diagnostic_sample(
                question=question,
                outcome="accepted",
                question_type=hard_question_type_by_object_id.get(
                    id(question),
                    req.question_style or "balanced",
                ),
                topic_category_by_name=topic_category_by_name,
            )
            for question in accepted_questions_sample
        ]
        print(
            "HARD GENERATION METRICS:",
            json.dumps(
                hard_generation_metrics,
                sort_keys=True,
            ),
        )
        persist_hard_generation_diagnostics(
            hard_generation_metrics,
            hard_rejection_samples + hard_accepted_samples,
        )

    try:
        validate_quiz_requested_count(
            req.num_questions,
            all_questions,
        )
    except ValueError as incomplete_quiz_error:
        db.close()
        print(
            "❌ QUIZ REQUESTED-COUNT INVARIANT FAILED:",
            str(incomplete_quiz_error)
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to generate the requested number of "
                "quality-approved questions. No partial quiz was saved."
            ),
        )

    db.close()
    db_save = SessionLocal()
    try:
        print("🔥 STO PER FARE INSERT QUIZ")
        print("💾 SAVING")
        print(json.dumps(q, indent=2))
        # ✅ INSERT QUIZ (UNA SOLA VOLTA)
        db_save.execute(
            text("""
                insert into quizzes (id, project_id, user_id, created_at, num_questions, difficulty,question_style)
                values (:id, :project_id, :user_id, now(), :num_questions, :difficulty, :question_style)
            """),
            {
                "id": quiz_id,
                "project_id": project_id,
                "user_id": user_id,
                "num_questions": req.num_questions,
                "difficulty": req.difficulty or "medium",
                "question_style": req.question_style or "balanced"
            }
        )

        print("🔥 INSERT QUIZ ESEGUITO")

        # ✅ ORA SALVI LE DOMANDE (CORRETTO)
        print("🔥 INIZIO SALVATAGGIO DOMANDE")
        # FINAL NORMALIZATION 😄

        for q in all_questions:

            if (
                "correct_answer" not in q
                or q.get("correct_answer") is None
            ):

                if "correct" in q:
                    q["correct_answer"] = q["correct"]

                elif "answer" in q:
                    q["correct_answer"] = q["answer"]

                else:
                    print("❌ MISSING CORRECT ANSWER:", q)
                    raise ValueError(
                        "Accepted quiz question is missing a correct answer"
                    )

            # NORMALIZE correct_answer TO INTEGER INDEX
            if isinstance(q["correct_answer"], str):

                answer_text = q["correct_answer"].strip()

                print("🔍 MATCHING ANSWER TEXT:")
                print(answer_text)

                matched_index = None

                for idx, option in enumerate(q.get("options", [])):

                    clean_option = str(option).strip()

                    print(f"   OPTION {idx}: {clean_option}")

                    if clean_option == answer_text:

                        matched_index = idx

                        print("✅ EXACT MATCH:", idx)

                        break

                    if clean_option.startswith(answer_text):

                        matched_index = idx

                        print("✅ OPTION STARTSWITH ANSWER:", idx)

                        break

                    if answer_text.startswith(clean_option):

                        matched_index = idx

                        print("✅ ANSWER STARTSWITH OPTION:", idx)

                        break

                if matched_index is not None:

                    print("🎯 FINAL MATCHED INDEX:", matched_index)

                    q["correct_answer"] = matched_index

                else:

                    print("❌ NO MATCH FOUND")
                    print("ANSWER:", answer_text)
                    print("OPTIONS:", q.get("options"))

                    q["correct_answer"] = 0

            else:
                q["correct_answer"] = int(q["correct_answer"])
            # 🔥 EXPLANATION FALLBACK

            if not str(q.get("explanation") or "").strip():

                q["explanation"] = (
                    q.get("explanation_long")
                    or q.get("reasoning")
                    or q.get("rationale")
                    or ""
                )

            if not str(q.get("explanation") or "").strip():
                q["explanation"] = await generate_missing_explanation(q)

            if not str(q.get("explanation") or "").strip():

                print("⚠️ MISSING EXPLANATION:")
                print(q.get("question"))
            
            print(
                "📝 EXPLANATION LENGTH:",
                len(q.get("explanation", ""))
            )

        saved_question_count = 0

        for i, q in enumerate(all_questions):
            if not q.get("question"):
                raise ValueError(
                    "Accepted quiz question is missing question text"
                )

            if q.get("correct_answer") is None:
                raise ValueError(
                    "Accepted quiz question is missing correct_answer"
                )

            if not q.get("options"):
                raise ValueError(
                    "Accepted quiz question is missing answer options"
                )
            print("🔑 KEYS:", list(q.keys()))
            print("💾 SAVING QUESTION")
            print("QUESTION:", q["question"])
            print("CORRECT_ANSWER SAVED:", q.get("correct_answer"))
            print("OPTIONS:", q.get("options"))

            print("👉 SALVO DOMANDA:", q["question"])

            result = db_save.execute(
                text("""
                    insert into quiz_questions (
                        quiz_id,
                        question,
                        correct_answer,
                        options,
                        topic,
                        question_order
                    )
                    values (
                        :quiz_id,
                        :question,
                        :correct_answer,
                        :options,
                        :topic,
                        :question_order
                    )
                    returning id
                """),
                {
                    "quiz_id": quiz_id,
                    "question": q["question"],
                    "correct_answer": q["correct_answer"],
                    "options": json.dumps(q["options"]),
                    "topic": q.get("topic"),
                    "question_order": i
                }
            )

            new_id = result.fetchone()[0]
            q["id"] = str(new_id)   # 👈 QUESTO È IL FIX CRITICO
            saved_question_count += 1

        if saved_question_count != req.num_questions:
            raise ValueError(
                "Quiz persistence invariant failed: "
                f"requested {req.num_questions}, "
                f"inserted {saved_question_count}"
            )

        persisted_question_count = db_save.execute(
            text("""
                select count(*)
                from quiz_questions
                where quiz_id = :quiz_id
            """),
            {"quiz_id": quiz_id}
        ).scalar()

        if persisted_question_count != req.num_questions:
            raise ValueError(
                "Quiz persistence verification failed: "
                f"requested {req.num_questions}, "
                f"found {persisted_question_count}"
            )

        db_save.commit()
        print("✅ COMMIT FATTO")
        print("💾 QUIZ STYLE SAVED:", req.question_style)
        

        

    except Exception as e:
        db_save.rollback()
        print("❌ ERRORE SALVATAGGIO QUIZ:", e)
        raise HTTPException(
            status_code=503,
            detail=(
                "Unable to persist the complete quiz. "
                "No partial quiz was saved."
            ),
        )

    finally:
        db_save.close()
    print(
        f"🚀 TOTAL QUIZ TIME: {time.time() - quiz_start:.1f}s"
    )
    print(
        f"📊 FINAL COUNT: {len(all_questions)}/{req.num_questions}"
    )
    print("🚀 RETURNING QUIZ")
    print("QUIZ ID:", quiz_id)
    print("QUESTIONS:", len(all_questions))
    return {
        "quiz_id": quiz_id,
        "questions": all_questions
    }

 # =========================================
# QUIZ STATS
# =========================================    
@app.get("/projects/{project_id}/quiz_stats")
def get_quiz_stats(project_id: str):

    db = SessionLocal()

    try:
        result = db.execute(text("""
            select 
                quiz_id,
                count(*) as attempts,
                max(score) as best_score,
                avg(score) as avg_score,
                (
                    select score 
                    from quiz_attempts qa2
                    where qa2.quiz_id = qa.quiz_id
                    order by created_at desc
                    limit 1
                ) as last_score
            from quiz_attempts qa
            where project_id = :project_id
            group by quiz_id
        """), {"project_id": project_id})

        rows = result.fetchall()

        return [
            {
                "quiz_id": r.quiz_id,
                "attempts": r.attempts,
                "best_score": r.best_score,
                "avg_score": r.avg_score,
                "last_score": r.last_score
            }
            for r in rows
        ]

    finally:
        db.close()




class AskRequest(BaseModel):
    project_id: str
    question: str
    topics: Optional[List[str]] = []
    history: list = []
    expand_search: bool = False

from typing import Optional

#  # =========================================
# # QUIZ STREAM
# # =========================================

# @app.post("/projects/{project_id}/generate_quiz_stream")
# async def generate_quiz_stream(
#     project_id: str,
#     req: QuizRequest,
#     user = Depends(verify_user)
# ):
#     user_id = user["id"]
#     db = SessionLocal()
    
#     # 1. Create the Quiz ID and Database Record FIRST
#     quiz_id = str(uuid.uuid4())
#     try:
#         db.execute(
#             text("""
#                 insert into quizzes (id, project_id, user_id, title, created_at)
#                 values (:id, :project_id, :user_id, :title, now())
#             """),
#             {
#                 "id": quiz_id,
#                 "project_id": project_id,
#                 "user_id": user_id,
#                 "title": f"Quiz on {', '.join(req.topics) if req.topics else 'Study Material'}"
#             }
#         )
#         db.commit()
#     except Exception as e:
#         print(f"Database Error: {e}")
#         db.rollback()
#     finally:
#         db.close()

#     async def quiz_generator():
#         # 2. Yield the ID to the frontend first
#         yield f"ID:{quiz_id}\n"
    

#         db = SessionLocal()

#         remaining = req.num_questions
#         topics = req.topics if hasattr(req, "topics") else []
#         first_batch = True
#         seen_questions = set()

#         # RETRIEVAL UNA SOLA VOLTA
#         emb = client.embeddings.create(
#             model="text-embedding-3-small",
#             input=f"study material concepts {req.language} {req.difficulty}"
#         )

#         query_embedding = emb.data[0].embedding

#         rows = db.execute(
#             text("""
#                 (
#                     select chunk_text, doc_title, page
#                     from chunks
#                     where project_id = :project_id
#                     order by embedding <-> CAST(:embedding AS vector)
#                     limit 40
#                 )

#                 union

#                 (
#                     select chunk_text, doc_title, page
#                     from chunks
#                     where project_id = :project_id
#                     order by random()
#                     limit 40
#                 )
#             """),
#             {
#                 "project_id": project_id,
#                 "embedding": query_embedding
#             }
#         ).fetchall()

#         chunk_topic_map = {r[0]: r[4] for r in rows if r[4]}

#         random.shuffle(rows)

#         db.close()

#         material_blocks = []

#         for r in rows:

#             text_chunk = r[0].lower()
#             chunk_topic = r[3] if (len(r) > 3 and r[3]) else r[1]

#             if topics:
#                 if not any(topic.lower() in text_chunk for topic in topics):
#                     continue   # 🔥 SCARTA CHUNK NON RILEVANTI

#             material_blocks.append(
#                 f"### TOPIC: {chunk_topic}\nFILE: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:500]}"
#             )
#         if len(material_blocks) == 0:
#             print("⚠️ No topic match, fallback to full material")

#             for r in rows:
#                 material_blocks.append(
#                     f"FILE: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:500]}"
#                 )

#         context = "\n\n---\n\n".join(material_blocks)

#         # =========================================
#         # REASONING UNIT EXTRACTION
#         # =========================================

#         reasoning_material = []

#         async def extract_reasoning_units(block):

#             extraction_prompt = f"""
#         You MUST use ONLY the provided material.

#         Your task is NOT to summarize facts.

#         Your task is to extract:
#         - dynamic relationships
#         - causal chains
#         - state transitions
#         - system responses
#         - compensatory mechanisms
#         - interactions between concepts
#         - consequences of changes
#         - comparisons between conditions
#         - applied interpretations

#         Avoid isolated factual statements.

#         GOOD extraction example:
#         "When condition A increases, mechanism B becomes inhibited, causing consequence C."

#         BAD extraction example:
#         "Mechanism B inhibits C."

#         Focus on:
#         - what changes
#         - what triggers something
#         - what happens as a consequence
#         - how systems react
#         - why outcomes differ
#         - what compensates for a disruption

#         The extracted units should naturally support:
#         - scenario-based questions
#         - applied reasoning
#         - inference
#         - interpretation
#         - comparison questions

#         Return STRICT JSON:

#         {{
#         "units": [
#             {{
#             "scenario": "...",
#             "reasoning": "...",
#             "consequence": "..."
#             }}
#         ]
#         }}

#         MATERIAL:
#         {block}
#         """

#             try:

#                 response = await asyncio.to_thread(
#                     client.chat.completions.create,
#                     model="gpt-4o-mini",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": extraction_prompt
#                         }
#                     ],
#                     temperature=0.3
#                 )

#                 content = response.choices[0].message.content.strip()

#                 content = (
#                     content
#                     .replace("```json", "")
#                     .replace("```", "")
#                     .strip()
#                 )

#                 data = json.loads(content)

#                 return data.get("units", [])

#             except Exception as e:

#                 print("⚠️ REASONING EXTRACTION ERROR:", e)

#                 return []
        
#         # EXTRACT REASONING UNITS

#         for block in material_blocks[:25]:

#             units = await extract_reasoning_units(block)

#             reasoning_material.extend(units)

#         print("🧠 TOTAL REASONING UNITS:", len(reasoning_material))

#         # =========================================
#         # DEDUPLICATE REASONING UNITS
#         # =========================================

#         seen_units = set()
#         deduped_reasoning = []

#         for unit in reasoning_material:

#             key = json.dumps(unit, sort_keys=True)

#             if key not in seen_units:

#                 seen_units.add(key)
#                 deduped_reasoning.append(unit)

#         reasoning_material = deduped_reasoning

#         print("✅ DEDUPED REASONING UNITS:", len(reasoning_material))

#         # GENERAZIONE QUIZ
#         batch_size = 4 if req.difficulty == "hard" else 8
#         num_batches = (req.num_questions + batch_size - 1) // batch_size

        
            

       

#         def fails_basic_reasoning_check(question_text):

#             normalized = (
#                 question_text
#                 .strip()
#                 .lower()
#             )
#             print("🔍 CHECKING:", normalized)
#             for pattern in LOW_QUALITY_PATTERNS:

#                 if normalized.startswith(pattern):

#                     print("🚫 MATCHED:", pattern)

#                     return True
                
#                 # NEW RULE 😄
#                 if "?" in normalized:

#                     first_part = normalized.split("?")[0]

#                     if len(first_part.split()) < 12:

#                         return True

#             return False
        
#         def score_reasoning_quality(question_text):

#             normalized = question_text.lower()

#             score = 0

#             scenario_indicators = [
#                 "during",
#                 "after",
#                 "when",
#                 "following",
#                 "in response",
#                 "as a result",
#                 "because",
#                 "while",
#                 "despite",
#                 "under",
#                 "increased",
#                 "decreased",
#                 "reduced",
#                 "elevated",
#                 "shift",
#                 "adaptation",
#                 "response"
#             ]

#             generic_patterns = [
#                 "what is",
#                 "which enzyme",
#                 "which molecule",
#                 "what role",
#                 "what is the main",
#                 "what is the primary",
#                 "which process",
#                 "which pathway",
#                 "what function"
#             ]

#             causal_words = [
#                 "because",
#                 "therefore",
#                 "leading to",
#                 "resulting in",
#                 "causing",
#                 "due to",
#                 "consequence"
#             ]

#             if any(x in normalized for x in scenario_indicators):
#                 score += 3

#             if any(x in normalized for x in causal_words):
#                 score += 2

#             if len(question_text.split()) > 18:
#                 score += 1

#             if any(x in normalized for x in generic_patterns):
#                 score -= 5

#             return score

#         async def validate_question(question):

#             print("✅ VALIDATOR CALLED")

#             question_text = question.get("question", "")

#             normalized = question_text.lower()

#             heuristic_score = score_reasoning_quality(
#                 question_text
#             )

#             print("📊 HEURISTIC SCORE:", heuristic_score)

#             print("🔍 VALIDATING:", question_text)

#             scenario_indicators = [
#                 "during",
#                 "after",
#                 "when",
#                 "following",
#                 "in response",
#                 "as a result",
#                 "because",
#                 "while",
#                 "despite",
#                 "under",
#                 "a patient",
#                 "a system",
#                 "an observed",
#                 "increased",
#                 "decreased",
#                 "reduced",
#                 "elevated"
#             ]

#             if not any(x in normalized for x in scenario_indicators):

#                 print("🚫 NO SCENARIO STRUCTURE")

#                 return False

#             if heuristic_score <= -2:

#                 print("🚫 HEURISTIC REJECTION")

#                 return False

#             if fails_basic_reasoning_check(question_text):

#                 print("🚫 BASIC FILTER REJECTED")

#                 return False

#             return True

#         def rewrite_question_opening(question_text):

#             replacements = [
#                 ("What is the role of", "During a system response,"),
#                 ("What is the primary role of", "In the context of a changing system,"),
#                 ("What is the main role of", "During a regulatory process,"),
#                 ("What is the function of", "Within an adaptive process,"),
#                 ("What is the primary function of", "Within a dynamic system,"),
#                 ("What is the significance of", "In a situation where"),
#                 ("What is the primary consequence of", "A prolonged alteration results in"),
#                 ("What is the main consequence of", "A system disruption leads to"),
#                 ("What happens to", "During a changing condition,"),
#                 ("What occurs", "During this transition,"),

#                 ("What is", "In this situation,"),
#                 ("Which of the following", "Among the following scenarios,"),
#                 ("Which process", "During a changing situation,"),
#                 ("Which pathway", "Within this sequence of events,"),
#                 ("Which mechanism", "The explanation that BEST fits this situation"),
#                 ("Which component", "An important element in this context"),
#                 ("Which factor", "A key factor in this situation"),
#                 ("Which condition", "Under these circumstances"),
#                 ("Which statement", "The most accurate interpretation"),
#                 ("Which enzyme", "A required element in this process"),
#                 ("Which molecule", "An important component in this situation"),
#                 ("Which amino acid", "In this context,"),
#                 ("Which metabolic process", "During this system change,"),
#             ]

#             for old, new in replacements:

#                 if question_text.startswith(old):

#                     question_text = question_text.replace(old, new, 1)

#                     break

#             return question_text

#         async def generate_reasoning_chain(
#             scenario_text,
#             reasoning_material,
#             client
#         ):

#             reasoning_prompt = f"""
#         You MUST analyze the following biological/metabolic scenario.

#         SCENARIO:
#         {scenario_text}

#         Using ONLY the supporting material below,
#         extract the DIAGNOSTIC REASONING BLUEPRINT
#         behind the scenario.

#         Focus on:
#         - what is OBSERVED
#         - what must be INFERRED
#         - what hidden mechanism explains the situation
#         - what incorrect interpretation a weak student might make
#         - what downstream consequence follows logically

#         Focus on:
#         - the condition
#         - the hidden mechanism
#         - the downstream consequence
#         - the key reasoning target

#         DO NOT generate a question yet.

#         Return STRICT JSON:

#         {{
#             "observable_change": "...",
#             "hidden_cause": "...",
#             "required_inference": "...",
#             "common_wrong_interpretation": "...",
#             "downstream_effect": "..."
#         }}

#         SUPPORTING MATERIAL:
#         {json.dumps(reasoning_material[:40], indent=2)}
#         """

#             try:

#                 response = await asyncio.to_thread(
#                     client.chat.completions.create,
#                     model="gpt-4o-mini",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": reasoning_prompt
#                         }
#                     ],
#                     temperature=0.3
#                 )

#                 content = (
#                     response
#                     .choices[0]
#                     .message.content
#                     .replace("```json", "")
#                     .replace("```", "")
#                     .strip()
#                 )

#                 data = json.loads(content)

#                 print("🧠 REASONING CHAIN:", data)

#                 return data

#             except Exception as e:

#                 print("❌ REASONING CHAIN ERROR:", e)

#                 return {
#                     "observable_change": "...",
#                     "hidden_cause": "...",
#                     "required_inference": "...",
#                     "common_wrong_interpretation": "...",
#                     "downstream_effect": "..."
#                 }
        
#         async def generate_scenarios(n):

#             scenario_prompt = f"""
#         You MUST use ONLY the provided material.

#         Create {n} short applied learning scenarios.

#         The scenarios must:
#         - avoid directly naming the target concept when possible
#         - prefer contextual clues over explicit naming
#         - focus on interpretation instead of direct recall
#         - require understanding of interactions, causes, consequences, or comparisons
#         - NOT directly reveal the answer
#         - NOT ask questions yet
#         - NOT contain multiple choice options
#         - NOT behave like flashcards
#         - NOT define concepts directly
#         The scenario MUST describe a CONCRETE observable biological situation.

#         The scenario should describe:
#         - measurable changes
#         - pathway activation or inhibition
#         - metabolite accumulation or depletion
#         - transport increases or decreases
#         - hormonal shifts
#         - energy source transitions
#         - physiological adaptation
#         - downstream consequences

#         The scenario MUST include:
#         - a condition
#         - at least TWO changing variables
#         - at least ONE consequence or adaptation

#         The scenario should feel like a real metabolic or physiological event,
#         NOT an abstract conceptual description.

#         GOOD scenario:
#         "During prolonged fasting, liver malonyl-CoA levels decrease while mitochondrial fatty acid transport increases and ketone production rises."

#         GOOD scenario:
#         "During intense physical activity, lactate production increases in skeletal muscle while hepatic glucose regeneration becomes more active."

#         GOOD scenario:
#         "Reduced insulin signaling decreases glucose uptake in peripheral tissues while fatty acid utilization progressively increases."

#         GOOD scenario:
#         "A reduction in one metabolic intermediate relieves inhibition of fatty acid transport into mitochondria, increasing beta-oxidation."

#         BAD scenario:
#         "A compensatory response stabilizes the system."

#         BAD scenario:
#         "A pathway becomes activated."

#         BAD scenario:
#         "A process becomes less efficient."

#         BAD scenario:
#         "A regulatory mechanism changes."

#         Return STRICT JSON:

#         {{
#         "scenarios": [
#             {{
#             "scenario": "..."
#             }}
#         ]
#         }}

#         MATERIAL:
#         {json.dumps(retrieved_chunks[:40], indent=2)}
#         """

#             try:

#                 response = await asyncio.to_thread(
#                     client.chat.completions.create,
#                     model="gpt-4o-mini",
#                     messages=[
#                         {
#                             "role": "user",
#                             "content": scenario_prompt
#                         }
#                     ],
#                     temperature=0.7
#                 )

#                 content = response.choices[0].message.content.strip()

#                 content = (
#                     content
#                     .replace("```json", "")
#                     .replace("```", "")
#                     .strip()
#                 )

#                 data = json.loads(content)

#                 return data.get("scenarios", [])

#             except Exception as e:

#                 print("⚠️ SCENARIO GENERATION ERROR:", e)

#                 return []

#         async def generate_batch(n):
#             scenarios = await generate_scenarios(n)

#             reasoning_chain = await generate_reasoning_chain(
#                 scenario_text,
#                 reasoning_material,
#                 client
#             )

#             validated_questions = []

#             for scenario_obj in scenarios[:n]:

#                 scenario_text = scenario_obj.get("scenario", "")
#                 prompt = f"""
#                 You MUST use ONLY the material provided below.

#                 Use the provided material as the ONLY factual source.

#                 You may infer:
#                 - mechanisms
#                 - consequences
#                 - relationships
#                 - adaptations
#                 - compensations
#                 - downstream effects

#                 ONLY if they are logically supported by the material.

#                 PRIMARY SCENARIO CONTEXT:

#                 {scenario_text}
        

#                 DIAGNOSTIC REASONING BLUEPRINT:

#                 Observable Change:
#                 {reasoning_chain.get("observable_change", "")}

#                 Hidden Cause:
#                 {reasoning_chain.get("hidden_cause", "")}

#                 Required Inference:
#                 {reasoning_chain.get("required_inference", "")}

#                 Common Wrong Interpretation:
#                 {reasoning_chain.get("common_wrong_interpretation", "")}

#                 Downstream Effect:
#                 {reasoning_chain.get("downstream_effect", "")}

#                 IMPORTANT:

#                 The ENTIRE question MUST be built around this scenario.

#                 The scenario is the PRIMARY source for the question.

#                 The provided material is ONLY supporting context.

#                 If the scenario is removed,
#                 the question MUST become impossible to answer correctly.

#                 The question MUST directly depend on:
#                 - the observed changes
#                 - the described condition
#                 - the adaptation
#                 - the consequence
#                 - the comparison
                

#                 contained in the scenario.

#                 DO NOT generate generic textbook questions.

#                 DO NOT ask about isolated definitions,
#                 rules,
#                 pathways,
#                 or static facts
#                 unless they are REQUIRED to interpret the scenario itself.

#                 Transform the scenario into an applied reasoning question.

#                 The FIRST sentence of the question MUST describe:
#                 - a condition
#                 - a system change
#                 - a dysfunction
#                 - a compensation
#                 - an adaptation
#                 - a metabolic transition
#                 or
#                 - an observed consequence

#                 derived directly from the scenario.

#                 The scenario itself MUST become the question stem.

#                 ONLY AFTER describing the situation,
#                 ask the reasoning question.

#                 The question MUST NOT start with:
#                 - What
#                 - Which
#                 - During which
#                 - In which
#                 - How does

#                 The question MUST start with a concrete situation, for example:
#                 - During...
#                 - When...
#                 - After...
#                 - If...
#                 - Following...
#                 - A system...
#                 - An observed change...
#                 - A prolonged condition...

#                 If the question starts with What, Which, During which, In which, or How does, it is INVALID.

#                 The student should need to:
#                 The question MUST specifically test the REQUIRED INFERENCE
#                 from the diagnostic reasoning blueprint above.

#                 The correct answer should ONLY become obvious
#                 after interpreting the observable change correctly.

#                 The wrong answers should reflect:
#                 - common misconceptions
#                 - superficial interpretations
#                 - incomplete causal reasoning
#                 or
#                 - confusion between related mechanisms.

#                 situation
#                 → interpretation
#                 → mechanism
#                 → consequence
#                 → answer

#                 NOT:

#                 keyword
#                 → memorized fact
#                 → answer

#                 The question MUST require reasoning about:
#                 - a change
#                 - a consequence
#                 - a comparison
#                 - a system response
#                 - a failure
#                 - a dependency
#                 or
#                 - a downstream effect

#                 Questions solvable through direct factual recall alone
#                 are LOW QUALITY.

#                 Avoid:
#                 - direct definitions
#                 - glossary-style questions
#                 - isolated factual recall
#                 - textbook-style prompts
#                 - single-step recognition questions

#                 BAD:
#                 "What is the role of glucagon?"

#                 BAD:
#                 "Which pathway produces NADPH?"

#                 GOOD:
#                 "During prolonged fasting, glucose utilization decreases in peripheral tissues while fatty acid oxidation increases.

#                 Which downstream metabolic adaptation would MOST likely occur?"

#                 GOOD:
#                 "A compensatory metabolic response temporarily stabilizes energy production after a regulatory pathway becomes less effective.

#                 Which mechanism BEST explains the observed adaptation?"

#                 IMPORTANT:
#                 - Each question MUST focus on a DIFFERENT concept
#                 - Do NOT repeat mechanisms or pathways
#                 - Cover different parts of the material
#                 - Exactly ONE answer must be correct
#                 - Do NOT use "All of the above"
#                 - Incorrect options must still be plausible

#                 Difficulty: {req.difficulty}

#                 EASY:
#                 - basic understanding
#                 - direct mechanisms
#                 - limited reasoning

#                 MEDIUM:
#                 - cause-effect reasoning
#                 - interactions between processes
#                 - moderate interpretation

#                 HARD:
#                 - applied reasoning
#                 - hidden mechanisms
#                 - interpretation of consequences
#                 - at least TWO reasoning steps
#                 - scenario-driven reasoning

#                 Language: {req.language}

#                 Return STRICT JSON:

#                 {{
#                 "questions": [
#                     {{
#                     "question": "...",
#                     "options": ["...", "...", "...", "...", "..."],
#                     "correct": 0,
#                     "topic": "...",
#                     "explanation": "Short explanation",
#                     "explanation_long": "2-3 sentences maximum",
#                     "source_document": "Exact file name",
#                     "source_page": "Page number"
#                     }}
#                 ]
#                 }}

#                 SUPPORTING MATERIAL ONLY:
#                 {reasoning_material_text}
#                 """

#             response = await asyncio.to_thread(
#                 client.chat.completions.create,
#                 model="gpt-4o-mini",
#                 messages=[{"role":"user","content":prompt}],
#                 temperature=0.7
#             )

#             content = response.choices[0].message.content.strip()

#             print("RAW RESPONSE:", content[:500])
            

#             try:

#                 content = content.replace("```json","").replace("```","").strip()

#                 data = json.loads(content)

#                 questions = data.get("questions", [])
#                 print("🚨 QUESTIONS PARSED:", len(questions))
#                 print("🚨 ABOUT TO ENTER VALIDATION LOOP")
#                 for q in questions:

#                     if "correct" in q and "correct_answer" not in q:
#                         q["correct_answer"] = q["correct"]

#                     if "correct_answer" in q:
#                         q["correct_answer"] = int(q["correct_answer"])
                
#                 validated_questions = []
#                 print("🚨 LOOP START")
#                 for q in questions:
#                     reasoning_score = score_reasoning_quality(
#                         q["question"]
#                     )

#                     print("🧠 QUESTION:")
#                     print(q["question"])

#                     print("🧠 REASONING SCORE:", reasoning_score)

                    
                    
#                     is_valid = await validate_question(q)

#                     if not is_valid:

#                         print("❌ QUESTION REJECTED")

#                         continue

#                     q["question"] = rewrite_question_opening(
#                         q["question"]
#                     )

#                     validated_questions.append(q)

                

#                 seen = set()
#                 unique_questions = []

#                 for q in validated_questions:

#                     text_q = q.get("question","").strip().lower()

#                     if text_q not in seen:
#                         seen.add(text_q)
#                         unique_questions.append(q)

#                 for q in validated_questions:

#                     if "correct_answer" not in q:

#                         if "correct" in q:
#                             q["correct_answer"] = int(q["correct"])

#                         else:
#                             q["correct_answer"] = 0 

#                 return unique_questions

#             except Exception as e:

#                 print("QUIZ JSON ERROR:", e)
#                 print("RAW GPT OUTPUT:", content)

#                 return []


#         tasks = []

#         for i in range(num_batches):

#             n = min(batch_size, req.num_questions - i * batch_size)

#             tasks.append(generate_batch(n))


#         results = await asyncio.gather(*tasks)

#         questions = []

#         for batch in results:
#             questions.extend(batch)

#         # REMOVE GLOBAL DUPLICATES

#         # --- 1. FILTRO DUPLICATI E PULIZIA ---
#         seen_questions = set()
#         unique_questions = []

#         for q in questions:
#             text_q = q.get("question", "").strip().lower()
#             key = " ".join(text_q.split()[:8])
#             if key not in seen_questions:
#                 seen_questions.add(key)
#                 unique_questions.append(q)

#         # --- 2. GENERAZIONE DOMANDE MANCANTI ---
#         if len(unique_questions) < req.num_questions:
#             missing = req.num_questions - len(unique_questions)
#             extra = await generate_batch(missing)
#             for q in extra:
#                 text_q = q.get("question", "").strip().lower()
#                 key = " ".join(text_q.split()[:8])
#                 if key not in seen_questions:
#                     seen_questions.add(key)
#                     unique_questions.append(q)
#                 if len(unique_questions) >= req.num_questions:
#                     break

#         # --- 3. SALVATAGGIO NEL DB E INVIO STREAM ---
#         for i, q in enumerate(unique_questions[:req.num_questions]):

#             correct_val = q.get("correct", q.get("correct_answer", 0))
#             q["correct"] = int(correct_val)

#             # SHUFFLE OPTIONS TO AVOID POSITION BIAS

#             try:

#                 # REMOVE DUPLICATE OPTIONS

#                 q["options"] = list(dict.fromkeys(q["options"]))

#                 # INVALID OPTION COUNT

#                 if len(q["options"]) < 4:
#                     continue

#                 # INVALID CORRECT INDEX

#                 if q["correct"] >= len(q["options"]):
#                     continue

#                 correct_text = q["options"][q["correct"]]

#                 random.shuffle(q["options"])

#                 q["correct"] = q["options"].index(correct_text)

#             except Exception as e:

#                 print("⚠️ Shuffle error:", e)

#                 continue

#             topic_val = q.get("topic")

#             if not topic_val or topic_val == "...":
#                 topic_val = topics[0] if topics else "General"

#             q["topic"] = topic_val

#             db_save = SessionLocal()
#             try:
#                 db_save.execute(
#                     text("""
#                         insert into quiz_questions 
#                         (quiz_id, question_order, question, options, correct, explanation, explanation_long, source_document, source_page, topic)
#                         values 
#                         (:quiz_id, :order, :question, :options, :correct, :explanation, :explanation_long, :doc, :page, :topic)
#                     """),
#                     {
#                         "quiz_id": quiz_id,
#                         "order": i,
#                         "question": q.get("question"),
#                         "options": json.dumps(q.get("options")),
#                         "correct": q["correct"],
#                         "explanation": q.get("explanation"),
#                         "explanation_long": q.get("explanation_long"),
#                         "doc": q.get("source_document"),
#                         "page": q.get("source_page"),
#                         "topic": q["topic"]
#                     }
#                 )
#                 db_save.commit()
#             except Exception as e:
#                 print(f"❌ Errore salvataggio domanda {i}: {e}")
#                 db_save.rollback()
#             finally:
#                 db_save.close()

#             yield json.dumps(q) + "\n"

#     # <-- QUESTO RETURN CHIUDE IL quiz_generator (4 spazi di rientro)
#     # <-- QUESTO RETURN CHIUDE LA FUNZIONE PRINCIPALE (allineato a 'async def')
#     return StreamingResponse(quiz_generator(), media_type="text/event-stream")

    
@app.post("/projects/{project_id}/generate_flashcards")
async def generate_flashcards(
    project_id: str,
    req: dict = Body(None),
    user = Depends(verify_user)
):
    gpt_text = "IA non ancora consultata" 
    db = SessionLocal()
    flashcards = [] # FONDAMENTALE: inizializza la lista qui
    
    try:
        num_cards = 10

        topics_list = []
        topic_ids = []

        if req and isinstance(req, dict):

            num_cards = req.get("num_cards", 10)

            topics_list = req.get("topics", [])
            topic_ids = req.get("topic_ids", [])

            if isinstance(topics_list, str):
                topics_list = [topics_list]

        print(f"🔥 START FLASHCARDS: Project {project_id} | Topics: {topics_list}")

        # =====================================
        # NEW LEARNING GRAPH RETRIEVAL
        # =====================================
        retrieval_start = time.time()
        scope = resolve_learning_scope(
            project_id=project_id,
            topic_ids=req.get("topic_ids", []),
            limit=30
        )

        print("🧠 FLASHCARD LEARNING SCOPE:", scope["scope_type"])

        context_chunks = scope["chunks"]

        print("📦 FLASHCARD CHUNKS:", len(context_chunks))

        if not context_chunks:
            print("❌ NESSUN TESTO TROVATO NEL PROGETTO")
            return {"flashcards": []}

        # 2. PREPARAZIONE CONTESTO E MAPPING
        context_blocks = []
        chunk_topic_map = {} 
        for i, c in enumerate(context_chunks):
            cid = f"CH-{i}"
            topic_name = c.get("topic") or "General"
            chunk_topic_map[cid] = topic_name
            # Usiamo 'text' perché search_project_chunks restituisce dizionari con 'text'
            context_blocks.append(
                f"### CHUNK_ID: {cid} | TOPIC: {topic_name}\n{c.get('chunk_text', '')}"
            )

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
        Generate EXACTLY {num_cards} academic flashcards in JSON format.
        
        STRICT RULES:
        1. Use ONLY the provided material.
        2. Assign the correct 'chunk_id' to each card (e.g., "CH-0").
        3. Aim for high-yield medical questions (Why, How, Mechanisms).

        Material:
        {context_text}

        Expected JSON Structure:
        {{
          "flashcards": [
            {{
              "question": "...",
              "answer": "...",
              "chunk_id": "CH-X",
              "difficulty": "medium"
            }}
          ]
        }}
        """

        # 3. CHIAMATA A GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a professional medical tutor. You MUST always respond in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        gpt_text = response.choices[0].message.content
        
        # 4. PARSING E FILTRO
        data = json.loads(gpt_text)
        raw_cards = data.get("flashcards", [])
        
        seen = set()
        for card in raw_cards:
            q = card.get("question", "").strip().lower()
            cid = str(card.get("chunk_id", "")).strip()
            
            # Verifichiamo che il chunk_id esista nella nostra mappa
            if q and cid in chunk_topic_map and q not in seen:
                card["topic"] = chunk_topic_map[cid]
                seen.add(q)
                flashcards.append(card)

        # Limitiamo al numero richiesto
        flashcards = flashcards[:num_cards]

        # 5. SALVATAGGIO NEL DATABASE
        for card in flashcards:
            # Usiamo text() per coerenza con il tuo stile
            result = db.execute(
                text("""
                    INSERT INTO flashcards (project_id, user_id, question, answer, topic)
                    VALUES (:project_id, :user_id, :question, :answer, :topic)
                    RETURNING id
                """),
                {
                    "project_id": project_id,
                    "user_id": user["id"],
                    "question": card.get("question"),
                    "answer": card.get("answer"),
                    "topic": card.get("topic")
                }
            )
            card["id"] = result.fetchone()[0]

        db.commit()
        return {"flashcards": flashcards}

    except Exception as e:
        if db: db.rollback()
        print(f"❌ ERRORE GENERAZIONE: {str(e)}")
        print(f"📝 RAW GPT OUTPUT: {gpt_text}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
    
def search_project_chunks(
    project_id: str,
    query: str = None,
    topics: list[str] = None,
    k: int = 20
):

    db = SessionLocal()

    # ======================
    # BUILD QUERY INTELLIGENTE
    # ======================

    if query and topics:
        full_query = f"{query} {' '.join(topics)}"
    elif topics:
        full_query = " ".join(topics)
    else:
        full_query = query or "important study concepts"

    # DEBUG
    print("🔍 RETRIEVAL QUERY:", full_query)
    def normalize_topic(t):
        return " ".join(str(t).split()).strip()

    

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=full_query
    )

    query_embedding = emb.data[0].embedding

    # ======================
    # VECTOR SEARCH + RANDOM MIX
    # ======================

    if topics:
        rows = []

        for topic in topics:
            topic_norm = normalize_string(topic)
            topic_keyword = topic_norm.split()[0] if topic_norm else ""

            topic_rows = db.execute(
                text("""
                    select chunk_text, doc_title, page, topic
                    from chunks
                    where project_id = :project_id
                    and (
                        lower(topic) like :topic_keyword
                        OR lower(chunk_text) like :topic_keyword
                    )
                    order by embedding <-> CAST(:embedding AS vector)
                    limit :k_per_topic
                """),
                {
                    "project_id": project_id,
                    "topic_keyword": f"%{topic_keyword.lower()}%",
                    "embedding": query_embedding,
                    "k_per_topic": max(3, k // max(len(topics), 1))
                }
            ).fetchall()

            print(f"📚 SEARCH_PROJECT_CHUNKS topic '{topic}' (keyword '{topic_keyword}') -> {len(topic_rows)} rows")

            rows.extend(topic_rows)
            unique = {}
            for r in rows:
                unique[r[0]] = r  # r[0] = id

            rows = list(unique.values())
            rows = rows[:15]
    else:
        rows = db.execute(
            text("""
                select chunk_text, doc_title, page, topic
                from chunks
                where project_id = :project_id
                order by embedding <-> CAST(:embedding AS vector)
                limit :k
            """),
            {
                "project_id": project_id,
                "k": k,
                "embedding": query_embedding
            }
        ).fetchall()
       
    

    db.close()

    # ======================
    # COSTRUZIONE CHUNKS
    # ======================

    chunks = []

    def normalize(s):
        return " ".join(str(s).lower().split())

    chunks = []

    for r in rows:

        text_chunk = r[0]
        chunk_topic = r[3]

        if (
            not chunk_topic
            or str(chunk_topic).lower().endswith(".pdf")
        ):
            chunk_topic = (
                normalize_string(r[1])
                .replace(".pdf", "")
                .replace("_", " ")
            )

        

        chunks.append({
            "text": text_chunk,
            "document": r[1],
            "page": r[2],
            "topic": chunk_topic
        })

    print("📦 CHUNKS RETRIEVED:", len(chunks))
    
    

    return chunks[:k]
from sqlalchemy import text as sql_text

@app.get("/projects/{project_id}/topics")
async def get_topics(project_id: str):
    db = SessionLocal()
    try:
        # We now select category, topic, and description
        result = db.execute(
            sql_text("""
                SELECT id, category, topic, description  
                FROM topics 
                WHERE project_id = :project_id
                AND topic IS NOT NULL
                AND is_display_topic = true
                ORDER BY category ASC, topic ASC
            """), 
            {"project_id": project_id}
        )
        rows = result.fetchall()
        
        # We format them into the structured object your UI needs
        return {
            "topics": [
                {
                    "id": str(r[0]),
                    "category": r[1] or "General",
                    "topic": r[2],
                    "description": r[3] or "",
                    "difficulty": "medium",
                    "accuracy": 50
                }
                for r in rows
            ]
        }
    finally:
        db.close()

@app.get("/projects/{project_id}/topic_status")

def get_topic_status(
    project_id: str,
    user = Depends(verify_user)
):
    from sqlalchemy.orm import sessionmaker

    FreshSession = sessionmaker(bind=engine)

    db = FreshSession()
    db.expire_all()
    print("🧠 TOPIC STATUS REQUEST FOR:", project_id)
    try:
        print("🔄 FORCING FRESH STATUS READ")
        row = db.execute(
            text("""
                select topic_status, last_processed_page
                from projects
                where id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()
        print("📦 RAW DB ROW:", row)
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        print("🔥 RAW STATUS FROM DB:", row[0])

        return {
            "status": row[0] or "idle",
            "last_processed_page": row[1] or 0
        }

    finally:
        db.close()

      
       
        


@app.get("/projects/{project_id}/quizzes")
async def list_project_quizzes(
    project_id: str,
    user = Depends(verify_user)
):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select id, created_at, num_questions, difficulty
            from quizzes
            where project_id = :project_id
            order by created_at desc
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    quizzes = []

    for r in rows:
        quizzes.append({
            "id": r[0],
            "created_at": str(r[1]),
            "num_questions": r[2],
            "difficulty": r[3]
        })

    return {"quizzes": quizzes}
@app.get("/projects/{project_id}/flashcards")
async def get_flashcards(project_id: str):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select id, question, answer, topic
            from flashcards
            where project_id = :project_id
            order by random()
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    flashcards = []

    for r in rows:
        flashcards.append({
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "topic": r[3]
        })

    return {"flashcards": flashcards}
# ======================
# PROJECT SUMMARY
# ======================

@app.get("/projects/{project_id}/summary")
async def project_summary(project_id: str, user = Depends(verify_user)):
    user_id = user["id"]
    db = SessionLocal()
    try:
        # 1. Statistiche generali Quiz
        quiz_stats = db.execute(
            text("""
                SELECT 
                    COUNT(qa.id), 
                    AVG((qa.score::float / NULLIF(qa.total_questions, 0)) * 100)
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                WHERE qa.user_id = :u_id AND q.project_id = :p_id
                AND qa.total_questions > 0
            """),
            {"u_id": user_id, "p_id": project_id}
        ).fetchone()

        # 2. Storico Quiz (history_rows)
        history_rows = db.execute(
            text("""
                SELECT q.title, qa.score, qa.total_questions, qa.completed_at
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                WHERE qa.user_id = :u_id AND q.project_id = :p_id
                ORDER BY qa.completed_at DESC
            """),
            {"u_id": user_id, "p_id": project_id}
        ).fetchall()

        quiz_history = [
            {"title": r[0], "score": r[1], "total": r[2], "date": r[3].isoformat() if r[3] else None} 
            for r in history_rows
        ]

        # 3. Dettaglio Topic Ibrido (topic_rows)
        topic_rows = db.execute(
            text("""
                WITH CombinedScores AS (
                    SELECT 
                        qq.topic, 
                        (qa.score::float / NULLIF(qa.total_questions, 0)) * 100 as score
                    FROM quiz_questions qq
                    JOIN quizzes q ON qq.quiz_id = q.id
                    JOIN quiz_attempts qa ON qa.quiz_id = q.id
                    WHERE q.project_id = :p_id AND qa.user_id = :u_id
                    
                    UNION ALL
                    
                    SELECT 
                        f.topic,
                        CASE WHEN fr.is_correct THEN 100 ELSE 0 END as score
                    FROM flashcards f
                    JOIN flashcard_reviews fr ON f.id = fr.flashcard_id
                    WHERE f.project_id = :p_id AND f.user_id = :u_id
                )
                SELECT topic, AVG(score) FROM CombinedScores GROUP BY topic
            """),
            {"p_id": project_id, "u_id": user_id}
        ).fetchall()

        topics_detail = [
            {"topic": r[0] or "General", "score": round(float(r[1]), 1) if r[1] is not None else 0} 
            for r in topic_rows
        ]

        # Conteggio flashcard riviste per il box (opzionale)
        f_count = db.execute(
            text("SELECT COUNT(DISTINCT flashcard_id) FROM flashcard_reviews fr JOIN flashcards f ON fr.flashcard_id = f.id WHERE f.project_id = :p_id AND f.user_id = :u_id"),
            {"p_id": project_id, "u_id": user_id}
        ).scalar() or 0

        # IL RETURN DEVE CONTENERE TUTTE LE CHIAVI
        return {
            "quiz_attempts": int(quiz_stats[0]) if quiz_stats[0] else 0,
            "messaggio_segreto": "Sto leggendo questo file!",
            "avg_score": round(float(quiz_stats[1]), 1) if quiz_stats[1] else 0,
            "topics_count": len(topic_mastery),
            "flashcards_reviewed": f_count, # La variabile del conteggio flashcard
            "quiz_history": quiz_history_list,    # <--- FONDAMENTALE
            "topics_detail": topics_detail_list,  # <--- FONDAMENTALE
            "topic_mastery": topics_detail_list   # Per la compatibilità con ResultsView
        }

    except Exception as e:
        print(f"ERRORE CRITICO: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def flashcards_detailed_stats(project_id: str, user = Depends(verify_user)):

    db = SessionLocal()

    # Cambiamo la query per raggruppare per topic
    # Nota: usiamo la tabella 'flashcards' per i nomi dei topic e 'flashcard_reviews' per i voti
    rows = db.execute(
        text("""
            SELECT 
                f.topic,
                COUNT(fr.id) as total,
                SUM(CASE WHEN fr.is_correct = false THEN 1 ELSE 0 END) as wrong,
                SUM(CASE WHEN fr.difficulty = 1 THEN 1 ELSE 0 END) as hard,
                SUM(CASE WHEN fr.difficulty = 2 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN fr.difficulty = 3 THEN 1 ELSE 0 END) as easy
            FROM flashcards f
            LEFT JOIN flashcard_reviews fr ON f.id = fr.flashcard_id
            WHERE f.project_id = :project_id 
              AND f.user_id = :user_id
            GROUP BY f.topic
        """),
        {
            "project_id": project_id,
            "user_id": user["id"]
        }
    ).fetchall()

    db.close()

    # Trasformiamo i risultati in un dizionario mappato per topic
    stats_by_topic = {}
    for r in rows:
        topic_name = r[0] or "General"
        total = r[1] or 0
        
        # Se non ci sono review per questo topic, mettiamo tutto a zero
        if total == 0:
            stats_by_topic[topic_name] = {
                "total": 0, "wrong": 0, "hard": 0, "good": 0, "easy": 0, "accuracy": 0
            }
            continue

        wrong = r[2] or 0
        hard = r[3] or 0
        good = r[4] or 0
        easy = r[5] or 0
        
        stats_by_topic[topic_name] = {
            "total": total,
            "wrong": wrong,
            "hard": hard,
            "good": good, # Il tuo frontend usa 'good' per il colore blu
            "easy": easy,
            "accuracy": round(((good + easy) / total) * 100, 1)
        }

    return stats_by_topic

# ======================
# PROJECT RESULTS
# ======================
@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def flashcards_detailed_stats(
    project_id: str,
    user = Depends(verify_user)
):
    db = SessionLocal() # Apre la connessione
    try:
        # La tua query SQL raggruppata per topic
        rows = db.execute(
            text("""
                select
                    topic,
                    sum(case when is_correct = false then 1 else 0 end) as wrong,
                    sum(case when difficulty = 1 then 1 else 0 end) as hard,
                    sum(case when difficulty = 2 then 1 else 0 end) as correct,
                    sum(case when difficulty = 3 then 1 else 0 end) as easy,
                    count(*) as total
                from flashcard_reviews
                where project_id = :project_id
                  and user_id = :user_id
                group by topic
            """),
            {"project_id": project_id, "user_id": user["id"]}
        ).fetchall()

        # Trasformazione dati
        stats_by_topic = {}
        for r in rows:
            topic_name = r[0] or "General"
            stats_by_topic[topic_name] = {
                "wrong": int(r[1] or 0),
                "hard": int(r[2] or 0),
                "good": int(r[3] or 0), # Manteniamo 'good' per coerenza frontend
                "easy": int(r[4] or 0),
                "total": int(r[5] or 0)
            }

        return stats_by_topic

    except Exception as e:
        print(f"❌ Error fetching detailed stats: {e}")
        # Opzionale: puoi sollevare un'eccezione HTTP qui
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    finally:
        db.close() # <--- SEMPRE QUI, garantisce che la connessione torni al pool


@app.get("/projects/{project_id}/results")
async def project_results(
    project_id: str,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    # verifica progetto
    project = db.execute(
        text("""
            select id
            from projects
            where id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    if not project:
        db.close()
        raise HTTPException(status_code=403, detail="Access denied")

    # ======================
    # QUIZ HISTORY
    # ======================

    quiz_rows = db.execute(
        text("""
            select qa.created_at,
                   qa.score,
                   qa.total_questions,
                   q.difficulty
            from quiz_attempts qa
            join quizzes q on qa.quiz_id = q.id
            where q.project_id = :project_id
            and qa.user_id = :user_id
            order by qa.created_at desc
            limit 20
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchall()

    quiz_history = []

    for r in quiz_rows:
        quiz_history.append({
            "date": str(r[0]),
            "score": r[1],
            "total": r[2],
            "difficulty": r[3]
        })

    # ======================
    # TOPIC ACCURACY
    # ======================
    attempt_rows = db.execute(
        text("""
            select answers
            from quiz_attempts
            where project_id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchall()

    topic_mastery_map = {}

    for row in attempt_rows:

        answers = row[0] or []

        for a in answers:

            topic = a.get("topic", "general")

            if topic not in topic_mastery_map:
                topic_mastery_map[topic] = {
                    "correct": 0,
                    "total": 0
                }

            topic_mastery_map[topic]["total"] += 1

            if a.get("is_correct"):
                topic_mastery_map[topic]["correct"] += 1

    topic_mastery = []

    for topic, stats in topic_mastery_map.items():

        accuracy = 0

        if stats["total"] > 0:
            accuracy = round(
                (stats["correct"] / stats["total"]) * 100,
                1
            )

        topic_mastery.append({
            "topic": topic,
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": accuracy
        })

    # ======================
    # GLOBAL QUIZ METRICS
    # ======================

    total_correct = sum(
        t["correct"] for t in topic_mastery
    )

    total_questions = sum(
        t["total"] for t in topic_mastery
    )

    total_wrong = total_questions - total_correct

    average_accuracy = 0

    if total_questions > 0:
        average_accuracy = round(
            (total_correct / total_questions) * 100,
            1
        )


    # ======================
    # FLASHCARD METRICS
    # ======================

    flashcard_stats = db.execute(
        text("""
            SELECT
                COUNT(*) as total_reviews,
                SUM(
                    CASE
                        WHEN is_correct THEN 1
                        ELSE 0
                    END
                ) as correct_reviews
            FROM flashcard_reviews
            WHERE project_id = :project_id
            AND user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    flashcard_reviews_count = int(flashcard_stats[0] or 0)
    flashcard_correct = int(flashcard_stats[1] or 0)

    flashcard_accuracy = 0

    if flashcard_reviews_count > 0:
        flashcard_accuracy = round(
            (flashcard_correct / flashcard_reviews_count) * 100,
            1
        )

    # ======================
    # DUE TODAY
    # ======================

    due_today = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and user_id = :user_id
            and next_review is not null
            and next_review <= now()
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).scalar()

    if due_today is None:
        due_today = 0

    # ======================
    # MOST FORGOTTEN TOPICS
    # ======================

    forgotten_rows = db.execute(
        text("""
            select
                topic,
                sum(
                    case
                        when is_correct then 1
                        else 0
                    end
                ) as correct,
                count(*) as total
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
            and topic is not null
            group by topic
            having count(*) >= 2
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchall()

    forgotten_topics = []

    for r in forgotten_rows:

        accuracy = 0

        if r[2] > 0:
            accuracy = round((r[1] / r[2]) * 100, 1)

        forgotten_topics.append({
            "topic": r[0],
            "accuracy": accuracy,
            "reviews": int(r[2])
        })

    forgotten_topics.sort(key=lambda x: x["accuracy"])

    # ======================
    # WEAK AREAS
    # ======================

    weak_areas = sorted(
        topic_mastery,
        key=lambda x: x["accuracy"]
    )[:3]

    db.close()


   
    # ======================
    # QUIZ ATTEMPTS
    # ======================

    quiz_attempts = db.execute(
        text("""
            select count(*)
            from quiz_attempts
            where project_id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).scalar()

    # ======================
    # AVG SCORE
    # ======================

    avg_score = db.execute(
        text("""
            select avg(
                (score::float / total_questions) * 100
            )
            from quiz_attempts
            where user_id = :user_id
            and quiz_id in (
                select id from quizzes where project_id = :project_id
            )
        """),
        {
            "user_id": user_id,
            "project_id": project_id
        }
    ).scalar()

    if avg_score is None:
        avg_score = 0

    # ======================
    # FLASHCARDS REVIEWED
    # ======================

    flashcards_reviewed = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and last_review is not null
        """),
        {
            "project_id": project_id
        }
    ).scalar()

    # ======================
    # TOPICS COUNT
    # ======================

    topics_count = db.execute(
        text("""
            select count(distinct topic)
            from quiz_questions qq
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
        """),
        {
            "project_id": project_id
        }
    ).scalar()

    db.close()

    print("🔥 FINAL RESULTS RESPONSE:")
    print({
        "quiz_history": quiz_history,
        "topic_mastery": topic_mastery,
        "topics_detail": topic_mastery,
        "quiz_attempts": quiz_attempts or 0,
        "avg_score": round(avg_score, 1),
    })

    return {
        "quiz_history": quiz_history,
        "topic_mastery": topic_mastery,
        "topics_detail": topic_mastery,

        # QUIZ METRICS
        "quiz_attempts": quiz_attempts or 0,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "average_accuracy": average_accuracy,

        # FLASHCARD METRICS
        "flashcard_reviews": flashcard_reviews_count,
        "flashcard_accuracy": flashcard_accuracy,

        # RETENTION INTELLIGENCE
        "due_today": due_today,
        "forgotten_topics": forgotten_topics[:3],

        # LEARNING METRICS
        "avg_score": round(avg_score, 1),
        "flashcards_reviewed": flashcards_reviewed or 0,
        "topics_count": topics_count or 0,

        # DIAGNOSTICS
        "weak_areas": weak_areas
    }
@app.get("/projects/{project_id}/flashcards_count")
async def flashcards_count(project_id: str):

    db = SessionLocal()

    row = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and next_review <= now()
        """),
        {"project_id": project_id}
    ).fetchone()

    db.close()

    return {"count": row[0]}
@app.get("/projects/{project_id}/study_flashcards")
async def study_flashcards(project_id: str, limit: int = 20):

    db = SessionLocal()

    rows = db.execute(
    text("""
        select id, question, answer
        from flashcards
        where project_id = :project_id
        order by id desc
        limit :limit
    """),
    {
        "project_id": project_id,
        "limit": limit
    }
    ).fetchall()

    db.close()

    cards = []

    for r in rows:
        cards.append({
            "id": r[0],
            "question": r[1],
            "answer": r[2]
        })

    return {"flashcards": cards}

@app.post("/review_flashcard")
async def review_flashcard(
    req: dict = Body(...),
    user = Depends(verify_user)
):

    db = SessionLocal()
    print("REVIEW_FLASHCARD req:", req)
    print("REVIEW_FLASHCARD flashcard_id:", req.get("flashcard_id"))
    print("REVIEW_FLASHCARD type:", type(req.get("flashcard_id")))
    try:

        difficulty = req.get("difficulty", 1)
        try:
            difficulty = int(difficulty)
        except:
            difficulty = 1
        flashcard_id = req.get("flashcard_id")
        if not flashcard_id:
            raise HTTPException(status_code=400, detail="flashcard_id missing")
        row = db.execute(
            text("""
                select next_review, last_review
                from flashcards
                where id = :flashcard_id
            """),
            {"flashcard_id": flashcard_id}
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Flashcard not found")
        is_correct = req.get("is_correct", False)

        from datetime import datetime, timedelta

        now = datetime.utcnow()

        # fallback
        current_interval_days = 1

        if row and row[0] and row[1]:
            delta = row[0] - row[1]
            current_interval_days = max(1, delta.days)

        # ======================
        # SPACED REPETITION LOGIC
        # ======================

        if not is_correct:
            new_interval_days = 1

        elif difficulty == 1:
            new_interval_days = max(1, current_interval_days // 2)

        elif difficulty == 2:
            new_interval_days = current_interval_days + 1

        elif difficulty == 3:
            new_interval_days = current_interval_days * 2

        else:
            new_interval_days = current_interval_days + 3

        next_review = now + timedelta(days=new_interval_days)

        db.execute(
            text("""
                update flashcards
                set
                    difficulty = :difficulty,
                    last_review = now(),
                    next_review = now() + (:days || ' days')::interval
                where id = :flashcard_id
            """),
            {
                "difficulty": difficulty,
                "flashcard_id": flashcard_id,
                "days": new_interval_days
            }
        )

        flashcard_row = db.execute(
            text("""
                select project_id, topic
                from flashcards
                where id = :flashcard_id
            """),
            {
                "flashcard_id": flashcard_id
            }
        ).fetchone()
        print("🔥 SAVING FLASHCARD REVIEW:", {
            "flashcard_id": flashcard_id,
            "project_id": flashcard_row[0],
            "user_id": user["id"],
            "topic": flashcard_row[1],
            "is_correct": is_correct
        })
        if flashcard_row:
            db.execute(
                text("""
                    insert into flashcard_reviews
                    (flashcard_id, project_id, user_id, is_correct, difficulty, elapsed_seconds, topic)
                    values
                    (:flashcard_id, :project_id, :user_id, :is_correct, :difficulty, :elapsed_seconds, :topic)
                """),
                {
                    "flashcard_id": flashcard_id,
                    "project_id": flashcard_row[0],
                    "user_id": user["id"],
                    "is_correct": is_correct,
                    "difficulty": difficulty,
                    "elapsed_seconds": req.get("elapsed_seconds", 0),
                    "topic": flashcard_row[1]
                }
            )

        db.commit()
    

        return {"status": "ok"}

    

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()
        print("ERROR review_flashcard:", e)
        raise HTTPException(status_code=500, detail="Internal error")

    finally:
        db.close()
    

@app.get("/projects/{project_id}/flashcard_results")
async def flashcard_results(
    project_id: str,
    user = Depends(verify_user)
):
    db = SessionLocal()

    total_reviews = db.execute(
        text("""
            select count(*)
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
        """),
        {"project_id": project_id, "user_id": user["id"]}
    ).scalar()

    correct_reviews = db.execute(
        text("""
            select count(*)
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
            and is_correct = true
        """),
        {"project_id": project_id, "user_id": user["id"]}
    ).scalar()

    avg_time = db.execute(
        text("""
            select avg(elapsed_seconds)
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
        """),
        {"project_id": project_id, "user_id": user["id"]}
    ).scalar()

    db.close()

    accuracy = 0
    if total_reviews and total_reviews > 0:
        accuracy = round((correct_reviews / total_reviews) * 100, 1)

    return {
        "total_reviews": total_reviews or 0,
        "correct_reviews": correct_reviews or 0,
        "accuracy": accuracy,
        "avg_time": round(avg_time or 0, 1)
    }

@app.get("/quizzes/{quiz_id}")
async def get_quiz(quiz_id: str):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select question, options, correct, explanation, explanation_long,
                   source_document, source_page, topic
            from quiz_questions
            where quiz_id = :quiz_id
            order by question_order
        """),
        {"quiz_id": quiz_id}
    ).fetchall()

    db.close()

    questions = []

    for r in rows:
        questions.append({
            "question": r[0],
            "options": r[1] if isinstance(r[1], list) else json.loads(r[1]),
            "correct": r[2],
            "explanation": r[3],
            "explanation_long": r[4],
            "source_document": r[5],
            "source_page": r[6],
            "topic": r[7],
        })

    return {"questions": questions}

@app.post("/save_quiz_attempt")
async def save_quiz_attempt(req: dict, user = Depends(verify_user)):
    quiz_id = req.get("quiz_id")
    user_id = user["id"]  # <--- Recuperiamo l'ID dell'utente autenticato
    answers = req.get("answers", [])

    db = SessionLocal()
    try:
        # 1. Salva il tentativo (se non lo fa già Supabase, o per sicurezza)
        # Se hai già una riga in quiz_attempts, questo passaggio serve a legare il tutto
        
        for a in answers:
            db.execute(
                text("""
                    insert into quiz_answers (quiz_id, question_id, is_correct, topic, user_id)
                    values (:quiz_id, :question_id, :is_correct, :topic, :user_id)
                """),
                {
                    "quiz_id": quiz_id,
                    "question_id": a.get("question_id"),
                    "is_correct": a.get("is_correct", False),
                    "topic": (a.get("topic") or "General").strip().lower(),
                    "user_id": user_id  # <--- Passiamo lo user_id al DB
                }
            )
        db.commit()
        print(f"✅ Salvate {len(answers)} risposte per l'utente {user_id}")
        return {"status": "saved"}
    except Exception as e:
        db.rollback()
        print(f"❌ Database Error: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()

@app.get("/projects/{project_id}/stats")
async def get_quiz_stats(project_id: str, user = Depends(verify_user)):
    db = SessionLocal()
    try:
        # Questa query unisce Quiz e Flashcards calcolando i totali per ogni topic
        query = text("""
            SELECT 
                LOWER(TRIM(qa.topic)) as topic,
                COUNT(*) FILTER (WHERE qa.is_correct) as correct_count,
                COUNT(*) as total_count
            FROM quiz_answers qa
            WHERE qa.user_id = :u_id
            AND qa.topic IS NOT NULL
            AND LOWER(TRIM(qa.topic)) != 'general'
            GROUP BY LOWER(TRIM(qa.topic))

            UNION ALL

            SELECT 
                LOWER(TRIM(topic)) as topic,
                COUNT(*) FILTER (WHERE is_correct) as correct_count,
                COUNT(*) as total_count
            FROM flashcard_reviews
            WHERE project_id = :p_id
            AND user_id = :u_id
            AND topic IS NOT NULL
            AND LOWER(TRIM(topic)) != 'general'
            GROUP BY LOWER(TRIM(topic))
        """)
        
        result = db.execute(
            query,
            {"p_id": project_id, "u_id": user["id"]}
        ).fetchall()
        print("🔥 RAW STATS RESULT:", result)

        merged = {}

        for r in result:
            topic = (r[0] or "").strip().lower()

            if not topic:
                continue

            if topic not in merged:
                merged[topic] = {
                    "correct": 0,
                    "total": 0
                }

            merged[topic]["correct"] += int(r[1] or 0)
            merged[topic]["total"] += int(r[2] or 0)

        return merged
    finally:
        db.close()

@app.get("/projects/{project_id}/quiz_attempts_summary")
async def quiz_attempts_summary(project_id: str, user = Depends(verify_user)):
    # --- AGGIUNGI SOLO QUESTO CONTROLLO ---
    if not project_id or project_id == "" or project_id == "undefined":
        return {"data": {}}
    # ---------------------------------------

    db = SessionLocal()
    try:
        # Il resto del tuo codice rimane identico
        rows = db.execute(
            text("""
                select 
                    qa.quiz_id,
                    count(*) as attempts,
                    max(qa.score) as best_score,
                    (
                        select qa2.score
                        from quiz_attempts qa2
                        where qa2.quiz_id = qa.quiz_id
                        and qa2.user_id = :user_id
                        order by qa2.id desc
                        limit 1
                    ) as last_score
                from quiz_attempts qa
                join quizzes q on qa.quiz_id = q.id
                where q.project_id = :project_id
                and qa.user_id = :user_id
                group by qa.quiz_id
            """),
            {
                "project_id": project_id,
                "user_id": user["id"]
            }
        ).fetchall()

        result = {}
        for r in rows:
            result[str(r[0])] = {
                "attempts": r[1],
                "best_score": r[2],
                "last_score": r[3]
            }
        return {"data": result}
    except Exception as e:
        print(f"Errore stats: {e}")
        return {"data": {}} # Protezione extra: se c'è un errore, ritorna dati vuoti
    finally:
        db.close()

@app.post("/ask")
async def ask_documents(req: AskRequest):
    print("HISTORY RECEIVED:", req.history)

    # 🔥 STEP 1 — COSTRUISCI SEARCH QUERY CON HISTORY
    search_query = req.question

    if req.history:
        last_user_messages = [
            m.get("content")
            for m in req.history
            if m.get("role") == "user"
        ][-2:]

        if last_user_messages:
            search_query = " ".join(last_user_messages)

    print("SEARCH QUERY:", search_query)

    # 🔥 STEP 2 — USA search_query (NON req.question)
    chunks = search_project_chunks(
        project_id=req.project_id,
        query=search_query,   # 👈 QUESTA È LA MODIFICA CHIAVE
        topics=req.topics,
        k=12
    )    
    print("CHUNKS FOUND:", len(chunks))
    if chunks:
        print("SAMPLE CHUNK:", chunks[0]["text"][:200])

    

    context_blocks = []

    for c in chunks:
        context_blocks.append(
            f"DOCUMENT: {c['document']} | PAGE: {c['page']}\nCONTENT:\n{c['text'][:600]}"
        )


    context = "\n\n---\n\n".join(context_blocks) 
    print("CONTEXT LENGTH:", len(context))
    # 3️⃣ costruzione contesto
    
    history_text = ""

    if req.history:
        for msg in req.history:
            role = msg.get("role")
            content = msg.get("content")

            if not content:
                continue

            if role == "user":
                history_text += f"Student: {content}\n"
            elif role == "assistant":
                history_text += f"Tutor: {content}\n"

    if getattr(req, 'expand_search', False):
        instruction_mode = """
        - You are in 'GLOBAL KNOWLEDGE' mode.
        - Start from the provided Context, but if it's not enough or you can explain better, 
          use your full AI knowledge base.
        - Provide a rich, detailed, and helpful explanation.
        """
        current_temp = 0.6 # Più creativo
    else:
        instruction_mode = """
        - You are in 'STRICT MODE'.
        - Use ONLY the material provided in the Context.
        - If the answer is not in the material, say: 'I'm sorry, I can't find this in your documents.'
        - DO NOT use external knowledge.
        """
        current_temp = 0.1 # Più preciso e fedele al testo

    prompt = f"""
    You are an expert study tutor helping a student understand material deeply.

    IMPORTANT:
    - This is an ongoing conversation.
    - The student may ask follow-up questions.
    - You MUST use previous conversation context to refine and expand your answers.
    - Do NOT restart explanations from scratch if the question is a follow-up.
    - Stay focused on the SAME concept unless the student changes topic.
    - If the Current question contains a quiz question under "Question:",
      identify the language of that quiz question and answer entirely in that
      same language. Do not infer the response language from UI labels or these
      prompt instructions.

    Rules:
    - If relevant info exists, explain it clearly
    - If partial, expand logically using the context
    - Be precise and avoid generic answers
    - When needed, connect the answer to previous messages
    - Use clear paragraphs or bullet points

    Context:
    {context}

    Conversation so far:
    {history_text}

    Current question:
    {req.question}

    Answer as a tutor helping the student progressively understand the topic.
    """


    # 4️⃣ GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful study tutor."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"answer": response.choices[0].message.content}



print("🔥 MODEL FIELDS:", ActiveRecallRequest.__fields__.keys())


# main.py

# main.py

@app.post("/projects/{project_id}/active_recall_question")
async def active_recall_question(project_id: str, req: ActiveRecallRequest, user = Depends(verify_user)):
    # 1. Normalizzazione stringhe (rimuoviamo ogni dubbio sugli spazi)
    def super_clean(s):
        return re.sub(r'\s+', ' ', str(s).replace('\xa0', ' ')).strip()

    topics = [super_clean(t) for t in req.topics if t]
    
    if not topics:
        return {"question": "No topics available", "concept": "General"}

    # 2. ROTAZIONE FORZATA
    # Usiamo l'indice del frontend per pescare il topic
    current_focus = topics[req.index % len(topics)]
    
    # Estraiamo la parola chiave (es. da "Block Actions" prendiamo "Block")
    # Questo serve per il matching nel DB se il nome intero è corrotto
    keyword = current_focus.split()[0] if current_focus else ""

    db = SessionLocal()
    
    # 🔥 PRENDI I TOPIC DAL FRONTEND
    topics = req.topics or []

    # 🔥 NORMALIZZA
    topics = [normalize_string(t) for t in topics if t]

    # 🔥 ROTAZIONE
    if topics:
        current_focus = topics[req.index % len(topics)]
    else:
        current_focus = "General"

    # 🔥 KEYWORD SEMPLICE (prima parola)
    keyword = current_focus.split(" ")[0]

    print("🎯 CURRENT FOCUS:", current_focus)
    print("🔑 KEYWORD:", keyword)
 
    query_text = " ".join(req.topics) if req.topics else "general study content"

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )

    query_embedding = emb.data[0].embedding

    active_recall_roles = ensure_project_chunk_roles(
        db,
        project_id,
    )
    db.commit()
    active_recall_teaching_count = sum(
        1
        for role in active_recall_roles
        if is_assignment_eligible_chunk_role(role)
    )
    print(
        "ELIGIBLE TEACHING CHUNKS:",
        active_recall_teaching_count
    )
    print(
        "EXCLUDED CHUNKS:",
        len(active_recall_roles) - active_recall_teaching_count
    )

    rows = db.execute(
        text("""
            SELECT chunk_text, doc_title, page, topic
            FROM chunks
            WHERE project_id = :project_id
            AND chunk_role = 'teaching'
            AND (
                topic ILIKE :full_focus             -- Esempio: %Block Actions%
                OR topic ILIKE :keyword_focus       -- Esempio: %Block%
                OR chunk_text ILIKE :keyword_focus  -- Cerca "Block" nel testo
            )
            ORDER BY embedding <-> CAST(:embedding AS vector)
            LIMIT 12
        """),
        {
            "project_id": project_id,
            "full_focus": f"%{current_focus}%",
            "keyword_focus": f"%{keyword}%",
            "embedding": query_embedding
        }
    ).fetchall()
    db.close()

    # ... (il resto del codice per generare la risposta con GPT)

    if not rows:
        return {"question": "No context found for this topic.", "concept": current_focus}

    # 5. Prepariamo il contesto per GPT
    random.shuffle(rows)
    selected_rows = rows[:5]
    
    context_blocks = []
    for r in selected_rows:
        block = f"SOURCE: {r[1]} (Page {r[2]}) | TOPIC: {r[3]}\nCONTENT: {r[0]}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    # 6. Prompt ottimizzato per varietà e precisione
    prompt = f"""
    You are a professional tutor specializing in the Active Recall method.
    TARGET TOPIC: {current_focus}
    
    CONTEXT MATERIAL:
    {context}

    STRICT RULES:
    1. Focus ONLY on "{current_focus}".
    2. Generate an open-ended question that requires reasoning (e.g., "How does...", "Why...", "What is the relationship...").
    3. Avoid simple "What is X?" definitions.
    4. Use ONLY the provided context.
    5. The response MUST be a valid JSON object.
    6. The question MUST be written entirely in {req.language}.
    7. Do not switch language under any circumstance.

    JSON FORMAT:
    {{
    "question": "...",
    "concept": "{current_focus}",
    "difficulty": "medium",
    "source_document": "...",
    "source_page": "..."
    }}
    """

    # 7. Chiamata a OpenAI con parametri di varietà (Temperature e Penalty)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate unique active recall questions. Never repeat the same question structure twice."},
            {"role": "user", "content": f"{prompt}\n\nSeed for variety: {time.time()}"}
        ],
        temperature=0.9,       # Aumenta la creatività
        presence_penalty=0.6,  # Evita di ripetere gli stessi concetti
        frequency_penalty=0.3, # Evita di usare le stesse parole
        response_format={ "type": "json_object" }
    )

    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
    except:
        # Fallback in caso di errore JSON
        data = {"question": f"Explain a key mechanism of {current_focus} based on the text.", "concept": current_focus}
    
    return data

from typing import Optional

class ActiveRecallEvaluateRequest(BaseModel):
    question: str
    student_answer: str
    history: Optional[list[str]] = None

@app.post("/generate_recovery_flashcards")
async def generate_recovery_flashcards(req: dict):
    data = req if isinstance(req, dict) else await req.json()

    topics = data.get("topics", [])

    project_id = data.get("project_id")

    print("🧠 RECOVERY TOPICS:", topics)

    db = SessionLocal()
    recovery_roles = ensure_project_chunk_roles(
        db,
        project_id,
    )
    db.commit()
    recovery_teaching_count = sum(
        1
        for role in recovery_roles
        if is_assignment_eligible_chunk_role(role)
    )
    print(
        "ELIGIBLE TEACHING CHUNKS:",
        recovery_teaching_count
    )
    print(
        "EXCLUDED CHUNKS:",
        len(recovery_roles) - recovery_teaching_count
    )

    # prendi chunk random ma piccoli (focus)
    if topics and len(topics) > 0:
        rows = db.execute(
            text("""
                select chunk_text
                from chunks
                where project_id = :project_id
                and chunk_role = 'teaching'
                and topic = :topic
                order by random()
                limit 5
            """),
            {
                "project_id": project_id,
                "topic": topics[0]
            }
        ).fetchall()

        # fallback se non trova chunk con quel topic
        if not rows:
            rows = db.execute(
                text("""
                    select chunk_text
                    from chunks
                    where project_id = :project_id
                    and chunk_role = 'teaching'
                    order by random()
                    limit 5
                """),
                {"project_id": project_id}
            ).fetchall()

    else:
        rows = db.execute(
            text("""
                select chunk_text
                from chunks
                where project_id = :project_id
                and chunk_role = 'teaching'
                order by random()
                limit 5
            """),
            {"project_id": project_id}
        ).fetchall()

    db.close()

    context = "\n\n".join([r[0][:300] for r in rows])

    prompt = f"""
Generate 3 recovery flashcards.

Rules:
- Focus on reinforcing misunderstood concepts
- Keep them simple and clear
- No duplicates
- Use ONLY provided material

Return JSON:

[
  {{
    "question": "...",
    "answer": "..."
  }}
]

Material:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    content = content.replace("```json","").replace("```","").strip()

    try:
        return {"flashcards": json.loads(content)}
    except:
        return {"flashcards": []}

@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def get_flashcards_detailed_stats(project_id: str, user = Depends(verify_user)):
    db = SessionLocal()
    try:
        # Recupera il conteggio delle flashcard raggruppate per topic e difficoltà
        rows = db.execute(
            text("""
                SELECT LOWER(TRIM(topic)), difficulty, COUNT(*) as count
                FROM flashcards
                WHERE project_id = :project_id
                GROUP BY topic, difficulty
            """),
            {"project_id": project_id}
        ).fetchall()

        stats = {}
        for row in rows:
            topic = row[0] or "General"
            diff = row[1] or "unseen" # Se non sono state ancora studiate
            count = row[2]

            if topic not in stats:
                stats[topic] = {"wrong": 0, "hard": 0, "good": 0, "easy": 0, "total": 0}
            
            # Mapping delle chiavi per il frontend
            if diff == "wrong": stats[topic]["wrong"] += count
            elif diff == "hard": stats[topic]["hard"] += count
            elif diff == "good": stats[topic]["good"] += count
            elif diff == "easy": stats[topic]["easy"] += count
            
            stats[topic]["total"] += count

        return stats
    finally:
        db.close()

@app.post("/active_recall_evaluate")
async def active_recall_evaluate(req: ActiveRecallEvaluateRequest):

    history_text = "\n".join(req.history or [])

    prompt = f"""
    You are a supportive study tutor evaluating a student's answer using semantic reasoning.

    Question:
    {req.question}

    Previous answers:
    {history_text}

    Latest answer:
    {req.student_answer}

    Evaluation rules:
    - CONCEPTUAL FOCUS: Identify the core concepts of a correct answer. If the student mentions a concept using synonyms or shorthand (e.g., "no more movement" instead of "cannot move anymore"), consider it PRESENT.
    - MEANING OVER WORDING: Do not penalize the student for using different vocabulary. If the "Main Idea" is there, it is CORRECT.
    - NO REDUNDANCY: Do NOT list a concept in the "missing" array if the student has already expressed it, even partially or briefly.
    

    Scoring:
    - correct: Core concepts are present (even if brief or using synonyms).
    - partial: Concept is understood but a CRITICAL, non-implied consequence is missing.
    - incorrect: The core concept is missing or fundamentally wrong.

    Return ONLY JSON:
    {{
    "evaluation": "correct | partial | incorrect",
    "score": 0-1,
    "feedback": "Concise, supportive feedback. If they are right, tell them!",
    "missing": ["Only list things truly not mentioned or implied"],
    "wrong_claims": ["..."],
    "hint": "...",
    "explanation": "Clear explanation of the concept"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You evaluate student answers."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json","").replace("```","").strip()

    try:
        data = json.loads(content)
    except:
        data = {
            "evaluation": "incorrect",
            "score": 0,
            "feedback": "The answer needs more explanation.",
            "hint": "Try explaining the concept step by step.",
            "explanation": "Review the concept and try again."
        }

    return data
# ======================
# STUDY SESSION
# ======================

@app.get("/projects/{project_id}/study_session")
async def study_session(
    project_id: str,
    topics: str = None,
    user = Depends(verify_user)
):
    
    def normalize(t):
        return t.lower().replace(" ", "")

    topics_list = [t.strip() for t in topics.split(",")] if topics else []

    print("🎯 RAW TOPICS:", topics_list)

    print("🚨 RAW TOPICS STRING:", topics)

    topics_list = topics.split(",") if topics else []

    print("🎯 TOPICS LIST:", topics_list)

    # 🔥 fallback per compatibilità col codice esistente
    topic = topics_list[0] if topics_list else None

    # 🔥 NORMALIZZA
    topics_list = list(set(topics_list))

    print("🎯 CLEAN TOPICS:", topics_list)
   
    db = SessionLocal()

    # ======================
    # DETECT WEAK TOPICS
    # ======================

    weak_topic_rows = db.execute(
        text("""
            select 
                qq.topic,
                sum(case when qa.is_correct then 1 else 0 end) as correct,
                count(*) as total,
                sum(case when qa.is_correct then 1 else 0 end)::float / count(*) as accuracy
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
            group by qq.topic
            order by accuracy asc
            limit 5
        """),
        {"project_id": project_id}
    ).fetchall()
    print("🔥 WEAK DEBUG topic_rows:", weak_topic_rows)
    weak_topics = [r[0] for r in weak_topic_rows if r[0]]

    # ======================
    # GENERATE FLASHCARDS
    # ======================

    scope = resolve_learning_scope(
        project_id=project_id,
        # topic_ids=topic_ids,
        limit=30
    )

    context_chunks = scope["chunks"]

    # fallback globale
    if not context_chunks:

        print("⚠️ NO CHUNKS FOUND → GLOBAL FALLBACK")

        scope = resolve_learning_scope(
            project_id=project_id,
            topic_ids=[],
            limit=30
        )

        context_chunks = scope["chunks"]

    # 🔥 funzione per pulire topic
    def clean(t):
        return " ".join(t.split()) if t else t

    # 🔥 estrai topic dai chunk
    chunk_topics = list(dict.fromkeys([
        clean(c.get("topic")) for c in context_chunks if c.get("topic")
    ]))

    context_blocks = []

    for c in context_chunks:
        context_blocks.append(f"""
    TOPIC: {c.get("topic")}

    CONTENT:
    {c.get("text", "")[:400]}
    """)

    context_text = "\n\n---\n\n".join(context_blocks)
    weak_topics_text = ", ".join(weak_topics) if weak_topics else "important concepts"

    prompt = f"""
    You are a strict study tutor generating flashcards for a focused study session.

    FOCUS TOPIC:
    {", ".join(topics_list) if topics_list else "GENERAL"}

    CRITICAL RULE:
    - "You MUST generate flashcards ONLY about these topics: \"{", ".join(topics_list) if topics_list else "GENERAL"}\""
    Each flashcard MUST include the topic it comes from.
    The topic MUST be exactly one of the topics written in the material.
    - Even if other topics appear, IGNORE them
    - If a chunk is not related to this topic, DO NOT use it
    Generate EXACTLY 15 flashcards.
    Focus especially on these weak topics:
    {weak_topics_text}

    Rules:
    - Use ONLY the provided material
    - Do NOT use external knowledge
    - Do NOT invent information
    - Each flashcard must cover a DIFFERENT concept
    - Avoid similar or repeated questions
    - Prefer "why", "how", "what happens if"
    - Avoid simple definitions unless necessary

    Difficulty:
    - easy → direct recall
    - medium → explanation or relation
    - hard → reasoning or consequences

    Return ONLY JSON:

    [
    {{
        "question": "...",
        "answer": "..."
        "topic": "..."
    }}
    ]

    Material:
    {context_text}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.7,
        messages=[
            {"role": "system", "content": "Generate study session flashcards."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json", "").replace("```", "").strip()

    try:
        generated_cards = json.loads(content)

        if not isinstance(generated_cards, list):
            generated_cards = []

        seen = set()
        unique_cards = []

        for c in generated_cards:
            q = c.get("question", "").strip().lower()

            if q and q not in seen:
                seen.add(q)
                unique_cards.append(c)

        generated_cards = unique_cards[:15]

    except Exception as e:
        print("❌ STUDY SESSION FLASHCARDS JSON ERROR:", e)
        print("RAW GPT OUTPUT:", content)
        generated_cards = []
    
    if not isinstance(generated_cards, list):
        generated_cards = []

    flashcards = []

    for c in generated_cards:

        flashcard_id = str(uuid.uuid4())
        assigned_topic = chunk_topics[len(flashcards) % len(chunk_topics)] if chunk_topics else topic

        db.execute(
            text("""
                insert into flashcards
                (id, project_id, user_id, question, answer, topic)
                values
                (:id, :project_id, :user_id, :question, :answer, :topic)
            """),
            {
                "id": flashcard_id,
                "project_id": project_id,
                "user_id": user["id"],
                "question": c.get("question"),
                "answer": c.get("answer"),
                "topic": assigned_topic
            }
        )

        flashcards.append({
            "id": flashcard_id,
            "question": c.get("question"),
            "answer": c.get("answer"),
            "topic": assigned_topic
        })
        print("🧠 FINAL FLASHCARD TOPICS:", [f["topic"] for f in flashcards])
    db.commit()

    # ======================
    # ADAPTIVE QUIZ CONFIG
    # ======================

    avg_accuracy_row = db.execute(
        text("""
            select avg(
                case when qa.is_correct then 1 else 0 end
            )
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
        """),
        {"project_id": project_id}
    ).fetchone()

    avg_accuracy = avg_accuracy_row[0] if avg_accuracy_row and avg_accuracy_row[0] else 0.5

    if avg_accuracy < 0.5:
        quiz_questions = 15
    elif avg_accuracy < 0.8:
        quiz_questions = 20
    else:
        quiz_questions = 25

    db.close()

    return {
        "flashcards": flashcards,
        "recall_topics": weak_topics,
        "quiz": {
            "num_questions": quiz_questions,
            "difficulty": "medium",
            "focus_topics": weak_topics
        }
    }
@app.delete("/projects/{project_id}/documents/{doc_title}")
def delete_document(
        project_id: str,
        doc_title: str,
        user = Depends(verify_user)
    ):

        doc_title = unquote(doc_title)  # 🔥 FIX URL encoding

        print("DELETE DOCUMENT:", project_id, doc_title)
        user_id = user["id"]
        db = SessionLocal()

        # verifica ownership
        project = db.execute(
            text("""
                select id
                from projects
                where id = :project_id
                and user_id = :user_id
            """),
            {
                "project_id": project_id,
                "user_id": user_id
            }
        ).fetchone()

        if not project:
            db.close()
            raise HTTPException(status_code=403, detail="Access denied")

        # 🔥 DELETE REAL (tutti i chunk del documento)
        db.execute(
            text("""
                delete from chunks
                where project_id = :project_id
                and doc_title = :doc_title
            """),
            {
                "project_id": project_id,
                "doc_title": doc_title
            }
        )
        print("ROWS DELETED")

        db.commit()
        db.close()

        return {"status": "deleted"}   
 
