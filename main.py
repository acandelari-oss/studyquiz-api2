import os
import uuid
import json
import re
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel

from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

from openai import OpenAI


# =====================
# ENV
# =====================
DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY = os.environ["BACKEND_API_KEY"]


# =====================
# INIT
# =====================
app = FastAPI()
client = OpenAI(api_key=OPENAI_API_KEY)

engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(bind=engine)


# =====================
# AUTH
# =====================
def verify_api_key(authorization: str = Header(None)):
    if authorization != f"Bearer {BACKEND_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")


# =====================
# MODELS
# =====================
class ProjectCreate(BaseModel):
    name: str
    user_id: Optional[str] = None  # per salvare progetto per utente


class ProjectOut(BaseModel):
    project_id: str
    name: str


class IngestDocument(BaseModel):
    title: str
    text: str
    page: Optional[int] = None  # con pdf-parse spesso non disponibile


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class QuizRequest(BaseModel):
    num_questions: int
    language: str
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool


class ExtendedExplanationRequest(BaseModel):
    question: str
    correct: str
    explanation: Optional[str] = None
    language: str = "English"


# =====================
# HEALTH
# =====================
@app.get("/health")
def health():
    return {"status": "ok"}


# =====================
# EMBEDDINGS
# =====================
def embed_texts(texts: List[str]):
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in res.data]


# =====================
# PROJECTS
# =====================
@app.post("/projects")
def create_project(data: ProjectCreate, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        project_id = str(uuid.uuid4())

        # Richiede che in DB esista colonna user_id in projects (vedi note sotto)
        db.execute(
            sql_text("""
                insert into projects (id, name, user_id)
                values (:id, :name, :user_id)
            """),
            {"id": project_id, "name": data.name, "user_id": data.user_id}
        )
        db.commit()
        return {"project_id": project_id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


@app.get("/projects")
def list_projects(user_id: str, api_key: str = Depends(verify_api_key)):
    """
    Ritorna i progetti dell'utente.
    """
    db = SessionLocal()
    try:
        rows = db.execute(
            sql_text("""
                select id, name
                from projects
                where user_id = :uid
                order by created_at desc nulls last
            """),
            {"uid": user_id}
        ).fetchall()

        return {
            "projects": [{"project_id": r[0], "name": r[1]} for r in rows]
        }
    finally:
        db.close()


@app.get("/projects/{project_id}/documents")
def list_documents(project_id: str, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        rows = db.execute(
            sql_text("""
                select distinct doc_title
                from chunks
                where project_id = :pid
                order by doc_title
            """),
            {"pid": project_id}
        ).fetchall()

        return {"documents": [{"title": r[0]} for r in rows]}
    finally:
        db.close()


# =====================
# INGEST
# =====================
@app.post("/projects/{project_id}/ingest")
def ingest(project_id: str, data: IngestRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        for doc in data.documents:
            # chunking minimale (1 chunk). In futuro puoi spezzare in più chunk.
            chunks = [doc.text]
            vectors = embed_texts(chunks)

            for chunk_text, vector in zip(chunks, vectors):
                db.execute(
                    sql_text("""
                        insert into chunks
                        (project_id, doc_id, doc_title, page, chunk_text, embedding)
                        values
                        (:project_id, :doc_id, :doc_title, :page, :chunk_text, CAST(:embedding AS vector))
                    """),
                    {
                        "project_id": project_id,
                        "doc_id": str(uuid.uuid4()),
                        "doc_title": doc.title,
                        "page": doc.page,
                        "chunk_text": chunk_text,
                        "embedding": vector
                    }
                )

        db.commit()
        return {"status": "ok"}

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# =====================
# GENERATE QUIZ
# =====================
@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(project_id: str, req: QuizRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()

    try:
        # recupera materiale (top N). In futuro: retrieval per similarità su query dinamica.
        rows = db.execute(
            sql_text("""
                select doc_title, coalesce(page, null) as page, chunk_text
                from chunks
                where project_id = :pid
                limit 10
            """),
            {"pid": project_id}
        ).fetchall()

        if not rows:
            raise HTTPException(status_code=400, detail="No study material found")

        # costruiamo contesto con titoli e (page)
        context_parts = []
        for title, page, text_ in rows:
            page_str = "unknown" if page is None else str(page)
            context_parts.append(f"[SOURCE: {title} | PAGE: {page_str}]\n{text_}")
        context = "\n\n".join(context_parts)

        prompt = f"""
You are generating a high-quality exam-style multiple choice quiz.

Constraints:
- Return ONLY a JSON array (no markdown, no prose).
- Each item must be an object with exactly:
  question: string
  options: array of 4 strings
  correct: one of the options exactly
  explanation: string
  source: string (must match one SOURCE title from the material)
  page: number or null (if unknown)

Rules:
- Difficulty: {req.difficulty}
- Language: {req.language}
- Exactly {req.num_questions} questions
- Options must all be plausible and relevant to the question.
- No duplicate questions.
- Do NOT mention slides.
- Keep correct answer as the full option string, not a letter.

Return format example:
[
  {{
    "question": "...",
    "options": ["...","...","...","..."],
    "correct": "...",
    "explanation": "...",
    "source": "SomeFile.pdf",
    "page": null
  }}
]

MATERIAL:
{context}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY valid JSON array. No markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        raw = response.choices[0].message.content or ""

        # estrai JSON array anche se il modello accidentalmente aggiunge testo
        match = re.search(r"\[\s*\{.*\}\s*\]\s*$", raw, re.S)
        if not match:
            # fallback: cerca prima/ultima parentesi quadra
            start = raw.find("[")
            end = raw.rfind("]")
            if start == -1 or end == -1 or end <= start:
                raise HTTPException(status_code=500, detail=f"No JSON array found in model output: {raw}")
            raw_json = raw[start:end+1]
        else:
            raw_json = match.group(0)

        try:
            quiz = json.loads(raw_json)
        except Exception:
            raise HTTPException(status_code=500, detail=f"Invalid JSON from model: {raw_json}")

        if not isinstance(quiz, list):
            raise HTTPException(status_code=500, detail="Quiz is not a JSON array")

        # validazione minima
        for i, q in enumerate(quiz):
            if not isinstance(q, dict):
                raise HTTPException(status_code=500, detail=f"Quiz item {i} is not an object")
            if "options" not in q or not isinstance(q["options"], list) or len(q["options"]) != 4:
                raise HTTPException(status_code=500, detail=f"Quiz item {i} must have 4 options")
            if "correct" not in q or q["correct"] not in q["options"]:
                raise HTTPException(status_code=500, detail=f"Quiz item {i} correct must match one option")

        return {"quiz": quiz}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# =====================
# EXTENDED EXPLANATION
# =====================
@app.post("/extended_explanation")
def extended_explanation(req: ExtendedExplanationRequest, api_key: str = Depends(verify_api_key)):
    prompt = f"""
Write an extended, step-by-step, UWorld-style explanation.

Language: {req.language}

Question:
{req.question}

Correct answer:
{req.correct}

Short explanation (if available):
{req.explanation or ""}

Requirements:
- Explain why the correct option is correct
- Explain why each wrong option is wrong (brief but clear)
- Include high-yield takeaways at the end
- Do not output JSON, output plain text only
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a top medical educator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return {"explanation": response.choices[0].message.content or ""}
