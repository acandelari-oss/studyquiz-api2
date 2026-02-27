import os
import uuid
import json
from typing import Optional, List, Dict, Any

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

engine = create_engine(DATABASE_URL)
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


class IngestDocument(BaseModel):
    title: str
    text: str
    page: Optional[int] = None


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class QuizRequest(BaseModel):
    num_questions: int
    language: str
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool


# =====================
# HEALTH
# =====================
@app.get("/health")
def health():
    return {"status": "ok"}


# =====================
# CREATE PROJECT
# =====================
@app.post("/projects")
def create_project(data: ProjectCreate, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        pid = str(uuid.uuid4())
        db.execute(
            sql_text("""
                insert into projects (id, name)
                values (:id, :name)
            """),
            {"id": pid, "name": data.name},
        )
        db.commit()
        return {"project_id": pid}
    finally:
        db.close()


# =====================
# LIST DOCUMENTS
# =====================
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
            {"pid": project_id},
        ).fetchall()

        return {"documents": [{"title": r[0]} for r in rows]}
    finally:
        db.close()


# =====================
# EMBEDDINGS
# =====================
def embed_texts(texts: List[str]) -> List[List[float]]:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in res.data]


# =====================
# INGEST
# =====================
@app.post("/projects/{project_id}/ingest")
def ingest(project_id: str, data: IngestRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        for doc in data.documents:
            # For now: 1 chunk = whole text (later we can chunk + store per page)
            chunk_text = doc.text.strip()
            if not chunk_text:
                continue

            vector = embed_texts([chunk_text])[0]

            db.execute(
                sql_text("""
                    insert into chunks
                    (project_id, doc_id, doc_title, chunk_text, embedding, page)
                    values
                    (:project_id, :doc_id, :doc_title, :chunk_text, CAST(:embedding AS vector), :page)
                """),
                {
                    "project_id": project_id,
                    "doc_id": str(uuid.uuid4()),
                    "doc_title": doc.title,
                    "chunk_text": chunk_text,
                    "embedding": vector,
                    "page": doc.page,
                },
            )

        db.commit()
        return {"status": "ok"}
    except Exception as e:
        db.rollback()
        print("INGEST ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# =====================
# QUIZ JSON helpers
# =====================
def extract_json_or_raise(raw: str) -> Dict[str, Any]:
    """
    Tries to parse JSON.
    If model returns text with code fences or extra text, we try to extract first JSON block.
    """
    raw = raw.strip()

    # remove ```json fences if present
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.replace("json", "", 1).strip()

    # direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # try to find a JSON object inside the text
    first_brace = raw.find("{")
    last_brace = raw.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        candidate = raw[first_brace:last_brace + 1]
        try:
            return json.loads(candidate)
        except Exception:
            pass

    raise ValueError("Model output is not valid JSON")


# =====================
# GENERATE QUIZ
# =====================
@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(project_id: str, req: QuizRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        # pull some context (you can later switch to vector search again)
        rows = db.execute(
            sql_text("""
                select chunk_text, doc_title, page
                from chunks
                where project_id = :pid
                limit 12
            """),
            {"pid": project_id},
        ).fetchall()

        if not rows:
            raise HTTPException(status_code=400, detail="No study material found. Upload at least one file.")

        context = "\n\n".join(
            [
                f"[SOURCE_FILE: {r[1]} | PAGE: {r[2]}]\n{r[0]}"
                for r in rows
            ]
        )

        # Force JSON output with a strict schema-like instruction
        prompt = f"""
You MUST return ONLY valid JSON (no markdown, no code fences).

Create exactly {req.num_questions} multiple choice questions based ONLY on the provided material.

Rules:
- language: {req.language}
- difficulty: {req.difficulty}
- Each question must have exactly 4 options.
- Exactly 1 correct answer; correct_answer MUST match one of the options exactly.
- explanation should be concise but clear.
- source_file and source_page must match the material tags you see in context (SOURCE_FILE / PAGE).

Return JSON in this format:
{{
  "questions": [
    {{
      "question": "....",
      "options": ["A", "B", "C", "D"],
      "correct_answer": "B",
      "explanation": "....",
      "source_file": "....",
      "source_page": 12
    }}
  ]
}}

MATERIAL:
{context}
"""

        # Ask model (and request JSON object if supported)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return strictly valid JSON only."},
                {"role": "user", "content": prompt},
            ],
            # If your OpenAI library supports it, this helps a lot:
            response_format={"type": "json_object"},
        )

        raw = resp.choices[0].message.content or ""

        try:
            parsed = extract_json_or_raise(raw)
        except Exception as e:
            # Return useful debug (first 600 chars) instead of generic 500
            snippet = raw[:600].replace("\n", "\\n")
            raise HTTPException(
                status_code=500,
                detail=f"Quiz generation produced invalid JSON. Error: {str(e)}. Raw snippet: {snippet}"
            )

        # Basic validation: ensure questions array exists
        questions = parsed.get("questions")
        if not isinstance(questions, list) or len(questions) == 0:
            raise HTTPException(status_code=500, detail="Quiz JSON missing 'questions' array or it's empty.")

        return parsed

    except HTTPException:
        raise
    except Exception as e:
        print("QUIZ ERROR:", repr(e))
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
