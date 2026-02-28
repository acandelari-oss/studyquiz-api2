import os
import uuid
import json
from typing import List, Literal, Optional

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel, Field

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


class IngestDocument(BaseModel):
    title: str
    text: str


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


Difficulty = Literal["high", "medium", "low"]


class QuizRequest(BaseModel):
    num_questions: int = Field(ge=1, le=200)
    language: str = "en"
    difficulty: Difficulty = "high"
    group_by_macro_topics: bool = True
    answers_at_end: bool = True  # exam mode: yes
    timer_minutes: Optional[int] = Field(default=None, ge=1, le=300)


# output schema (what we want from the model)
class QuizChoice(BaseModel):
    label: Literal["A", "B", "C", "D"]
    text: str


class QuizQuestion(BaseModel):
    id: str
    macro_topic: str
    stem: str
    choices: List[QuizChoice]  # always 4
    correct_label: Literal["A", "B", "C", "D"]
    explanation: str


class QuizPayload(BaseModel):
    title: str
    language: str
    difficulty: str
    questions: List[QuizQuestion]


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
    project_id = str(uuid.uuid4())

    db.execute(
        sql_text("""
        insert into projects (id, name)
        values (:id, :name)
        """),
        {"id": project_id, "name": data.name},
    )
    db.commit()
    db.close()
    return {"project_id": project_id}


# =====================
# EMBEDDINGS
# =====================
def embed_texts(texts):
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [r.embedding for r in response.data]


# =====================
# INGEST
# =====================
@app.post("/projects/{project_id}/ingest")
def ingest(project_id: str, data: IngestRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        for doc in data.documents:
            document_text = doc.text or ""
            document_title = doc.title or "Untitled"

            # naive: 1 chunk. (you can chunk later)
            chunks = [document_text]
            vectors = embed_texts(chunks)

            for chunk_text, vector in zip(chunks, vectors):
                db.execute(
                    sql_text("""
                    insert into chunks (
                      project_id,
                      doc_id,
                      doc_title,
                      chunk_text,
                      embedding
                    )
                    values (
                      :project_id,
                      :doc_id,
                      :doc_title,
                      :chunk_text,
                      CAST(:embedding AS vector)
                    )
                    """),
                    {
                        "project_id": project_id,
                        "doc_id": str(uuid.uuid4()),
                        "doc_title": document_title,
                        "chunk_text": chunk_text,
                        "embedding": vector,
                    },
                )

        db.commit()
        return {"status": "ok"}

    except Exception as e:
        db.rollback()
        print("INGEST ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()


# =====================
# GENERATE QUIZ (JSON)
# =====================
@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(project_id: str, req: QuizRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()
    try:
        # retrieve relevant context
        query_vector = embed_texts(["create an exam style quiz"])[0]
        rows = db.execute(
            sql_text("""
            select chunk_text
            from chunks
            where project_id = :project_id
            order by embedding <=> CAST(:query_vector AS vector)
            limit 10
            """),
            {"project_id": project_id, "query_vector": query_vector},
        ).fetchall()

        if not rows:
            raise HTTPException(status_code=400, detail="No study material found")

        context = "\n\n".join([r[0] for r in rows])

        # strong, strict JSON instruction
        system = (
            "You are an expert exam-writer. "
            "You MUST output ONLY valid JSON, no markdown, no commentary, no code fences."
        )

        user_prompt = f"""
Create an EXAM MODE multiple-choice quiz based ONLY on the material.

Rules:
- Language: {req.language}
- Difficulty: {req.difficulty} (make it genuinely challenging)
- Number of questions: {req.num_questions}
- Focus: cell biology only (exclude virology/virus-related content)
- No references to slides/pages/numbers (no 'in slide 12' etc.)
- Each question must have exactly 4 options labeled A, B, C, D
- Exactly ONE correct option
- All options must be plausible and relevant to the stem (no random distractors)
- No duplicated questions, no duplicated answer options within the same question
- Group questions by macro topics (put macro_topic on each question)
- Provide an explanation for the correct answer (concise but high quality)
- Do NOT reveal correct answers inside stem/choices; only put it in correct_label field

Output JSON schema EXACTLY:
{{
  "title": "string",
  "language": "string",
  "difficulty": "string",
  "questions": [
    {{
      "id": "q1",
      "macro_topic": "string",
      "stem": "string",
      "choices": [
        {{ "label": "A", "text": "..." }},
        {{ "label": "B", "text": "..." }},
        {{ "label": "C", "text": "..." }},
        {{ "label": "D", "text": "..." }}
      ],
      "correct_label": "A|B|C|D",
      "explanation": "string"
    }}
  ]
}}

Material:
{context}
"""

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_prompt},
            ],
        )

        raw = resp.choices[0].message.content or ""
        # Parse + validate JSON
        try:
            payload_dict = json.loads(raw)
            payload = QuizPayload(**payload_dict)
        except Exception as e:
            print("QUIZ JSON PARSE/VALIDATION ERROR:", e)
            # return raw for debugging
            raise HTTPException(status_code=500, detail=f"Model returned invalid JSON. Raw: {raw[:800]}")

        return {"quiz": payload_dict}

    except HTTPException:
        raise
    except Exception as e:
        print("QUIZ ERROR:", e)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
