import os
import uuid
import json

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
    file_name: str | None = None
    page_number: int | None = None


class IngestRequest(BaseModel):
    documents: list[IngestDocument]


class QuizRequest(BaseModel):
    num_questions: int
    language: str
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool


class ExpandExplanationRequest(BaseModel):
    question: str
    options: dict
    correct: str
    short_explanation: str
    source_file: str | None = None
    source_page: int | None = None
    source_excerpt: str | None = None


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
        {"id": project_id, "name": data.name}
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
            chunks = [doc.text]
            vectors = embed_texts(chunks)

            for chunk_text, vector in zip(chunks, vectors):
                db.execute(
                    sql_text("""
                    insert into chunks
                    (
                      project_id,
                      doc_id,
                      doc_title,
                      chunk_text,
                      embedding,
                      file_name,
                      page_number
                    )
                    values
                    (
                      :project_id,
                      :doc_id,
                      :doc_title,
                      :chunk_text,
                      CAST(:embedding AS vector),
                      :file_name,
                      :page_number
                    )
                    """),
                    {
                        "project_id": project_id,
                        "doc_id": str(uuid.uuid4()),
                        "doc_title": doc.title,
                        "chunk_text": chunk_text,
                        "embedding": vector,
                        "file_name": doc.file_name,
                        "page_number": doc.page_number
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
# GENERATE QUIZ (JSON + SOURCE)
# =====================

@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(project_id: str, req: QuizRequest, api_key: str = Depends(verify_api_key)):
    db = SessionLocal()

    try:
        # Retrieve top chunks via vector search
        query_vector = embed_texts(["quiz generation query"])[0]

        rows = db.execute(
            sql_text("""
            select
              chunk_text,
              file_name,
              page_number
            from chunks
            where project_id = :project_id
            order by embedding <=> CAST(:query_vector AS vector)
            limit 12
            """),
            {"project_id": project_id, "query_vector": query_vector}
        ).fetchall()

        if not rows:
            raise HTTPException(status_code=400, detail="No study material found")

        # Build context with source labels
        # We keep each chunk tagged with its file/page so the model can cite it.
        context_blocks = []
        for (chunk_text, file_name, page_number) in rows:
            src = f"Source: {file_name or 'Unknown file'} | Page: {page_number or 'Unknown'}"
            context_blocks.append(f"{src}\n{chunk_text}")

        context = "\n\n---\n\n".join(context_blocks)

        prompt = f"""
Create a multiple-choice quiz and return ONLY valid JSON.

JSON schema (strict):
{{
  "questions": [
    {{
      "question": "string",
      "options": {{
        "A": "string",
        "B": "string",
        "C": "string",
        "D": "string"
      }},
      "correct": "A|B|C|D",
      "explanation": "string (short explanation, 2-4 sentences)",
      "source_file": "string | null",
      "source_page": "number | null",
      "source_excerpt": "string (a short excerpt from the provided context that supports the answer, max 250 chars)"
    }}
  ]
}}

Rules:
- Number of questions: {req.num_questions}
- Difficulty: {req.difficulty}
- Language: {req.language}
- Options must be plausible and relevant (no nonsense distractors).
- Exactly 4 options (A-D), exactly 1 correct.
- Do NOT include any references like "slide 56" or "as shown above".
- Use ONLY the provided context.
- For each question choose ONE best source from the context and fill source_file/source_page and a short source_excerpt from that source.

Context:
{context}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You generate high-quality quizzes. Return only JSON."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        content = response.choices[0].message.content
        quiz_json = json.loads(content)
        return quiz_json

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        db.close()


# =====================
# EXPAND EXPLANATION (MORE DETAILS)
# =====================

@app.post("/projects/{project_id}/expand_explanation")
def expand_explanation(project_id: str, req: ExpandExplanationRequest, api_key: str = Depends(verify_api_key)):
    try:
        prompt = f"""
You are helping a student understand a quiz question in depth.
Return ONLY plain text.

Question:
{req.question}

Options:
A) {req.options.get("A")}
B) {req.options.get("B")}
C) {req.options.get("C")}
D) {req.options.get("D")}

Correct answer: {req.correct}

Short explanation:
{req.short_explanation}

Source (from study material):
File: {req.source_file}
Page: {req.source_page}
Excerpt: {req.source_excerpt}

Task:
Write an expanded explanation with:
- A deeper conceptual explanation
- Why the correct option is correct
- Why each incorrect option is wrong
- 1–2 memory tips / mnemonics if appropriate
Keep it rigorous and aligned with the excerpt. Do not invent facts not supported by the excerpt.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You explain biomedical concepts clearly and rigorously."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2
        )

        return {"expanded_explanation": response.choices[0].message.content}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
