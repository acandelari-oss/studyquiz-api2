from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uuid, os
from sqlalchemy import text
from db import SessionLocal
from rag import chunk_text, embed
from openai import OpenAI

client = OpenAI()
app = FastAPI(title="StudyQuiz API")

API_KEY = os.getenv("QUIZTEST_API_KEY")

def require_key(auth):
    if not auth or not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    if auth.split("Bearer ")[1] != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ---------- SCHEMAS ----------

class CreateProjectIn(BaseModel):
    name: str

class IngestDoc(BaseModel):
    doc_id: str
    title: str
    text: str

class IngestIn(BaseModel):
    documents: list[IngestDoc]

class GenerateQuizIn(BaseModel):
    language: str
    num_questions: int
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool

# ---------- ENDPOINTS ----------

@app.post("/projects")
def create_project(payload: CreateProjectIn, authorization: str = Header(None)):
    require_key(authorization)
    pid = uuid.uuid4()
    db = SessionLocal()
    db.execute(text("insert into projects (id, name) values (:id, :name)"),
               {"id": pid, "name": payload.name})
    db.commit()
    return {"project_id": str(pid)}

@app.post("/projects/{project_id}/ingest")
def ingest(project_id: str, payload: IngestIn, authorization: str = Header(None)):
    require_key(authorization)
    db = SessionLocal()

    for d in payload.documents:
        db.execute(text("""
          insert into documents (id, project_id, title, text)
          values (:id, :pid, :title, :text)
        """), {"id": d.doc_id, "pid": project_id, "title": d.title, "text": d.text})

        chunks = chunk_text(d.text)
        vectors = embed(chunks)

        for c, v in zip(chunks, vectors):
            db.execute(text("""
              insert into chunks (project_id, document_id, doc_title, chunk_text, embedding)
              values (:pid, :did, :title, :text, :emb)
            """), {
                "pid": project_id,
                "did": d.doc_id,
                "title": d.title,
                "text": c,
                "emb": v
            })

    db.commit()
    return {"ingested": len(payload.documents)}

@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(project_id: str, payload: GenerateQuizIn, authorization: str = Header(None)):
    require_key(authorization)
    db = SessionLocal()

    query_embedding = embed(["cell biology advanced quiz"])[0]

    rows = db.execute(text("""
      select chunk_text from chunks
      where project_id = :pid
      order by embedding <=> :emb
      limit 8
    """), {"pid": project_id, "emb": query_embedding}).fetchall()

    context = "\n\n".join(r[0] for r in rows)

    prompt = f"""
Create {payload.num_questions} high-difficulty multiple-choice questions
about CELL BIOLOGY only, based strictly on the material below.

Rules:
- 4 options (A–D), one correct
- No slide/page references
- No viruses
- All options must be plausible
- Answers at the end

Material:
{context}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"quiz_text": resp.choices[0].message.content}
@app.get("/health")
def health():
    return {"ok": True}
