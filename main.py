from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import openai
import uuid
import os

# ======================
# CONFIG
# ======================

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY = os.environ["BACKEND_API_KEY"]

openai.api_key = OPENAI_API_KEY

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_recycle=300,
)

SessionLocal = sessionmaker(bind=engine)

app = FastAPI()
security = HTTPBearer()


# ======================
# AUTH
# ======================

def require_key(
    credentials: HTTPAuthorizationCredentials = Depends(security),
):
    if credentials.credentials != BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ======================
# MODELS
# ======================

class ProjectCreate(BaseModel):
    name: str


class IngestDoc(BaseModel):
    doc_id: str
    title: str
    text: str


class IngestRequest(BaseModel):
    project_id: str
    documents: list[IngestDoc]


class QuizRequest(BaseModel):
    num_questions: int
    language: str
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool


# ======================
# HELPERS
# ======================

def embed(texts: list[str]) -> list[list[float]]:
    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [d.embedding for d in res.data]


# ======================
# ROUTES
# ======================

@app.post("/projects")
def create_project(
    payload: ProjectCreate,
    _: HTTPAuthorizationCredentials = Depends(require_key),
):
    project_id = str(uuid.uuid4())
    return {"project_id": project_id}


@app.post("/projects/{project_id}/ingest")
def ingest(
    project_id: str,
    payload: IngestRequest,
    _: HTTPAuthorizationCredentials = Depends(require_key),
):
    db = SessionLocal()

    try:
        chunks = []
        for doc in payload.documents:
            chunks.append(doc.text)

        vectors = embed(chunks)

        for text_chunk, vec in zip(chunks, vectors):
            db.execute(
                text("""
                    insert into chunks (project_id, chunk_text, embedding)
                    values (:pid, :text, :emb)
                """),
                {
                    "pid": project_id,
                    "text": text_chunk,
                    "emb": vec,
                },
            )

        db.commit()
        return {"status": "ok"}

    finally:
        db.close()


@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(
    project_id: str,
    payload: QuizRequest,
    _: HTTPAuthorizationCredentials = Depends(require_key),
):
    db = SessionLocal()

    try:
        query_embedding = embed(
            [f"Generate a {payload.difficulty} quiz"]
        )[0]

        rows = db.execute(
            text("""
                select chunk_text
                from chunks
                where project_id = :pid
                order by embedding <=> (:emb)::vector
                limit 8
            """),
            {
                "pid": project_id,
                "emb": query_embedding,
            },
        ).fetchall()

        if not rows:
            raise HTTPException(
                status_code=400,
                detail="No content ingested for this project",
            )

        context = "\n".join(r[0] for r in rows)

        prompt = f"""
Create a {payload.difficulty} difficulty quiz in {payload.language}.
Use ONLY the content below.

{context}

Rules:
- {payload.num_questions} questions
- 4 options per question
- One correct answer
- No references to slides
- Biology only
- {"Group by macro topics" if payload.group_by_macro_topics else ""}
- {"Put answers at the end" if payload.answers_at_end else ""}
"""

        completion = openai.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        return {
            "quiz_text": completion.choices[0].message.content
        }

    finally:
        db.close()
