import os
import uuid
from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI

# =========================
# ENV VARIABLES
# =========================

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY = os.environ["BACKEND_API_KEY"]

# =========================
# INIT
# =========================

app = FastAPI()

client = OpenAI(api_key=OPENAI_API_KEY)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)

# =========================
# AUTH
# =========================

def verify_api_key(authorization: str = Header(None)):
    if authorization != f"Bearer {BACKEND_API_KEY}":
        raise HTTPException(status_code=401, detail="Invalid API key")

# =========================
# MODELS
# =========================

class ProjectCreate(BaseModel):
    name: str

class IngestRequest(BaseModel):
    project_id: str
    documents: list

class QuizRequest(BaseModel):
    project_id: str
    num_questions: int
    language: str
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():
    return {"status": "ok"}

# =========================
# CREATE PROJECT
# =========================

@app.post("/projects")
def create_project(
    data: ProjectCreate,
    api_key: str = Depends(verify_api_key)
):
    db = SessionLocal()

    project_id = str(uuid.uuid4())

    db.execute(text("""
        insert into projects
        (id, name)
        values
        (:id, :name)
    """), {
        "id": project_id,
        "name": data.name
    })

    db.commit()

    return {"project_id": project_id}


# =========================
# EMBEDDING FUNCTION
# =========================

def embed(texts):

    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [r.embedding for r in res.data]


# =========================
# INGEST TEXT
# =========================

@app.post("/projects/{project_id}/ingest")
def ingest(
    project_id: str,
    data: IngestRequest,
    api_key: str = Depends(verify_api_key)
):

    db = SessionLocal()

    for doc in data.documents:

        text = doc["text"]

        chunks = [text]

        vectors = embed(chunks)

        for chunk, vector in zip(chunks, vectors):

            doc_id = str(uuid.uuid4())
            doc_title = doc.get("title", "Study Material")

            db.execute(text("""

                insert into chunks
                (
                project_id,
                doc_id,
                doc_title,
                chunk_text,
                embedding
                )

                values
                (
                :pid,
                :doc_id,
                :doc_title,
                :text,
                CAST(:emb AS vector)
                )

            """), {

                "pid": project_id,
                "doc_id": doc_id,
                "doc_title": doc_title,
                "text": chunk,
                "emb": vector

            })

    db.commit()

    return {"status": "ok"}


# =========================
# GENERATE QUIZ
# =========================

@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(
    project_id: str,
    req: QuizRequest,
    api_key: str = Depends(verify_api_key)
):

    db = SessionLocal()

    query_embedding = embed(["generate quiz"])[0]

    rows = db.execute(text("""

        select chunk_text from chunks
        where project_id = :pid
        order by embedding <=> CAST(:emb AS vector)
        limit 8

    """), {
        "pid": project_id,
        "emb": query_embedding
    }).fetchall()

    if not rows:
        raise HTTPException(status_code=400, detail="No material found")

    context = "\n\n".join([r[0] for r in rows])

    prompt = f"""

Create {req.num_questions} multiple choice questions.

Language: {req.language}
Difficulty: {req.difficulty}

Rules:

• Biology only
• No virus questions
• 4 options
• Only one correct answer
• High quality medical exam level

Material:

{context}

Return quiz.

"""

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {
                "role": "system",
                "content": "You create medical quizzes."
            },

            {
                "role": "user",
                "content": prompt
            }

        ],

        temperature=0.7

    )

    quiz = response.choices[0].message.content

    return {

        "quiz": quiz

    }
