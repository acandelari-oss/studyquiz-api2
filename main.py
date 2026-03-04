import os
import uuid
import base64
import io
from typing import List

import requests
from fastapi import FastAPI, Depends, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader


# ======================
# LOAD ENV
# ======================

load_dotenv()

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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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


class IngestDocument(BaseModel):
    title: str
    file_bytes: str  # PDF base64


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


# ======================
# HEALTH
# ======================

@app.get("/health")
def health():
    return {"status": "ok"}


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
        "documents": [{"title": r[0]} for r in rows]
    }


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
        for doc in data.documents:

            pdf_bytes = base64.b64decode(doc.file_bytes)
            pdf_stream = io.BytesIO(pdf_bytes)

            reader = PdfReader(pdf_stream)

            for page_index, page in enumerate(reader.pages):
                page_text = page.extract_text()

                if not page_text or not page_text.strip():
                    continue

                embedding = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=page_text
                ).data[0].embedding

                db.execute(
                    text("""
                        insert into chunks
                        (project_id, doc_title, chunk_text, embedding, page)
                        values
                        (:project_id, :doc_title, :chunk_text, CAST(:embedding AS vector), :page)
                    """),
                    {
                        "project_id": project_id,
                        "doc_title": doc.title,
                        "chunk_text": page_text,
                        "embedding": embedding,
                        "page": page_index + 1
                    }
                )

        db.commit()

    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=str(e))

    db.close()
    return {"status": "ok"}


# ======================
# GENERATE QUIZ
# ======================

@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(
    project_id: str,
    req: QuizRequest,
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

    # create embedding for the quiz query
query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input="generate quiz questions from the study material"
).data[0].embedding


quiz_results = []

for i in range(req.num_questions):

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=f"study question topic {i}"
    ).data[0].embedding


    rows = db.execute(
        text("""
            select chunk_text, doc_title, page
            from chunks
            where project_id = :project_id
            order by embedding <-> CAST(:embedding AS vector)
            limit 5
        """),
        {
            "project_id": project_id,
            "embedding": query_embedding
        }
    ).fetchall()


    material_blocks = []

    for r in rows:
        material_blocks.append(
            f"FILE: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0]}"
        )

    context = "\n\n---\n\n".join(material_blocks)

    prompt = f"""
You MUST use ONLY the material provided below.

You are NOT allowed to use external knowledge.

If the answer cannot be found in the material,
you must skip the question and generate another one.

Every question MUST be supported by the material.
Every question MUST include the exact source page.

Generate {req.num_questions} high-quality multiple choice questions.
Difficulty: {req.difficulty}
Language: {req.language}

Questions must:
- test understanding of the material
- avoid trivial questions
- avoid repeating the same concept
- cover different parts of the material
Wrong options must be plausible but incorrect.
Avoid obviously wrong answers.
Each question must have EXACTLY 5 options labeled A), B), C), D), E).


Return STRICT JSON:

[
  {{
    "question": "...",
    "options": ["...", "...", "...", "...", "..."],
    "correct": "A",
    "explanation": "Short explanation",
    "explanation_long": "Detailed explanation",
    "source_document": "Exact file name used",
    "source_page": "Page number"
  }}
]

Material:
{context}
"""

    response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": prompt}],
    temperature=0.3
)

content = response.choices[0].message.content.strip()
content = content.replace("```json", "").replace("```", "").strip()

quiz_results.append(content)

   return {"quiz": quiz_results}