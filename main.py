import os
import uuid

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


class IngestRequest(BaseModel):

    documents: list[IngestDocument]


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

def create_project(

    data: ProjectCreate,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    project_id = str(uuid.uuid4())

    db.execute(

        sql_text("""

        insert into projects

        (id, name)

        values

        (:id, :name)

        """),

        {

            "id": project_id,

            "name": data.name

        }

    )

    db.commit()

    db.close()

    return {"project_id": project_id}


# =====================
# LIST DOCUMENTS
# =====================

@app.get("/projects/{project_id}/documents")

def list_documents(

    project_id: str,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    try:

        rows = db.execute(

            sql_text("""

            select distinct doc_title

            from chunks

            where project_id = :project_id

            order by doc_title

            """),

            {

                "project_id": project_id

            }

        ).fetchall()

        documents = [

            {

                "title": r[0]

            }

            for r in rows

        ]

        return {

            "documents": documents

        }

    finally:

        db.close()


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

def ingest(

    project_id: str,

    data: IngestRequest,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    try:

        for doc in data.documents:

            vectors = embed_texts([doc.text])

            db.execute(

                sql_text("""

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

                    "doc_title": doc.title,

                    "chunk_text": doc.text,

                    "embedding": vectors[0]

                }

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
# GENERATE QUIZ
# =====================

@app.post("/projects/{project_id}/generate_quiz")

def generate_quiz(

    project_id: str,

    req: QuizRequest,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    try:

        rows = db.execute(

            sql_text("""

            select chunk_text

            from chunks

            where project_id = :project_id

            limit 8

            """),

            {

                "project_id": project_id

            }

        ).fetchall()

        context = "\n".join([r[0] for r in rows])


        prompt = f"""

Create {req.num_questions} multiple choice questions.

Language: {req.language}

Difficulty: {req.difficulty}

Material:

{context}

Return JSON.

"""


        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {

                    "role": "system",

                    "content": "You create quizzes."

                },

                {

                    "role": "user",

                    "content": prompt

                }

            ]

        )


        return {

            "quiz": response.choices[0].message.content

        }

    finally:

        db.close()
