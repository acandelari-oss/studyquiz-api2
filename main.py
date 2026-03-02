import os
import uuid
import json

from typing import List, Optional

from fastapi import FastAPI, Depends, HTTPException, Header

from pydantic import BaseModel

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from openai import OpenAI

from dotenv import load_dotenv


# LOAD ENV
load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_API_KEY = os.getenv("BACKEND_API_KEY")


if not DATABASE_URL:
    raise Exception("DATABASE_URL missing")


# APP
app = FastAPI()

client = OpenAI(api_key=OPENAI_API_KEY)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)


# AUTH
def verify_api_key(authorization: str = Header(None)):

    if authorization != f"Bearer {BACKEND_API_KEY}":

        raise HTTPException(status_code=401, detail="Invalid API key")


# MODELS
class ProjectCreate(BaseModel):

    name: str


class QuizRequest(BaseModel):

    num_questions: int

    difficulty: str

    language: str


class IngestDocument(BaseModel):

    title: str

    text: str


class IngestRequest(BaseModel):

    documents: List[IngestDocument]


# HEALTH
@app.get("/health")

def health():

    return {"status": "ok"}


# CREATE PROJECT
@app.post("/projects")

def create_project(

    data: ProjectCreate,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    project_id = str(uuid.uuid4())

    db.execute(

        text("""

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

    # IMPORTANTISSIMO

    return {

        "project_id": project_id,

        "name": data.name

    }


# LIST PROJECTS
@app.get("/projects")

def list_projects(

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    rows = db.execute(

        text("""

        select id, name

        from projects

        order by name

        """)

    ).fetchall()

    db.close()

    return {

        "projects":

        [

            {

                "id": r[0],

                "name": r[1]

            }

            for r in rows

        ]

    }


# INGEST
@app.post("/projects/{project_id}/ingest")

def ingest(

    project_id: str,

    data: IngestRequest,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    try:

        for doc in data.documents:

            embedding = client.embeddings.create(

                model="text-embedding-3-small",

                input=doc.text

            ).data[0].embedding


            db.execute(

                text("""

                insert into chunks

                (

                    project_id,

                    doc_title,

                    chunk_text,

                    embedding

                )

                values

                (

                    :project_id,

                    :doc_title,

                    :chunk_text,

                    CAST(:embedding AS vector)

                )

                """),

                {

                    "project_id": project_id,

                    "doc_title": doc.title,

                    "chunk_text": doc.text,

                    "embedding": embedding

                }

            )

        db.commit()

        return {"status": "ok"}


    except Exception as e:

        db.rollback()

        raise HTTPException(status_code=500, detail=str(e))


    finally:

        db.close()


# GENERATE QUIZ
@app.post("/projects/{project_id}/generate_quiz")

def generate_quiz(

    project_id: str,

    req: QuizRequest,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    rows = db.execute(

        text("""

        select chunk_text

        from chunks

        where project_id = :project_id

        limit 5

        """),

        {

            "project_id": project_id

        }

    ).fetchall()

    db.close()


    context = "\n\n".join(

        r[0]

        for r in rows

    )


    prompt = f"""

Generate {req.num_questions} questions.

Difficulty: {req.difficulty}

Language: {req.language}


Format STRICT JSON:

[

{{
question:

options: ["A","B","C","D","E"]

correct:

explanation:

}}

]

Material:

{context}

"""


    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {

                "role":"user",

                "content":prompt

            }

        ]

    )


    return {

        "quiz":

        response.choices[0].message.content

    }