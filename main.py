import os
import uuid
import json

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel

from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

from openai import OpenAI


DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY = os.environ["BACKEND_API_KEY"]


app = FastAPI()

client = OpenAI(api_key=OPENAI_API_KEY)

engine = create_engine(DATABASE_URL)

SessionLocal = sessionmaker(bind=engine)


def verify_api_key(authorization: str = Header(None)):

    if authorization != f"Bearer {BACKEND_API_KEY}":

        raise HTTPException(status_code=401, detail="Invalid API key")


class ProjectCreate(BaseModel):

    name: str


class IngestDocument(BaseModel):

    title: str
    text: str
    page: int | None = None


class IngestRequest(BaseModel):

    documents: list[IngestDocument]


class QuizRequest(BaseModel):

    num_questions: int
    language: str
    difficulty: str
    group_by_macro_topics: bool
    answers_at_end: bool


@app.get("/projects/{project_id}/documents")

def list_documents(project_id: str, api_key: str = Depends(verify_api_key)):

    db = SessionLocal()

    rows = db.execute(

        sql_text("""

        select distinct doc_title

        from chunks

        where project_id=:pid

        order by doc_title

        """),

        {"pid": project_id}

    ).fetchall()

    db.close()

    return {

        "documents":

        [

            {"title": r[0]}

            for r in rows

        ]

    }


@app.post("/projects")

def create_project(data: ProjectCreate, api_key: str = Depends(verify_api_key)):

    db = SessionLocal()

    pid = str(uuid.uuid4())

    db.execute(

        sql_text("""

        insert into projects

        (id,name)

        values

        (:id,:name)

        """),

        {"id": pid, "name": data.name}

    )

    db.commit()

    db.close()

    return {"project_id": pid}


def embed(text):

    res = client.embeddings.create(

        model="text-embedding-3-small",

        input=text

    )

    return res.data[0].embedding


@app.post("/projects/{project_id}/ingest")

def ingest(project_id: str, req: IngestRequest, api_key: str = Depends(verify_api_key)):

    db = SessionLocal()

    for doc in req.documents:

        emb = embed(doc.text)

        db.execute(

            sql_text("""

            insert into chunks

            (

            project_id,

            doc_id,

            doc_title,

            chunk_text,

            embedding,

            page

            )

            values

            (

            :pid,

            :docid,

            :title,

            :text,

            cast(:emb as vector),

            :page

            )

            """),

            {

                "pid": project_id,

                "docid": str(uuid.uuid4()),

                "title": doc.title,

                "text": doc.text,

                "emb": emb,

                "page": doc.page

            }

        )

    db.commit()

    db.close()

    return {"status": "ok"}


@app.post("/projects/{project_id}/generate_quiz")

def generate_quiz(project_id: str, req: QuizRequest, api_key: str = Depends(verify_api_key)):

    db = SessionLocal()

    rows = db.execute(

        sql_text("""

        select chunk_text,doc_title,page

        from chunks

        where project_id=:pid

        limit 10

        """),

        {"pid": project_id}

    ).fetchall()

    db.close()

    context = "\n".join(

        [

            f"{r[0]} (source:{r[1]} page:{r[2]})"

            for r in rows

        ]

    )

    prompt = f"""

Create {req.num_questions} MCQ.

Return JSON:

questions:

question

options

correct_answer

explanation

source_file

source_page

Material:

{context}

"""

    res = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {"role": "system", "content": "Return valid JSON only"},

            {"role": "user", "content": prompt}

        ]

    )

    return json.loads(res.choices[0].message.content)
