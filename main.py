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

            document_id = str(uuid.uuid4())

            # SAVE DOCUMENT

            db.execute(sql_text("""

                insert into documents

                (id, project_id, filename)

                values

                (:id, :project_id, :filename)

            """), {

                "id": document_id,

                "project_id": project_id,

                "filename": doc.title

            })


            chunks = [doc.text]

            vectors = embed_texts(chunks)


            for chunk_text, vector in zip(chunks, vectors):

                db.execute(sql_text("""

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

                """), {

                    "project_id": project_id,

                    "doc_id": document_id,

                    "doc_title": doc.title,

                    "chunk_text": chunk_text,

                    "embedding": vector

                })


        db.commit()

        return {"status": "ok"}


    except Exception as e:

        db.rollback()

        raise HTTPException(status_code=500, detail=str(e))


    finally:

        db.close()


# =====================
# LIST DOCUMENTS
# =====================

@app.get("/projects/{project_id}/documents")

def list_documents(

    project_id: str,

    api_key: str = Depends(verify_api_key)

):

    db = SessionLocal()

    rows = db.execute(sql_text("""

        select id, filename, uploaded_at

        from documents

        where project_id = :project_id

        order by uploaded_at desc

    """), {

        "project_id": project_id

    }).fetchall()

    db.close()

    return {

        "documents": [

            {

                "id": r[0],

                "filename": r[1],

                "uploaded_at": str(r[2])

            }

            for r in rows

        ]

    }


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

        query_vector = embed_texts(["quiz"])[0]

        rows = db.execute(

            sql_text("""

            select chunk_text

            from chunks

            where project_id = :project_id

            order by embedding <=> CAST(:query_vector AS vector)

            limit 8

            """),

            {

                "project_id": project_id,

                "query_vector": query_vector

            }

        ).fetchall()


        context = "\n\n".join([r[0] for r in rows])


        prompt = f"""

Create {req.num_questions} multiple choice questions.

Material:

{context}

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


    except Exception as e:

        raise HTTPException(status_code=500, detail=str(e))


    finally:

        db.close()
