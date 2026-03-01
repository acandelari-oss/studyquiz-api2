import os
import uuid
import json

from fastapi import FastAPI, Depends, HTTPException, Header
from pydantic import BaseModel

from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

from openai import OpenAI


# ENV

DATABASE_URL=os.environ["DATABASE_URL"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY=os.environ["BACKEND_API_KEY"]


# INIT

app=FastAPI()

client=OpenAI(api_key=OPENAI_API_KEY)

engine=create_engine(DATABASE_URL)

SessionLocal=sessionmaker(bind=engine)


# AUTH

def verify_api_key(authorization:str=Header(None)):

    if authorization!=f"Bearer {BACKEND_API_KEY}":

        raise HTTPException(status_code=401,detail="Invalid API key")


# MODELS

class ProjectCreate(BaseModel):

    name:str


class IngestDocument(BaseModel):

    title:str
    text:str


class IngestRequest(BaseModel):

    documents:list[IngestDocument]


class QuizRequest(BaseModel):

    num_questions:int
    language:str
    difficulty:str
    group_by_macro_topics:bool
    answers_at_end:bool


# CREATE PROJECT

@app.post("/projects")

def create_project(

    data:ProjectCreate,

    api_key:str=Depends(verify_api_key)

):

    db=SessionLocal()

    pid=str(uuid.uuid4())

    db.execute(

        sql_text("""

        insert into projects

        (id,name)

        values

        (:id,:name)

        """),

        {"id":pid,"name":data.name}

    )

    db.commit()

    db.close()

    return {"project_id":pid}


# LIST DOCUMENTS

@app.get("/projects/{project_id}/documents")

def list_documents(

    project_id:str,

    api_key:str=Depends(verify_api_key)

):

    db=SessionLocal()

    rows=db.execute(

        sql_text("""

        select distinct doc_title

        from chunks

        where project_id=:pid

        """),

        {"pid":project_id}

    ).fetchall()

    db.close()

    return {

        "documents":[r[0] for r in rows]

    }


# EMBEDDINGS

def embed(text):

    res=client.embeddings.create(

        model="text-embedding-3-small",

        input=text

    )

    return res.data[0].embedding


# INGEST

@app.post("/projects/{project_id}/ingest")

def ingest(

    project_id:str,

    data:IngestRequest,

    api_key:str=Depends(verify_api_key)

):

    db=SessionLocal()

    try:

        for doc in data.documents:

            emb=embed(doc.text)

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

                :pid,

                :doc_id,

                :title,

                :text,

                CAST(:emb AS vector)

                )

                """),

                {

                    "pid":project_id,

                    "doc_id":str(uuid.uuid4()),

                    "title":doc.title,

                    "text":doc.text,

                    "emb":emb

                }

            )

        db.commit()

        return {"status":"ok"}

    except Exception as e:

        db.rollback()

        raise HTTPException(

            status_code=500,

            detail=str(e)

        )

    finally:

        db.close()


# GENERATE QUIZ

@app.post("/projects/{project_id}/generate_quiz")

def generate_quiz(

    project_id:str,

    req:QuizRequest,

    api_key:str=Depends(verify_api_key)

):

    db=SessionLocal()

    rows=db.execute(

        sql_text("""

        select chunk_text,doc_title

        from chunks

        where project_id=:pid

        limit 8

        """),

        {"pid":project_id}

    ).fetchall()

    db.close()

    if not rows:

        raise HTTPException(

            status_code=400,

            detail="No material"

        )


    context="\n\n".join(

        [

            f"{r[1]}:\n{r[0]}"

            for r in rows

        ]

    )


    prompt=f"""

Create {req.num_questions} quiz questions.

Language: {req.language}

Difficulty: {req.difficulty}

Return ONLY JSON array:

[
{{
"question":"",
"options":["A","B","C","D"],
"correct":"",
"explanation":"",
"source":"file name",
"page":1
}}
]

Material:

{context}

"""


    response=client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[

            {"role":"system","content":"Return only JSON"},

            {"role":"user","content":prompt}

        ]

    )


    raw=response.choices[0].message.content


    try:

        quiz=json.loads(raw)

    except:

        raise HTTPException(

            status_code=500,

            detail=f"Invalid JSON: {raw}"

        )


    return {"quiz":quiz}
