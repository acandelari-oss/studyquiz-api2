import os
import uuid
import json

from fastapi import FastAPI, Depends, HTTPException, Header

from pydantic import BaseModel

from sqlalchemy import create_engine, text as sql_text
from sqlalchemy.orm import sessionmaker

from openai import OpenAI


DATABASE_URL=os.environ["DATABASE_URL"]
OPENAI_API_KEY=os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY=os.environ["BACKEND_API_KEY"]


app=FastAPI()

client=OpenAI(api_key=OPENAI_API_KEY)

engine=create_engine(DATABASE_URL)

SessionLocal=sessionmaker(bind=engine)



def verify_api_key(authorization:str=Header(None)):

    if authorization!=f"Bearer {BACKEND_API_KEY}":

        raise HTTPException(status_code=401,detail="Invalid API key")



class QuizRequest(BaseModel):

    num_questions:int
    language:str
    difficulty:str
    group_by_macro_topics:bool
    answers_at_end:bool



def embed(text):

    res=client.embeddings.create(

        model="text-embedding-3-small",

        input=text

    )

    return res.data[0].embedding



@app.post("/projects/{project_id}/generate_quiz")

def generate_quiz(

    project_id:str,

    req:QuizRequest,

    api_key:str=Depends(verify_api_key)

):

    db=SessionLocal()

    try:

        query_vector=embed("quiz")


        rows=db.execute(

            sql_text("""

            select chunk_text

            from chunks

            where project_id=:pid

            limit 8

            """),

            {"pid":project_id}

        ).fetchall()


        if not rows:

            raise HTTPException(

                status_code=400,

                detail="No content"

            )


        context="\n\n".join([r[0] for r in rows])


        prompt=f"""

Create {req.num_questions} quiz questions.

Language: {req.language}

Difficulty: {req.difficulty}

Return ONLY valid JSON in this exact format:

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

                {"role":"system","content":"Return ONLY JSON."},

                {"role":"user","content":prompt}

            ],

            temperature=0.3

        )


        raw=response.choices[0].message.content


        print("OPENAI RAW:",raw)


        try:

            quiz=json.loads(raw)

        except:

            raise HTTPException(

                status_code=500,

                detail=f"Invalid JSON from OpenAI: {raw}"

            )


        if not isinstance(quiz,list):

            raise HTTPException(

                status_code=500,

                detail="Quiz is not list"

            )


        return {"quiz":quiz}


    finally:

        db.close()
