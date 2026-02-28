# ADD these fields in chunks table:

# doc_title
# page_number


# ======================
# GENERATE QUIZ UPDATED
# ======================


@app.post("/projects/{project_id}/generate_quiz")

def generate_quiz(project_id: str, req: QuizRequest, api_key: str = Depends(verify_api_key)):


    db = SessionLocal()


    try:


        query_vector = embed_texts(["quiz"])[0]


        rows = db.execute(sql_text("""

        select

            chunk_text,
            doc_title,
            page_number

        from chunks

        where project_id = :project_id

        order by embedding <=> CAST(:query_vector AS vector)

        limit 10

        """),

        {

            "project_id": project_id,
            "query_vector": query_vector

        }).fetchall()



        context = "\n\n".join([

            f"""

SOURCE: {r.doc_title}

PAGE: {r.page_number}

TEXT:

{r.chunk_text}

"""

            for r in rows

        ])



        prompt = f"""

Create {req.num_questions} questions.


STRICT FORMAT JSON:

[

{{
question:

answers:

correct:

explanation:

explanation_extended:

source_file:

page:

}}

]

Material:

{context}

"""



        response = client.chat.completions.create(

            model="gpt-4o-mini",

            messages=[

                {

                    "role": "system",

                    "content":

                    "You create medical exam questions with full explanations and source."

                },

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            temperature=0.3

        )



        return {

            "quiz": response.choices[0].message.content

        }



    finally:

        db.close()
