import os
import uuid
import base64
import io
from typing import List, Optional
import json
import requests
from fastapi import FastAPI, Depends, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import asyncio
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image


pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"




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
    
    print("AUTH HEADER:", authorization)

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
# DELETE PROJECT
# ======================

@app.delete("/projects/{project_id}")
def delete_project(
    project_id: str,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    # verifica che il progetto appartenga all'utente
    project = db.execute(
        text("""
            select id
            from projects
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
        raise HTTPException(status_code=404, detail="Project not found")

    # cancella i dati collegati
    db.execute(
        text("delete from chunks where project_id = :project_id"),
        {"project_id": project_id}
    )

    db.execute(
        text("delete from quizzes where project_id = :project_id"),
        {"project_id": project_id}
    )

    db.execute(
        text("delete from flashcards where project_id = :project_id"),
        {"project_id": project_id}
    )

    # cancella il progetto
    db.execute(
        text("delete from projects where id = :project_id"),
        {"project_id": project_id}
    )

    db.commit()
    db.close()

    return {"status": "deleted"}
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
# OCR FALLBACK
# ======================

def ocr_pdf_page(pdf_bytes, page_index):

    try:

        images = convert_from_bytes(
            pdf_bytes,
            first_page=page_index + 1,
            last_page=page_index + 1
        )

        if not images:
            return ""

        img = images[0]

        text = pytesseract.image_to_string(img)

        return text.strip()

    except Exception as e:

        print("OCR ERROR:", e)

        return ""

# ======================
# TEXT CHUNKING
# ======================

def chunk_text(text, chunk_size=800, overlap=150):

    chunks = []

    start = 0
    length = len(text)

    while start < length:

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

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

                    print("EMPTY PAGE → OCR FALLBACK")

                    page_text = ocr_pdf_page(pdf_bytes, page_index)

                    if not page_text:
                        continue

                chunks = chunk_text(page_text)

                for chunk in chunks:

                    emb = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )

                    embedding = emb.data[0].embedding
                    embedding_str = "[" + ",".join(map(str, embedding)) + "]"

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
                            "chunk_text": chunk,
                            "embedding": embedding_str,
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


    # ======================
    # RETRIEVAL (UNA SOLA VOLTA)
    # ======================

    print("START EMBEDDING")

    query_embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input="generate study quiz questions"
    ).data[0].embedding

    print("EMBEDDING DONE")

    rows = db.execute(
        text("""
            select chunk_text, doc_title, page
            from chunks
            where project_id = :project_id
            order by embedding <-> CAST(:embedding AS vector)
            limit 25
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
    print("CONTEXT LENGTH:", len(context))
    if len(context) < 100:
        print("WARNING: context too small")


    # ======================
    # QUIZ GENERATION
    # ======================

    quiz_results = []
    used_concepts = []

    remaining = req.num_questions
    batch_size = 20


    while remaining > 0:

        n = min(batch_size, remaining)
        print("CONTEXT LENGTH:", len(context))
        
        prompt = f"""
You MUST use ONLY the material provided below.

You are NOT allowed to use external knowledge.

If the material does NOT contain enough information,
return an empty JSON array: []

Do NOT guess or invent information.

If the answer cannot be found in the material, skip the question.

Avoid generating questions similar to these already generated questions:
{used_concepts}

Material:
{context}

Generate {n} high-quality multiple choice questions.

Difficulty: {req.difficulty}
Language: {req.language}

Questions must:
- test understanding of mechanisms and cause-effect relationships
- prefer application or reasoning questions over definitions
- compare processes when possible
- avoid trivial definition questions
- cover different concepts from the material

Each question must have EXACTLY 5 options.
Distractors must be plausible and conceptually related to the topic.
Avoid obviously wrong answers.
Incorrect options should represent common misconceptions when possible.
Avoid generating multiple questions about the same concept.

Return STRICT JSON ARRAY like this:

[
{{
"question": "...",
"options": ["...","...","...","...","..."],
"correct": "A",
"topic": "...",
"explanation": "...",
"explanation_long": "...",
"source_document": "...",
"source_page": "..."
}}
]


""" 
        print("PROMPT SAMPLE:", prompt[:500])


        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.3
        )


        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()


        parsed = []

        try:
            parsed = json.loads(content)

            if isinstance(parsed, list):
                quiz_results.extend(parsed)
            else:
                quiz_results.append(parsed)

        except Exception as e:
            print("JSON parse error:", e)
            print(content)

        for q in parsed:
            used_concepts.append(q.get("question","")[:120])
        used_concepts.append(content[:120])

        remaining -= n


   

    seen = set()
    unique_questions = []

    for q in quiz_results:

        text_q = q.get("question","").strip().lower()[:80]

        if text_q not in seen:
            seen.add(text_q)
            unique_questions.append(q)

    db.close()

    return {"quiz": unique_questions}

    





class AskRequest(BaseModel):
    project_id: str
    question: str

from typing import Optional

class ActiveRecallRequest(BaseModel):
    topic: Optional[str] = None

@app.post("/projects/{project_id}/generate_quiz_stream")
async def generate_quiz_stream(
    project_id: str,
    req: QuizRequest,
    user = Depends(verify_user)
):
    user_id = user["id"]
    async def quiz_generator():

        db = SessionLocal()

        remaining = req.num_questions
        first_batch = True
        seen_questions = set()

        # RETRIEVAL UNA SOLA VOLTA
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input=f"study material concepts {req.language} {req.difficulty}"
        )

        query_embedding = emb.data[0].embedding

        rows = db.execute(
            text("""
                (
                    select chunk_text, doc_title, page
                    from chunks
                    where project_id = :project_id
                    order by embedding <-> CAST(:embedding AS vector)
                    limit 40
                )

                union

                (
                    select chunk_text, doc_title, page
                    from chunks
                    where project_id = :project_id
                    order by random()
                    limit 40
                )
            """),
            {
                "project_id": project_id,
                "embedding": query_embedding
            }
        ).fetchall()

        db.close()

        material_blocks = []

        for r in rows:
            material_blocks.append(
                f"FILE: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:500]}"
            )

        context = "\n\n---\n\n".join(material_blocks)

        # GENERAZIONE QUIZ
        batch_size = 8
        num_batches = (req.num_questions + batch_size - 1) // batch_size


        async def generate_batch(n):

            prompt = f"""
        You MUST use ONLY the material provided below.

        You are NOT allowed to use external knowledge.

        Material:
        {context}

        Generate {n} high-quality multiple choice study questions.

        IMPORTANT:
        Each question MUST focus on a DIFFERENT concept from the material.

        The quiz must cover as many different topics from the material as possible.

        Avoid generating multiple questions about the same biological mechanism.

        Difficulty: {req.difficulty}
        Language: {req.language}

        Rules:
        - Questions must test understanding of the material
        - Avoid trivial definitions
        - Use different concepts from the material
        - Each question must have EXACTLY 5 options labeled A,B,C,D,E
        - You MUST cite the document and page used
        - Questions MUST cover DIFFERENT topics from the material
        - Cover as many different topics from the material as possible
        - Avoid generating questions about the same concept repeatedly
        - Each question must focus on a different part of the material
        - Avoid repeating the same question phrasing

        Return STRICT JSON ARRAY like this:

        [
        {{
        "question": "...",
        "options": ["...", "...", "...", "...", "..."],
        "correct": "A",
        "explanation": "Short explanation",
        "explanation_long": "2-3 sentences maximum",
        "source_document": "Exact file name",
        "source_page": "Page number"
        }}
        ]
        """

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.3
            )

            content = response.choices[0].message.content.strip()

            try:

                content = content.replace("```json","").replace("```","").strip()

                questions = json.loads(content)

                if not isinstance(questions, list):
                    questions = [questions]

                seen = set()
                unique_questions = []

                for q in questions:

                    text_q = q.get("question","").strip().lower()

                    if text_q not in seen:
                        seen.add(text_q)
                        unique_questions.append(q)

                return unique_questions

            except Exception as e:

                print("QUIZ JSON ERROR:", e)
                print("RAW GPT OUTPUT:", content)

                return []


        tasks = []

        for i in range(num_batches):

            n = min(batch_size, req.num_questions - i * batch_size)

            tasks.append(generate_batch(n))


        results = await asyncio.gather(*tasks)

        questions = []

        for batch in results:
            questions.extend(batch)

        # REMOVE GLOBAL DUPLICATES

        seen_questions = set()
        unique_questions = []

        for q in questions:

            text_q = q.get("question","").strip().lower()

            if text_q in seen_questions:
                continue

            seen_questions.add(text_q)
            unique_questions.append(q)

        return unique_questions[:req.num_questions]


    questions = await quiz_generator()

    # ======================
    # SAVE QUIZ
    # ======================

    quiz_id = str(uuid.uuid4())

    db = SessionLocal()

    db.execute(
        text("""
        insert into quizzes
        (id, project_id, user_id, num_questions, difficulty, language)
        values
        (:id, :project_id, :user_id, :num_questions, :difficulty, :language)
        """),
        {
            "id": quiz_id,
            "project_id": project_id,
            "user_id": user_id,
            "num_questions": req.num_questions,
            "difficulty": req.difficulty,
            "language": req.language
        }
    )

    for i, q in enumerate(questions):

        db.execute(
            text("""
            insert into quiz_questions
            (quiz_id, question_order, question, options, correct, explanation, explanation_long, source_document, source_page)
            values
            (:quiz_id, :order, :question, :options, :correct, :explanation, :explanation_long, :doc, :page)
            """),
            {
                "quiz_id": quiz_id,
                "order": i,
                "question": q.get("question"),
                "options": json.dumps(q.get("options")),
                "correct": q.get("correct"),
                "explanation": q.get("explanation"),
                "explanation_long": q.get("explanation_long"),
                "doc": q.get("source_document"),
                "page": q.get("source_page"),
                "topic": q.get("topic")
            }
        )

    db.commit()
    db.close()

    return {
        "quiz_id": quiz_id,
        "questions": questions
    }
@app.post("/projects/{project_id}/generate_flashcards")
async def generate_flashcards(
    project_id: str,
    req: dict = None,
    user = Depends(verify_user)
):

    num_cards = 10
    user_id = user["id"]
    if req and isinstance(req, dict):
        num_cards = req.get("num_cards", 10)

    # recuperiamo contesto dal vector DB
    context_chunks = search_project_chunks(project_id, k=20)

    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
You are an expert tutor.

Create {num_cards} flashcards from the study material.
Flashcards must cover DIFFERENT topics from the material.
Avoid creating multiple flashcards about the same concept.

Each flashcard must have:
- question
- answer

Return ONLY valid JSON in this format:

[
  {{
    "question": "...",
    "answer": "..."
  }}
]

Study material:
{context_text}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You generate study flashcards."},
            {"role": "user", "content": prompt}
        ]
    )

    gpt_text = response.choices[0].message.content

    try:

        gpt_text = gpt_text.replace("```json", "").replace("```", "").strip()
        flashcards = json.loads(gpt_text)

        # remove duplicate questions

        seen = set()
        unique_flashcards = []

        for card in flashcards:

            q = card.get("question","").strip().lower()

            if q not in seen:
                seen.add(q)
                unique_flashcards.append(card)

        flashcards = unique_flashcards

        if not isinstance(flashcards, list):
            flashcards = []

    except Exception as e:

        print("FLASHCARDS JSON ERROR:", e)
        print("RAW GPT OUTPUT:", gpt_text)

        flashcards = []
    # ======================
    # SAVE FLASHCARDS
    # ======================

    db = SessionLocal()

    for card in flashcards:

        db.execute(
            text("""
            insert into flashcards
            (project_id, user_id, question, answer)
            values
            (:project_id, :user_id, :question, :answer)
            """),
            {
                "project_id": project_id,
                "user_id": user_id,
                "question": card.get("question"),
                "answer": card.get("answer")
            }
        )

    db.commit()
    db.close()

    return {"flashcards": flashcards}
def search_project_chunks(project_id: str, k: int = 20):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select chunk_text, doc_title, page
            from chunks
            where project_id = :project_id
            limit :k
        """),
        {
            "project_id": project_id,
            "k": k
        }
    ).fetchall()

    db.close()

    chunks = []

    for r in rows:
        chunks.append({
            "text": r[0],
            "document": r[1],
            "page": r[2]
        })

    return chunks
@app.get("/projects/{project_id}/topics")
async def get_project_topics(
    project_id: str,
    user = Depends(verify_user)
):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select chunk_text
            from chunks
            where project_id = :project_id
            limit 80
        """),
        {"project_id": project_id}
    ).fetchall()

    print("ROWS FOUND:", len(rows))
    if not rows:
        db.close()
        return {"topics": []}
        

    text_blocks = [r[0] for r in rows]

    context = "\n\n".join(text_blocks)

    prompt = f"""
You are analyzing medical study material.

Your task:
Identify the main study concepts students must learn.

Return JSON in this format:

{{ "topics": ["Apoptosis","Necrosis","Inflammation","Cell injury","Oxidative stress"] }}

Material:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Return ONLY JSON."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    # pulizia markdown restituito da GPT
    content = content.replace("```json", "").replace("```", "").strip()

    import json

    try:
        data = json.loads(content)

        topics = data.get("topics", [])

        if not isinstance(topics, list):
            topics = []

        # ======================
        # CLEAN AND MERGE SIMILAR TOPICS
        # ======================

        cleaned = []
        seen = set()

        for t in topics:

            base = t.lower()

            # normalizzazione semplice
            base = base.replace("pathway","")
            base = base.replace("process","")
            base = base.replace("mechanism","")
            base = base.replace("regulation","")

            base = base.strip()

            if base not in seen:
                seen.add(base)
                cleaned.append(t)

        topics = cleaned

        # ======================
        # COUNT TOPIC OCCURRENCES IN MATERIAL
        # ======================

        topic_counts = {}

        full_text = context.lower()

        for t in topics:

            key = t.lower()

            count = full_text.count(key)

            topic_counts[key] = max(count,1)

    except Exception as e:
        print("TOPIC PARSE ERROR:", e)
        print("RAW GPT OUTPUT:", content)
        topics = []

    print("TOPICS GENERATED:", topics)

    # ora aggiungiamo difficulty e suggerimenti
    import random

    topics_with_stats = []

    for t in topics:

        accuracy_row = db.execute(
            text("""
                select avg(
                    case when qa.is_correct then 1 else 0 end
                )
                from quiz_answers qa
                join quiz_questions qq on qa.question_id = qq.id
                where qq.topic = :topic
            """),
            {"topic": t}
        ).fetchone()

        accuracy = accuracy_row[0] if accuracy_row and accuracy_row[0] is not None else 0.5


        if accuracy > 0.8:
            difficulty = "easy"
        elif accuracy > 0.5:
            difficulty = "medium"
        else:
            difficulty = "hard"

        base = t.lower().replace("pathway","").replace("process","").replace("mechanism","").replace("regulation","").strip()

        topics_with_stats.append({
            "topic": t,
            "difficulty": difficulty,
            "suggested_page": random.randint(1,300),
            
        })
    db.close()
    return {"topics": topics_with_stats}
@app.get("/projects/{project_id}/quizzes")
async def list_project_quizzes(
    project_id: str,
    user = Depends(verify_user)
):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select id, created_at, num_questions, difficulty
            from quizzes
            where project_id = :project_id
            order by created_at desc
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    quizzes = []

    for r in rows:
        quizzes.append({
            "id": r[0],
            "created_at": str(r[1]),
            "num_questions": r[2],
            "difficulty": r[3]
        })

    return {"quizzes": quizzes}
@app.get("/projects/{project_id}/flashcards")
async def get_flashcards(project_id: str):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select question, answer
            from flashcards
            where project_id = :project_id
            order by created_at desc
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    flashcards = []

    for r in rows:
        flashcards.append({
            "question": r[0],
            "answer": r[1]
        })

    return {"flashcards": flashcards}
# ======================
# PROJECT SUMMARY
# ======================

@app.get("/projects/{project_id}/summary")
async def project_summary(
    project_id: str,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    project = db.execute(
        text("""
            select id
            from projects
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

    quiz_attempts = db.execute(
        text("""
            select count(*)
            from quiz_attempts
            where user_id = :user_id
            and quiz_id in (
                select id from quizzes where project_id = :project_id
            )
        """),
        {
            "user_id": user_id,
            "project_id": project_id
        }
    ).scalar()

    avg_score = db.execute(
        text("""
            select avg(
                (score::float / total_questions) * 100
            )
            from quiz_attempts
            where user_id = :user_id
            and quiz_id in (
                select id from quizzes where project_id = :project_id
            )
        """),
        {
            "user_id": user_id,
            "project_id": project_id
        }
    ).scalar()

    if avg_score is None:
        avg_score = 0

    flashcards_reviewed = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and last_review is not null
        """),
        {
            "project_id": project_id
        }
    ).scalar()

    topics_count = db.execute(
        text("""
            select count(distinct topic)
            from quiz_questions qq
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
        """),
        {
            "project_id": project_id
        }
    ).scalar()

    db.close()

    return {
        "quiz_attempts": quiz_attempts or 0,
        "avg_score": round(avg_score, 1),
        "flashcards_reviewed": flashcards_reviewed or 0,
        "topics_count": topics_count or 0
    }


# ======================
# PROJECT RESULTS
# ======================

@app.get("/projects/{project_id}/results")
async def project_results(
    project_id: str,
    user = Depends(verify_user)
):

    user_id = user["id"]
    db = SessionLocal()

    # verifica progetto
    project = db.execute(
        text("""
            select id
            from projects
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

    # ======================
    # QUIZ HISTORY
    # ======================

    quiz_rows = db.execute(
        text("""
            select qa.created_at,
                   qa.score,
                   qa.total_questions,
                   q.difficulty
            from quiz_attempts qa
            join quizzes q on qa.quiz_id = q.id
            where q.project_id = :project_id
            and qa.user_id = :user_id
            order by qa.created_at desc
            limit 20
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchall()

    quiz_history = []

    for r in quiz_rows:
        quiz_history.append({
            "date": str(r[0]),
            "score": r[1],
            "total": r[2],
            "difficulty": r[3]
        })

    # ======================
    # TOPIC ACCURACY
    # ======================

    topic_rows = db.execute(
        text("""
            select qq.topic,
                   avg(case when qa.is_correct then 1 else 0 end) as accuracy
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
            group by qq.topic
        """),
        {"project_id": project_id}
    ).fetchall()

    topic_mastery = []

    for r in topic_rows:
        topic_mastery.append({
            "topic": r[0],
            "accuracy": round((r[1] or 0) * 100,1)
        })

    db.close()

    return {
        "quiz_history": quiz_history,
        "topic_mastery": topic_mastery
    }
    # ======================
    # QUIZ ATTEMPTS
    # ======================

    quiz_attempts = db.execute(
        text("""
            select count(*)
            from quiz_attempts
            where user_id = :user_id
            and quiz_id in (
                select id from quizzes where project_id = :project_id
            )
        """),
        {
            "user_id": user_id,
            "project_id": project_id
        }
    ).scalar()

    # ======================
    # AVG SCORE
    # ======================

    avg_score = db.execute(
        text("""
            select avg(
                (score::float / total_questions) * 100
            )
            from quiz_attempts
            where user_id = :user_id
            and quiz_id in (
                select id from quizzes where project_id = :project_id
            )
        """),
        {
            "user_id": user_id,
            "project_id": project_id
        }
    ).scalar()

    if avg_score is None:
        avg_score = 0

    # ======================
    # FLASHCARDS REVIEWED
    # ======================

    flashcards_reviewed = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and last_review is not null
        """),
        {
            "project_id": project_id
        }
    ).scalar()

    # ======================
    # TOPICS COUNT
    # ======================

    topics_count = db.execute(
        text("""
            select count(distinct topic)
            from quiz_questions qq
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
        """),
        {
            "project_id": project_id
        }
    ).scalar()

    db.close()

    return {
        "quiz_attempts": quiz_attempts or 0,
        "avg_score": round(avg_score, 1),
        "flashcards_reviewed": flashcards_reviewed or 0,
        "topics_count": topics_count or 0
    }
@app.get("/projects/{project_id}/flashcards_count")
async def flashcards_count(project_id: str):

    db = SessionLocal()

    row = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and next_review <= now()
        """),
        {"project_id": project_id}
    ).fetchone()

    db.close()

    return {"count": row[0]}
@app.get("/projects/{project_id}/study_flashcards")
async def study_flashcards(project_id: str, limit: int = 20):

    db = SessionLocal()

    rows = db.execute(
    text("""
        select id, question, answer
        from flashcards
        where project_id = :project_id
        order by id desc
        limit :limit
    """),
    {
        "project_id": project_id,
        "limit": limit
    }
    ).fetchall()

    db.close()

    cards = []

    for r in rows:
        cards.append({
            "id": r[0],
            "question": r[1],
            "answer": r[2]
        })

    return {"flashcards": cards}
@app.post("/review_flashcard")
async def review_flashcard(req: dict):

    db = SessionLocal()

    
    difficulty = req.get("difficulty", 1)
    flashcard_id = req.get("flashcard_id")
    is_correct = req.get("is_correct", False)
    

    if not is_correct:
        interval = "1 day"
    if difficulty == 1:
        interval = "1 day"
    elif difficulty == 2:
        interval = "3 days"
    elif difficulty == 3:
        interval = "7 days"
    else:
        interval = "14 days"

    db.execute(
        text(f"""
            update flashcards
            set
                difficulty = :difficulty,
                last_review = now(),
                next_review = now() + interval '{interval}'
            where id = :flashcard_id
        """),
        {
            "difficulty": difficulty,
            "flashcard_id": flashcard_id
        }
    )

    db.commit()
    db.close()

    return {"status": "ok"}
@app.get("/quizzes/{quiz_id}")
async def get_quiz(quiz_id: str):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select question, options, correct, explanation, explanation_long,
                   source_document, source_page
            from quiz_questions
            where quiz_id = :quiz_id
            order by question_order
        """),
        {"quiz_id": quiz_id}
    ).fetchall()

    db.close()

    questions = []

    for r in rows:
        questions.append({
            "question": r[0],
            "options": r[1] if isinstance(r[1], list) else json.loads(r[1]),
            "correct": r[2],
            "explanation": r[3],
            "explanation_long": r[4],
            "source_document": r[5],
            "source_page": r[6]
        })

    return {"questions": questions}

@app.post("/save_quiz_attempt")
async def save_quiz_attempt(
    req: dict,
    user = Depends(verify_user)
):

    db = SessionLocal()

    db.execute(
        text("""
            insert into quiz_attempts
            (quiz_id, user_id, score, total_questions)
            values
            (:quiz_id, :user_id, :score, :total_questions)
        """),
        {
            "quiz_id": req["quiz_id"],
            "user_id": user["id"],
            "score": req["score"],
            "total_questions": req["total_questions"]
        }
    )

    db.commit()
    db.close()

    return {"status": "saved"}

@app.post("/ask")
async def ask_documents(req: AskRequest):

    # 1️⃣ embedding della domanda
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=f"medical concept explanation {req.question}"

    )

    query_embedding = emb.data[0].embedding


    # 2️⃣ vector search sui chunks
    db = SessionLocal()

    rows = db.execute(
        text("""
            select chunk_text, doc_title, page
            from chunks
            where project_id = :project_id
            order by embedding <-> CAST(:embedding AS vector)
            limit 20
        """),
        {
            "project_id": req.project_id,
            "embedding": query_embedding
        }
    ).fetchall()
    rows = rows[:12]
    print("ROWS FOUND:", len(rows))
    if rows:
        print("SAMPLE CHUNK:", rows[0][0][:200])

    db.close()

    context_blocks = []
    for r in rows:
        context_blocks.append(f"DOCUMENT: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:400]}")


    context = "\n\n---\n\n".join(context_blocks) 
    print("CONTEXT LENGTH:", len(context))
    # 3️⃣ costruzione contesto
    

    prompt = f"""
You are an expert study tutor.
You MUST answer using ONLY the material provided below.
You MUST answer using the provided material.

Rules:
- If the material contains relevant information, explain it clearly in your own words.
- If the material contains partial information, use it to build the best possible explanation.
- Only say that the documents do not contain enough information if the concept is completely absent.
- Be helpful and explanatory, not overly strict.
- When possible, mention the document and page.

Context:
{context}

Question:
{req.question}

Answer clearly for a student.
"""


    # 4️⃣ GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful study tutor."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer
    }




@app.post("/projects/{project_id}/active_recall_question")
async def active_recall_question(
    project_id: str,
    req: ActiveRecallRequest,
    user = Depends(verify_user)
):

    # embedding query
    # ======================
    # WEAK TOPIC PRIORITY
    # ======================

    weak_topic = None

    db = SessionLocal()

    rows = db.execute(
        text("""
            select qq.topic,
                avg(case when qa.is_correct then 1 else 0 end) as accuracy
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
            group by qq.topic
            order by accuracy asc
            limit 1
        """),
        {"project_id": project_id}
    ).fetchone()

    db.close()

    if rows and rows[0]:
        weak_topic = rows[0]

    query_text = weak_topic if weak_topic else "important study concepts"

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )

    query_embedding = emb.data[0].embedding

    db = SessionLocal()

    rows = db.execute(
        text("""
            select chunk_text, doc_title, page
            from chunks
            where project_id = :project_id
            order by embedding <-> CAST(:embedding AS vector)
            limit 10
        """),
        {
            "project_id": project_id,
            "embedding": query_embedding
        }
    ).fetchall()

    db.close()

    context_blocks = []

    for r in rows:
        context_blocks.append(
            f"DOCUMENT: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:400]}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
You are a tutor helping a student practice ACTIVE RECALL.

Your task:
Generate ONE open-ended question that forces the student
to recall and explain a concept from the material.

Rules:
- Do NOT ask multiple choice questions
- Ask a question that requires explanation
- Focus on mechanisms, processes, cause-effect
- Avoid trivial definitions

Material:
{context}

Return ONLY JSON:

{{
 "question": "...",
 "source_document": "...",
 "source_page": "..."
}}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.4,
        messages=[
            {"role": "system", "content": "You generate active recall questions."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json","").replace("```","").strip()

    try:
        data = json.loads(content)
    except:
        data = {"question": "Explain an important concept from the material."}

    return data

from typing import Optional

class ActiveRecallEvaluateRequest(BaseModel):
    question: str
    student_answer: str
    history: Optional[list[str]] = None


@app.post("/active_recall_evaluate")
async def active_recall_evaluate(req: ActiveRecallEvaluateRequest):

    history_text = "\n".join(req.history or [])

    prompt = f"""
    You are a supportive study tutor evaluating a student's answer.

    Question:
    {req.question}

    Previous answers:
    {history_text}

    Latest answer:
    {req.student_answer}

    Evaluation rules:

    - Evaluate the student's overall understanding across ALL answers.
    - If the student progressively improves, recognize it.
    - Accept correct ideas even if the explanation is short.
    - Do NOT repeat the same feedback.
    - If the student already explained the main concepts, mark the answer as correct.
    - If the student gives up, explain the concept clearly.

    Return ONLY JSON:

    {{
    "evaluation": "correct | partial | incorrect",
    "score": 0-1,
    "feedback": "short supportive feedback",
    "hint": "optional hint",
    "explanation": "clear explanation if needed"
    }}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You evaluate student answers."},
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content.strip()
    content = content.replace("```json","").replace("```","").strip()

    try:
        data = json.loads(content)
    except:
        data = {
            "evaluation": "incorrect",
            "score": 0,
            "feedback": "The answer needs more explanation.",
            "hint": "Try explaining the concept step by step.",
            "explanation": "Review the concept and try again."
        }

    return data

    # ======================
    # STUDY SESSION
    # ======================

    @app.get("/projects/{project_id}/study_session")
    async def study_session(
        project_id: str,
        user = Depends(verify_user)
    ):

    db = SessionLocal()
    # ======================
    # DETECT WEAK TOPICS
    # ======================

    weak_topic_rows = db.execute(
        text("""
            select qq.topic,
                avg(case when qa.is_correct then 1 else 0 end) as accuracy
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
            group by qq.topic
            order by accuracy asc
            limit 5
        """),
        {"project_id": project_id}
    ).fetchall()

    weak_topics = [r[0] for r in weak_topic_rows if r[0]]

        # ======================
        # SESSION FLASHCARDS (15) - GENERATED FRESH
        # ======================

        context_chunks = search_project_chunks(project_id, k=20)

        context_text = "\n\n".join([c["text"] for c in context_chunks])

        weak_topics_text = ", ".join(weak_topics) if weak_topics else "important concepts"

        prompt = f"""
        Create 15 NEW flashcards for a study session.

        Focus especially on these weak topics:
        {weak_topics_text}

        Rules:
        - Cover different concepts
        - Avoid duplicates
        - Make flashcards useful for testing understanding
        - Prefer mechanisms and cause-effect relationships

        Return ONLY JSON:

        [
        {{
            "question": "...",
            "answer": "..."
        }}
        ]

        Material:
        {context_text}
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Generate study session flashcards."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        try:
            generated_cards = json.loads(content)
            if not isinstance(generated_cards, list):
                generated_cards = []
        except:
            generated_cards = []

        flashcards = []

        for i, card in enumerate(generated_cards):
            flashcards.append({
                "id": i + 1,
                "question": card.get("question"),
                "answer": card.get("answer")
            })
    # ======================
    # RECALL TOPICS
    # ======================

    topic_rows = db.execute(
        text("""
            select qq.topic,
                   avg(case when qa.is_correct then 1 else 0 end) as accuracy
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
            group by qq.topic
            order by accuracy asc
            limit 5
        """),
        {"project_id": project_id}
    ).fetchall()

    recall_topics = []

    for r in topic_rows:
        recall_topics.append(r[0])


    # ======================
    # QUIZ CONFIG
    # ======================

    # ======================
    # ADAPTIVE SESSION
    # ======================

    avg_accuracy_row = db.execute(
        text("""
            select avg(
                case when qa.is_correct then 1 else 0 end
            )
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
        """),
        {"project_id": project_id}
    ).fetchone()

    avg_accuracy = avg_accuracy_row[0] if avg_accuracy_row and avg_accuracy_row[0] else 0.5


    if avg_accuracy < 0.5:

        recall_count = 8
        quiz_questions = 15

    elif avg_accuracy < 0.8:

        recall_count = 5
        quiz_questions = 20

    else:

        recall_count = 3
        quiz_questions = 25


    quiz_config = {
        "num_questions": quiz_questions,
        "difficulty": "medium",
        "focus_topics": weak_topics
    }

    db.close()

    return {
        "flashcards": flashcards,
        "recall_topics": recall_topics[:recall_count],
        "quiz": quiz_config
    }