import os
import uuid
import base64
import io
from typing import List, Optional
import json
import requests
from fastapi import FastAPI, Depends, HTTPException, Header, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import text as sql_text
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from openai import OpenAI
from dotenv import load_dotenv
from pypdf import PdfReader
import asyncio
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
from urllib.parse import unquote
from fastapi import HTTPException
from fastapi import Body



pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"




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
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://quiz-ui-ruddy.vercel.app"
    ],
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
    "documents": [
        {"id": r[0], "title": r[0]}
        for r in rows
    ]
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

def chunk_text(text, max_chars=1000, overlap=200):

    import re

    paragraphs = re.split(r'\n\s*\n', text)

    chunks = []
    current_chunk = ""

    for p in paragraphs:

        p = p.strip()

        if not p:
            continue

        if len(p) > max_chars:

            for i in range(0, len(p), max_chars):
                sub = p[i:i+max_chars]
                chunks.append(sub)

            continue

        if len(current_chunk) + len(p) < max_chars:

            current_chunk += "\n\n" + p

        else:

            chunks.append(current_chunk.strip())

            current_chunk = current_chunk[-overlap:] + "\n\n" + p

    if current_chunk:
        chunks.append(current_chunk.strip())

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

                if page_index > 30:
                    print("PAGE LIMIT REACHED → STOP")
                    break
                print(f"PAGE {page_index+1}")
                page_text = page.extract_text()

                if not page_text or not page_text.strip():

                    print(f"OCR PAGE {page_index+1}")

                    page_text = ocr_pdf_page(pdf_bytes, page_index)

                    if not page_text:
                        continue

                chunks = chunk_text(page_text)
                chunks = [c for c in chunks if len(c) > 100]
                print("CHUNKS CREATED:", len(chunks))

                for chunk in chunks:

                    emb = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=chunk
                    )
                    print("EMBEDDING CHUNK...")
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
                    
        print("START DOCUMENT:", doc.title)
        db.commit()

    except Exception as e:
        db.rollback()
        db.close()
        raise HTTPException(status_code=500, detail=str(e))

    db.close()
    return {"status": "ok"}

def process_topics_task(project_id: str):
    db = SessionLocal()

    try:
        print("BACKGROUND TOPICS START:", project_id)

        db.execute(
            text("""
                update projects
                set topic_status = 'processing'
                where id = :project_id
            """),
            {"project_id": project_id}
        )
        db.commit()

        rows = db.execute(
            text("""
                select chunk_text
                from chunks
                where project_id = :project_id
                limit 500
            """),
            {"project_id": project_id}
        ).fetchall()

        full_text = "\n\n".join([r[0] for r in rows if r[0]])

        print("BACKGROUND TEXT LENGTH:", len(full_text))

        if not full_text.strip():
            db.execute(
                text("""
                    update projects
                    set topic_status = 'error'
                    where id = :project_id
                """),
                {"project_id": project_id}
            )
            db.commit()
            print("BACKGROUND TOPICS ERROR: no content")
            return

        prompt = f"""
Extract the MAIN academic topics.

RULES:
- Max 40 topics
- Academic names
- No duplicates
- Use ONLY topics explicitly present in the text

Return JSON:
{{ "topics": ["topic1","topic2"] }}

TEXT:
{full_text[:12000]}
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return ONLY JSON."},
                {"role": "user", "content": prompt}
            ]
        )

        content = response.choices[0].message.content.strip()
        content = content.replace("```json", "").replace("```", "").strip()

        print("BACKGROUND RAW GPT:", content[:500])

        data_json = json.loads(content)
        topics = data_json.get("topics", [])

        db.execute(
            text("delete from topics where project_id = :project_id"),
            {"project_id": project_id}
        )

        seen = set()

        for t in topics:
            topic_name = (t or "").strip()
            if not topic_name:
                continue

            key = topic_name.lower()
            if key in seen:
                continue

            seen.add(key)

            db.execute(
                text("""
                    insert into topics (project_id, topic)
                    values (:project_id, :topic)
                """),
                {
                    "project_id": project_id,
                    "topic": topic_name
                }
            )

        db.execute(
            text("""
                update projects
                set topic_status = 'completed'
                where id = :project_id
            """),
            {"project_id": project_id}
        )

        db.commit()
        print("BACKGROUND TOPICS SAVED:", len(seen))

    except Exception as e:
        db.rollback()
        print("BACKGROUND TOPICS ERROR:", e)

        try:
            db.execute(
                text("""
                    update projects
                    set topic_status = 'error'
                    where id = :project_id
                """),
                {"project_id": project_id}
            )
            db.commit()
        except Exception as inner_e:
            print("BACKGROUND STATUS UPDATE ERROR:", inner_e)

    finally:
        db.close()

@app.post("/projects/{project_id}/ingest_stream")
async def ingest_stream(
    project_id: str,
    data: IngestRequest,
    background_tasks: BackgroundTasks,
    user = Depends(verify_user)
):
    docs = data.documents

    async def generate():
        db = SessionLocal()

        try:
            db.execute(
                text("""
                    update projects
                    set topic_status = 'processing'
                    where id = :project_id
                """),
                {"project_id": project_id}
            )
            db.commit()

            yield "Starting upload...\n"

            # ======================
            # SAVE CHUNKS
            # ======================
            for doc in docs:
                yield f"Processing document: {doc.title}\n"

                pdf_bytes = base64.b64decode(doc.file_bytes)
                pdf_stream = io.BytesIO(pdf_bytes)

                reader = PdfReader(pdf_stream)

                for page_index, page in enumerate(reader.pages):
                    if page_index > 30:
                        yield "Page limit reached → stopping\n"
                        break

                    yield f"Page {page_index+1}\n"

                    page_text = page.extract_text()

                    if not page_text or not page_text.strip():
                        yield f"OCR page {page_index+1}\n"
                        page_text = ocr_pdf_page(pdf_bytes, page_index)

                        if not page_text:
                            continue

                    chunks = chunk_text(page_text)
                    chunks = [c for c in chunks if len(c) > 100]

                    yield f"{len(chunks)} chunks created\n"

                    for i, chunk in enumerate(chunks):
                        yield f"Embedding chunk {i+1}/{len(chunks)}\n"
                        await asyncio.sleep(0)

                        emb = await asyncio.to_thread(
                            client.embeddings.create,
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

            

            yield "Upload complete ✅\n"

            background_tasks.add_task(process_topics_task, project_id)

        except Exception as e:
            db.rollback()
            yield f"Upload failed: {str(e)}\n"
        finally:
            db.close()

    return StreamingResponse(
        generate(),
        media_type="text/plain"
    )

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
    topics: Optional[List[str]] = []

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
        topics = req.topics if hasattr(req, "topics") else []
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

            text_chunk = r[0].lower()

            if topics:
                if not any(topic.lower() in text_chunk for topic in topics):
                    continue   # 🔥 SCARTA CHUNK NON RILEVANTI

            material_blocks.append(
                f"FILE: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:500]}"
            )
        if len(material_blocks) == 0:
            print("⚠️ No topic match, fallback to full material")

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
        Each question MUST focus on a COMPLETELY DIFFERENT concept.

        CRITICAL:
        - Do NOT generate questions about the same topic even if phrased differently
        - Avoid paraphrasing the same concept
        - Each question must test a UNIQUE concept
        - If you cannot find enough different concepts, generate fewer questions instead

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

            # filtro più intelligente (prime parole)
            key = " ".join(text_q.split()[:8])

            if key in seen_questions:
                continue

            seen_questions.add(key)
            unique_questions.append(q)

        # FILL MISSING QUESTIONS
        if len(unique_questions) < req.num_questions:

            missing = req.num_questions - len(unique_questions)
            print(f"Filling {missing} missing questions...")

            extra = await generate_batch(missing)

            for q in extra:

                text_q = q.get("question","").strip().lower()
                key = " ".join(text_q.split()[:8])

                if key in seen_questions:
                    continue

                seen_questions.add(key)
                unique_questions.append(q)

                if len(unique_questions) >= req.num_questions:
                    break

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
    topics = []

    if req and isinstance(req, dict):
        num_cards = req.get("num_cards", 10)
        topics = req.get("topics", [])

    # recuperiamo contesto dal vector DB
    query_text = " ".join(topics) if topics else "important concepts"

    # 🔥 COSTRUIAMO QUERY FORTE BASATA SUI TOPICS
    if topics:
        topic_query = " ".join(topics)
        query_text = f"study material about {topic_query}"
    else:
        query_text = "important study concepts"

    # 🔥 RETRIEVAL PIÙ PRECISO (NO RANDOM)
    db = SessionLocal()

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )

    query_embedding = emb.data[0].embedding

    rows = db.execute(
        text("""
            select chunk_text, doc_title, page
            from chunks
            where project_id = :project_id
            order by embedding <-> CAST(:embedding AS vector)
            limit 12
        """),
        {
            "project_id": project_id,
            "embedding": query_embedding
        }
    ).fetchall()

    db.close()

    context_chunks = [
        {
            "text": r[0],
            "document": r[1],
            "page": r[2]
        }
        for r in rows
    ]
    

    context_text = "\n\n".join([c["text"] for c in context_chunks])

    prompt = f"""
    You MUST generate EXACTLY {num_cards} flashcards.
    You are a strict study tutor.

    You MUST follow these rules:

    - Use ONLY the provided material
    - DO NOT use external knowledge
    - If information is missing → SKIP the card
    - Each flashcard must test ONE clear concept
    - Each flashcard MUST include a "topic"
    - The topic must be a specific concept explicitly present in the material
    - Avoid generic or vague topics
    - Avoid generic or vague questions
    - Avoid repeating similar questions
    - Avoid simple definition-only questions when possible
    - Prefer cause-effect, mechanisms, reasoning

    Return EXACTLY {num_cards} items.

    IMPORTANT:
    Each flashcard must be DIFFERENT in concept.

    Return ONLY JSON:

    [
    {{
        "question": "...",
        "answer": "...",
        "concept": "...",
        "topic": "...",
        "difficulty": "easy | medium | hard"
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

        # remove duplicate + low quality flashcards

        seen = set()
        unique_flashcards = []

        for card in flashcards:

            q = card.get("question","").strip().lower()
            a = card.get("answer","").strip()

            # filtri qualità base
            if len(q) < 10 or len(a) < 5:
                continue

            if "what is" in q and len(a.split()) < 5:
                continue

            if q not in seen:
                seen.add(q)
                unique_flashcards.append(card)

        flashcards = unique_flashcards
        flashcards = flashcards[:num_cards]

        if not isinstance(flashcards, list):
            flashcards = []

        # 🔥 FORCE EXACT NUMBER
        flashcards = flashcards[:num_cards]

        # 🔥 SE MANCANO → GENERA FINO A COMPLETARE
        while len(flashcards) < num_cards:

            missing = num_cards - len(flashcards)

            print(f"⚠️ Filling missing flashcards: {missing}")

            extra_prompt = f"""
        Generate {missing} NEW flashcards.

        Rules:
        - Do NOT repeat previous concepts
        - Avoid similar questions
        - Be specific and concrete
        - Use ONLY the material

        Study material:
        {context_text}
        """

            extra_response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You generate study flashcards."},
                    {"role": "user", "content": extra_prompt}
                ]
            )

            extra_text = extra_response.choices[0].message.content
            extra_text = extra_text.replace("```json", "").replace("```", "").strip()

            try:
                extra_cards = json.loads(extra_text)
            except:
                break

            for card in extra_cards:

                if len(flashcards) >= num_cards:
                    break

                q = card.get("question","").strip().lower()

                if q not in seen:
                    seen.add(q)
                    flashcards.append(card)

            attempt += 1

    except Exception as e:

        print("FLASHCARDS JSON ERROR:", e)
        print("RAW GPT OUTPUT:", gpt_text)

        flashcards = []
    # ======================
    # SAVE FLASHCARDS
    # ======================

    db = SessionLocal()

    for card in flashcards:

        result = db.execute(
            text("""
            insert into flashcards
            (project_id, user_id, question, answer, topic)
            values
            (:project_id, :user_id, :question, :answer, :topic)
            returning id
            """),
            {
                "project_id": project_id,
                "user_id": user_id,
                "question": card.get("question"),
                "answer": card.get("answer"),
                "topic": card.get("topic") or card.get("concept")
            }
        )

        new_id = result.fetchone()[0]
        card["id"] = new_id

    db.commit()
    db.close()

    return {"flashcards": flashcards}
def search_project_chunks(
    project_id: str,
    query: str = None,
    topics: list[str] = None,
    k: int = 20
):

    db = SessionLocal()

    # ======================
    # BUILD QUERY INTELLIGENTE
    # ======================

    if query and topics:
        full_query = f"{query} {' '.join(topics)}"
    elif topics:
        full_query = " ".join(topics)
    else:
        full_query = query or "important study concepts"

    # DEBUG
    print("🔍 RETRIEVAL QUERY:", full_query)
    print("🎯 TOPICS:", topics)

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=full_query
    )

    query_embedding = emb.data[0].embedding

    # ======================
    # VECTOR SEARCH + RANDOM MIX
    # ======================

    rows = db.execute(
        text("""
            (
                select chunk_text, doc_title, page
                from chunks
                where project_id = :project_id
                order by embedding <-> CAST(:embedding AS vector)
                limit :k
            )

            union

            (
                select chunk_text, doc_title, page
                from chunks
                where project_id = :project_id
                order by random()
                limit :k
            )
        """),
        {
            "project_id": project_id,
            "k": k,
            "embedding": query_embedding
        }
    ).fetchall()

    db.close()

    # ======================
    # COSTRUZIONE CHUNKS
    # ======================

    chunks = []

    for r in rows:
        chunks.append({
            "text": r[0],
            "document": r[1],
            "page": r[2]
        })

    print("📦 CHUNKS RETRIEVED:", len(chunks))

    return chunks[:k]
from sqlalchemy import text as sql_text

@app.get("/projects/{project_id}/topics")
async def get_topics(project_id: str):

    db = SessionLocal()

    try:

        result = db.execute(
            sql_text("""
                SELECT DISTINCT topic
                FROM topics
                WHERE project_id = :project_id
                AND topic IS NOT NULL
            """),
            {"project_id": project_id}
        )

        rows = result.fetchall()

        topics = []

        for r in rows:
            topic_name = (r[0] or "").strip()

            if not topic_name:
                continue

            topics.append({
                "topic": topic_name,
                "difficulty": "medium",
                "accuracy": 50,
                "suggested_page": None
            })

        return {"topics": topics}

    finally:
        db.close()

@app.get("/projects/{project_id}/topic_status")
def get_topic_status(
    project_id: str,
    user = Depends(verify_user)
):
    db = SessionLocal()

    try:
        row = db.execute(
            text("""
                select topic_status
                from projects
                where id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()

        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        return {
            "status": row[0] or "idle"
        }

    finally:
        db.close()

@app.post("/projects/{project_id}/generate_topics")
def generate_topics(
        project_id: str,
        user = Depends(verify_user)
    ):
        db = SessionLocal()

        try:
            print("START TOPICS GENERATION:", project_id)

            rows = db.execute(
                text("""
                    SELECT chunk_text
                    FROM chunks
                    WHERE project_id = :project_id
                    LIMIT 500
                """),
                {"project_id": project_id}
            ).fetchall()

            full_text = "\n\n".join([r[0] for r in rows if r[0]])

            print("TEXT LENGTH:", len(full_text))

            if not full_text.strip():
                raise HTTPException(status_code=400, detail="No content")

            prompt = f"""
    Extract the MAIN academic topics.

    RULES:
    - Max 40 topics
    - Academic names
    - No duplicates
    - Use ONLY topics explicitly present in the text

    Return JSON:
    {{ "topics": ["topic1","topic2"] }}

    TEXT:
    {full_text[:12000]}
    """

            print("CALLING GPT...")

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return ONLY JSON."},
                    {"role": "user", "content": prompt}
                ]
            )

            content = response.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()

            print("RAW GPT:", content[:500])

            data_json = json.loads(content)
            topics = data_json.get("topics", [])

            print("TOPICS FOUND:", len(topics))

            db.execute(
                text("DELETE FROM topics WHERE project_id = :project_id"),
                {"project_id": project_id}
            )

            seen = set()

            for t in topics:
                topic_name = (t or "").strip()

                if not topic_name:
                    continue

                key = topic_name.lower()
                if key in seen:
                    continue

                seen.add(key)

                db.execute(
                    text("""
                        INSERT INTO topics (project_id, topic)
                        VALUES (:project_id, :topic)
                    """),
                    {
                        "project_id": project_id,
                        "topic": topic_name
                    }
                )

            db.commit()

            print("TOPICS SAVED")

            return {
                "status": "ok",
                "topics_count": len(seen)
            }

        except Exception as e:
            db.rollback()
            print("TOPICS ERROR:", e)
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            db.close()        
       
        


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

@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def flashcards_detailed_stats(
    project_id: str,
    user = Depends(verify_user)
):

    db = SessionLocal()

    rows = db.execute(
        text("""
            select
                count(*) as total,
                sum(case when is_correct = false then 1 else 0 end) as wrong,
                sum(case when difficulty = 1 then 1 else 0 end) as hard,
                sum(case when difficulty = 2 then 1 else 0 end) as correct,
                sum(case when difficulty = 3 then 1 else 0 end) as easy
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user["id"]
        }
    ).fetchone()

    db.close()

    total = rows[0] or 1

    return {
        "total": total,
        "wrong": rows[1] or 0,
        "hard": rows[2] or 0,
        "correct": rows[3] or 0,
        "easy": rows[4] or 0,
        "accuracy": round(((rows[3] + rows[4]) / total) * 100, 1)
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


async def review_flashcard(
    req: dict = Body(...),
    user = Depends(verify_user)
):

    db = SessionLocal()
    print("REVIEW_FLASHCARD req:", req)
    print("REVIEW_FLASHCARD flashcard_id:", req.get("flashcard_id"))
    print("REVIEW_FLASHCARD type:", type(req.get("flashcard_id")))
    try:

        difficulty = req.get("difficulty", 1)
        try:
            difficulty = int(difficulty)
        except:
            difficulty = 1
        flashcard_id = req.get("flashcard_id")
        if not flashcard_id:
            raise HTTPException(status_code=400, detail="flashcard_id missing")
        row = db.execute(
            text("""
                select next_review, last_review
                from flashcards
                where id = :flashcard_id
            """),
            {"flashcard_id": flashcard_id}
        ).fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Flashcard not found")
        is_correct = req.get("is_correct", False)

        from datetime import datetime, timedelta

        now = datetime.utcnow()

        # fallback
        current_interval_days = 1

        if row and row[0] and row[1]:
            delta = row[0] - row[1]
            current_interval_days = max(1, delta.days)

        # ======================
        # SPACED REPETITION LOGIC
        # ======================

        if not is_correct:
            new_interval_days = 1

        elif difficulty == 1:
            new_interval_days = max(1, current_interval_days // 2)

        elif difficulty == 2:
            new_interval_days = current_interval_days + 1

        elif difficulty == 3:
            new_interval_days = current_interval_days * 2

        else:
            new_interval_days = current_interval_days + 3

        next_review = now + timedelta(days=new_interval_days)

        db.execute(
            text("""
                update flashcards
                set
                    difficulty = :difficulty,
                    last_review = now(),
                    next_review = now() + (:days || ' days')::interval
                where id = :flashcard_id
            """),
            {
                "difficulty": difficulty,
                "flashcard_id": flashcard_id,
                "days": new_interval_days
            }
        )

        flashcard_row = db.execute(
            text("""
                select project_id, user_id
                from flashcards
                where id = :flashcard_id
            """),
            {
                "flashcard_id": flashcard_id
            }
        ).fetchone()

        if flashcard_row:
            db.execute(
                text("""
                    insert into flashcard_reviews
                    (flashcard_id, project_id, user_id, is_correct, difficulty, elapsed_seconds)
                    values
                    (:flashcard_id, :project_id, :user_id, :is_correct, :difficulty, :elapsed_seconds)
                """),
                {
                    "flashcard_id": flashcard_id,
                    "project_id": flashcard_row[0],
                    "user_id": user["id"],
                    "is_correct": is_correct,
                    "difficulty": difficulty,
                    "elapsed_seconds": req.get("elapsed_seconds", 0)
                }
            )

        db.commit()
    

        return {"status": "ok"}

    

    except HTTPException:
        raise

    except Exception as e:
        db.rollback()
        print("ERROR review_flashcard:", e)
        raise HTTPException(status_code=500, detail="Internal error")

    finally:
        db.close()
    

@app.get("/projects/{project_id}/flashcard_results")
async def flashcard_results(
    project_id: str,
    user = Depends(verify_user)
):
    db = SessionLocal()

    total_reviews = db.execute(
        text("""
            select count(*)
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
        """),
        {"project_id": project_id, "user_id": user["id"]}
    ).scalar()

    correct_reviews = db.execute(
        text("""
            select count(*)
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
            and is_correct = true
        """),
        {"project_id": project_id, "user_id": user["id"]}
    ).scalar()

    avg_time = db.execute(
        text("""
            select avg(elapsed_seconds)
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
        """),
        {"project_id": project_id, "user_id": user["id"]}
    ).scalar()

    db.close()

    accuracy = 0
    if total_reviews and total_reviews > 0:
        accuracy = round((correct_reviews / total_reviews) * 100, 1)

    return {
        "total_reviews": total_reviews or 0,
        "correct_reviews": correct_reviews or 0,
        "accuracy": accuracy,
        "avg_time": round(avg_time or 0, 1)
    }

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

@app.get("/projects/{project_id}/quiz_attempts_summary")
    
async def quiz_attempts_summary(
        project_id: str,
        user = Depends(verify_user)
    ):
        print("QUIZ STATS CALLED", project_id)
        db = SessionLocal()

        rows = db.execute(
            text("""
                select 
                    qa.quiz_id,
                    count(*) as attempts,
                    max(qa.score) as best_score,
                    (
                        select qa2.score
                        from quiz_attempts qa2
                        where qa2.quiz_id = qa.quiz_id
                        and qa2.user_id = :user_id
                        order by qa2.id desc
                        limit 1
                    ) as last_score
                from quiz_attempts qa
                join quizzes q on qa.quiz_id = q.id
                where q.project_id = :project_id
                and qa.user_id = :user_id
                group by qa.quiz_id
            """),
            {
                "project_id": project_id,
                "user_id": user["id"]
            }
        ).fetchall()

        db.close()

        result = {}

        for r in rows:
            result[r[0]] = {
                "attempts": r[1],
                "best_score": r[2],
                "last_score": r[3]
            }

        return {"data": result}    

@app.post("/ask")
async def ask_documents(req: AskRequest):
    chunks = search_project_chunks(
        project_id=req.project_id,
        query=req.question,
        topics=req.topics,
        k=12
    )    
    print("CHUNKS FOUND:", len(chunks))
    if chunks:
        print("SAMPLE CHUNK:", chunks[0]["text"][:200])

    

    context_blocks = []

    for c in chunks:
        context_blocks.append(
            f"DOCUMENT: {c['document']} | PAGE: {c['page']}\nCONTENT:\n{c['text'][:400]}"
        )


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
- Be concise
- Avoid repeating the same concept
- Use clear paragraphs
- Use bullet points when useful

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
            (
                select chunk_text, doc_title, page
                from chunks
                where project_id = :project_id
                order by embedding <-> CAST(:embedding AS vector)
                limit 4
            )

            union

            (
                select chunk_text, doc_title, page
                from chunks
                where project_id = :project_id
                order by random()
                limit 8
            )
        """),
        {
            "project_id": project_id,
            "embedding": query_embedding
        }
    ).fetchall()

    import random

    rows = list(rows)
    random.shuffle(rows)

    # 🔥 PRENDI SOLO 5 CHUNK (invece di tutti)
    rows = rows[:5]

    db.close()

    context_blocks = []

    for r in rows:
        context_blocks.append(
            f"DOCUMENT: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:400]}"
        )

    context = "\n\n---\n\n".join(context_blocks)

    prompt = f"""
    You are a strict study tutor generating ACTIVE RECALL questions.

    You MUST follow these rules:

    - Use ONLY the provided material
    - DO NOT use external knowledge
    - If the material is unclear → ask a simpler question
    - Ask ONLY ONE question
    - The question must focus on ONE clear concept
    - Avoid vague or generic questions
    - Avoid simple definitions when possible
    - Prefer "why", "how", "what happens if"

    GOOD examples:
    - Why does X lead to Y?
    - How does X affect Y?
    - What happens if X is disrupted?

    BAD examples:
    - What is X?
    - Explain everything about X

    Material:
    {context}

    IMPORTANT:
    - Use ONLY the information explicitly written above
    - Do NOT infer missing details
    - Do NOT add knowledge not present in the text
    - If unsure, ask a simpler question based strictly on the text

    Difficulty rules:
    - easy → direct recall from text
    - medium → requires explanation or connection
    - hard → requires reasoning or implications

    Return ONLY JSON:

    {{
    "question": "...",
    "concept": "...",
    "difficulty": "easy | medium | hard",
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

@app.post("/generate_recovery_flashcards")
async def generate_recovery_flashcards(req: dict):

    project_id = req.get("project_id")

    db = SessionLocal()

    # prendi chunk random ma piccoli (focus)
    rows = db.execute(
        text("""
            select chunk_text
            from chunks
            where project_id = :project_id
            order by random()
            limit 5
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    context = "\n\n".join([r[0][:300] for r in rows])

    prompt = f"""
Generate 3 recovery flashcards.

Rules:
- Focus on reinforcing misunderstood concepts
- Keep them simple and clear
- No duplicates
- Use ONLY provided material

Return JSON:

[
  {{
    "question": "...",
    "answer": "..."
  }}
]

Material:
{context}
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.5,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    content = response.choices[0].message.content
    content = content.replace("```json","").replace("```","").strip()

    try:
        return {"flashcards": json.loads(content)}
    except:
        return {"flashcards": []}

@app.post("/active_recall_evaluate")
async def active_recall_evaluate(req: ActiveRecallEvaluateRequest):

    history_text = "\n".join(req.history or [])

    prompt = f"""
    You are a strict but supportive study tutor evaluating a student's answer.

    Question:
    {req.question}

    Previous answers:
    {history_text}

    Latest answer:
    {req.student_answer}

    Evaluation rules:

    - Determine if the student truly understood the concept
    - Do NOT be overly permissive
    - A vague or generic answer is NOT correct
    - The answer must include the key idea of the concept
    - Partial answers should be marked as "partial"
    - Only mark "correct" if the core concept is clearly expressed

    Scoring:

    - correct → clear understanding, key concept present
    - partial → some correct ideas but incomplete or unclear
    - incorrect → wrong or missing core idea

    Also:

    - Identify if important parts are missing
    - Detect incorrect claims if present
    - Provide a short helpful feedback
    - Provide a hint ONLY if the answer is not fully correct

    Return ONLY JSON:

    {{
    "evaluation": "correct | partial | incorrect",
    "score": 0-1,
    "feedback": "...",
    "missing": ["..."],
    "wrong_claims": ["..."],
    "hint": "...",
    "explanation": "clear explanation of the concept"
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
    # GENERATE FLASHCARDS
    # ======================

    context_chunks = search_project_chunks(project_id, k=20)
    context_text = "\n\n".join([c["text"][:400] for c in context_chunks])
    weak_topics_text = ", ".join(weak_topics) if weak_topics else "important concepts"

    prompt = f"""
    You are a strict study tutor generating flashcards for a focused study session.
    Generate EXACTLY 15 flashcards.
    Focus especially on these weak topics:
    {weak_topics_text}

    Rules:
    - Use ONLY the provided material
    - Do NOT use external knowledge
    - Do NOT invent information
    - Each flashcard must cover a DIFFERENT concept
    - Avoid similar or repeated questions
    - Prefer "why", "how", "what happens if"
    - Avoid simple definitions unless necessary

    Difficulty:
    - easy → direct recall
    - medium → explanation or relation
    - hard → reasoning or consequences

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
        temperature=0.7,
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
    
    if not isinstance(generated_cards, list):
        generated_cards = []

    flashcards = []

    for c in generated_cards:

        flashcard_id = str(uuid.uuid4())

        db.execute(
            text("""
                insert into flashcards
                (id, project_id, user_id, question, answer)
                values
                (:id, :project_id, :user_id, :question, :answer)
            """),
            {
                "id": flashcard_id,
                "project_id": project_id,
                "user_id": user["id"],
                "question": c.get("question"),
                "answer": c.get("answer")
            }
        )

        flashcards.append({
            "id": flashcard_id,
            "question": c.get("question"),
            "answer": c.get("answer")
        })
    db.commit()

    # ======================
    # ADAPTIVE QUIZ CONFIG
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
        quiz_questions = 15
    elif avg_accuracy < 0.8:
        quiz_questions = 20
    else:
        quiz_questions = 25

    db.close()

    return {
        "flashcards": flashcards,
        "recall_topics": weak_topics,
        "quiz": {
            "num_questions": quiz_questions,
            "difficulty": "medium",
            "focus_topics": weak_topics
        }
    }
@app.delete("/projects/{project_id}/documents/{doc_title}")
def delete_document(
        project_id: str,
        doc_title: str,
        user = Depends(verify_user)
    ):

        doc_title = unquote(doc_title)  # 🔥 FIX URL encoding

        print("DELETE DOCUMENT:", project_id, doc_title)
        user_id = user["id"]
        db = SessionLocal()

        # verifica ownership
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

        # 🔥 DELETE REAL (tutti i chunk del documento)
        db.execute(
            text("""
                delete from chunks
                where project_id = :project_id
                and doc_title = :doc_title
            """),
            {
                "project_id": project_id,
                "doc_title": doc_title
            }
        )
        print("ROWS DELETED")

        db.commit()
        db.close()

        return {"status": "deleted"}   
 
