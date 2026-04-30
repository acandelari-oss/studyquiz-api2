import os
import uuid
import base64
import io
from typing import List, Optional
import json
import requests
import random
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
from typing import Optional, List
import time
import re

def normalize_string(s: str) -> str:
    if not s: return ""
    # Sostituisce \xa0 (non-breaking space) con spazio normale
    s = str(s).replace('\xa0', ' ')
    # Riduce spazi multipli a uno solo
    return re.sub(r'\s+', ' ', s).strip()

class ActiveRecallRequest(BaseModel):
    topics: Optional[List[str]] = None
    index: int = 0

print("✅ ActiveRecallRequest model loaded with topics")



pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"




# ======================
# LOAD ENV
# ======================

load_dotenv()

topic_index = 0

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
    allow_origins=["*"],  # in dev
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
    topics: Optional[List[str]] = []
    topic: Optional[str] = None 

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

            start = 0

            while start < len(p):
                end = start + max_chars
                sub = p[start:end]

                chunks.append(sub.strip())

                start += max_chars - overlap

            continue

        if len(current_chunk) + len(p) < max_chars:

            current_chunk += "\n\n" + p

        else:

            chunks.append(current_chunk.strip())

            overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            current_chunk = overlap_text + "\n\n" + p

    if current_chunk:
        chunks.append(current_chunk.strip())
    print("📄 TOTAL CHUNKS:", len(chunks))
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
                    clean_topic = normalize_string(doc.title) if doc.title else "General"

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
                            "page": page_index + 1,
                            "topic": clean_topic
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
        # 1. Update status and clear old topics
        db.execute(text("update projects set topic_status = 'processing' where id = :project_id"), {"project_id": project_id})
        db.execute(text("delete from topics where project_id = :project_id"), {"project_id": project_id})
        db.commit()

        # Fetch chunks to analyze (up to 120 chunks)
        rows = db.execute(
            text("select chunk_text from chunks where project_id = :project_id order by page asc limit 1000"),
            {"project_id": project_id}
        ).fetchall()

        all_text_chunks = [r[0] for r in rows if r[0]]
        # Group chunks into batches of 20
        chunk_groups = [all_text_chunks[i:i+20] for i in range(0, len(all_text_chunks), 20)]
        
        seen_titles = set()

        for group in chunk_groups:
            group_text = "\n\n".join(group)
            
            # UNIVERSAL PROMPT: Works for any discipline (Medicine, Law, Engineering, etc.)
            prompt = f"""
            Act as a specialist in Instructional Design and Knowledge Organization. 
            Analyze the provided text to extract its fundamental conceptual hierarchy.
            
            GOAL:
            Organize the information into a logical structure of Macro-Categories and specific Topics to facilitate academic study.
            
            STRICT RULES:
            1. CATEGORY: Identify the primary organizational unit or domain of the text.
            - Medicine: Use anatomical regions or systems (e.g., 'NECK ANATOMY', 'DIGESTIVE SYSTEM').
            - Sciences: Use branches or chemical groups (e.g., 'INORGANIC CHEMISTRY', 'THERMODYNAMICS').
            - Law: Use codes, areas, or legal entities (e.g., 'CIVIL LAW', 'CONTRACTS').
            - General: Use major chapter themes.
            - ALWAYS USE UPPERCASE for categories.
            
            2. TOPIC: The specific concept, entity, rule, or theory (1-4 words).
            
            3. DESCRIPTION: Provide a dense, academic, and precise definition. Focus on 'What it is' and 'Its primary function or rule' (max 200 characters).
            
            4. COVERAGE: Exhaustively extract all technical terms. Do not skip any significant educational content.
            
            5. NO MICRO-CATEGORIES: Do not create a category for a single topic. Group related topics under a broader, logical Macro-Category.

            FORMAT RULES:
            - Return ONLY valid JSON.[cite: 2]
            - Importance: Rate 1-10 based on how fundamental the concept is to the subject.[cite: 2]

            JSON STRUCTURE:
            {{
            "categories": [
                {{
                "name": "CATEGORY NAME",
                "topics": [
                    {{ 
                    "title": "Topic Name", 
                    "description": "Clear academic definition", 
                    "importance": 7 
                    }}
                ]
                }}
            ]
            }}

            CONTENT TO ANALYZE:
            {group_text}
            """

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={ "type": "json_object" } # Ensures stable JSON output[cite: 2]
            )

            try:
                data = json.loads(response.choices[0].message.content.replace("```json", "").replace("```", "").strip())

                for cat in data.get("categories", []):
                    # 2. LOGICA DI UNIFICAZIONE CATEGORIE
                    raw_category = (cat.get("name") or "GENERAL").strip().upper()
                    
                    # Controlliamo se nel DB per questo progetto esiste già una categoria simile
                    # Questo evita di avere "MUSCLE ACTION" e "MUSCLE ACTIONS" separati
                    existing_cat_row = db.execute(
                        text("select category from topics where project_id = :pid and category ilike :c limit 1"),
                        {"pid": project_id, "c": raw_category}
                    ).fetchone()
                    
                    category_name = existing_cat_row[0] if existing_cat_row else raw_category
                    
                    for t_obj in cat.get("topics", []):
                        topic_title = (t_obj.get("title") or "").strip()
                        description = (t_obj.get("description") or "").strip()
                        # Recuperiamo l'importanza se presente, altrimenti default a 5
                        importance = t_obj.get("importance", 5) 

                        if not topic_title:
                            continue

                        # Manteniamo il titolo pulito senza regex aggressive che 
                        # possono rovinare acronimi medici o tecnici
                        topic_title = topic_title.strip()

                        key = topic_title.lower().strip()
                        if key in seen_titles:
                            continue
                        seen_titles.add(key)

                        print(f"➡️ TOPIC: {topic_title} [{category_name}]")

                        # 🔥 EMBEDDING
                        try:
                            print("🔥 CREO EMBEDDING...")
                            # Includiamo anche la categoria nell'input dell'embedding 
                            # per migliorare la precisione della ricerca
                            emb_input = f"{category_name} - {topic_title}: {description}"
                            
                            emb = client.embeddings.create(
                                model="text-embedding-3-small",
                                input=emb_input
                            )

                            embedding = emb.data[0].embedding
                            embedding_str = "[" + ",".join(map(str, embedding)) + "]"

                        except Exception as e:
                            print("❌ EMBEDDING ERROR:", e)
                            continue

                        # 🔥 INSERT 
                        # Ho aggiunto la colonna 'importance' nel caso volessi usarla, 
                        # se non l'hai nel DB, rimuovila dalla query
                        db.execute(
                            text("""
                                insert into topics 
                                (project_id, category, topic, description, embedding)
                                values 
                                (:project_id, :category, :topic, :description, CAST(:embedding AS vector))
                            """),
                            {
                                "project_id": project_id, 
                                "category": category_name, 
                                "topic": topic_title, 
                                "description": description,
                                "embedding": embedding_str
                            }
                        )
                db.commit() 

            except Exception as e:
                print(f"Error parsing JSON in chunk: {e}")
                continue 
                
        assign_topics_to_chunks(project_id)

        # 2. Final status update
        db.execute(text("update projects set topic_status = 'completed' where id = :project_id"), {"project_id": project_id})
        db.commit()
        print("TOPIC GENERATION COMPLETE")

    except Exception as e:
        db.rollback()
        print("BACKGROUND TOPICS ERROR:", e)
        db.execute(text("update projects set topic_status = 'error' where id = :project_id"), {"project_id": project_id})
        db.commit()
    finally:
        db.close()

def assign_topics_to_chunks(project_id: str):
    db = SessionLocal()

    try:
        print("🔗 START TOPIC ASSIGNMENT")

        db.execute(text("""
            update chunks c
            set topic = sub.topic
            from (
                select
                    c.id as chunk_id,
                    t.topic,
                    row_number() over (
                        partition by c.id
                        order by c.embedding <-> t.embedding
                    ) as rn
                from chunks c
                join topics t
                    on c.project_id = t.project_id
                where c.project_id = :project_id
            ) sub
            where c.id = sub.chunk_id
              and sub.rn = 1
        """), {"project_id": project_id})

        db.commit()
        print("✅ TOPIC ASSIGNMENT DONE")

    except Exception as e:
        db.rollback()
        print("❌ TOPIC ASSIGNMENT ERROR:", e)

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
            
            import re

            def clean_text(text):
                if not text:
                    return ""

                # separa parole attaccate tipo "TEAMBefore"
                text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)

                # separa numeri e lettere
                text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
                text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)

                # aggiunge spazio dopo punto
                text = re.sub(r'\.(\w)', r'. \1', text)

                # normalizza spazi
                text = re.sub(r'\s+', ' ', text)

                return text.strip()
            # ======================
            # SAVE CHUNKS
            # ======================
            for doc in docs:
                yield f"Processing document: {doc.title}\n"

                pdf_bytes = base64.b64decode(doc.file_bytes)
                pdf_stream = io.BytesIO(pdf_bytes)

                reader = PdfReader(pdf_stream)

                for page_index, page in enumerate(reader.pages):
                    

                    yield f"Page {page_index+1}\n"

                    page_text = page.extract_text()

                    page_text = clean_text(page_text)

                    if not page_text or not page_text.strip():
                        yield f"OCR page {page_index+1}\n"
                        page_text = ocr_pdf_page(pdf_bytes, page_index)

                        if not page_text:
                            continue

                    chunks = chunk_text(page_text)
                    chunks = [clean_text(c) for c in chunks]
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

                        # --- AGGIUNGI QUI IL LOG DI DEBUG ---
                        print(f"DEBUG: Processando chunk {i} per il documento: {doc.title}")
                        if not doc.title:
                            print("ATTENZIONE: doc.title è vuoto o None!")
    # ------------------------------------

                        db.execute(
                            text("""
                                insert into chunks
                                (project_id, doc_title, chunk_text, embedding, page, topic)
                                values
                                (:project_id, :doc_title, :chunk_text, CAST(:embedding AS vector), :page, :topic)
                            """),
                            {
                                "project_id": project_id,
                                "doc_title": doc.title,
                                "chunk_text": chunk,
                                "embedding": embedding_str,
                                "page": page_index + 1,      
                                "topic": normalize_string(doc.title) if doc.title else "General"
                            }
                        )
                        db.commit()
                        print(f"CHUNK {i} SALVATO CON TOPIC: {doc.title}")

            

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

def expand_topics_with_db(project_id, topics, db):
    if not topics:
        return []

    expanded = []

    try:
        rows = db.execute(
            text("""
                select topic
                from chunks
                where project_id = :project_id
                and topic is not null
                group by topic
                limit 50
            """),
            {
                "project_id": project_id
            }
        ).fetchall()

        all_topics = [normalize_string(row[0]) for row in rows]

        for t in topics:
            t_norm = normalize_string(t)

            matches = [
                topic for topic in all_topics
                if t_norm.lower().replace("actions", "action") == topic.lower()
            ]

            print(f"🔎 AUTO-MATCH for '{t}' (norm: '{t_norm}'):", matches)

            expanded.extend(matches)
    except Exception as e:
        print("❌ ERROR IN expand_topics_with_db:", e)
        return topics

    if not expanded:
        return topics

    return list(set(expanded))
# ======================
# GENERATE QUIZ
# ======================

@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(
    project_id: str,
    req: QuizRequest,
    user = Depends(verify_user)
):
    # 🔥 PRINT 1 — INIZIO FUNZIONE
    print("🔥 ENTER generate_quiz")

    user_id = user["id"]
    db = SessionLocal()

    existing_questions = db.execute(
        text("""
            select question
            from quiz_questions q
            join quizzes z on q.quiz_id = z.id
            where z.project_id = :project_id
        """),
        {"project_id": project_id}
    ).fetchall()

    existing_texts = set(q[0].strip().lower() for q in existing_questions)

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

    print(f"DEBUG: Avvio generazione quiz per progetto {project_id}")
    print("📥 TOPICS:", req.topics)
    
    # 1. RETRIEVAL POTENZIATO (120 chunk)
    query_text = " ".join(req.topics) if req.topics else "General overview of the provided documents"
    emb_res = client.embeddings.create(model="text-embedding-3-small", input=query_text)
    query_embedding = emb_res.data[0].embedding

    rows = []

    # 🔥 prendiamo tutti i chunk una volta sola
    all_rows = db.execute(
        text("""
            SELECT id, chunk_text, doc_title, page, topic
            FROM chunks
            WHERE project_id = :project_id
        """),
        {"project_id": project_id}
    ).fetchall()

    if req.topics:
        normalized_targets = [normalize_string(t) for t in req.topics]

        for r in all_rows:
            topic_db = normalize_string(r[4])

            for target in normalized_targets:
                if (
                    topic_db == target
                    or topic_db.startswith(target)
                    or target.startswith(topic_db)
                ):
                    rows.append(r)
                    break

        # 🔥 rimuovi duplicati
        rows = list({r[0]: r for r in rows}.values())

        print("🎯 FILTERED QUIZ ROWS:", len(rows))

        import random
        random.shuffle(rows)

        rows = rows[:min(len(rows), 80)]

    else:
        print("🌍 GLOBAL QUIZ MODE (SAFE)")

    if req.topics is None:
        # 👉 SOLO per quiz da sidebar
        rows = db.execute(
            text("""
                SELECT id, chunk_text, doc_title, page, topic
                FROM chunks
                WHERE project_id = :project_id
                ORDER BY embedding <-> CAST(:embedding AS vector)
                LIMIT 30
            """),
            {
                "project_id": project_id,
                "embedding": str(query_embedding)
            }
        ).fetchall()
    else:
        # 👉 fallback sicurezza (non dovrebbe mai succedere)
        import random
        random.shuffle(all_rows)
        rows = all_rows[:min(len(all_rows), 80)]

    # 🔍 DEBUG CHUNKS
    for r in rows[:3]:
        print("📄 SAMPLE CHUNK:", r[1][:200])

    chunk_topic_map = {
        str(r[0]): " ".join(str(r[4]).split()) 
        for r in rows if r[4]
    }

    if req.topics:
        active_topics = [normalize_string(t) for t in req.topics]
    else:
        active_topics = list(set(
            normalize_string(r[4]) for r in rows if r[4]
        )) or ["General"]

    # 4. Fallback se non ci sono topic
    if not active_topics:
        active_topics = ["General"]

    print("🎯 ACTIVE TOPICS (CLEANED):", active_topics)
    

    if not rows:
        db.close()
        return {"quiz": []}

    

    # 2. GENERAZIONE A BATCH (Garantisce 30 domande e qualità)
    all_questions = []
    seen_texts = set()
    max_attempts = 6
    target_count = req.num_questions if req.num_questions else 30

    while len(all_questions) < target_count and max_attempts > 0:
        import random

        # 🔥 scegli chunk diversi ogni volta
        valid_rows = [
            r for r in rows
            if r[1] and r[4] and normalize_string(r[4]) in active_topics
        ]

        sample_size = min(15, len(valid_rows))
        sampled_rows = random.sample(valid_rows, sample_size)

        context = "\n\n".join([
            f"### CHUNK_ID: {r[0]} | TOPIC: {r[4]}\n{r[1]}"
            for r in sampled_rows
        ])
        max_attempts -= 1  # Decrementiamo qui una volta sola
        remaining = target_count - len(all_questions)
        batch_size = min(15, remaining) 
        
        print(f"DEBUG: Tentativo {6 - max_attempts}, domande rimanenti: {remaining}")

        topics_str = ", ".join(active_topics)
        avoid_str = ", ".join(list(existing_texts)[:20])

        system_prompt = f"""
        You are an academic researcher. Generate {batch_size} high-quality questions in {req.language}.
        Available Topics: {topics_str}
        Avoid these topics: {avoid_str}

        STRICT RULES:
        1. Use ONLY the provided material.
        2. Professional tone.
        3. Return ONLY valid JSON.
        """

        user_prompt = f"""
        Material:
        {context[:15000]}

        Return EXACTLY this JSON structure:
        {{
          "questions": [
            {{
              "question": "The question text",
              "options": ["A", "B", "C", "D", "E"],
              "correct_answer": 0,
              "topic": "Topic name",
              "explanation": "Short explanation",
              "chunk_id": "Must match the CHUNK_ID from the text"
            }}
          ]
        }}
        """

        try:
            # Rimuovi qualsiasi riga che faccia prompt = prompt.replace(...)
            
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt}, # Usiamo system_prompt
                    {"role": "user", "content": user_prompt}     # Usiamo user_prompt
                ],
                response_format={ "type": "json_object" }
            )
            
            content = response.choices[0].message.content
            print("🧠 GPT RAW:", content[:500])
            
            data = json.loads(content)
            current_batch = data.get("questions", [])
            
            if isinstance(current_batch, list):
                for q in current_batch:
                    if len(all_questions) >= target_count:
                        break

                    txt = q.get("question", "").strip().lower()
                    chunk_id = str(q.get("chunk_id", ""))

                    if txt and txt not in seen_texts and txt not in existing_texts:
                        seen_texts.add(txt)

                        correct_val = q.get("correct_answer", q.get("correct", 0))
                        
                        # 2. Inseriamo ENTRAMBI i nomi campo per compatibilità totale
                        q["correct_answer"] = correct_val
                        q["correct"] = correct_val

                        print("🧩 CHUNK ID:", chunk_id)
                        print("🧩 TOPIC FROM MAP:", chunk_topic_map.get(chunk_id))
                        print("🧠 TOPIC FROM GPT:", q.get("topic"))

                        # 🔥 SINCRONIZZAZIONE TOPIC E CAMPI
                        topic_from_db = chunk_topic_map.get(chunk_id)
                        if topic_from_db:
                            q["topic"] = topic_from_db
                        else:
                             # ❗ niente macro-topic → fallback neutro
                            print("⚠️ FALLBACK USATO per chunk_id:", chunk_id)
                            q["topic"] = "General"

                        # Assicuriamoci che il campo per QuizView sia corretto
                        # Se GPT ha generato 'correct', lo rinominiamo per il frontend
                        if "correct" in q and "correct_answer" not in q:
                            q["correct_answer"] = q["correct"]
                        
                        print("✅ FINAL TOPIC USATO:", q["topic"])
                        all_questions.append(q)
                        

        except Exception as e:
            print(f"❌ Errore nel batch: {e}")
            continue

    db.close()
    quiz_id = str(uuid.uuid4())

    db_save = SessionLocal()
    try:
        print("🔥 STO PER FARE INSERT QUIZ")

        # ✅ INSERT QUIZ (UNA SOLA VOLTA)
        db_save.execute(
            text("""
                insert into quizzes (id, project_id, user_id, created_at, num_questions, difficulty)
                values (:id, :project_id, :user_id, now(), :num_questions, :difficulty)
            """),
            {
                "id": quiz_id,
                "project_id": project_id,
                "user_id": user_id,
                "num_questions": len(all_questions),
                "difficulty": req.difficulty or "medium"
            }
        )

        print("🔥 INSERT QUIZ ESEGUITO")

        # ✅ ORA SALVI LE DOMANDE (CORRETTO)
        print("🔥 INIZIO SALVATAGGIO DOMANDE")

        for i, q in enumerate(all_questions):
            print("👉 SALVO DOMANDA:", q["question"])

            result = db_save.execute(
                text("""
                    insert into quiz_questions (
                        quiz_id,
                        question,
                        correct_answer,
                        options,
                        topic,
                        question_order
                    )
                    values (
                        :quiz_id,
                        :question,
                        :correct_answer,
                        :options,
                        :topic,
                        :question_order
                    )
                    returning id
                """),
                {
                    "quiz_id": quiz_id,
                    "question": q["question"],
                    "correct_answer": q["correct_answer"],
                    "options": json.dumps(q["options"]),
                    "topic": q.get("topic"),
                    "question_order": i
                }
            )

            new_id = result.fetchone()[0]
            q["id"] = str(new_id)   # 👈 QUESTO È IL FIX CRITICO
        db_save.commit()
        print("✅ COMMIT FATTO")

    except Exception as e:
        db_save.rollback()
        print("❌ ERRORE SALVATAGGIO QUIZ:", e)

    finally:
        db_save.close()

    return {
        "quiz_id": quiz_id,
        "questions": all_questions
    }

    
@app.get("/projects/{project_id}/quiz_stats")
def get_quiz_stats(project_id: str):

    db = SessionLocal()

    try:
        result = db.execute(text("""
            select 
                quiz_id,
                count(*) as attempts,
                max(score) as best_score,
                avg(score) as avg_score,
                (
                    select score 
                    from quiz_attempts qa2
                    where qa2.quiz_id = qa.quiz_id
                    order by created_at desc
                    limit 1
                ) as last_score
            from quiz_attempts qa
            where project_id = :project_id
            group by quiz_id
        """), {"project_id": project_id})

        rows = result.fetchall()

        return [
            {
                "quiz_id": r.quiz_id,
                "attempts": r.attempts,
                "best_score": r.best_score,
                "avg_score": r.avg_score,
                "last_score": r.last_score
            }
            for r in rows
        ]

    finally:
        db.close()




class AskRequest(BaseModel):
    project_id: str
    question: str
    topics: Optional[List[str]] = []
    history: list = []
    expand_search: bool = False

from typing import Optional



@app.post("/projects/{project_id}/generate_quiz_stream")
async def generate_quiz_stream(
    project_id: str,
    req: QuizRequest,
    user = Depends(verify_user)
):
    user_id = user["id"]
    db = SessionLocal()
    
    # 1. Create the Quiz ID and Database Record FIRST
    quiz_id = str(uuid.uuid4())
    try:
        db.execute(
            text("""
                insert into quizzes (id, project_id, user_id, title, created_at)
                values (:id, :project_id, :user_id, :title, now())
            """),
            {
                "id": quiz_id,
                "project_id": project_id,
                "user_id": user_id,
                "title": f"Quiz on {', '.join(req.topics) if req.topics else 'Study Material'}"
            }
        )
        db.commit()
    except Exception as e:
        print(f"Database Error: {e}")
        db.rollback()
    finally:
        db.close()

    async def quiz_generator():
        # 2. Yield the ID to the frontend first
        yield f"ID:{quiz_id}\n"
    

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

        chunk_topic_map = {r[0]: r[4] for r in rows if r[4]}

        random.shuffle(rows)

        db.close()

        material_blocks = []

        for r in rows:

            text_chunk = r[0].lower()
            chunk_topic = r[3] if (len(r) > 3 and r[3]) else r[1]

            if topics:
                if not any(topic.lower() in text_chunk for topic in topics):
                    continue   # 🔥 SCARTA CHUNK NON RILEVANTI

            material_blocks.append(
                f"### TOPIC: {chunk_topic}\nFILE: {r[1]} | PAGE: {r[2]}\nCONTENT:\n{r[0][:500]}"
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
        Each section starts with '### TOPIC: [Name]'.

        You are NOT allowed to use external knowledge.

        Material:
        {context}

        Generate {n} high-quality multiple choice study questions.

        IMPORTANT:
        Each question MUST focus on a COMPLETELY DIFFERENT concept.

        CRITICAL INSTRUCTIONS FOR QUESTION VARIETY:
        - NEVER start a question with "What is", "What are", or "Define".
        - Focus on APPLICATION: Create scenarios where the user must apply a rule or concept.
        - Focus on MECHANISMS: Ask "How does [X] affect [Y]?" or "In what sequence does [X] occur?"
        - Focus on COMPARISON: "Which feature distinguishes [X] from [Y]?"
        - Avoid trivial definitions; test for deep understanding and cause-effect relationships.

        CRITICAL:
        - Do NOT generate questions about the same topic even if phrased differently
        - Avoid paraphrasing the same concept
        - Each question must test a UNIQUE concept
        - If you cannot find enough different concepts, generate fewer questions instead
        - Each question MUST include the CHUNK_ID it was generated from.
        - Use the CHUNK_ID from the chunk you used to generate the question.
        - Each question must be based on a DIFFERENT part of the material.
        - Do NOT repeat questions or concepts.
        - Exactly ONE answer must be correct.
        - All other options must be clearly incorrect.
        - Do NOT use "All of the above".
        - All answer options must be relevant to the question.

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
        - Return STRICT JSON ARRAY format.

        

        Return a valid JSON object with this structure:

        {
        "questions": [
            {
            "question": "...",
            "options": ["...", "...", "...", "...", "..."],
            "correct": 0,
            "topic": "...",
            "explanation": "Short explanation",
            "explanation_long": "2-3 sentences maximum",
            "source_document": "Exact file name",
            "source_page": "Page number"
            }
        ]
        }
        """

            response = await asyncio.to_thread(
                client.chat.completions.create,
                model="gpt-4o-mini",
                messages=[{"role":"user","content":prompt}],
                temperature=0.7
            )

            content = response.choices[0].message.content.strip()

            print("RAW RESPONSE:", content[:500])
            

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

        # --- 1. FILTRO DUPLICATI E PULIZIA ---
        seen_questions = set()
        unique_questions = []

        for q in questions:
            text_q = q.get("question", "").strip().lower()
            key = " ".join(text_q.split()[:8])
            if key not in seen_questions:
                seen_questions.add(key)
                unique_questions.append(q)

        # --- 2. GENERAZIONE DOMANDE MANCANTI ---
        if len(unique_questions) < req.num_questions:
            missing = req.num_questions - len(unique_questions)
            extra = await generate_batch(missing)
            for q in extra:
                text_q = q.get("question", "").strip().lower()
                key = " ".join(text_q.split()[:8])
                if key not in seen_questions:
                    seen_questions.add(key)
                    unique_questions.append(q)
                if len(unique_questions) >= req.num_questions:
                    break

        # --- 3. SALVATAGGIO NEL DB E INVIO STREAM ---
        for i, q in enumerate(unique_questions[:req.num_questions]):
            correct_val = q.get("correct", q.get("correct_answer", 0))
            q["correct"] = int(correct_val)
            
            topic_val = q.get("topic")
            if not topic_val or topic_val == "...":
                topic_val = topics[0] if topics else "General"
            q["topic"] = topic_val

            db_save = SessionLocal()
            try:
                db_save.execute(
                    text("""
                        insert into quiz_questions 
                        (quiz_id, question_order, question, options, correct, explanation, explanation_long, source_document, source_page, topic)
                        values 
                        (:quiz_id, :order, :question, :options, :correct, :explanation, :explanation_long, :doc, :page, :topic)
                    """),
                    {
                        "quiz_id": quiz_id,
                        "order": i,
                        "question": q.get("question"),
                        "options": json.dumps(q.get("options")),
                        "correct": q["correct"],
                        "explanation": q.get("explanation"),
                        "explanation_long": q.get("explanation_long"),
                        "doc": q.get("source_document"),
                        "page": q.get("source_page"),
                        "topic": q["topic"]
                    }
                )
                db_save.commit()
            except Exception as e:
                print(f"❌ Errore salvataggio domanda {i}: {e}")
                db_save.rollback()
            finally:
                db_save.close()

            yield json.dumps(q) + "\n"

    # <-- QUESTO RETURN CHIUDE IL quiz_generator (4 spazi di rientro)
    # <-- QUESTO RETURN CHIUDE LA FUNZIONE PRINCIPALE (allineato a 'async def')
    return StreamingResponse(quiz_generator(), media_type="text/event-stream")

    
@app.post("/projects/{project_id}/generate_flashcards")
async def generate_flashcards(
    project_id: str,
    req: dict = None,
    user = Depends(verify_user)
):

    num_cards = 10
    user_id = user["id"]

    def normalize_topic(t):
        return " ".join(str(t).split()).strip()

    topics = []
    db = SessionLocal()

    if req and isinstance(req, dict):
        num_cards = req.get("num_cards", 10)
        raw_topics = req.get("topics", [])
        topics = [normalize_topic(t) for t in raw_topics]
        topics = [normalize_string(t) for t in raw_topics if t]
        print("🎯 CLEAN TOPICS:", topics)

        

    print("🎯 NORMALIZED TOPICS:", topics)

    
    # 🔥 RETRIEVAL BILANCIATO PER TOPIC
    

    rows = []
    seen_ids = set()

    if topics:
        per_topic_limit = max(4, (num_cards // max(len(topics), 1)) + 2)

        rows = db.execute(
            text("""
                select id, chunk_text, doc_title, page, topic
                from chunks
                where project_id = :project_id
                and (
                    """ + " OR ".join([
                        f"lower(topic) like :kw{i}" for i in range(len(topics))
                    ]) + """
                )
                limit :limit
            """),
            {
                "project_id": project_id,
                "limit": num_cards * 4,
                **{
                    f"kw{i}": f"%{t.split()[0].lower()}%"
                    for i, t in enumerate(topics)
                }
            }
        ).fetchall()

    else:
        emb = client.embeddings.create(
            model="text-embedding-3-small",
            input="important study concepts"
        )
        query_embedding = emb.data[0].embedding

        rows = db.execute(
            text("""
                select id, chunk_text, doc_title, page, topic
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

    topic_counts = {}
    for r in rows:
        t = r[4] or "UNKNOWN"
        topic_counts[t] = topic_counts.get(t, 0) + 1

    print("🧠 FLASHCARD RETRIEVAL DISTRIBUTION:", topic_counts)

    db.close()

    chunk_topic_map = {
        str(r[0]): " ".join(str(r[4]).split())
        for r in rows if r[4]
    }

    context_text = "\n\n".join([
        f"### CHUNK_ID: {r[0]} | TOPIC: {r[4]} | FILE: {r[2]} | PAGE: {r[3]}\n{r[1]}"
        for r in rows if r[1]
    ])

    prompt = f"""
    You MUST generate EXACTLY {num_cards} flashcards.

    FOCUS TOPIC:
    {", ".join(topics) if topics else "GENERAL"}

    CRITICAL RULE (VERY IMPORTANT):
    - Distribute flashcards across ALL provided topics
    - Each topic MUST be covered
    - Do NOT focus on only one topic


    You are a strict study tutor.

    STRICT RULES:
    - Use ONLY the provided material
    - DO NOT use external knowledge
    - If information is missing, skip the card
    - Each flashcard must test ONE clear concept
    - Each flashcard MUST include a valid chunk_id
    - The chunk_id MUST match one CHUNK_ID from the material
    - You are NOT allowed to invent chunk IDs
    - Do NOT generate the topic yourself
    - The topic will be assigned by the system from the chunk
    - Avoid generic or vague questions
    - Avoid repeating similar questions
    - Avoid simple definition-only questions when possible
    - Prefer cause-effect, mechanisms, reasoning

    Return EXACTLY {num_cards} items.

    Return ONLY JSON:

    [
    {{
        "question": "...",
        "answer": "...",
        "concept": "...",
        "chunk_id": "Must match the CHUNK_ID from the material",
        "difficulty": "easy | medium | hard"
    }}
    ]

    Study material:
    {context_text}
    """

    print("🎯 USING TOPIC:", topics)
    print("🧠 CONTEXT PREVIEW:", context_text[:300])

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

            q = card.get("question", "").strip().lower()
            a = card.get("answer", "").strip()
            chunk_id = str(card.get("chunk_id", "")).strip()

            # filtri qualità base
            if len(q) < 10 or len(a) < 5:
                continue

            if "what is" in q and len(a.split()) < 5:
                continue

            if chunk_id not in chunk_topic_map:
                print("❌ INVALID CHUNK_ID:", chunk_id)
                continue

            topic_from_db = chunk_topic_map.get(chunk_id)
            if not topic_from_db:
                print("⚠️ NO TOPIC FOUND FOR CHUNK:", chunk_id)
                continue

            card["topic"] = topic_from_db

            if q not in seen:
                seen.add(q)
                unique_flashcards.append(card)

        flashcards = unique_flashcards[:num_cards]

        # 🔥 SE MANCANO → GENERA FINO A COMPLETARE
        attempt = 0
        max_attempts = 3
        while len(flashcards) < num_cards and attempt < max_attempts:

            missing = num_cards - len(flashcards)

            print(f"⚠️ Filling missing flashcards: {missing}")

            extra_prompt = f"""
            Generate {missing} NEW flashcards.

            STRICT RULES:
            - Use ONLY the provided material
            - DO NOT use external knowledge
            - Each flashcard MUST include a valid chunk_id
            - The chunk_id MUST match one CHUNK_ID from the material
            - You are NOT allowed to invent chunk IDs
            - Do NOT generate the topic yourself
            - Avoid repeating previous concepts
            - Avoid similar questions
            - Be specific and concrete

            Return ONLY JSON:

            [
            {{
                "question": "...",
                "answer": "...",
                "concept": "...",
                "chunk_id": "Must match the CHUNK_ID from the material",
                "difficulty": "easy | medium | hard"
            }}
            ]

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

                q = card.get("question", "").strip().lower()
                a = card.get("answer", "").strip()
                chunk_id = str(card.get("chunk_id", "")).strip()

                # filtri qualità base
                if len(q) < 10 or len(a) < 5:
                    continue

                if chunk_id not in chunk_topic_map:
                    print("❌ INVALID CHUNK_ID (extra):", chunk_id)
                    continue

                topic_from_db = chunk_topic_map.get(chunk_id)
                if not topic_from_db:
                    print("⚠️ NO TOPIC FOUND FOR EXTRA CHUNK:", chunk_id)
                    continue

                card["topic"] = topic_from_db

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
        if not card.get("topic"):
            print("⚠️ FLASHCARD WITHOUT TOPIC SKIPPED:", card.get("question"))
            continue

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
                "topic": card.get("topic")
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
    def normalize_topic(t):
        return " ".join(str(t).split()).strip()

    

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=full_query
    )

    query_embedding = emb.data[0].embedding

    # ======================
    # VECTOR SEARCH + RANDOM MIX
    # ======================

    if topics:
        rows = []

        for topic in topics:
            topic_norm = normalize_string(topic)
            topic_keyword = topic_norm.split()[0] if topic_norm else ""

            topic_rows = db.execute(
                text("""
                    select chunk_text, doc_title, page, topic
                    from chunks
                    where project_id = :project_id
                    and (
                        lower(topic) like :topic_keyword
                        OR lower(chunk_text) like :topic_keyword
                    )
                    order by embedding <-> CAST(:embedding AS vector)
                    limit :k_per_topic
                """),
                {
                    "project_id": project_id,
                    "topic_keyword": f"%{topic_keyword.lower()}%",
                    "embedding": query_embedding,
                    "k_per_topic": max(3, k // max(len(topics), 1))
                }
            ).fetchall()

            print(f"📚 SEARCH_PROJECT_CHUNKS topic '{topic}' (keyword '{topic_keyword}') -> {len(topic_rows)} rows")

            rows.extend(topic_rows)
            unique = {}
            for r in rows:
                unique[r[0]] = r  # r[0] = id

            rows = list(unique.values())
            rows = rows[:15]
    else:
        rows = db.execute(
            text("""
                select chunk_text, doc_title, page, topic
                from chunks
                where project_id = :project_id
                order by embedding <-> CAST(:embedding AS vector)
                limit :k
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

    def normalize(s):
        return " ".join(str(s).lower().split())

    chunks = []

    for r in rows:

        text_chunk = r[0]
        chunk_topic = r[3]

        

        chunks.append({
            "text": text_chunk,
            "document": r[1],
            "page": r[2],
            "topic": chunk_topic
        })

    print("📦 CHUNKS RETRIEVED:", len(chunks))
    
    

    return chunks[:k]
from sqlalchemy import text as sql_text

@app.get("/projects/{project_id}/topics")
async def get_topics(project_id: str):
    db = SessionLocal()
    try:
        # We now select category, topic, and description
        result = db.execute(
            sql_text("""
                SELECT category, topic, description 
                FROM topics 
                WHERE project_id = :project_id
                AND topic IS NOT NULL 
                ORDER BY category ASC, topic ASC
            """), 
            {"project_id": project_id}
        )
        rows = result.fetchall()
        
        # We format them into the structured object your UI needs
        return {"topics": [
            {
                "category": r[0] or "General",
                "topic": r[1],
                "description": r[2] or "",
                "difficulty": "medium",
                "accuracy": 50
            } for r in rows
        ]}
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
                    sum(case when qa.is_correct then 1 else 0 end) as correct,
                    count(*) as total
            from quiz_answers qa
            join quiz_questions qq on qa.question_id = qq.id
            join quizzes q on qq.quiz_id = q.id
            where q.project_id = :project_id
            group by qq.topic
        """),
        
        {"project_id": project_id}
    
    ).fetchall()
    print("🔥 ACCURACY DEBUG topic_rows :", topic_rows)
    topic_mastery = {
    r[0]: {
        "correct": int(r[1]),
        "total": int(r[2])
    }
    for r in topic_rows
}

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
async def save_quiz_attempt(req: dict, user = Depends(verify_user)):
    # --- FIX: Controllo validità quiz_id ---
    quiz_id = req.get("quiz_id")
    project_id = req.get("project_id")
    if not quiz_id or quiz_id == "":
        # Se non c'è l'ID, non salviamo ma non facciamo crashare il server
        return {"status": "error", "message": "quiz_id missing"}

    db = SessionLocal()
    try:
        db.execute(
            text("""
                insert into quiz_attempts
                (quiz_id, user_id, project_id, score, total_questions)
                values
                (:quiz_id, :user_id, :project_id, :score, :total_questions)
            """),
            {
                "quiz_id": quiz_id,
                "user_id": user["id"],
                "project_id": project_id,
                "score": req.get("score", 0),
                "total_questions": req.get("total_questions", 0)
            }
        )
        answers = req.get("answers", [])

        for a in answers:
            db.execute(
                text("""
                    insert into quiz_answers (question_id, is_correct)
                    values (:question_id, :is_correct)
                """),
                {
                    "quiz_id": quiz_id,
                    "question_id": a.get("question_id"),
                    "is_correct": a.get("is_correct", False)
                }
            )
        db.commit()
    except Exception as e:
        db.rollback()
        print(f"Database Error: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()

    return {"status": "saved"}

@app.get("/projects/{project_id}/quiz_attempts_summary")
async def quiz_attempts_summary(project_id: str, user = Depends(verify_user)):
    # --- AGGIUNGI SOLO QUESTO CONTROLLO ---
    if not project_id or project_id == "" or project_id == "undefined":
        return {"data": {}}
    # ---------------------------------------

    db = SessionLocal()
    try:
        # Il resto del tuo codice rimane identico
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

        result = {}
        for r in rows:
            result[str(r[0])] = {
                "attempts": r[1],
                "best_score": r[2],
                "last_score": r[3]
            }
        return {"data": result}
    except Exception as e:
        print(f"Errore stats: {e}")
        return {"data": {}} # Protezione extra: se c'è un errore, ritorna dati vuoti
    finally:
        db.close()

@app.post("/ask")
async def ask_documents(req: AskRequest):
    print("HISTORY RECEIVED:", req.history)

    # 🔥 STEP 1 — COSTRUISCI SEARCH QUERY CON HISTORY
    search_query = req.question

    if req.history:
        last_user_messages = [
            m.get("content")
            for m in req.history
            if m.get("role") == "user"
        ][-2:]

        if last_user_messages:
            search_query = " ".join(last_user_messages)

    print("SEARCH QUERY:", search_query)

    # 🔥 STEP 2 — USA search_query (NON req.question)
    chunks = search_project_chunks(
        project_id=req.project_id,
        query=search_query,   # 👈 QUESTA È LA MODIFICA CHIAVE
        topics=req.topics,
        k=12
    )    
    print("CHUNKS FOUND:", len(chunks))
    if chunks:
        print("SAMPLE CHUNK:", chunks[0]["text"][:200])

    

    context_blocks = []

    for c in chunks:
        context_blocks.append(
            f"DOCUMENT: {c['document']} | PAGE: {c['page']}\nCONTENT:\n{c['text'][:600]}"
        )


    context = "\n\n---\n\n".join(context_blocks) 
    print("CONTEXT LENGTH:", len(context))
    # 3️⃣ costruzione contesto
    
    history_text = ""

    if req.history:
        for msg in req.history:
            role = msg.get("role")
            content = msg.get("content")

            if not content:
                continue

            if role == "user":
                history_text += f"Student: {content}\n"
            elif role == "assistant":
                history_text += f"Tutor: {content}\n"

    if getattr(req, 'expand_search', False):
        instruction_mode = """
        - You are in 'GLOBAL KNOWLEDGE' mode.
        - Start from the provided Context, but if it's not enough or you can explain better, 
          use your full AI knowledge base.
        - Provide a rich, detailed, and helpful explanation.
        """
        current_temp = 0.6 # Più creativo
    else:
        instruction_mode = """
        - You are in 'STRICT MODE'.
        - Use ONLY the material provided in the Context.
        - If the answer is not in the material, say: 'I'm sorry, I can't find this in your documents.'
        - DO NOT use external knowledge.
        """
        current_temp = 0.1 # Più preciso e fedele al testo

    prompt = f"""
    You are an expert study tutor helping a student understand material deeply.

    IMPORTANT:
    - This is an ongoing conversation.
    - The student may ask follow-up questions.
    - You MUST use previous conversation context to refine and expand your answers.
    - Do NOT restart explanations from scratch if the question is a follow-up.
    - Stay focused on the SAME concept unless the student changes topic.

    Rules:
    - If relevant info exists, explain it clearly
    - If partial, expand logically using the context
    - Be precise and avoid generic answers
    - When needed, connect the answer to previous messages
    - Use clear paragraphs or bullet points

    Context:
    {context}

    Conversation so far:
    {history_text}

    Current question:
    {req.question}

    Answer as a tutor helping the student progressively understand the topic.
    """


    # 4️⃣ GPT
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful study tutor."},
            {"role": "user", "content": prompt}
        ]
    )

    return {"answer": response.choices[0].message.content}



print("🔥 MODEL FIELDS:", ActiveRecallRequest.__fields__.keys())


# main.py

# main.py

@app.post("/projects/{project_id}/active_recall_question")
async def active_recall_question(project_id: str, req: ActiveRecallRequest, user = Depends(verify_user)):
    # 1. Normalizzazione stringhe (rimuoviamo ogni dubbio sugli spazi)
    def super_clean(s):
        return re.sub(r'\s+', ' ', str(s).replace('\xa0', ' ')).strip()

    topics = [super_clean(t) for t in req.topics if t]
    
    if not topics:
        return {"question": "No topics available", "concept": "General"}

    # 2. ROTAZIONE FORZATA
    # Usiamo l'indice del frontend per pescare il topic
    current_focus = topics[req.index % len(topics)]
    
    # Estraiamo la parola chiave (es. da "Block Actions" prendiamo "Block")
    # Questo serve per il matching nel DB se il nome intero è corrotto
    keyword = current_focus.split()[0] if current_focus else ""

    db = SessionLocal()
    
    # 🔥 PRENDI I TOPIC DAL FRONTEND
    topics = req.topics or []

    # 🔥 NORMALIZZA
    topics = [normalize_string(t) for t in topics if t]

    # 🔥 ROTAZIONE
    if topics:
        current_focus = topics[req.index % len(topics)]
    else:
        current_focus = "General"

    # 🔥 KEYWORD SEMPLICE (prima parola)
    keyword = current_focus.split(" ")[0]

    print("🎯 CURRENT FOCUS:", current_focus)
    print("🔑 KEYWORD:", keyword)
 
    query_text = " ".join(req.topics) if req.topics else "general study content"

    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=query_text
    )

    query_embedding = emb.data[0].embedding

    rows = db.execute(
        text("""
            SELECT chunk_text, doc_title, page, topic
            FROM chunks
            WHERE project_id = :project_id
            AND (
                topic ILIKE :full_focus             -- Esempio: %Block Actions%
                OR topic ILIKE :keyword_focus       -- Esempio: %Block%
                OR chunk_text ILIKE :keyword_focus  -- Cerca "Block" nel testo
            )
            ORDER BY embedding <-> CAST(:embedding AS vector)
            LIMIT 12
        """),
        {
            "project_id": project_id,
            "full_focus": f"%{current_focus}%",
            "keyword_focus": f"%{keyword}%",
            "embedding": query_embedding
        }
    ).fetchall()
    db.close()

    # ... (il resto del codice per generare la risposta con GPT)

    if not rows:
        return {"question": "No context found for this topic.", "concept": current_focus}

    # 5. Prepariamo il contesto per GPT
    random.shuffle(rows)
    selected_rows = rows[:5]
    
    context_blocks = []
    for r in selected_rows:
        block = f"SOURCE: {r[1]} (Page {r[2]}) | TOPIC: {r[3]}\nCONTENT: {r[0]}"
        context_blocks.append(block)

    context = "\n\n---\n\n".join(context_blocks)

    # 6. Prompt ottimizzato per varietà e precisione
    prompt = f"""
    You are a professional tutor specializing in the Active Recall method.
    TARGET TOPIC: {current_focus}
    
    CONTEXT MATERIAL:
    {context}

    STRICT RULES:
    1. Focus ONLY on "{current_focus}".
    2. Generate an open-ended question that requires reasoning (e.g., "How does...", "Why...", "What is the relationship...").
    3. Avoid simple "What is X?" definitions.
    4. Use ONLY the provided context.
    5. The response MUST be a valid JSON object.
    6. Language: English.

    JSON FORMAT:
    {{
    "question": "...",
    "concept": "{current_focus}",
    "difficulty": "medium",
    "source_document": "...",
    "source_page": "..."
    }}
    """

    # 7. Chiamata a OpenAI con parametri di varietà (Temperature e Penalty)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You generate unique active recall questions. Never repeat the same question structure twice."},
            {"role": "user", "content": f"{prompt}\n\nSeed for variety: {time.time()}"}
        ],
        temperature=0.9,       # Aumenta la creatività
        presence_penalty=0.6,  # Evita di ripetere gli stessi concetti
        frequency_penalty=0.3, # Evita di usare le stesse parole
        response_format={ "type": "json_object" }
    )

    content = response.choices[0].message.content.strip()
    
    try:
        data = json.loads(content)
    except:
        # Fallback in caso di errore JSON
        data = {"question": f"Explain a key mechanism of {current_focus} based on the text.", "concept": current_focus}
    
    return data

from typing import Optional

class ActiveRecallEvaluateRequest(BaseModel):
    question: str
    student_answer: str
    history: Optional[list[str]] = None

@app.post("/generate_recovery_flashcards")
async def generate_recovery_flashcards(req: dict):
    data = req if isinstance(req, dict) else await req.json()

    topics = data.get("topics", [])

    project_id = data.get("project_id")

    print("🧠 RECOVERY TOPICS:", topics)

    db = SessionLocal()

    # prendi chunk random ma piccoli (focus)
    if topics and len(topics) > 0:
        rows = db.execute(
            text("""
                select chunk_text
                from chunks
                where project_id = :project_id
                and topic = :topic
                order by random()
                limit 5
            """),
            {
                "project_id": project_id,
                "topic": topics[0]
            }
        ).fetchall()

        # fallback se non trova chunk con quel topic
        if not rows:
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

    else:
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
    You are a supportive study tutor evaluating a student's answer using semantic reasoning.

    Question:
    {req.question}

    Previous answers:
    {history_text}

    Latest answer:
    {req.student_answer}

    Evaluation rules:
    - CONCEPTUAL FOCUS: Identify the core concepts of a correct answer. If the student mentions a concept using synonyms or shorthand (e.g., "no more movement" instead of "cannot move anymore"), consider it PRESENT.
    - MEANING OVER WORDING: Do not penalize the student for using different vocabulary. If the "Main Idea" is there, it is CORRECT.
    - NO REDUNDANCY: Do NOT list a concept in the "missing" array if the student has already expressed it, even partially or briefly.
    

    Scoring:
    - correct: Core concepts are present (even if brief or using synonyms).
    - partial: Concept is understood but a CRITICAL, non-implied consequence is missing.
    - incorrect: The core concept is missing or fundamentally wrong.

    Return ONLY JSON:
    {{
    "evaluation": "correct | partial | incorrect",
    "score": 0-1,
    "feedback": "Concise, supportive feedback. If they are right, tell them!",
    "missing": ["Only list things truly not mentioned or implied"],
    "wrong_claims": ["..."],
    "hint": "...",
    "explanation": "Clear explanation of the concept"
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
    topics: str = None,
    user = Depends(verify_user)
):
    
    def normalize(t):
        return t.lower().replace(" ", "")

    topics_list = [t.strip() for t in topics.split(",")] if topics else []

    print("🎯 RAW TOPICS:", topics_list)

    print("🚨 RAW TOPICS STRING:", topics)

    topics_list = topics.split(",") if topics else []

    print("🎯 TOPICS LIST:", topics_list)

    # 🔥 fallback per compatibilità col codice esistente
    topic = topics_list[0] if topics_list else None

    # 🔥 NORMALIZZA
    topics_list = list(set(topics_list))

    print("🎯 CLEAN TOPICS:", topics_list)
   
    db = SessionLocal()

    # ======================
    # DETECT WEAK TOPICS
    # ======================

    weak_topic_rows = db.execute(
        text("""
            select 
                qq.topic,
                sum(case when qa.is_correct then 1 else 0 end) as correct,
                count(*) as total,
                sum(case when qa.is_correct then 1 else 0 end)::float / count(*) as accuracy
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
    print("🔥 WEAK DEBUG topic_rows:", weak_topic_rows)
    weak_topics = [r[0] for r in weak_topic_rows if r[0]]

    # ======================
    # GENERATE FLASHCARDS
    # ======================

    context_chunks = search_project_chunks(
        project_id=project_id,
        topics=topics_list if len(topics_list) > 0 else None,
        k=20
    )

    # 🔥 fallback se pochi chunk
    if not context_chunks:
        print("⚠️ NO CHUNKS FOUND → fallback")
        context_chunks = search_project_chunks(
            project_id=project_id,
            k=20
        )

    # 🔥 funzione per pulire topic
    def clean(t):
        return " ".join(t.split()) if t else t

    # 🔥 estrai topic dai chunk
    chunk_topics = list(dict.fromkeys([
        clean(c.get("topic")) for c in context_chunks if c.get("topic")
    ]))

    context_blocks = []

    for c in context_chunks:
        context_blocks.append(f"""
    TOPIC: {c.get("topic")}

    CONTENT:
    {c.get("text", "")[:400]}
    """)

    context_text = "\n\n---\n\n".join(context_blocks)
    weak_topics_text = ", ".join(weak_topics) if weak_topics else "important concepts"

    prompt = f"""
    You are a strict study tutor generating flashcards for a focused study session.

    FOCUS TOPIC:
    {", ".join(topics_list) if topics_list else "GENERAL"}

    CRITICAL RULE:
    - "You MUST generate flashcards ONLY about these topics: \"{", ".join(topics_list) if topics_list else "GENERAL"}\""
    Each flashcard MUST include the topic it comes from.
    The topic MUST be exactly one of the topics written in the material.
    - Even if other topics appear, IGNORE them
    - If a chunk is not related to this topic, DO NOT use it
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
        "topic": "..."
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

        seen = set()
        unique_cards = []

        for c in generated_cards:
            q = c.get("question", "").strip().lower()

            if q and q not in seen:
                seen.add(q)
                unique_cards.append(c)

        generated_cards = unique_cards[:15]

    except Exception as e:
        print("❌ STUDY SESSION FLASHCARDS JSON ERROR:", e)
        print("RAW GPT OUTPUT:", content)
        generated_cards = []
    
    if not isinstance(generated_cards, list):
        generated_cards = []

    flashcards = []

    for c in generated_cards:

        flashcard_id = str(uuid.uuid4())
        assigned_topic = chunk_topics[len(flashcards) % len(chunk_topics)] if chunk_topics else topic

        db.execute(
            text("""
                insert into flashcards
                (id, project_id, user_id, question, answer, topic)
                values
                (:id, :project_id, :user_id, :question, :answer, :topic)
            """),
            {
                "id": flashcard_id,
                "project_id": project_id,
                "user_id": user["id"],
                "question": c.get("question"),
                "answer": c.get("answer"),
                "topic": assigned_topic
            }
        )

        flashcards.append({
            "id": flashcard_id,
            "question": c.get("question"),
            "answer": c.get("answer"),
            "topic": assigned_topic
        })
        print("🧠 FINAL FLASHCARD TOPICS:", [f["topic"] for f in flashcards])
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
 
