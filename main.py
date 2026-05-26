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

MAX_RECOMMENDED_PAGES = 150
MAX_WARNING_PAGES = 100
MAX_STRONG_WARNING_PAGES = 180

MAX_TOPIC_PROCESSING_SECONDS = 240
MAX_ASSIGNMENT_MATCHES = 30000

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

    # 🔥 NUOVO SISTEMA
    topic_ids: Optional[List[str]] = []

    # 🔥 LEGACY TEMPORANEO
    topics: Optional[List[str]] = []

    # 🔥 LEGACY TEMPORANEO
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

def should_hide_topic(topic_title: str):

    t = topic_title.lower().strip()

    weak_patterns = [
        "deep muscles",
        "superficial muscles",
        "compartments of",
        "muscles of the posterior compartment",
        "muscles of the anterior compartment",
        "pronators and supinators",
        "muscle origins and insertions",
        "actions and innervation",
    ]

    return any(p in t for p in weak_patterns)

def rebalance_taxonomy(final_data):

    MIN_TOPICS_PER_CATEGORY = 2

    categories = final_data.get("categories", [])

    strong_categories = []
    weak_topics = []

    # separa categorie forti/deboli
    for cat in categories:

        topics = cat.get("topics", [])

        if len(topics) >= MIN_TOPICS_PER_CATEGORY:
            strong_categories.append(cat)
        else:
            weak_topics.extend(topics)

    # fallback
    if not strong_categories:
        return final_data

    # category più grande
    main_category = max(
        strong_categories,
        key=lambda c: len(c.get("topics", []))
    )

    # sposta topic deboli
    main_category["topics"].extend(weak_topics)

    return {
        "categories": strong_categories
    }

def process_topics_task(project_id: str):
    db = SessionLocal()
    try:
        print("BACKGROUND TOPICS START:", project_id)
        import time
        topic_start_time = time.time()
        # 1. Update status and clear old topics
        db.execute(text("update projects set topic_status = 'processing' where id = :project_id"), {"project_id": project_id})
        db.execute(text("delete from topics where project_id = :project_id"), {"project_id": project_id})
        db.commit()

        # Fetch chunks to analyze (up to 120 chunks)
        rows = db.execute(
            text("""
                select chunk_text, section
                from chunks
                where project_id = :project_id
                order by page asc
                limit 1000
            """),
            {"project_id": project_id}
        ).fetchall()

        all_chunks = [
            {
                "text": r[0],
                "section": r[1] or "GENERAL"
            }
            for r in rows
            if r[0]
        ]
        # Group chunks into batches of 20
        from collections import defaultdict

        section_groups = defaultdict(list)

        for chunk in all_chunks:
            section_groups[chunk["section"]].append(chunk["text"])
        
        seen_titles = set()
        all_candidate_topics = []

        for section_name, section_chunks in section_groups.items():
            if time.time() - topic_start_time > MAX_TOPIC_PROCESSING_SECONDS:
                print("⏰ TOPIC PROCESSING TIMEOUT - stopping safely")

                db.execute(
                    text("""
                        UPDATE projects
                        SET topic_status = 'partial'
                        WHERE id = :project_id
                    """),
                    {"project_id": project_id}
                )

                db.commit()
                break

            mini_groups = [
                section_chunks[i:i+20]
                for i in range(0, len(section_chunks), 20)
            ]
            for mini_group in mini_groups:
                if time.time() - topic_start_time > MAX_TOPIC_PROCESSING_SECONDS:
                    print("⏰ TOPIC MINI-GROUP TIMEOUT - stopping safely")

                    db.execute(
                        text("""
                            UPDATE projects
                            SET topic_status = 'partial'
                            WHERE id = :project_id
                        """),
                        {"project_id": project_id}
                    )

                    db.commit()
                    break

                group_text = "\n\n".join(mini_group)
            
                # UNIVERSAL PROMPT: Works for any discipline (Medicine, Law, Engineering, etc.)
                # OPTIMIZED PROMPT: Focused on Pedagogical Hierarchy and Topic Consolidation
                prompt = f"""
                Act as a specialist in Instructional Design and Knowledge Organization. 
                Analyze the provided text to extract its fundamental conceptual hierarchy.
                
                GOAL:
                Organize the information into a logical structure of Macro-Categories and robust Study Topics.
                
                STRICT RULES:
                1. CATEGORY RULES:

                - The provided DOCUMENT SECTION is the Category.

                - You MUST reuse the exact DOCUMENT SECTION name as the Category.

                - Do NOT invent new Categories.
                - Do NOT generalize Categories.
                - Do NOT rename Categories.
                - Do NOT merge Categories.

                - Categories should preserve the educational structure of the source material while grouping concepts into semantically coherent learning domains.

                - Avoid creating categories unrelated to the document structure.

                - DO NOT invent unrelated or alternative high-level Categories.

                - Keep the hierarchy aligned with the actual structure of the source document.

                - Categories must preserve the educational organization already present in the material.

                - Use UPPERCASE for Category names.
                -Each topic must semantically belong to its parent category.

                -Topics and categories must describe the same conceptual domain.
                -Topics must remain narrow enough to represent a focused retrievable study unit.

                -Avoid placing unrelated concepts inside the same category.

                GOOD CATEGORY EXAMPLES:
                - MUSCLES OF THE LOWER LIMB
                - MUSCLES OF THE UPPER LIMB
                - FASCIAL STRUCTURES
                - CELL BIOLOGY
                - CONTRACT LAW
                - THERMODYNAMICS

                BAD CATEGORY EXAMPLES:
                - DEEP MUSCLES OF THE POSTERIOR COMPARTMENT
                - ROTATOR CUFF MUSCLES
                - ANTERIOR MUSCLES OF THE LEG

                - Categories should describe a broad educational area, not a specific Study Topic.

                - ALWAYS USE UPPERCASE.
                
                2. TOPIC CREATION RULES:

                - Create ONLY pedagogically meaningful Study Topics.
                - A Study Topic must represent a coherent unit suitable for an actual study session.
                - A Topic title must not duplicate or trivially restate its parent Category name.
                - DO NOT create Topics for:
                - isolated terms
                - single definitions
                - individual list items
                - tiny subcomponents
                - concepts explained with minimal context
                unless they are universally recognized as major standalone concepts in the discipline.

                - Consolidate strongly related concepts into broader educational Topics.

                - Prefer broader conceptual Topics over highly granular fragmentation.

                GOOD TOPIC CHARACTERISTICS:
                - Represents a coherent study unit
                - Covers a meaningful conceptual area
                - Supports multiple quiz questions
                - Supports multiple flashcards
                - Can reasonably correspond to a focused study session
                - Contains concepts that are commonly studied together
                - Reflects the actual educational structure of the source material

                BAD TOPIC CHARACTERISTICS:
                - Isolated keywords or vocabulary terms
                - Extremely narrow details with little standalone relevance
                - Topics containing only one trivial fact
                - Fragmented micro-topics that do not support meaningful study
                - Artificially broad topics combining unrelated concepts

                - A Topic should:
                - support a complete study session
                - contain multiple internal concepts
                - support multiple quiz questions
                - support multiple flashcards
                - represent meaningful educational scope

                - Prefer FEWER but STRONGER Topics.

                - Topics should normally aggregate multiple related concepts internally, even if those concepts are not explicitly listed in the title.
                
                3. DESCRIPTION: Provide a dense, academic definition. If the Topic is a consolidation of multiple items, the description must briefly summarize all of them.
                
                4. IMPORTANCE: Rate 1-10 based on how fundamental the concept is to the subject.

                FORMAT RULES:
                - Return ONLY valid JSON.
                - Ensure names are professional and academically accurate.

                JSON STRUCTURE:
                {{
                "categories": [
                    {{
                    "name": "CATEGORY NAME",
                    "topics": [
                        {{ 
                        "title": "Consolidated Topic Name", 
                        "description": "Comprehensive academic definition covering the grouped concepts", 
                        "importance": 8 
                        }}
                    ]
                    }}
                ]
                }}

                DOCUMENT SECTION:
                {section_name}

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
                        # KEEP ORIGINAL DOCUMENT CATEGORY
                        raw_category = (
                            cat.get("name") or "GENERAL"
                        ).strip().upper()

                        category_name = raw_category
                        
                        
                        
                        for t_obj in cat.get("topics", []):
                            topic_title = (t_obj.get("title") or "").strip()
                            description = (t_obj.get("description") or "").strip()

                            # Recuperiamo l'importanza se presente, altrimenti default a 5
                            importance = t_obj.get("importance", 5)

                            # 🔥 DISPLAY FILTER LOGIC

                            is_display_topic = True

                            #if should_hide_topic(topic_title):
                                #is_display_topic = False

                        

                            #if importance <= 4:
                                #is_display_topic = False

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
                            all_candidate_topics.append({
                                "category": category_name,
                                "topic": topic_title,
                                "description": description,
                                "importance": importance
    })

                           
                            

                except Exception as e:
                    print(f"Error parsing JSON in chunk: {e}")
                    continue 
        print("🌍 STARTING GLOBAL TOPIC CONSOLIDATION")
        from collections import defaultdict

        section_map = defaultdict(list)

        for t in all_candidate_topics:

            category = t["category"]

            section_map[category].append(t)

        topics_text = ""

        MAX_TOPICS_FOR_CONSOLIDATION = 40

        for category, topics in section_map.items():

            print("🧠 CONSOLIDATION CATEGORY:", category)
            print("🧠 CONSOLIDATION TOPICS:", len(topics))

            topics_text += f"\n\n=== CATEGORY: {category} ===\n"

            for t in topics[:MAX_TOPICS_FOR_CONSOLIDATION]:

                topics_text += f"""
        TOPIC: {t['topic']}
        DESCRIPTION: {t['description']}
        IMPORTANCE: {t['importance']}
        """

        global_prompt = f"""
        Act as a senior Instructional Designer and Knowledge Architect.

        You are given MANY candidate Study Topics extracted from different portions of the same document.

        Your task is to CONSOLIDATE them into a SINGLE coherent educational taxonomy.

        GOALS:
        - Merge ONLY genuinely redundant or overlapping Topics.
        - NEVER merge Topics belonging to different Categories.
        - Categories represent hard educational boundaries.
        - Topics from different Categories must always remain separated even if semantically related.
        - Preserve important educational subdomains as separate Topics.
        - Do NOT merge Topics that belong to different conceptual depths.
        Examples:
        - "Forearm Muscles" and "Pronator Teres" are different depths.
        - "Truth Tables" and "Sentential Logic" are different depths.
        - Only merge Topics representing the same pedagogical level.
        - Do NOT merge Topics that represent distinct study areas, even if related.
        - Remove redundant Topics
        - Eliminate overly granular Topics
        - Create pedagogically balanced Study Topics with moderate granularity
        - Ensure complete document coverage
        - Preserve ALL major educational concepts

        IMPORTANT:
        - Categories must be broad academic domains.
        - Topics must represent meaningful study units.
        - Avoid fragmentation.
        - Avoid duplicate or overlapping Topics.
        - Prefer pedagogically balanced Topics.
        Examples of Topics that SHOULD remain separate:
        - Concepts that are independently studied
        - Topics with distinct educational objectives
        - Topics that support separate quiz generation
        - Topics with substantially different conceptual scope
        - Topics that are commonly taught as separate units
        - Topics that require different reasoning processes or learning goals
        - Avoid excessive fragmentation.
        - Avoid overly broad mega-topics.
        - A Topic should represent a focused but complete study unit.
        - Topics should normally correspond to 15-30 minutes of focused study.
        - Preserve important subdomains when they have substantial educational relevance.

        Return ONLY valid JSON.

        FORMAT:

        {{
        "categories": [
            {{
            "name": "CATEGORY",
            "topics": [
                {{
                "title": "TOPIC",
                "description": "DESCRIPTION",
                "importance": 8
                }}
            ]
            }}
        ]
        }}

        CANDIDATE TOPICS:

        {topics_text}
        """
        print("🚀 STARTING FINAL CONSOLIDATION CALL")
        print("🧠 CONSOLIDATION INPUT LENGTH:", len(topics_text))
        print("🧠 TOTAL CATEGORIES:", len(section_map))

        try:

            final_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "user", "content": global_prompt}
                ],
                response_format={"type": "json_object"}
            )

            final_data = json.loads(
                final_response.choices[0].message.content
            )

            print("✅ FINAL CONSOLIDATION RESPONSE RECEIVED")
            print("✅ GLOBAL CONSOLIDATION COMPLETE")
            print(json.dumps(final_data, indent=2))

        except Exception as e:

            print("❌ FINAL CONSOLIDATION FAILED:", e)

            # FALLBACK:
            # usa direttamente i candidate topics
            final_data = {
                "categories": []
            }

            for category, topics in section_map.items():

                final_data["categories"].append({
                    "name": category,
                    "topics": topics[:40]
                })

            print("⚠️ USING FALLBACK TOPIC STRUCTURE")
        final_data = rebalance_taxonomy(final_data)        
       
        # 🔥 DELETE OLD TOPIC-CHUNK LINKS

        db.execute(
            text("""
                DELETE FROM topic_chunks
                WHERE topic_id IN (
                    SELECT id
                    FROM topics
                    WHERE project_id = :project_id
                )
            """),
            {"project_id": project_id}
        )

        # 🔥 DELETE OLD TOPICS

        db.execute(
            text("""
                DELETE FROM topics
                WHERE project_id = :project_id
            """),
            {"project_id": project_id}
        )

        db.commit()

        print("🧹 OLD TOPICS + TOPIC_CHUNKS REMOVED")
        for cat in final_data.get("categories", []):

            category_name = (
                cat.get("name") or "GENERAL"
            ).strip().upper()

            for t_obj in cat.get("topics", []):

                topic_title = (
                    t_obj.get("title") or ""
                ).strip()

                description = (
                    t_obj.get("description") or ""
                ).strip()

                importance = t_obj.get("importance", 5)

                if not topic_title:
                    continue

                is_display_topic = True

                #if should_hide_topic(topic_title):
                    #is_display_topic = False

                #if importance <= 4:
                    #is_display_topic = False

                emb_input = (
                    f"{category_name} - "
                    f"{topic_title}: "
                    f"{description}"
                )

                emb = client.embeddings.create(
                    model="text-embedding-3-small",
                    input=emb_input
                )

                embedding = emb.data[0].embedding

                embedding_str = "[" + ",".join(
                    map(str, embedding)
                ) + "]"

                db.execute(
                    text("""
                        insert into topics
                        (
                            project_id,
                            category,
                            topic,
                            description,
                            embedding,
                            is_display_topic,
                            source_section
                        )
                        values
                        (
                            :project_id,
                            :category,
                            :topic,
                            :description,
                            CAST(:embedding AS vector),
                            :is_display_topic,
                            :source_section
                        )
                    """),
                    {
                        "project_id": project_id,
                        "category": category_name,
                        "topic": topic_title,
                        "description": description,
                        "embedding": embedding_str,
                        "is_display_topic": is_display_topic,
                        "source_section": category_name
                    }
                )

        db.commit()
        print("🧠 STARTING TOPIC-CHUNK ASSIGNMENT")

        try:
            assign_topics_to_chunks(project_id)
            print("✅ TOPIC-CHUNK ASSIGNMENT COMPLETED")
        except Exception as e:
            print("❌ TOPIC-CHUNK ASSIGNMENT FAILED:", e)

        
        # ======================
        # FINAL STATUS UPDATE
        # ======================

        

        final_db = SessionLocal()

        final_db.execute(
            text("""
                UPDATE projects
                SET topic_status = 'completed'
                WHERE id = :project_id
            """),
            {"project_id": project_id}
        )

        final_db.commit()

        check = final_db.execute(
            text("""
                SELECT topic_status
                FROM projects
                WHERE id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()

        print("✅ PROJECT TOPIC STATUS = COMPLETED")
        print("🔥 STATUS READBACK:", check[0])

        final_db.close()
        

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
        print("🧪 ENTER assign_topics_to_chunks")

        # pulizia link precedenti
        db.execute(
            text("""
                delete from topic_chunks
                where topic_id in (
                    select id
                    from topics
                    where project_id = :project_id
                )
            """),
            {"project_id": project_id}
        )

        # 🔥 RECUPERA MATCH CANDIDATI

        matches = db.execute(
            text("""
                select
                    c.id as chunk_id,
                    c.chunk_text,
                    c.section,

                    t.id as topic_id,
                    t.topic,
                    t.category,
                    t.source_section,

                    (
                        c.embedding <#> t.embedding
                    ) as similarity

                from chunks c
                join topics t
                    on c.project_id = t.project_id

                where
                    c.project_id = :project_id
            """),
            {"project_id": project_id}
        ).fetchall()
        print("🔢 TOPIC ASSIGNMENT MATCHES:", len(matches))
        print("🧪 MATCH QUERY COMPLETED")
        print("🧪 TOTAL MATCHES:", len(matches))

        if len(matches) > MAX_ASSIGNMENT_MATCHES:
            print("⚠️ TOO MANY MATCHES - applying safety cap")
            matches = matches[:MAX_ASSIGNMENT_MATCHES]

        links_created = 0
        assignments = set()

        processed_matches = 0

        for row in matches:
            processed_matches += 1

            if processed_matches % 5000 == 0:
                print("🧪 PROCESSED MATCHES:", processed_matches)

            chunk_id = row[0]
            chunk_text = (row[1] or "").lower()
            chunk_section = (row[2] or "").lower()

            topic_id = row[3]
            topic_name = (row[4] or "").lower()
            topic_category = (row[5] or "").lower()
            topic_section = (row[6] or "").lower()

            similarity = row[7]
            same_section = (
                chunk_section == topic_section
            )

            # 🔥 KEYWORD OVERLAP

            topic_words = [
                w for w in topic_name.split()
                if len(w) > 4
            ]

            keyword_overlap = sum(
                1 for w in topic_words
                if w in chunk_text
            )

            # 🔥 SECTION BONUS

            

            section_bonus = 0

            if same_section:
                section_bonus += 0.20

            # 🔥 FINAL SCORE

            final_score = (
                similarity
                + section_bonus
                + (keyword_overlap * 0.05)
            )

            if final_score > -0.55:

                key = (topic_id, chunk_id)

                if key in assignments:
                    continue

                assignments.add(key)

                db.execute(
                    text("""
                        insert into topic_chunks
                        (topic_id, chunk_id)
                        values
                        (:topic_id, :chunk_id)
                    """),
                    {
                        "topic_id": topic_id,
                        "chunk_id": chunk_id
                    }
                )

                links_created += 1

        db.commit()
        print("🧪 FINAL ASSIGNMENT COMMIT DONE")

        count = db.execute(
            text("""
                select count(*)
                from topic_chunks tc
                join topics t
                    on tc.topic_id = t.id
                where t.project_id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()[0]

        print(f"✅ SAVED {links_created} TOPIC-CHUNK LINKS")

    except Exception as e:

        db.rollback()
        print("❌ TOPIC ASSIGNMENT ERROR:", e)

    finally:
        db.close()

def detect_section_title(text: str, current_section="GENERAL"):

    if not text:
        return current_section

    lines = [
        l.strip()
        for l in text.split("\n")
        if l.strip()
    ]

    strong_candidates = []

    for line in lines[:15]:

        clean = line.strip()

        upper_ratio = (
            sum(1 for c in clean if c.isupper())
            / max(len(clean), 1)
        )

        if (
            upper_ratio > 0.7
            and 2 <= len(clean.split()) <= 8
        ):
            strong_candidates.append(clean.upper())

        anatomical_keywords = [
            "upper limb",
            "lower limb",
            "shoulder",
            "arm",
            "forearm",
            "hand",
            "hip",
            "thigh",
            "leg",
            "foot",
            "pelvis"
        ]

        if any(
            kw in clean.lower()
            for kw in anatomical_keywords
        ):
            strong_candidates.append(clean.upper())

        

        if len(clean) < 5 or len(clean) > 120:
            continue

        

        

        if (
            any(k in clean.lower() for k in anatomical_keywords)
            and upper_ratio > 0.5
        ):

            if upper_ratio > 0.45:
                strong_candidates.append(clean)

    if strong_candidates:

        best = max(
            strong_candidates,
            key=len
        )

        return best.upper()

    return current_section

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

                total_pages = len(reader.pages)

                yield f"FILE_ANALYSIS|pages={total_pages}\n"

                if total_pages > MAX_WARNING_PAGES:
                    yield (
                        "LARGE_FILE_WARNING|"
                        f"pages={total_pages}|"
                        "message=Large academic file detected. "
                        "We can process it, but topic generation may take longer. "
                        "For best results, consider splitting very large files by chapter.\n"
                    )

                current_section = (
                    clean_text(doc.title)
                    .upper()
                )
                for page_index, page in enumerate(reader.pages):
                    

                    yield f"Page {page_index+1}\n"
                    db.execute(
                        text("""
                            UPDATE projects
                            SET last_processed_page = :page
                            WHERE id = :project_id
                        """),
                        {
                            "page": page_index + 1,
                            "project_id": project_id
                        }
                    )

                    db.commit()

                    page_text = page.extract_text()

                    page_text = clean_text(page_text)
                    current_section = detect_section_title(
                        page_text,
                        current_section
                    )

                    section_title = current_section
                    print(f"📚 DETECTED SECTION: {section_title}")

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
                                (project_id, doc_title, chunk_text, embedding, page, topic, section)
                                values
                                (:project_id, :doc_title, :chunk_text, CAST(:embedding AS vector), :page, :topic, :section)
                            """),
                            {
                                "project_id": project_id,
                                "doc_title": doc.title,
                                "chunk_text": chunk,
                                "embedding": embedding_str,
                                "page": page_index + 1,      
                                "topic": None,
                                "section": section_title
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
# LEARNING SCOPE
# ======================

def resolve_learning_scope(project_id: str, topic_ids=None, limit: int = 80):
    db = SessionLocal()

    try:
        topic_ids = topic_ids or []

        if topic_ids:
            scope_type = "single_topic" if len(topic_ids) == 1 else "multi_topic"

            rows = db.execute(
                text("""
                    SELECT
                        c.id,
                        c.chunk_text,
                        c.doc_title,
                        c.page,
                        t.topic,
                        t.id as topic_id
                    FROM topic_chunks tc
                    JOIN chunks c
                        ON c.id = tc.chunk_id
                    JOIN topics t
                        ON t.id = tc.topic_id
                    WHERE t.project_id = :project_id
                    AND tc.topic_id IN :topic_ids
                    AND c.chunk_text IS NOT NULL
                    AND length(c.chunk_text) > 100
                    LIMIT :limit
                """),
                {
                    "project_id": project_id,
                    "topic_ids": tuple(topic_ids),
                    "limit": limit
                }
            ).fetchall()

        else:
            scope_type = "global"

            rows = db.execute(
                text("""
                    SELECT
                        c.id,
                        c.chunk_text,
                        c.doc_title,
                        c.page,
                        COALESCE(t.topic, 'General') as topic,
                        t.id as topic_id
                    FROM chunks c
                    LEFT JOIN topic_chunks tc
                        ON tc.chunk_id = c.id
                    LEFT JOIN topics t
                        ON t.id = tc.topic_id
                    WHERE c.project_id = :project_id
                    AND c.chunk_text IS NOT NULL
                    AND length(c.chunk_text) > 100
                    ORDER BY RANDOM()
                    LIMIT :limit
                """),
                {
                    "project_id": project_id,
                    "limit": limit
                }
            ).fetchall()

        chunks = []

        for r in rows:
            chunks.append({
                "chunk_id": str(r[0]),
                "chunk_text": r[1],
                "doc_title": r[2],
                "page": r[3],
                "topic": r[4],
                "topic_id": str(r[5]) if r[5] else None
            })

        topic_map = {
            c["chunk_id"]: c["topic"]
            for c in chunks
            if c.get("topic")
        }

        topics = {}

        for c in chunks:
            if c["topic_id"]:
                topics[c["topic_id"]] = {
                    "id": c["topic_id"],
                    "topic": c["topic"]
                }

        return {
            "scope_type": scope_type,
            "topic_ids": topic_ids,
            "topics": list(topics.values()),
            "chunks": chunks,
            "topic_map": topic_map,
            "chunk_count": len(chunks)
        }

    finally:
        db.close()
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
    print("🔥 FULL REQUEST:", req.dict())

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

    # 🔥 NUOVO SISTEMA
    topic_ids = req.topic_ids or []

    scope = resolve_learning_scope(project_id, topic_ids, limit=80)

    print("🧠 LEARNING SCOPE:", scope["scope_type"])
    print("📦 SCOPE CHUNKS:", scope["chunk_count"])
    print("🎯 SCOPE TOPICS:", scope["topics"][:3])

    # 🔥 LEGACY
    legacy_topics = req.topics or []

    print("📥 TOPIC IDS:", topic_ids)
    print("📥 LEGACY TOPICS:", legacy_topics)
    print("🔥 QUIZ REQUEST RECEIVED")
    print("topic_ids:", topic_ids)
    print("legacy_topics:", legacy_topics)
    print("num_questions:", req.num_questions)
    
    # 1. RETRIEVAL POTENZIATO (120 chunk)
    query_text = " ".join(req.topics) if req.topics else "General overview of the provided documents"
    emb_res = client.embeddings.create(model="text-embedding-3-small", input=query_text)
    query_embedding = emb_res.data[0].embedding

    rows = []

    rows = []

    # =====================================
    # NEW LEARNING GRAPH RETRIEVAL
    # =====================================

    scope = resolve_learning_scope(
        project_id=project_id,
        topic_ids=topic_ids,
        limit=80
    )

    print("🧠 LEARNING SCOPE:", scope["scope_type"])

    retrieved_chunks = scope["chunks"]

    print("📦 RETRIEVED CHUNKS:", len(retrieved_chunks))

    topic_map = scope["topic_map"]


    for c in retrieved_chunks[:5]:
        print(
            f"📄 CHUNK DA TOPIC: {c['topic']} | TESTO: {c['chunk_text'][:100]}..."
        )

    chunk_topic_map = {
        c["chunk_id"]: " ".join(str(c["topic"]).split())
        for c in retrieved_chunks
        if c["topic"]
    }
    print("🧠 CHUNK TOPIC MAP SAMPLE:")
    print(list(chunk_topic_map.items())[:5])

    if req.topics:
        active_topics = [normalize_string(t) for t in req.topics]
    else:
        active_topics = list(set(
            normalize_string(c["topic"])
            for c in retrieved_chunks
            if c["topic"]
        )) or ["General"]

    # 4. Fallback se non ci sono topic
    if not active_topics:
        active_topics = ["General"]

    print("🎯 ACTIVE TOPICS (CLEANED):", active_topics)
    

    if not retrieved_chunks:
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
        valid_chunks = [
            c for c in retrieved_chunks
            if c["chunk_text"]
            and c["topic"]
            and normalize_string(c["topic"]) in active_topics
        ]

        sample_size = min(15, len(valid_chunks))

        sampled_chunks = random.sample(valid_chunks, sample_size)

        context = "\n\n".join([
            f"### CHUNK_ID: {c['chunk_id']} | TOPIC: {c['topic']}\n{c['chunk_text']}"
            for c in sampled_chunks
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
        Distribute questions evenly across different subtopics and concepts found in the material.
        Prioritize conceptual coverage breadth over repetition depth.
        Avoid these topics: {avoid_str}

        STRICT RULES FOR DISTRACTORS:
        1. SEMANTIC COHERENCE: All 5 options (A, B, C, D, E) must belong to the same category or anatomical system as the correct answer.
        2. PLAUSIBILITY: Distractors must be plausible enough to challenge a prepared student. Do not use obviously unrelated terms.
        3. NO OVERLAP: Ensure only one answer is strictly correct according to the provided material.
        4. Distractors must be partially plausible but scientifically incorrect in the specific context of the question.
        5. Only one answer must remain correct even under expert-level interpretation.
        6. TARGET TOPICS: Focus specifically on {topics_str}.
        7. Each question must test a DIFFERENT concept.
        8. Avoid semantic overlap between questions.
        9. Do not ask multiple questions about the same enzyme, mechanism, pathway, or biological function.
        10. Max 1 question per specific molecular mechanism unless absolutely necessary.

        STRICT RULES:
        1. Use ONLY the provided material.
        2. Professional tone.
        IMPORTANT:
        Two questions are considered duplicates even if wording changes but the biological concept being tested is the same.
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

                       

                        topic_from_db = chunk_topic_map.get(chunk_id)

                        
                        # 🔥 SINCRONIZZAZIONE TOPIC E CAMPI
                        
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

        # =========================================
        # REASONING UNIT EXTRACTION
        # =========================================

        reasoning_material = []

        async def extract_reasoning_units(block):

            extraction_prompt = f"""
            You MUST use ONLY the provided material.

            Extract:
            - causes
            - effects
            - relationships
            - comparisons
            - transformations
            - conditions

            Focus ONLY on concepts explicitly supported by the material.

            Return STRICT JSON:

            {{
            "units": [
                {{
                "condition": "...",
                "mechanism": "...",
                "effect": "..."
                }}
            ]
            }}

            MATERIAL:
            {block}
            """

            try:

                response = await asyncio.to_thread(
                    client.chat.completions.create,
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "user",
                            "content": extraction_prompt
                        }
                    ],
                    temperature=0.3
                )

                content = response.choices[0].message.content.strip()

                content = (
                    content
                    .replace("```json", "")
                    .replace("```", "")
                    .strip()
                )

                data = json.loads(content)

                return data.get("units", [])

            except Exception as e:

                print("⚠️ REASONING EXTRACTION ERROR:", e)

                return []
        
        # EXTRACT REASONING UNITS

        for block in material_blocks[:25]:

            units = await extract_reasoning_units(block)

            reasoning_material.extend(units)

        print("🧠 TOTAL REASONING UNITS:", len(reasoning_material))

        # GENERAZIONE QUIZ
        batch_size = 4 if req.difficulty == "hard" else 8
        num_batches = (req.num_questions + batch_size - 1) // batch_size


        async def generate_batch(n):

            prompt = f"""
        You MUST use ONLY the material provided below.
        Each section starts with '### TOPIC: [Name]'.

        Use the provided material as the ONLY factual source.

        You may infer relationships, consequences, mechanisms, comparisons, or applications ONLY if they are logically supported by the material.
        Do not invent concepts, facts, mechanisms, laws, events, or terminology not grounded in the provided content.

        Reasoning Units:
        {json.dumps(reasoning_material[:80], indent=2)}

        Generate {n} short assessment scenarios.

        Each scenario must:
        - describe a condition, change, interaction, comparison, consequence, transformation, or system behavior
        - require the student to infer the correct answer
        - evaluate reasoning and conceptual understanding
        - avoid direct factual recall
        - avoid textbook-style definitions

The student should need to interpret the situation, not simply recognize a memorized fact.
        IMPORTANT:
        Each question MUST focus on a COMPLETELY DIFFERENT concept.

        CRITICAL INSTRUCTIONS FOR QUESTION VARIETY:
        - NEVER start a question with "What is", "What are", or "Define".
        - Focus on APPLICATION: Create scenarios where the user must apply a rule or concept.
        - Focus on relationships, consequences, interactions, transformations, comparisons, interpretations, or applied understanding.
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

        If Difficulty is HARD:
        - EVERY question MUST be built as a short applied situation.
        - Do NOT ask direct recall questions.
        - Do NOT ask "What is", "Which enzyme", "What role", "What function", "Which pathway" questions.
        - The question must describe a condition, change, case, conflict, mechanism failure, interpretation problem, or applied scenario.
        - The correct answer must require reasoning from the material, not simply recognizing a term.
        - If a question can be answered by matching one keyword to one definition, reject it and generate another.

        Difficulty Rules:

        EASY:
        - Focus on definitions, terminology, and direct recall
        - Questions should test basic recognition of concepts
        - Avoid multi-step reasoning
        - Avoid clinical or applied scenarios

        MEDIUM:
        - Focus on mechanisms, relationships, and cause-effect reasoning
        - Include pathway interactions and functional understanding
        - Require moderate reasoning to identify the correct answer

        HARD:
        - Generate applied, analytical, or reasoning-focused questions
        - Avoid direct definition or recall questions
        - The student must infer the answer from a situation, mechanism, consequence, comparison, or relationship
        - Questions should require analysis, not memorization
        - Prefer short applied scenarios over direct factual prompts

        BAD HARD QUESTION:
        "What is the role of X?"

        GOOD HARD QUESTION:
        "A mechanism responsible for X is altered.
        Which consequence would MOST likely occur?"

        BAD HARD QUESTION:
        "Which process performs Y?"

        GOOD HARD QUESTION:
        "A system changes from condition A to condition B.
        Which explanation BEST accounts for the change?"

        BAD HARD QUESTION:
        "What is the function of Y?"

        GOOD HARD QUESTION:
        "Two related mechanisms produce different outcomes under the same condition.
        Which explanation BEST accounts for the difference?"
        
    
        - Avoid starting most questions with:
        "What is..."
        "Which is..."
        "What role..."
        unless necessary.

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
            # SHUFFLE OPTIONS TO AVOID POSITION BIAS

            try:

                correct_text = q["options"][q["correct"]]

                random.shuffle(q["options"])

                q["correct"] = q["options"].index(correct_text)

            except Exception as e:

                print("⚠️ Shuffle error:", e)
            
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
    req: dict = Body(None),
    user = Depends(verify_user)
):
    gpt_text = "IA non ancora consultata" 
    db = SessionLocal()
    flashcards = [] # FONDAMENTALE: inizializza la lista qui
    
    try:
        num_cards = 10

        topics_list = []
        topic_ids = []

        if req and isinstance(req, dict):

            num_cards = req.get("num_cards", 10)

            topics_list = req.get("topics", [])
            topic_ids = req.get("topic_ids", [])

            if isinstance(topics_list, str):
                topics_list = [topics_list]

        print(f"🔥 START FLASHCARDS: Project {project_id} | Topics: {topics_list}")

        # =====================================
        # NEW LEARNING GRAPH RETRIEVAL
        # =====================================

        scope = resolve_learning_scope(
            project_id=project_id,
            topic_ids=req.get("topic_ids", []),
            limit=30
        )

        print("🧠 FLASHCARD LEARNING SCOPE:", scope["scope_type"])

        context_chunks = scope["chunks"]

        print("📦 FLASHCARD CHUNKS:", len(context_chunks))

        if not context_chunks:
            print("❌ NESSUN TESTO TROVATO NEL PROGETTO")
            return {"flashcards": []}

        # 2. PREPARAZIONE CONTESTO E MAPPING
        context_blocks = []
        chunk_topic_map = {} 
        for i, c in enumerate(context_chunks):
            cid = f"CH-{i}"
            topic_name = c.get("topic") or "General"
            chunk_topic_map[cid] = topic_name
            # Usiamo 'text' perché search_project_chunks restituisce dizionari con 'text'
            context_blocks.append(
                f"### CHUNK_ID: {cid} | TOPIC: {topic_name}\n{c.get('chunk_text', '')}"
            )

        context_text = "\n\n".join(context_blocks)

        prompt = f"""
        Generate EXACTLY {num_cards} academic flashcards in JSON format.
        
        STRICT RULES:
        1. Use ONLY the provided material.
        2. Assign the correct 'chunk_id' to each card (e.g., "CH-0").
        3. Aim for high-yield medical questions (Why, How, Mechanisms).

        Material:
        {context_text}

        Expected JSON Structure:
        {{
          "flashcards": [
            {{
              "question": "...",
              "answer": "...",
              "chunk_id": "CH-X",
              "difficulty": "medium"
            }}
          ]
        }}
        """

        # 3. CHIAMATA A GPT
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.4,
            messages=[
                {"role": "system", "content": "You are a professional medical tutor. You MUST always respond in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        gpt_text = response.choices[0].message.content
        
        # 4. PARSING E FILTRO
        data = json.loads(gpt_text)
        raw_cards = data.get("flashcards", [])
        
        seen = set()
        for card in raw_cards:
            q = card.get("question", "").strip().lower()
            cid = str(card.get("chunk_id", "")).strip()
            
            # Verifichiamo che il chunk_id esista nella nostra mappa
            if q and cid in chunk_topic_map and q not in seen:
                card["topic"] = chunk_topic_map[cid]
                seen.add(q)
                flashcards.append(card)

        # Limitiamo al numero richiesto
        flashcards = flashcards[:num_cards]

        # 5. SALVATAGGIO NEL DATABASE
        for card in flashcards:
            # Usiamo text() per coerenza con il tuo stile
            result = db.execute(
                text("""
                    INSERT INTO flashcards (project_id, user_id, question, answer, topic)
                    VALUES (:project_id, :user_id, :question, :answer, :topic)
                    RETURNING id
                """),
                {
                    "project_id": project_id,
                    "user_id": user["id"],
                    "question": card.get("question"),
                    "answer": card.get("answer"),
                    "topic": card.get("topic")
                }
            )
            card["id"] = result.fetchone()[0]

        db.commit()
        return {"flashcards": flashcards}

    except Exception as e:
        if db: db.rollback()
        print(f"❌ ERRORE GENERAZIONE: {str(e)}")
        print(f"📝 RAW GPT OUTPUT: {gpt_text}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()
    
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

        if (
            not chunk_topic
            or str(chunk_topic).lower().endswith(".pdf")
        ):
            chunk_topic = (
                normalize_string(r[1])
                .replace(".pdf", "")
                .replace("_", " ")
            )

        

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
                SELECT id, category, topic, description  
                FROM topics 
                WHERE project_id = :project_id
                AND topic IS NOT NULL
                AND is_display_topic = true
                ORDER BY category ASC, topic ASC
            """), 
            {"project_id": project_id}
        )
        rows = result.fetchall()
        
        # We format them into the structured object your UI needs
        return {
            "topics": [
                {
                    "id": str(r[0]),
                    "category": r[1] or "General",
                    "topic": r[2],
                    "description": r[3] or "",
                    "difficulty": "medium",
                    "accuracy": 50
                }
                for r in rows
            ]
        }
    finally:
        db.close()

@app.get("/projects/{project_id}/topic_status")

def get_topic_status(
    project_id: str,
    user = Depends(verify_user)
):
    from sqlalchemy.orm import sessionmaker

    FreshSession = sessionmaker(bind=engine)

    db = FreshSession()
    db.expire_all()
    print("🧠 TOPIC STATUS REQUEST FOR:", project_id)
    try:
        print("🔄 FORCING FRESH STATUS READ")
        row = db.execute(
            text("""
                select topic_status, last_processed_page
                from projects
                where id = :project_id
            """),
            {"project_id": project_id}
        ).fetchone()
        print("📦 RAW DB ROW:", row)
        if not row:
            raise HTTPException(status_code=404, detail="Project not found")

        print("🔥 RAW STATUS FROM DB:", row[0])

        return {
            "status": row[0] or "idle",
            "last_processed_page": row[1] or 0
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
            select id, question, answer, topic
            from flashcards
            where project_id = :project_id
            order by random()
        """),
        {"project_id": project_id}
    ).fetchall()

    db.close()

    flashcards = []

    for r in rows:
        flashcards.append({
            "id": r[0],
            "question": r[1],
            "answer": r[2],
            "topic": r[3]
        })

    return {"flashcards": flashcards}
# ======================
# PROJECT SUMMARY
# ======================

@app.get("/projects/{project_id}/summary")
async def project_summary(project_id: str, user = Depends(verify_user)):
    user_id = user["id"]
    db = SessionLocal()
    try:
        # 1. Statistiche generali Quiz
        quiz_stats = db.execute(
            text("""
                SELECT 
                    COUNT(qa.id), 
                    AVG((qa.score::float / NULLIF(qa.total_questions, 0)) * 100)
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                WHERE qa.user_id = :u_id AND q.project_id = :p_id
                AND qa.total_questions > 0
            """),
            {"u_id": user_id, "p_id": project_id}
        ).fetchone()

        # 2. Storico Quiz (history_rows)
        history_rows = db.execute(
            text("""
                SELECT q.title, qa.score, qa.total_questions, qa.completed_at
                FROM quiz_attempts qa
                JOIN quizzes q ON qa.quiz_id = q.id
                WHERE qa.user_id = :u_id AND q.project_id = :p_id
                ORDER BY qa.completed_at DESC
            """),
            {"u_id": user_id, "p_id": project_id}
        ).fetchall()

        quiz_history = [
            {"title": r[0], "score": r[1], "total": r[2], "date": r[3].isoformat() if r[3] else None} 
            for r in history_rows
        ]

        # 3. Dettaglio Topic Ibrido (topic_rows)
        topic_rows = db.execute(
            text("""
                WITH CombinedScores AS (
                    SELECT 
                        qq.topic, 
                        (qa.score::float / NULLIF(qa.total_questions, 0)) * 100 as score
                    FROM quiz_questions qq
                    JOIN quizzes q ON qq.quiz_id = q.id
                    JOIN quiz_attempts qa ON qa.quiz_id = q.id
                    WHERE q.project_id = :p_id AND qa.user_id = :u_id
                    
                    UNION ALL
                    
                    SELECT 
                        f.topic,
                        CASE WHEN fr.is_correct THEN 100 ELSE 0 END as score
                    FROM flashcards f
                    JOIN flashcard_reviews fr ON f.id = fr.flashcard_id
                    WHERE f.project_id = :p_id AND f.user_id = :u_id
                )
                SELECT topic, AVG(score) FROM CombinedScores GROUP BY topic
            """),
            {"p_id": project_id, "u_id": user_id}
        ).fetchall()

        topics_detail = [
            {"topic": r[0] or "General", "score": round(float(r[1]), 1) if r[1] is not None else 0} 
            for r in topic_rows
        ]

        # Conteggio flashcard riviste per il box (opzionale)
        f_count = db.execute(
            text("SELECT COUNT(DISTINCT flashcard_id) FROM flashcard_reviews fr JOIN flashcards f ON fr.flashcard_id = f.id WHERE f.project_id = :p_id AND f.user_id = :u_id"),
            {"p_id": project_id, "u_id": user_id}
        ).scalar() or 0

        # IL RETURN DEVE CONTENERE TUTTE LE CHIAVI
        return {
            "quiz_attempts": int(quiz_stats[0]) if quiz_stats[0] else 0,
            "messaggio_segreto": "Sto leggendo questo file!",
            "avg_score": round(float(quiz_stats[1]), 1) if quiz_stats[1] else 0,
            "topics_count": len(topic_mastery),
            "flashcards_reviewed": f_count, # La variabile del conteggio flashcard
            "quiz_history": quiz_history_list,    # <--- FONDAMENTALE
            "topics_detail": topics_detail_list,  # <--- FONDAMENTALE
            "topic_mastery": topics_detail_list   # Per la compatibilità con ResultsView
        }

    except Exception as e:
        print(f"ERRORE CRITICO: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        db.close()

@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def flashcards_detailed_stats(project_id: str, user = Depends(verify_user)):

    db = SessionLocal()

    # Cambiamo la query per raggruppare per topic
    # Nota: usiamo la tabella 'flashcards' per i nomi dei topic e 'flashcard_reviews' per i voti
    rows = db.execute(
        text("""
            SELECT 
                f.topic,
                COUNT(fr.id) as total,
                SUM(CASE WHEN fr.is_correct = false THEN 1 ELSE 0 END) as wrong,
                SUM(CASE WHEN fr.difficulty = 1 THEN 1 ELSE 0 END) as hard,
                SUM(CASE WHEN fr.difficulty = 2 THEN 1 ELSE 0 END) as good,
                SUM(CASE WHEN fr.difficulty = 3 THEN 1 ELSE 0 END) as easy
            FROM flashcards f
            LEFT JOIN flashcard_reviews fr ON f.id = fr.flashcard_id
            WHERE f.project_id = :project_id 
              AND f.user_id = :user_id
            GROUP BY f.topic
        """),
        {
            "project_id": project_id,
            "user_id": user["id"]
        }
    ).fetchall()

    db.close()

    # Trasformiamo i risultati in un dizionario mappato per topic
    stats_by_topic = {}
    for r in rows:
        topic_name = r[0] or "General"
        total = r[1] or 0
        
        # Se non ci sono review per questo topic, mettiamo tutto a zero
        if total == 0:
            stats_by_topic[topic_name] = {
                "total": 0, "wrong": 0, "hard": 0, "good": 0, "easy": 0, "accuracy": 0
            }
            continue

        wrong = r[2] or 0
        hard = r[3] or 0
        good = r[4] or 0
        easy = r[5] or 0
        
        stats_by_topic[topic_name] = {
            "total": total,
            "wrong": wrong,
            "hard": hard,
            "good": good, # Il tuo frontend usa 'good' per il colore blu
            "easy": easy,
            "accuracy": round(((good + easy) / total) * 100, 1)
        }

    return stats_by_topic

# ======================
# PROJECT RESULTS
# ======================
@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def flashcards_detailed_stats(
    project_id: str,
    user = Depends(verify_user)
):
    db = SessionLocal() # Apre la connessione
    try:
        # La tua query SQL raggruppata per topic
        rows = db.execute(
            text("""
                select
                    topic,
                    sum(case when is_correct = false then 1 else 0 end) as wrong,
                    sum(case when difficulty = 1 then 1 else 0 end) as hard,
                    sum(case when difficulty = 2 then 1 else 0 end) as correct,
                    sum(case when difficulty = 3 then 1 else 0 end) as easy,
                    count(*) as total
                from flashcard_reviews
                where project_id = :project_id
                  and user_id = :user_id
                group by topic
            """),
            {"project_id": project_id, "user_id": user["id"]}
        ).fetchall()

        # Trasformazione dati
        stats_by_topic = {}
        for r in rows:
            topic_name = r[0] or "General"
            stats_by_topic[topic_name] = {
                "wrong": int(r[1] or 0),
                "hard": int(r[2] or 0),
                "good": int(r[3] or 0), # Manteniamo 'good' per coerenza frontend
                "easy": int(r[4] or 0),
                "total": int(r[5] or 0)
            }

        return stats_by_topic

    except Exception as e:
        print(f"❌ Error fetching detailed stats: {e}")
        # Opzionale: puoi sollevare un'eccezione HTTP qui
        raise HTTPException(status_code=500, detail="Internal Server Error")
    
    finally:
        db.close() # <--- SEMPRE QUI, garantisce che la connessione torni al pool


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
    attempt_rows = db.execute(
        text("""
            select answers
            from quiz_attempts
            where project_id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchall()

    topic_mastery_map = {}

    for row in attempt_rows:

        answers = row[0] or []

        for a in answers:

            topic = a.get("topic", "general")

            if topic not in topic_mastery_map:
                topic_mastery_map[topic] = {
                    "correct": 0,
                    "total": 0
                }

            topic_mastery_map[topic]["total"] += 1

            if a.get("is_correct"):
                topic_mastery_map[topic]["correct"] += 1

    topic_mastery = []

    for topic, stats in topic_mastery_map.items():

        accuracy = 0

        if stats["total"] > 0:
            accuracy = round(
                (stats["correct"] / stats["total"]) * 100,
                1
            )

        topic_mastery.append({
            "topic": topic,
            "correct": stats["correct"],
            "total": stats["total"],
            "accuracy": accuracy
        })

    # ======================
    # GLOBAL QUIZ METRICS
    # ======================

    total_correct = sum(
        t["correct"] for t in topic_mastery
    )

    total_questions = sum(
        t["total"] for t in topic_mastery
    )

    total_wrong = total_questions - total_correct

    average_accuracy = 0

    if total_questions > 0:
        average_accuracy = round(
            (total_correct / total_questions) * 100,
            1
        )


    # ======================
    # FLASHCARD METRICS
    # ======================

    flashcard_stats = db.execute(
        text("""
            SELECT
                COUNT(*) as total_reviews,
                SUM(
                    CASE
                        WHEN is_correct THEN 1
                        ELSE 0
                    END
                ) as correct_reviews
            FROM flashcard_reviews
            WHERE project_id = :project_id
            AND user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchone()

    flashcard_reviews_count = int(flashcard_stats[0] or 0)
    flashcard_correct = int(flashcard_stats[1] or 0)

    flashcard_accuracy = 0

    if flashcard_reviews_count > 0:
        flashcard_accuracy = round(
            (flashcard_correct / flashcard_reviews_count) * 100,
            1
        )

    # ======================
    # DUE TODAY
    # ======================

    due_today = db.execute(
        text("""
            select count(*)
            from flashcards
            where project_id = :project_id
            and user_id = :user_id
            and next_review is not null
            and next_review <= now()
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).scalar()

    if due_today is None:
        due_today = 0

    # ======================
    # MOST FORGOTTEN TOPICS
    # ======================

    forgotten_rows = db.execute(
        text("""
            select
                topic,
                sum(
                    case
                        when is_correct then 1
                        else 0
                    end
                ) as correct,
                count(*) as total
            from flashcard_reviews
            where project_id = :project_id
            and user_id = :user_id
            and topic is not null
            group by topic
            having count(*) >= 2
        """),
        {
            "project_id": project_id,
            "user_id": user_id
        }
    ).fetchall()

    forgotten_topics = []

    for r in forgotten_rows:

        accuracy = 0

        if r[2] > 0:
            accuracy = round((r[1] / r[2]) * 100, 1)

        forgotten_topics.append({
            "topic": r[0],
            "accuracy": accuracy,
            "reviews": int(r[2])
        })

    forgotten_topics.sort(key=lambda x: x["accuracy"])

    # ======================
    # WEAK AREAS
    # ======================

    weak_areas = sorted(
        topic_mastery,
        key=lambda x: x["accuracy"]
    )[:3]

    db.close()


   
    # ======================
    # QUIZ ATTEMPTS
    # ======================

    quiz_attempts = db.execute(
        text("""
            select count(*)
            from quiz_attempts
            where project_id = :project_id
            and user_id = :user_id
        """),
        {
            "project_id": project_id,
            "user_id": user_id
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

    print("🔥 FINAL RESULTS RESPONSE:")
    print({
        "quiz_history": quiz_history,
        "topic_mastery": topic_mastery,
        "topics_detail": topic_mastery,
        "quiz_attempts": quiz_attempts or 0,
        "avg_score": round(avg_score, 1),
    })

    return {
        "quiz_history": quiz_history,
        "topic_mastery": topic_mastery,
        "topics_detail": topic_mastery,

        # QUIZ METRICS
        "quiz_attempts": quiz_attempts or 0,
        "total_correct": total_correct,
        "total_wrong": total_wrong,
        "average_accuracy": average_accuracy,

        # FLASHCARD METRICS
        "flashcard_reviews": flashcard_reviews_count,
        "flashcard_accuracy": flashcard_accuracy,

        # RETENTION INTELLIGENCE
        "due_today": due_today,
        "forgotten_topics": forgotten_topics[:3],

        # LEARNING METRICS
        "avg_score": round(avg_score, 1),
        "flashcards_reviewed": flashcards_reviewed or 0,
        "topics_count": topics_count or 0,

        # DIAGNOSTICS
        "weak_areas": weak_areas
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
                select project_id, topic
                from flashcards
                where id = :flashcard_id
            """),
            {
                "flashcard_id": flashcard_id
            }
        ).fetchone()
        print("🔥 SAVING FLASHCARD REVIEW:", {
            "flashcard_id": flashcard_id,
            "project_id": flashcard_row[0],
            "user_id": user["id"],
            "topic": flashcard_row[1],
            "is_correct": is_correct
        })
        if flashcard_row:
            db.execute(
                text("""
                    insert into flashcard_reviews
                    (flashcard_id, project_id, user_id, is_correct, difficulty, elapsed_seconds, topic)
                    values
                    (:flashcard_id, :project_id, :user_id, :is_correct, :difficulty, :elapsed_seconds, :topic)
                """),
                {
                    "flashcard_id": flashcard_id,
                    "project_id": flashcard_row[0],
                    "user_id": user["id"],
                    "is_correct": is_correct,
                    "difficulty": difficulty,
                    "elapsed_seconds": req.get("elapsed_seconds", 0),
                    "topic": flashcard_row[1]
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
    quiz_id = req.get("quiz_id")
    user_id = user["id"]  # <--- Recuperiamo l'ID dell'utente autenticato
    answers = req.get("answers", [])

    db = SessionLocal()
    try:
        # 1. Salva il tentativo (se non lo fa già Supabase, o per sicurezza)
        # Se hai già una riga in quiz_attempts, questo passaggio serve a legare il tutto
        
        for a in answers:
            db.execute(
                text("""
                    insert into quiz_answers (quiz_id, question_id, is_correct, topic, user_id)
                    values (:quiz_id, :question_id, :is_correct, :topic, :user_id)
                """),
                {
                    "quiz_id": quiz_id,
                    "question_id": a.get("question_id"),
                    "is_correct": a.get("is_correct", False),
                    "topic": (a.get("topic") or "General").strip().lower(),
                    "user_id": user_id  # <--- Passiamo lo user_id al DB
                }
            )
        db.commit()
        print(f"✅ Salvate {len(answers)} risposte per l'utente {user_id}")
        return {"status": "saved"}
    except Exception as e:
        db.rollback()
        print(f"❌ Database Error: {e}")
        return {"status": "error", "detail": str(e)}
    finally:
        db.close()

@app.get("/projects/{project_id}/stats")
async def get_quiz_stats(project_id: str, user = Depends(verify_user)):
    db = SessionLocal()
    try:
        # Questa query unisce Quiz e Flashcards calcolando i totali per ogni topic
        query = text("""
            SELECT 
                LOWER(TRIM(qa.topic)) as topic,
                COUNT(*) FILTER (WHERE qa.is_correct) as correct_count,
                COUNT(*) as total_count
            FROM quiz_answers qa
            WHERE qa.user_id = :u_id
            AND qa.topic IS NOT NULL
            AND LOWER(TRIM(qa.topic)) != 'general'
            GROUP BY LOWER(TRIM(qa.topic))

            UNION ALL

            SELECT 
                LOWER(TRIM(topic)) as topic,
                COUNT(*) FILTER (WHERE is_correct) as correct_count,
                COUNT(*) as total_count
            FROM flashcard_reviews
            WHERE project_id = :p_id
            AND user_id = :u_id
            AND topic IS NOT NULL
            AND LOWER(TRIM(topic)) != 'general'
            GROUP BY LOWER(TRIM(topic))
        """)
        
        result = db.execute(
            query,
            {"p_id": project_id, "u_id": user["id"]}
        ).fetchall()
        print("🔥 RAW STATS RESULT:", result)

        merged = {}

        for r in result:
            topic = (r[0] or "").strip().lower()

            if not topic:
                continue

            if topic not in merged:
                merged[topic] = {
                    "correct": 0,
                    "total": 0
                }

            merged[topic]["correct"] += int(r[1] or 0)
            merged[topic]["total"] += int(r[2] or 0)

        return merged
    finally:
        db.close()

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

@app.get("/projects/{project_id}/flashcards_detailed_stats")
async def get_flashcards_detailed_stats(project_id: str, user = Depends(verify_user)):
    db = SessionLocal()
    try:
        # Recupera il conteggio delle flashcard raggruppate per topic e difficoltà
        rows = db.execute(
            text("""
                SELECT LOWER(TRIM(topic)), difficulty, COUNT(*) as count
                FROM flashcards
                WHERE project_id = :project_id
                GROUP BY topic, difficulty
            """),
            {"project_id": project_id}
        ).fetchall()

        stats = {}
        for row in rows:
            topic = row[0] or "General"
            diff = row[1] or "unseen" # Se non sono state ancora studiate
            count = row[2]

            if topic not in stats:
                stats[topic] = {"wrong": 0, "hard": 0, "good": 0, "easy": 0, "total": 0}
            
            # Mapping delle chiavi per il frontend
            if diff == "wrong": stats[topic]["wrong"] += count
            elif diff == "hard": stats[topic]["hard"] += count
            elif diff == "good": stats[topic]["good"] += count
            elif diff == "easy": stats[topic]["easy"] += count
            
            stats[topic]["total"] += count

        return stats
    finally:
        db.close()

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

    scope = resolve_learning_scope(
        project_id=project_id,
        # topic_ids=topic_ids,
        limit=30
    )

    context_chunks = scope["chunks"]

    # fallback globale
    if not context_chunks:

        print("⚠️ NO CHUNKS FOUND → GLOBAL FALLBACK")

        scope = resolve_learning_scope(
            project_id=project_id,
            topic_ids=[],
            limit=30
        )

        context_chunks = scope["chunks"]

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
 
