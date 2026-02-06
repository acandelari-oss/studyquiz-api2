from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uuid
import os

app = FastAPI(title="StudyQuiz API")

# ====== AUTH ======
API_KEY = os.getenv("QUIZTEST_API_KEY", "")

def require_key(authorization: str | None):
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

# ====== IN-MEMORY STORES (prototype only) ======
PROJECT_DOCS = {}   # project_id -> list of documents {doc_id,title,text}
QUIZZES = {}        # quiz_id -> quiz json (macro_topics + answer_key + questions map)

# ====== SCHEMAS ======
class CreateProjectIn(BaseModel):
    name: str

class IngestDoc(BaseModel):
    doc_id: str
    title: str
    text: str

class IngestIn(BaseModel):
    documents: list[IngestDoc]

class GenerateQuizIn(BaseModel):
    language: str
    num_questions: int
    difficulty: str  # "medium" or "high"
    include_topics: list[str] | None = None
    exclude_topics: list[str] | None = None
    group_by_macro_topics: bool
    answers_at_end: bool

class ClarifyIn(BaseModel):
    quiz_id: str
    qid: str
    user_question: str

# ====== ENDPOINTS ======
@app.get("/health")
def health():
    return {"ok": True}

@app.post("/projects")
def create_project(payload: CreateProjectIn, authorization: str | None = Header(default=None)):
    require_key(authorization)
    project_id = str(uuid.uuid4())
    PROJECT_DOCS[project_id] = []
    return {"project_id": project_id}

@app.post("/projects/{project_id}/ingest")
def ingest(project_id: str, payload: IngestIn, authorization: str | None = Header(default=None)):
    require_key(authorization)
    if project_id not in PROJECT_DOCS:
        raise HTTPException(status_code=404, detail="Project not found")

    for d in payload.documents:
        PROJECT_DOCS[project_id].append(d.model_dump())
    return {"ingested": len(payload.documents)}

@app.post("/projects/{project_id}/generate_quiz")
def generate_quiz(project_id: str, payload: GenerateQuizIn, authorization: str | None = Header(default=None)):
    require_key(authorization)
    if project_id not in PROJECT_DOCS:
        raise HTTPException(status_code=404, detail="Project not found")

    # Prototype: return a MOCK quiz to validate wiring.
    quiz_id = str(uuid.uuid4())

    macro_topics = [{
        "name": "Cell Biology (Mock)",
        "questions": []
    }]

    answer_key = []

    # Create mock questions up to num_questions (cap to 10 for sanity in prototype)
    n = min(payload.num_questions, 10)
    for i in range(1, n + 1):
        qid = f"Q{i}"
        macro_topics[0]["questions"].append({
            "qid": qid,
            "stem": f"Mock high-level cell biology question #{i} based on ingested material.",
            "options": {
                "A": "Option A (relevant distractor)",
                "B": "Option B (relevant distractor)",
                "C": "Option C (relevant distractor)",
                "D": "Option D (relevant distractor)"
            }
        })
        answer_key.append({"qid": qid, "correct": "A"})

    quiz = {"quiz_id": quiz_id, "macro_topics": macro_topics, "answer_key": answer_key}
    QUIZZES[quiz_id] = quiz
    return quiz

@app.post("/projects/{project_id}/clarify")
def clarify(project_id: str, payload: ClarifyIn, authorization: str | None = Header(default=None)):
    require_key(authorization)
    if project_id not in PROJECT_DOCS:
        raise HTTPException(status_code=404, detail="Project not found")

    quiz = QUIZZES.get(payload.quiz_id)
    if not quiz:
        raise HTTPException(status_code=404, detail="Quiz not found")

    # Prototype: return mock evidence extracted from first document (if any)
    evidence = []
    docs = PROJECT_DOCS.get(project_id, [])
    if docs:
        snippet = docs[0]["text"][:400]
        evidence.append({"doc_title": docs[0]["title"], "snippet": snippet})

    return {
        "answer": "Prototype clarification (mock). This will later be generated using only the ingested evidence.",
        "evidence": evidence
    }
