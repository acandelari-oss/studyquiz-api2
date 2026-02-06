from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import uuid
import os

app = FastAPI(title="StudyQuiz API")

# Set this in Render environment variables
API_KEY = os.getenv("QUIZTEST_API_KEY", "")

class CreateProjectIn(BaseModel):
    name: str

def require_key(authorization: str | None):
    # If no key is configured, allow requests (useful for early testing)
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing API key")
    token = authorization.split("Bearer ", 1)[1].strip()
    if token != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/projects")
def create_project(payload: CreateProjectIn, authorization: str | None = Header(default=None)):
    require_key(authorization)
    return {"project_id": str(uuid.uuid4())}
