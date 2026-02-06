from fastapi import FastAPI
from pydantic import BaseModel
import uuid

app = FastAPI(title="StudyQuiz API")

class CreateProjectIn(BaseModel):
    name: str

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/projects")
def create_project(payload: CreateProjectIn):
    return {"project_id": str(uuid.uuid4())}
