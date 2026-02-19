from fastapi import FastAPI, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import openai
import uuid
import os

DATABASE_URL = os.environ["DATABASE_URL"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
BACKEND_API_KEY = os.environ["BACKEND_API_KEY"]

openai.api_key = OPENAI_API_KEY

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(bind=engine)

app = FastAPI()

security = HTTPBearer()


def require_key(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    if credentials.credentials != BACKEND_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class ProjectCreate(BaseModel):
    name: str


class IngestDoc(BaseModel):
    doc_id: str
    title: str
    text: str


class IngestRequest(BaseModel):
    project_id: str
    documents: list[IngestDoc]


@app.post("/projects")
def create_project(
    payload: ProjectCreate,
    _: HTTPAuthorizationCredentials = Depends(require_key),
):
    return {"project_id": str(uuid.uuid4())}


def embed(texts):

    res = openai.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )

    return [d.embedding for d in res.data]


@app.post("/projects/{project_id}/ingest")
def ingest(
    project_id: str,
    payload: IngestRequest,
    _: HTTPAuthorizationCredentials = Depends(require_key),
):

    db = SessionLocal()

    try:

        texts = [doc.text for doc in payload.documents]

        vectors = embed(texts)

        for text_chunk, vec in zip(texts, vectors):

            db.execute(

                text("""

                insert into chunks
                (project_id, chunk_text, embedding)

                values

                (
                    :pid,
                    :text,
                    CAST(:emb AS vector)
                )

                """),

                {
                    "pid": project_id,
                    "text": text_chunk,
                    "emb": vec,
                }

            )

        db.commit()

        return {"status": "ok"}

    except Exception as e:

        db.rollback()

        print("INGEST ERROR:", str(e))

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:

        db.close()
