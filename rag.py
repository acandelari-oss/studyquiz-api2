import re
from openai import OpenAI

client = OpenAI()

def chunk_text(text: str, size=1200, overlap=200):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, i = [], 0
    while i < len(text):
        chunks.append(text[i:i+size])
        i += size - overlap
    return chunks

def embed(texts: list[str]) -> list[list[float]]:
    res = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in res.data]
