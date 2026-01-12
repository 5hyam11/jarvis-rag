from fastapi import FastAPI, UploadFile, File
from pathlib import Path

from rag import JarvisRAG

app = FastAPI(title="Jarvis RAG")
jarvis = JarvisRAG()

UPLOAD_DIR = Path("data/uploads")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / file.filename

    contents = await file.read()
    path.write_bytes(contents)

    chunks = jarvis.ingest_file(path)
    return {"filename": file.filename, "chunks_added": chunks}


@app.post("/chat")
def chat(question: str, top_k: int = 8):
    return jarvis.query(question, top_k=top_k)
