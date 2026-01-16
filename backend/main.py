"""
Jarvis API - FastAPI Backend for Render
Handles chat, voice transcription (STT), text-to-speech (TTS), and RAG
"""

import os
import tempfile
from pathlib import Path
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

from openai import OpenAI

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb


# ============================================================
# Configuration
# ============================================================

DATA_DIR = Path("data/uploads")
STORAGE_DIR = Path("data/chroma")
COLLECTION_NAME = "jarvis"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")

FRONTEND_URL = os.getenv("FRONTEND_URL", "*")


# ============================================================
# Global State
# ============================================================

class AppState:
    index: Optional[VectorStoreIndex] = None
    openai_client: Optional[OpenAI] = None
    chroma_client = None
    chroma_collection = None


state = AppState()


# ============================================================
# Lifespan (startup/shutdown)
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    
    state.openai_client = OpenAI(api_key=OPENAI_API_KEY)
    
    Settings.llm = LlamaOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    Settings.embed_model = OpenAIEmbedding(
        model="text-embedding-3-small", 
        api_key=OPENAI_API_KEY
    )
    
    state.chroma_client = chromadb.PersistentClient(path=str(STORAGE_DIR))
    state.chroma_collection = state.chroma_client.get_or_create_collection(COLLECTION_NAME)
    
    vector_store = ChromaVectorStore(chroma_collection=state.chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)
    
    state.index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )
    
    print("✅ Jarvis API initialized")
    yield
    print("👋 Jarvis API shutting down")


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="Jarvis API",
    description="Voice-enabled RAG assistant API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL] if FRONTEND_URL != "*" else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Pydantic Models
# ============================================================

class ChatRequest(BaseModel):
    message: str
    use_rag: bool = True
    top_k: int = 5


class ChatResponse(BaseModel):
    answer: str
    sources: List[dict] = []


class TranscriptionResponse(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str
    voice: str = "alloy"


# ============================================================
# Endpoints
# ============================================================

@app.get("/")
def root():
    return {"message": "Jarvis API is running", "docs": "/docs"}


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "rag_enabled": state.index is not None,
        "documents_count": state.chroma_collection.count() if state.chroma_collection else 0
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        sources = []
        
        if request.use_rag and state.index and state.chroma_collection.count() > 0:
            query_engine = state.index.as_query_engine(
                similarity_top_k=request.top_k,
                response_mode="compact",
            )
            response = query_engine.query(request.message)
            answer = str(response)
            
            for node in response.source_nodes or []:
                meta = node.node.metadata or {}
                sources.append({
                    "file": meta.get("file_name", "Unknown"),
                    "score": round(node.score, 3) if node.score else None,
                    "excerpt": node.node.get_text()[:300] + "..."
                })
        else:
            response = state.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are Jarvis, a helpful AI assistant. Be concise and helpful."},
                    {"role": "user", "content": request.message}
                ],
                max_tokens=1000,
            )
            answer = response.choices[0].message.content
        
        return ChatResponse(answer=answer, sources=sources)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...)):
    try:
        suffix = Path(audio.filename).suffix if audio.filename else ".webm"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await audio.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        with open(tmp_path, "rb") as audio_file:
            transcript = state.openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        
        os.unlink(tmp_path)
        return TranscriptionResponse(text=transcript.strip())
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/tts")
async def text_to_speech(request: TTSRequest):
    try:
        text = request.text[:4096]
        
        response = state.openai_client.audio.speech.create(
            model="tts-1",
            voice=request.voice,
            input=text,
            response_format="mp3"
        )
        
        audio_stream = io.BytesIO(response.content)
        
        return StreamingResponse(
            audio_stream,
            media_type="audio/mpeg",
            headers={"Content-Disposition": "inline; filename=speech.mp3"}
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TTS failed: {str(e)}")


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    try:
        allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"File type not supported. Allowed: {allowed_extensions}"
            )
        
        file_path = DATA_DIR / file.filename
        content = await file.read()
        file_path.write_bytes(content)
        
        reader = SimpleDirectoryReader(input_files=[str(file_path)])
        documents = reader.load_data()
        
        splitter = SentenceSplitter(chunk_size=900, chunk_overlap=120)
        nodes = splitter.get_nodes_from_documents(documents)
        
        state.index.insert_nodes(nodes)
        
        return {
            "filename": file.filename,
            "chunks_added": len(nodes),
            "total_documents": state.chroma_collection.count()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/documents")
def list_documents():
    files = list(DATA_DIR.glob("*")) if DATA_DIR.exists() else []
    return {
        "documents": [f.name for f in files if f.is_file()],
        "total_chunks": state.chroma_collection.count() if state.chroma_collection else 0
    }


@app.delete("/documents/{filename}")
def delete_document(filename: str):
    file_path = DATA_DIR / filename
    if file_path.exists():
        file_path.unlink()
        return {"deleted": filename}
    raise HTTPException(status_code=404, detail="File not found")


