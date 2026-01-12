from pathlib import Path
import os
from typing import List, Dict, Any

import chromadb

from llama_index.core import (
    VectorStoreIndex,
    StorageContext,
    Settings,
    SimpleDirectoryReader,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI


DATA_DIR = Path("data/uploads")
PERSIST_DIR = Path("storage")
COLLECTION_NAME = "jarvis"


class JarvisRAG:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        PERSIST_DIR.mkdir(parents=True, exist_ok=True)

        Settings.llm = OpenAI(
            model="gpt-4o-mini",
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY"),
        )

        splitter = SentenceSplitter(
            chunk_size=900,
            chunk_overlap=120,
        )

        chroma_client = chromadb.PersistentClient(path=str(PERSIST_DIR))
        chroma_collection = chroma_client.get_or_create_collection(
            COLLECTION_NAME
        )

        vector_store = ChromaVectorStore(
            chroma_collection=chroma_collection
        )

        storage_context = StorageContext.from_defaults(
            vector_store=vector_store
        )

        self.index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
        )

        self.splitter = splitter

    def ingest_file(self, filepath: Path) -> int:
        reader = SimpleDirectoryReader(
            input_files=[str(filepath)],
            filename_as_id=True,
        )
        documents = reader.load_data()

        nodes = self.splitter.get_nodes_from_documents(documents)
        self.index.insert_nodes(nodes)
        return len(nodes)

    def query(self, question: str, top_k: int = 8) -> Dict[str, Any]:
        engine = self.index.as_query_engine(
            similarity_top_k=top_k,
            response_mode="compact",
        )
        response = engine.query(question)

        citations = []
        for src in response.source_nodes or []:
            node = src.node
            meta = node.metadata or {}
            citations.append(
                {
                    "source": meta.get("file_name", ""),
                    "score": float(src.score) if src.score else None,
                    "text": node.get_text()[:600],
                }
            )

        return {
            "answer": str(response),
            "citations": citations,
        }
