import sys
from pathlib import Path
from rag import JarvisRAG

if len(sys.argv) < 2:
    print("Usage: python ingest.py <file>")
    sys.exit(1)

jarvis = JarvisRAG()
count = jarvis.ingest_file(Path(sys.argv[1]))
print(f"Ingested {count} chunks")
