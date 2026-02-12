from src.ingestion.vector_store import FAISSVectorStore

store = FAISSVectorStore()
store.load("data/vector_store", "aviation_vectors")

# Find chunks from Page 11 of Meteorology book
print("Searching for Page 11 content...")
for meta in store.metadata:
    if "Meteorology" in meta.get("document_name", "") and meta.get("page_number") == 11:
        print("-" * 50)
        print(f"CHUNK ID: {meta.get('chunk_id')}")
        print(f"TEXT: {meta.get('text_preview')}")