from src.ingestion.vector_store import FAISSVectorStore
from src.ingestion.embeddings import EmbeddingGenerator, EmbeddingConfig

# Load vector store
store = FAISSVectorStore()
store.load("data/vector_store", "aviation_vectors")
print(f"Loaded {len(store.metadata)} vectors ✅")

# Test search
config = EmbeddingConfig()
gen = EmbeddingGenerator(config)
query_emb = gen.embed_text("stall recovery procedure")
results = store.search(query_emb, k=3)

for i, r in enumerate(results, 1):
    print(f"{i}. Score: {r['score']:.3f} - Page {r['metadata']['page_number']}")