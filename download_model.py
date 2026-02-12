# Create file: download_model.py
from sentence_transformers import SentenceTransformer

print("Downloading embedding model...")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
print("✅ Model downloaded successfully!")
print(f"Model dimension: {model.get_sentence_embedding_dimension()}")