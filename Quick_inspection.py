# Quick inspection script
from src.ingestion.pdf_loader import load_aviation_pdfs
from src.ingestion.chunking import AviationChunker, ChunkingStrategy

docs = load_aviation_pdfs("data/raw")
filename, (pages, meta) = list(docs.items())[0]

chunker = AviationChunker(
    chunk_size=1000,
    chunk_overlap=200,
    strategy=ChunkingStrategy.AVIATION
)

chunks = chunker.chunk_pages(pages)

# Look at 10 random chunks
import random
for chunk in random.sample(chunks, 10):
    print(f"\n{'='*60}")
    print(f"Chunk {chunk.chunk_id} (Page {chunk.metadata['page_number']})")
    print('='*60)
    print(chunk.text)
    print()
    input("Press Enter for next chunk...")