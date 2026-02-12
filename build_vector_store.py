"""
Complete Pipeline: Build Vector Store
--------------------------------------
Runs the full pipeline from PDFs to searchable vector store.

Usage:
    python build_vector_store.py
"""

import sys
import os

from src.ingestion.pdf_loader import load_aviation_pdfs
from src.ingestion.chunking import chunk_documents, AviationChunker
from src.ingestion.embeddings import EmbeddingGenerator, EmbeddingConfig
from src.ingestion.vector_store import FAISSVectorStore, VectorStoreConfig
import numpy as np
from datetime import datetime


def main():
    """Run complete pipeline"""
    
    print("="*70)
    print("🚀 AVIATION RAG - VECTOR STORE BUILDER")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    PDF_DIR = "data/raw"
    PROCESSED_DIR = "data/processed"
    VECTOR_STORE_DIR = "data/vector_store"
    
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    BATCH_SIZE = 32
    
    # Create directories
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    
    # =========================================================================
    # STEP 1: Load PDFs
    # =========================================================================
    print("─" * 70)
    print("STEP 1: Loading PDF Documents")
    print("─" * 70)
    
    if not os.path.exists(PDF_DIR):
        print(f"❌ Error: {PDF_DIR} directory not found!")
        print(f"   Please create it and add your aviation PDFs.")
        return
    
    documents = load_aviation_pdfs(PDF_DIR)
    
    if not documents:
        print(f"❌ Error: No PDFs found in {PDF_DIR}!")
        return
    
    print(f"\n✅ Loaded {len(documents)} document(s):")
    for filename, (pages, metadata) in documents.items():
        print(f"   📄 {filename}")
        print(f"      Pages: {len(pages)}")
        print(f"      Size: {metadata.file_size_mb} MB")
        total_chars = sum(len(p.text) for p in pages)
        print(f"      Total characters: {total_chars:,}")
    
    # =========================================================================
    # STEP 2: Chunk Documents
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 2: Chunking Documents")
    print("─" * 70)
    print(f"   Chunk size: {CHUNK_SIZE} characters")
    print(f"   Overlap: {CHUNK_OVERLAP} characters")
    print(f"   Strategy: Aviation-optimized")
    print()
    
    chunks_dict = chunk_documents(
        documents,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    
    total_chunks = sum(len(chunks) for chunks in chunks_dict.values())
    
    print(f"\n✅ Created {total_chunks} chunks:")
    for doc_name, chunks in chunks_dict.items():
        print(f"   📄 {doc_name}: {len(chunks)} chunks")
        
        # Show statistics
        chunker = AviationChunker(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        stats = chunker.get_chunk_statistics(chunks)
        print(f"      Avg length: {stats['avg_length']:.0f} chars")
        print(f"      Range: {stats['min_length']}-{stats['max_length']} chars")
    
    # =========================================================================
    # STEP 3: Generate Embeddings
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 3: Generating Embeddings")
    print("─" * 70)
    print(f"   Model: {EMBEDDING_MODEL}")
    print(f"   Batch size: {BATCH_SIZE}")
    print()
    
    # Initialize generator
    config = EmbeddingConfig(
        model_name=EMBEDDING_MODEL,
        batch_size=BATCH_SIZE,
        device="auto",
        normalize=True
    )
    
    generator = EmbeddingGenerator(config)
    
    # Generate embeddings for each document
    all_embeddings = []
    all_metadata = []
    
    for doc_name, chunks in chunks_dict.items():
        print(f"   Processing {doc_name}...")
        
        embeddings, metadata = generator.embed_chunks(chunks, show_progress=True)
        
        # Add document name to all metadata
        for meta in metadata:
            meta['document_name'] = doc_name
        
        all_embeddings.append(embeddings)
        all_metadata.extend(metadata)
        
        # Save individual document embeddings
        safe_name = doc_name.replace('.pdf', '').replace(' ', '_')
        generator.save_embeddings(
            embeddings,
            metadata,
            PROCESSED_DIR,
            filename=f"{safe_name}_embeddings"
        )
        
        print(f"      ✅ {len(embeddings)} embeddings saved")
    
    # Combine all embeddings (Filter out empty ones first!)
    valid_embeddings = [e for e in all_embeddings if len(e) > 0]
    
    if not valid_embeddings:
        print("❌ Error: No embeddings were generated. Check your PDF content.")
        return

    all_embeddings = np.vstack(valid_embeddings)
    
    print(f"\n✅ Total embeddings generated: {len(all_embeddings)}")
    print(f"   Embedding dimension: {all_embeddings.shape[1]}")
    print(f"   Total size: {all_embeddings.nbytes / (1024**2):.2f} MB")
    
    # =========================================================================
    # STEP 4: Build Vector Store
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 4: Building FAISS Vector Store")
    print("─" * 70)
    
    store_config = VectorStoreConfig(
        index_type="Flat",
        dimension=all_embeddings.shape[1],
        metric="cosine"
    )
    
    store = FAISSVectorStore(store_config)
    
    print("   Adding embeddings to index...")
    store.add_embeddings(all_embeddings, all_metadata, normalize=True)
    
    print(f"\n✅ Vector store created:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    # =========================================================================
    # STEP 5: Save Vector Store
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 5: Saving Vector Store to Disk")
    print("─" * 70)
    
    store.save(VECTOR_STORE_DIR, "aviation_vectors")
    
    print(f"\n✅ Vector store saved to {VECTOR_STORE_DIR}/")
    print("   Files created:")
    print(f"   - aviation_vectors.index (FAISS index)")
    print(f"   - aviation_vectors_metadata.json (metadata)")
    print(f"   - aviation_vectors_config.json (configuration)")
    print(f"   - aviation_vectors_id_mapping.pkl (ID mapping)")
    
    # =========================================================================
    # STEP 6: Test Search
    # =========================================================================
    print("\n" + "─" * 70)
    print("STEP 6: Testing Search Functionality")
    print("─" * 70)
    
    test_queries = [
        "What are the emergency procedures for engine failure during takeoff?",
        "What is the stall speed of an aircraft?",
        "VFR weather minimums for flight",
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: '{query}'")
        
        # Generate query embedding
        query_embedding = generator.embed_text(query)
        
        # Search
        results = store.search(query_embedding, k=3)
        
        print(f"   Top 3 results:")
        for i, result in enumerate(results, 1):
            meta = result['metadata']
            score = result['score']
            
            print(f"\n   {i}. Similarity: {score:.4f}")
            print(f"      Document: {meta.get('document_name', 'Unknown')}")
            print(f"      Page: {meta.get('page_number', '?')}")
            print(f"      Chunk: {meta.get('chunk_id', '?')}")
            
            # Show text preview
            preview = meta.get('text_preview', '')
            if preview:
                print(f"      Preview: {preview[:80]}...")
    
    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETE!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"   Documents processed: {len(documents)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total embeddings: {len(all_embeddings)}")
    print(f"   Vector dimension: {all_embeddings.shape[1]}")
    print(f"   Index type: {store_config.index_type}")
    print(f"   Metric: {store_config.metric}")
    
    print(f"\n📁 Output files:")
    print(f"   Processed: {PROCESSED_DIR}/")
    print(f"   Vector Store: {VECTOR_STORE_DIR}/")
    
    print(f"\n🎯 Next steps:")
    print(f"   1. Test search with various queries")
    print(f"   2. Build the RAG pipeline (retrieval + generation)")
    print(f"   3. Create the API endpoints")
    print(f"   4. Run evaluation on 50 questions")
    
    print(f"\n⏱️  Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 Check the error above and try again")