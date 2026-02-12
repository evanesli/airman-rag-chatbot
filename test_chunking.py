"""
Test Script for Chunking System
Tests different chunking strategies and displays results
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.pdf_loader import load_aviation_pdfs
from src.ingestion.chunking import (
    AviationChunker, 
    ChunkingStrategy,
    chunk_documents
)


def test_basic_chunking():
    """Test basic chunking functionality"""
    print("\n" + "="*70)
    print("TEST 1: Basic Chunking")
    print("="*70)
    
    # Load documents
    print("\n📂 Loading PDFs from data/raw/...")
    try:
        documents = load_aviation_pdfs("data/raw")
        
        if not documents:
            print("❌ No documents loaded!")
            return False
        
        # Get first document
        filename, (pages, metadata) = list(documents.items())[0]
        
        print(f"\n✅ Testing with: {filename}")
        print(f"   Pages: {len(pages)}")
        print(f"   Total chars: {sum(len(p.text) for p in pages):,}")
        
        # Create chunker
        chunker = AviationChunker(
            chunk_size=1000,
            chunk_overlap=200,
            strategy=ChunkingStrategy.AVIATION,
            preserve_sections=True
        )
        
        # Chunk pages
        print("\n🔪 Chunking pages...")
        chunks = chunker.chunk_pages(pages)
        
        print(f"\n✅ Created {len(chunks)} chunks")
        
        # Display sample chunks
        print("\n📄 Sample Chunks:")
        print("-" * 70)
        
        for i in range(min(5, len(chunks))):
            chunk = chunks[i]
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Page: {chunk.metadata.get('page_number', '?')}")
            print(f"  Length: {len(chunk.text)} chars")
            
            if "section_type" in chunk.metadata:
                print(f"  Type: {chunk.metadata['section_type']}")
            
            if "section_header" in chunk.metadata:
                print(f"  Section: {chunk.metadata['section_header']}")
            
            # Show text preview
            preview = chunk.text[:200].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        
        # Statistics
        print("\n" + "="*70)
        print("📊 Chunking Statistics")
        print("="*70)
        
        stats = chunker.get_chunk_statistics(chunks)
        
        print(f"\nTotal chunks: {stats['total_chunks']}")
        print(f"Average length: {stats['avg_length']:.0f} chars")
        print(f"Min length: {stats['min_length']} chars")
        print(f"Max length: {stats['max_length']} chars")
        print(f"Total characters: {stats['total_chars']:,}")
        print(f"Chunks with sections: {stats.get('chunks_with_sections', 0)}")
        
        # Calculate chunks per page
        chunks_per_page = len(chunks) / len(pages)
        print(f"Average chunks per page: {chunks_per_page:.1f}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_strategy_comparison():
    """Compare different chunking strategies"""
    print("\n" + "="*70)
    print("TEST 2: Strategy Comparison")
    print("="*70)
    
    try:
        # Load documents
        documents = load_aviation_pdfs("data/raw")
        if not documents:
            return False
        
        filename, (pages, metadata) = list(documents.items())[0]
        print(f"\nComparing strategies on: {filename}")
        
        strategies = [
            ("FIXED", ChunkingStrategy.FIXED),
            ("SEMANTIC", ChunkingStrategy.SEMANTIC),
            ("SLIDING", ChunkingStrategy.SLIDING),
            ("AVIATION", ChunkingStrategy.AVIATION),
        ]
        
        results = {}
        
        for name, strategy in strategies:
            print(f"\n{'─' * 70}")
            print(f"Testing {name} strategy...")
            
            chunker = AviationChunker(
                chunk_size=1000,
                chunk_overlap=200,
                strategy=strategy
            )
            
            chunks = chunker.chunk_pages(pages[:10])  # Test on first 10 pages
            stats = chunker.get_chunk_statistics(chunks)
            
            results[name] = {
                "chunks": len(chunks),
                "avg_length": stats['avg_length'],
                "min_length": stats['min_length'],
                "max_length": stats['max_length'],
                "with_sections": stats.get('chunks_with_sections', 0)
            }
            
            print(f"  ✅ Created {len(chunks)} chunks")
            print(f"     Avg: {stats['avg_length']:.0f} chars")
            print(f"     Range: {stats['min_length']}-{stats['max_length']} chars")
        
        # Summary comparison
        print("\n" + "="*70)
        print("📊 Strategy Comparison Summary")
        print("="*70)
        print(f"\n{'Strategy':<12} {'Chunks':<10} {'Avg Size':<12} {'With Headers':<15}")
        print("-" * 70)
        
        for name, data in results.items():
            print(f"{name:<12} {data['chunks']:<10} {data['avg_length']:<12.0f} {data['with_sections']:<15}")
        
        # Recommendation
        print("\n💡 Recommendation:")
        print("   AVIATION strategy is recommended because:")
        print("   - Preserves procedures and warnings")
        print("   - Includes section headers")
        print("   - Smart boundary detection")
        print("   - Optimized for aviation content")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_chunk_quality():
    """Test chunk quality and content preservation"""
    print("\n" + "="*70)
    print("TEST 3: Chunk Quality Analysis")
    print("="*70)
    
    try:
        documents = load_aviation_pdfs("data/raw")
        if not documents:
            return False
        
        filename, (pages, metadata) = list(documents.items())[0]
        
        chunker = AviationChunker(
            chunk_size=1000,
            chunk_overlap=200,
            strategy=ChunkingStrategy.AVIATION
        )
        
        chunks = chunker.chunk_pages(pages)
        
        # Quality checks
        print("\n🔍 Quality Checks:")
        print("-" * 70)
        
        # Check 1: No empty chunks
        empty_chunks = [c for c in chunks if len(c.text.strip()) == 0]
        if empty_chunks:
            print(f"⚠️  Found {len(empty_chunks)} empty chunks")
        else:
            print("✅ No empty chunks")
        
        # Check 2: Proper overlaps
        overlapping = 0
        for i in range(len(chunks) - 1):
            if chunks[i].next_chunk_id == chunks[i+1].chunk_id:
                overlapping += 1
        
        print(f"✅ {overlapping}/{len(chunks)-1} chunks properly linked")
        
        # Check 3: Page number tracking
        chunks_with_pages = sum(1 for c in chunks if 'page_number' in c.metadata)
        print(f"✅ {chunks_with_pages}/{len(chunks)} chunks have page numbers")
        
        # Check 4: Find procedure chunks
        procedure_chunks = [c for c in chunks if 'section_type' in c.metadata]
        print(f"✅ {len(procedure_chunks)} chunks identified as special sections")
        
        # Check 5: Size distribution
        sizes = [len(c.text) for c in chunks]
        too_small = sum(1 for s in sizes if s < 100)
        too_large = sum(1 for s in sizes if s > 1000)
        
        if too_small > 0:
            print(f"⚠️  {too_small} chunks are very small (<100 chars)")
        else:
            print("✅ No chunks too small")
        
        if too_large > 0:
            print(f"⚠️  {too_large} chunks are very large (>1000 chars)")
        else:
            print("✅ No chunks too large")
        
        # Show examples of different chunk types
        print("\n📋 Example Chunks by Type:")
        print("-" * 70)
        
        # Find a procedure chunk
        proc_chunks = [c for c in chunks if 'section_type' in c.metadata]
        if proc_chunks:
            print("\n🔧 Procedure/Warning Chunk:")
            print(f"   {proc_chunks[0].text[:200]}...")
        
        # Find a regular chunk with header
        header_chunks = [c for c in chunks if 'section_header' in c.metadata]
        if header_chunks:
            print("\n📖 Chunk with Section Header:")
            print(f"   {header_chunks[0].text[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_documents():
    """Test chunking all documents"""
    print("\n" + "="*70)
    print("TEST 4: Multiple Documents")
    print("="*70)
    
    try:
        # Load all documents
        print("\n📂 Loading all documents...")
        documents = load_aviation_pdfs("data/raw")
        
        if not documents:
            return False
        
        # Chunk all documents
        print(f"\n🔪 Chunking {len(documents)} document(s)...")
        
        all_chunks = chunk_documents(
            documents,
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Display results
        print("\n" + "="*70)
        print("📊 Results by Document")
        print("="*70)
        
        total_chunks = 0
        total_chars = 0
        
        for filename, chunks in all_chunks.items():
            chunk_stats = AviationChunker(chunk_size=1000, chunk_overlap=200).get_chunk_statistics(chunks)
            
            print(f"\n📄 {filename}")
            print(f"   Chunks: {len(chunks)}")
            print(f"   Avg size: {chunk_stats['avg_length']:.0f} chars")
            print(f"   Total chars: {chunk_stats['total_chars']:,}")
            
            total_chunks += len(chunks)
            total_chars += chunk_stats['total_chars']
        
        print("\n" + "="*70)
        print(f"📈 Overall: {total_chunks} total chunks, {total_chars:,} total characters")
        print("="*70)
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 CHUNKING SYSTEM - TEST SUITE")
    print("="*70)
    
    # Check if data exists
    if not os.path.exists("data/raw"):
        print("\n❌ data/raw/ directory not found!")
        print("   Please add your PDFs and run pdf_loader first!")
        return
    
    # Run tests
    tests = [
        ("Basic Chunking", test_basic_chunking),
        ("Strategy Comparison", test_strategy_comparison),
        ("Chunk Quality", test_chunk_quality),
        ("Multiple Documents", test_multiple_documents)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*70)
    print("📋 TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{status} - {test_name}")
    
    total = len(results)
    passed = sum(1 for r in results.values() if r)
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All chunking tests passed!")
        print("\n✅ Your chunking system is ready!")
        print("\nNext steps:")
        print("1. Review chunk quality manually")
        print("2. Adjust chunk_size/overlap if needed")
        print("3. Move on to embeddings generation")
    else:
        print("\n⚠️  Some tests failed. Review errors above.")


if __name__ == "__main__":
    main()