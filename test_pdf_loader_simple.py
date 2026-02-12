"""
Simple Test Script for PDF Loader
Run this to verify your PDF loader is working correctly
"""

import os
import sys
from tqdm import tqdm  # Progress bar
# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ingestion.pdf_loader import AviationPDFLoader, load_aviation_pdfs


def test_single_pdf():
    """Test loading a single PDF file"""
    print("\n" + "="*70)
    print("TEST 1: Loading a Single PDF")
    print("="*70)
    
    # Find first PDF in data/raw
    raw_dir = "data/raw"
    
    if not os.path.exists(raw_dir):
        print(f"❌ Directory not found: {raw_dir}")
        print("   Please create it and add your aviation PDFs!")
        return False
    
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print(f"❌ No PDF files found in {raw_dir}")
        print("   Please add your aviation PDFs there!")
        return False
    
    # Load first PDF
    pdf_path = os.path.join(raw_dir, pdf_files[0])
    print(f"📄 Loading: {pdf_files[0]}")
    
    try:
        loader = AviationPDFLoader(
            extraction_method="auto",
            remove_headers_footers=True,
            clean_text=True
        )
        
        pages, metadata = loader.load_pdf(pdf_path)
        
        # Print results
        print(f"\n✅ SUCCESS! Loaded document:")
        print(f"   Filename: {metadata.filename}")
        print(f"   Pages: {metadata.total_pages}")
        print(f"   Size: {metadata.file_size_mb} MB")
        if metadata.title:
            print(f"   Title: {metadata.title}")
        if metadata.author:
            print(f"   Author: {metadata.author}")
        
        # Show statistics
        total_chars = sum(len(p.text) for p in pages)
        avg_chars = total_chars / len(pages)
        
        print(f"\n📊 Content Statistics:")
        print(f"   Total characters: {total_chars:,}")
        print(f"   Average per page: {avg_chars:.0f}")
        print(f"   Shortest page: {min(len(p.text) for p in pages)} chars")
        print(f"   Longest page: {max(len(p.text) for p in pages)} chars")
        
        # Show sample from first page
        print(f"\n📖 First Page Preview (200 chars):")
        print("-" * 70)
        print(pages[0].text[:200] + "...")
        print("-" * 70)
        
        # Check if extraction was good
        if avg_chars < 50:
            print("\n⚠️  WARNING: Very low character count per page!")
            print("   This might be a scanned PDF that needs OCR.")
        else:
            print("\n✅ Text extraction looks good!")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_multiple_pdfs():
    """Test loading all PDFs from directory"""
    print("\n" + "="*70)
    print("TEST 2: Loading All PDFs from Directory")
    print("="*70)
    
    try:
        documents = load_aviation_pdfs("data/raw")
        
        if not documents:
            print("❌ No PDFs loaded")
            return False
        
        print(f"\n✅ Loaded {len(documents)} document(s):\n")
        
        for filename, (pages, metadata) in documents.items():
            total_chars = sum(len(p.text) for p in pages)
            avg_chars = total_chars / len(pages)
            
            print(f"📄 {filename}")
            print(f"   Pages: {len(pages)}")
            print(f"   Total characters: {total_chars:,}")
            print(f"   Avg chars/page: {avg_chars:.0f}")
            print()
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extraction_methods():
    """Test different extraction methods"""
    print("\n" + "="*70)
    print("TEST 3: Comparing Extraction Methods")
    print("="*70)
    
    raw_dir = "data/raw"
    pdf_files = [f for f in os.listdir(raw_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        print("❌ No PDFs to test")
        return False
    
    pdf_path = os.path.join(raw_dir, pdf_files[0])
    print(f"📄 Testing with: {pdf_files[0]}\n")
    
    methods = ["pypdf", "pymupdf", "pdfplumber"]
    results = {}
    
    for method in methods:
        try:
            print(f"Testing {method}...", end=" ")
            loader = AviationPDFLoader(extraction_method=method, clean_text=False)
            pages, _ = loader.load_pdf(pdf_path)
            
            total_chars = sum(len(p.text) for p in pages)
            avg_chars = total_chars / len(pages)
            
            results[method] = {
                "success": True,
                "total_chars": total_chars,
                "avg_chars": avg_chars
            }
            print(f"✅ {avg_chars:.0f} chars/page")
            
        except Exception as e:
            results[method] = {"success": False, "error": str(e)}
            print(f"❌ Failed: {e}")
    
    # Show comparison
    print(f"\n📊 Comparison:")
    print("-" * 70)
    successful = [m for m, r in results.items() if r.get("success")]
    
    if successful:
        best = max(successful, key=lambda m: results[m]["avg_chars"])
        print(f"🏆 Best method: {best} ({results[best]['avg_chars']:.0f} chars/page)")
        print(f"\n   Use this in your config for this document type!")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print("🧪 AVIATION PDF LOADER - TEST SUITE")
    print("="*70)
    
    # Check if data/raw exists
    if not os.path.exists("data/raw"):
        print("\n❌ data/raw/ directory not found!")
        print("\nPlease:")
        print("1. Create data/raw/ directory")
        print("2. Add your aviation PDF files there")
        print("3. Run this test again")
        return
    
    # Run tests
    tests = [
        ("Single PDF", test_single_pdf),
        ("Multiple PDFs", test_multiple_pdfs),
        ("Extraction Methods", test_extraction_methods)
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
        print("\n🎉 All tests passed! Your PDF loader is working perfectly!")
        print("\nNext steps:")
        print("1. Review the extracted text to ensure quality")
        print("2. Move on to building the chunking system")
        print("3. Start creating embeddings")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("\nCommon issues:")
        print("- PDFs not in data/raw/ directory")
        print("- Scanned PDFs (need OCR)")
        print("- Missing dependencies (run: pip install -r requirements.txt)")


if __name__ == "__main__":
    main()