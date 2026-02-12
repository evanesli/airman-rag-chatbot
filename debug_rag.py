"""
Debug Script - Check What's Going Wrong
Shows actual answers and debug info
"""

import requests
import json

API_URL = "http://localhost:8000"

# Test a simple question with debug info
test_questions = [
    "What is a stall?",
    "What does VFR stand for?",
    "What is the stall speed?"
]

print("="*70)
print("DEBUG: Checking RAG System Responses")
print("="*70)

for question in test_questions:
    print(f"\n{'='*70}")
    print(f"Question: {question}")
    print('='*70)
    
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": question,
            "top_k": 5,
            "min_similarity": 0.1,  # Lower threshold
            "include_debug": True
        }
    )
    
    result = response.json()
    
    print(f"\n📝 ANSWER:")
    print(result['answer'])
    
    print(f"\n📊 CONFIDENCE: {result['confidence']}")
    
    print(f"\n📚 CITATIONS: {len(result.get('citations', []))}")
    for cit in result.get('citations', []):
        print(f"  - {cit['document']}, page {cit['page']}, score: {cit.get('relevance_score', 0):.3f}")
    
    if result.get('debug_info'):
        debug = result['debug_info']
        print(f"\n🔍 DEBUG INFO:")
        print(f"  Retrieved chunks: {debug.get('retrieved_chunks', 0)}")
        print(f"  Filtered chunks: {debug.get('filtered_chunks', 0)}")
        
        if debug.get('similarity_scores'):
            scores = debug['similarity_scores']
            print(f"  Similarity scores: {', '.join([f'{s:.3f}' for s in scores[:5]])}")
        
        if debug.get('chunks'):
            print(f"\n  First retrieved chunk:")
            first_chunk = debug['chunks'][0]
            print(f"    Score: {first_chunk.get('score', 0):.3f}")
            print(f"    Document: {first_chunk.get('document')}")
            print(f"    Page: {first_chunk.get('page')}")
            print(f"    Preview: {first_chunk.get('preview', '')[:150]}...")
        
        if debug.get('grounding_check'):
            gc = debug['grounding_check']
            print(f"\n  Grounding check:")
            print(f"    Is grounded: {gc.get('is_grounded')}")
            print(f"    Confidence: {gc.get('confidence')}")
            print(f"    Reason: {gc.get('reason')}")
    
    print()
    input("Press Enter for next question...")

print("\n" + "="*70)
print("✅ Debug complete!")
print("="*70)
