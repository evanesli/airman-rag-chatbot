"""
Quick Fix: Test with Lower Thresholds
This will help identify the right threshold for your documents
"""

import requests
import json

API_URL = "http://localhost:8000"

test_question = "What is a stall?"

print("="*70)
print("Testing Different Similarity Thresholds")
print("="*70)

thresholds = [0.3, 0.2, 0.5, 0.2, 0.7]

for threshold in thresholds:
    print(f"\n{'='*70}")
    print(f"Testing with min_similarity = {threshold}")
    print('='*70)
    
    response = requests.post(
        f"{API_URL}/ask",
        json={
            "question": test_question,
            "top_k": 5,
            "min_similarity": threshold,
            "include_debug": True
        }
    )
    
    result = response.json()
    
    print(f"Confidence: {result['confidence']:.2f}")
    print(f"Answer: {result['answer'][:100]}...")
    print(f"Citations: {len(result.get('citations', []))}")
    
    if result.get('debug_info'):
        debug = result['debug_info']
        print(f"Retrieved: {debug.get('retrieved_chunks', 0)}, Filtered: {debug.get('filtered_chunks', 0)}")
        if debug.get('similarity_scores'):
            print(f"Top score: {max(debug['similarity_scores']):.3f}")

print("\n" + "="*70)
print("RECOMMENDATION:")
print("="*70)
print("Use the threshold that gives:")
print("- Filtered chunks > 0")
print("- Confidence > 0.5")
print("- Actual answer (not 'not available')")
print("\nThen update evaluate_rag.py line 95:")
print("    min_similarity: 0.X  # Use the value you found")
print("="*70)
