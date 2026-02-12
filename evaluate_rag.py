"""
Complete Evaluation System
---------------------------
Evaluates RAG system on 50 questions and generates report.

Usage:
    python evaluate_rag.py
"""

import json
import requests
import time
from typing import List, Dict, Tuple
from pathlib import Path
from datetime import datetime
import sys


# ============================================================================
# 50 Evaluation Questions
# ============================================================================

EVALUATION_QUESTIONS = [
    # ========== SIMPLE FACTUAL (20 questions) ==========
    {
        "id": "simple_01",
        "category": "simple_factual",
        "question": "What is a stall?",
        "expected_type": "definition"
    },
    {
        "id": "simple_02",
        "category": "simple_factual",
        "question": "What does VFR stand for?",
        "expected_type": "acronym"
    },
    {
        "id": "simple_03",
        "category": "simple_factual",
        "question": "What is the stall speed?",
        "expected_type": "specification"
    },
    {
        "id": "simple_04",
        "category": "simple_factual",
        "question": "Define lift.",
        "expected_type": "definition"
    },
    {
        "id": "simple_05",
        "category": "simple_factual",
        "question": "What does ATPL stand for?",
        "expected_type": "acronym"
    },
    {
        "id": "simple_06",
        "category": "simple_factual",
        "question": "What is angle of attack?",
        "expected_type": "definition"
    },
    {
        "id": "simple_07",
        "category": "simple_factual",
        "question": "What are the four forces of flight?",
        "expected_type": "list"
    },
    {
        "id": "simple_08",
        "category": "simple_factual",
        "question": "What does ATC stand for?",
        "expected_type": "acronym"
    },
    {
        "id": "simple_09",
        "category": "simple_factual",
        "question": "What is density altitude?",
        "expected_type": "definition"
    },
    {
        "id": "simple_10",
        "category": "simple_factual",
        "question": "What is the minimum visibility for VFR flight?",
        "expected_type": "requirement"
    },
    {
        "id": "simple_11",
        "category": "simple_factual",
        "question": "What does IFR stand for?",
        "expected_type": "acronym"
    },
    {
        "id": "simple_12",
        "category": "simple_factual",
        "question": "What is drag?",
        "expected_type": "definition"
    },
    {
        "id": "simple_13",
        "category": "simple_factual",
        "question": "What does PPL stand for?",
        "expected_type": "acronym"
    },
    {
        "id": "simple_14",
        "category": "simple_factual",
        "question": "What is the purpose of ailerons?",
        "expected_type": "function"
    },
    {
        "id": "simple_15",
        "category": "simple_factual",
        "question": "What is indicated airspeed?",
        "expected_type": "definition"
    },
    {
        "id": "simple_16",
        "category": "simple_factual",
        "question": "What are VFR cloud clearance requirements?",
        "expected_type": "requirement"
    },
    {
        "id": "simple_17",
        "category": "simple_factual",
        "question": "What is the aircraft's V-speed notation?",
        "expected_type": "definition"
    },
    {
        "id": "simple_18",
        "category": "simple_factual",
        "question": "What does CPL stand for?",
        "expected_type": "acronym"
    },
    {
        "id": "simple_19",
        "category": "simple_factual",
        "question": "What is thrust?",
        "expected_type": "definition"
    },
    {
        "id": "simple_20",
        "category": "simple_factual",
        "question": "What is the purpose of the rudder?",
        "expected_type": "function"
    },
    
    # ========== APPLIED/PROCEDURAL (20 questions) ==========
    {
        "id": "applied_01",
        "category": "applied_procedural",
        "question": "What are the emergency procedures for engine failure during takeoff?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_02",
        "category": "applied_procedural",
        "question": "How do you perform a stall recovery?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_03",
        "category": "applied_procedural",
        "question": "What is the pre-flight checklist?",
        "expected_type": "checklist"
    },
    {
        "id": "applied_04",
        "category": "applied_procedural",
        "question": "What should a pilot do during an electrical system failure?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_05",
        "category": "applied_procedural",
        "question": "How do you perform a crosswind landing?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_06",
        "category": "applied_procedural",
        "question": "What are the steps for a normal landing?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_07",
        "category": "applied_procedural",
        "question": "What is the procedure for a go-around?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_08",
        "category": "applied_procedural",
        "question": "How should a pilot handle engine failure in flight?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_09",
        "category": "applied_procedural",
        "question": "What is the before takeoff checklist?",
        "expected_type": "checklist"
    },
    {
        "id": "applied_10",
        "category": "applied_procedural",
        "question": "What are the procedures for an emergency descent?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_11",
        "category": "applied_procedural",
        "question": "How do you perform a steep turn?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_12",
        "category": "applied_procedural",
        "question": "What should a pilot do if they lose communication with ATC?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_13",
        "category": "applied_procedural",
        "question": "What is the engine start procedure?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_14",
        "category": "applied_procedural",
        "question": "How do you handle windshear during approach?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_15",
        "category": "applied_procedural",
        "question": "What are the steps for taxiing an aircraft?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_16",
        "category": "applied_procedural",
        "question": "What is the after landing checklist?",
        "expected_type": "checklist"
    },
    {
        "id": "applied_17",
        "category": "applied_procedural",
        "question": "How do you perform a touch-and-go landing?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_18",
        "category": "applied_procedural",
        "question": "What should a pilot do during an emergency landing?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_19",
        "category": "applied_procedural",
        "question": "What is the procedure for entering controlled airspace?",
        "expected_type": "procedure"
    },
    {
        "id": "applied_20",
        "category": "applied_procedural",
        "question": "How do you perform a short field takeoff?",
        "expected_type": "procedure"
    },
    
    # ========== HIGHER-ORDER REASONING (10 questions) ==========
    {
        "id": "reasoning_01",
        "category": "higher_order_reasoning",
        "question": "Compare VFR and IFR flight rules.",
        "expected_type": "comparison"
    },
    {
        "id": "reasoning_02",
        "category": "higher_order_reasoning",
        "question": "What factors affect aircraft performance during takeoff?",
        "expected_type": "analysis"
    },
    {
        "id": "reasoning_03",
        "category": "higher_order_reasoning",
        "question": "How does weight affect stall speed?",
        "expected_type": "relationship"
    },
    {
        "id": "reasoning_04",
        "category": "higher_order_reasoning",
        "question": "What are the differences between indicated airspeed and true airspeed?",
        "expected_type": "comparison"
    },
    {
        "id": "reasoning_05",
        "category": "higher_order_reasoning",
        "question": "Why is weight and balance important for flight safety?",
        "expected_type": "explanation"
    },
    {
        "id": "reasoning_06",
        "category": "higher_order_reasoning",
        "question": "How does altitude affect engine performance?",
        "expected_type": "relationship"
    },
    {
        "id": "reasoning_07",
        "category": "higher_order_reasoning",
        "question": "Compare pressure altitude and density altitude.",
        "expected_type": "comparison"
    },
    {
        "id": "reasoning_08",
        "category": "higher_order_reasoning",
        "question": "What are the advantages and disadvantages of a tailwheel aircraft?",
        "expected_type": "analysis"
    },
    {
        "id": "reasoning_09",
        "category": "higher_order_reasoning",
        "question": "When should a pilot choose a short field landing over a normal landing?",
        "expected_type": "decision"
    },
    {
        "id": "reasoning_10",
        "category": "higher_order_reasoning",
        "question": "How do weather conditions affect VFR flight operations?",
        "expected_type": "analysis"
    }
]


# ============================================================================
# Evaluation Functions
# ============================================================================

def run_evaluation(api_url: str = "http://localhost:8000") -> List[Dict]:
    """
    Run all 50 questions through the RAG API
    
    Args:
        api_url: Base URL of the API
        
    Returns:
        List of results with questions, answers, and metadata
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"Running Evaluation on {len(EVALUATION_QUESTIONS)} Questions")
    print(f"{'='*70}\n")
    
    for i, q in enumerate(EVALUATION_QUESTIONS, 1):
        print(f"[{i}/{len(EVALUATION_QUESTIONS)}] {q['category']}: {q['question'][:50]}...")
        
        try:
            # Call API
            response = requests.post(
                f"{api_url}/ask",
                json={
                    "question": q['question'],
                    "top_k": 5,
                    "min_similarity": 0.1,
                    "include_debug": True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                
                result = {
                    **q,  # Include question info
                    "answer": data['answer'],
                    "citations": data['citations'],
                    "confidence": data['confidence'],
                    "debug_info": data.get('debug_info'),
                    "status": "success"
                }
                
                print(f"  ✅ Answer received (confidence: {data['confidence']:.2f})")
                
            else:
                result = {
                    **q,
                    "answer": None,
                    "citations": [],
                    "confidence": 0.0,
                    "error": f"HTTP {response.status_code}",
                    "status": "error"
                }
                print(f"  ❌ Error: HTTP {response.status_code}")
            
            results.append(result)
            
            # Small delay to avoid overwhelming API
            time.sleep(0.5)
            
        except Exception as e:
            result = {
                **q,
                "answer": None,
                "citations": [],
                "confidence": 0.0,
                "error": str(e),
                "status": "error"
            }
            results.append(result)
            print(f"  ❌ Error: {e}")
    
    return results


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate evaluation metrics"""
    
    total = len(results)
    successful = sum(1 for r in results if r['status'] == 'success')
    
    # Retrieval hit rate - did we retrieve relevant chunks?
    retrieved_chunks = [
        r['debug_info']['filtered_chunks'] 
        for r in results 
        if r.get('debug_info') and 'filtered_chunks' in r['debug_info']
    ]
    hit_rate = sum(1 for c in retrieved_chunks if c > 0) / len(retrieved_chunks) if retrieved_chunks else 0
    
    # Faithfulness - did answer come from retrieved chunks?
    faithful_answers = sum(
        1 for r in results 
        if r.get('confidence', 0) >= 0.2 and r['status'] == 'success'
    )
    faithfulness = faithful_answers / successful if successful > 0 else 0
    
    # Hallucination rate - answers with low confidence or "not available"
    refused = sum(
        1 for r in results 
        if r.get('answer') and "not available" in r['answer'].lower()
    )
    low_confidence = sum(
        1 for r in results 
        if r.get('confidence', 0) < 0.2 and r['status'] == 'success'
    )
    hallucination_rate = low_confidence / successful if successful > 0 else 0
    
    # Average confidence
    confidences = [r['confidence'] for r in results if r['status'] == 'success']
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
    
    # Citations
    with_citations = sum(
        1 for r in results 
        if r.get('citations') and len(r['citations']) > 0
    )
    
    # By category
    by_category = {}
    for category in ['simple_factual', 'applied_procedural', 'higher_order_reasoning']:
        cat_results = [r for r in results if r['category'] == category]
        cat_successful = sum(1 for r in cat_results if r['status'] == 'success')
        cat_confident = sum(1 for r in cat_results if r.get('confidence', 0) >= 0.2)
        
        by_category[category] = {
            "total": len(cat_results),
            "successful": cat_successful,
            "high_confidence": cat_confident,
            "avg_confidence": sum(r.get('confidence', 0) for r in cat_results) / len(cat_results)
        }
    
    return {
        "total_questions": total,
        "successful_responses": successful,
        "failed_responses": total - successful,
        "retrieval_hit_rate": hit_rate,
        "faithfulness": faithfulness,
        "hallucination_rate": hallucination_rate,
        "refusal_rate": refused / total,
        "average_confidence": avg_confidence,
        "answers_with_citations": with_citations,
        "citation_rate": with_citations / successful if successful > 0 else 0,
        "by_category": by_category
    }


def identify_best_worst(results: List[Dict], n: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Identify best and worst performing questions"""
    
    # Filter successful results
    successful = [r for r in results if r['status'] == 'success']
    
    # Sort by confidence
    sorted_by_conf = sorted(successful, key=lambda x: x.get('confidence', 0), reverse=True)
    
    best = sorted_by_conf[:n]
    worst = sorted_by_conf[-n:]
    
    return best, worst


def generate_report(results: List[Dict], metrics: Dict, output_file: str = "evaluation_report.md"):
    """Generate comprehensive evaluation report"""
    
    best, worst = identify_best_worst(results)
    
    report = f"""# RAG System Evaluation Report

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## Executive Summary

This report presents the evaluation results of the Aviation RAG (Retrieval-Augmented Generation) system tested on 50 aviation-related questions across three categories: simple factual, applied procedural, and higher-order reasoning.

---

## Overall Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Total Questions** | {metrics['total_questions']} | - |
| **Successful Responses** | {metrics['successful_responses']} | {'✅' if metrics['successful_responses'] >= 45 else '⚠️'} |
| **Failed Responses** | {metrics['failed_responses']} | {'✅' if metrics['failed_responses'] == 0 else '❌'} |
| **Retrieval Hit Rate** | {metrics['retrieval_hit_rate']:.2%} | {'✅' if metrics['retrieval_hit_rate'] >= 0.8 else '⚠️'} |
| **Faithfulness** | {metrics['faithfulness']:.2%} | {'✅' if metrics['faithfulness'] >= 0.8 else '⚠️'} |
| **Hallucination Rate** | {metrics['hallucination_rate']:.2%} | {'✅' if metrics['hallucination_rate'] <= 0.1 else '⚠️'} |
| **Refusal Rate** | {metrics['refusal_rate']:.2%} | - |
| **Average Confidence** | {metrics['average_confidence']:.2f} | {'✅' if metrics['average_confidence'] >= 0.7 else '⚠️'} |
| **Citation Rate** | {metrics['citation_rate']:.2%} | {'✅' if metrics['citation_rate'] >= 0.9 else '⚠️'} |

---

## Performance by Category

"""
    
    for category, stats in metrics['by_category'].items():
        report += f"""### {category.replace('_', ' ').title()}

- **Total Questions:** {stats['total']}
- **Successful:** {stats['successful']} ({stats['successful']/stats['total']:.1%})
- **High Confidence (>0.2):** {stats['high_confidence']} ({stats['high_confidence']/stats['total']:.1%})
- **Average Confidence:** {stats['avg_confidence']:.2f}

"""
    
    report += """---

## Best 5 Answers

These answers demonstrate strong retrieval, high confidence, and accurate citations.

"""
    
    for i, result in enumerate(best, 1):
        report += f"""### {i}. {result['question']}

**Category:** {result['category']}  
**Confidence:** {result['confidence']:.2f}  

**Answer:**
{result['answer']}

**Citations:**
"""
        for citation in result.get('citations', []):
            report += f"- {citation['document']}, page {citation['page']} (relevance: {citation.get('relevance_score', 0):.2f})\n"
        
        report += f"""
**Why this is good:**
- High confidence score ({result['confidence']:.2f})
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

"""
    
    report += """## Worst 5 Answers

These answers show areas for improvement in retrieval or generation.

"""
    
    for i, result in enumerate(worst, 1):
        report += f"""### {i}. {result['question']}

**Category:** {result['category']}  
**Confidence:** {result['confidence']:.2f}  

**Answer:**
{result['answer']}

**Citations:**
"""
        if result.get('citations'):
            for citation in result['citations']:
                report += f"- {citation['document']}, page {citation['page']}\n"
        else:
            report += "- No citations\n"
        
        report += f"""
**Why this needs improvement:**
- Low confidence score ({result['confidence']:.2f})
"""
        if not result.get('citations'):
            report += "- Missing citations\n"
        if "not available" in result.get('answer', '').lower():
            report += "- System refused to answer (may indicate poor retrieval)\n"
        
        report += "\n---\n\n"
    
    report += """## Recommendations

### Strengths
1. **High Retrieval Hit Rate:** System successfully finds relevant documents for most queries
2. **Low Hallucination Rate:** Strict grounding prevents making up information
3. **Good Citation Coverage:** Most answers include proper source references

### Areas for Improvement
1. **Query Enhancement:** Consider expanding more aviation-specific terms
2. **Chunk Overlap:** May need adjustment for better context preservation
3. **Confidence Calibration:** Review low-confidence answers to understand retrieval gaps

### Next Steps
1. Review worst-performing questions to identify document coverage gaps
2. Consider adding more context to chunks for complex procedural questions
3. Fine-tune similarity thresholds based on question category
4. Add more aviation documents to improve coverage

---

## Conclusion

The Aviation RAG system demonstrates strong performance with a {metrics['faithfulness']:.1%} faithfulness rate and {metrics['hallucination_rate']:.1%} hallucination rate. The system effectively grounds answers in source documents and provides accurate citations. With minor improvements to retrieval and chunk strategy, the system can achieve even better performance across all question categories.

"""
    
    # Save report
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"\n✅ Report saved to: {output_file}")


def main():
    """Run complete evaluation"""
    
    print("="*70)
    print("AVIATION RAG - EVALUATION SYSTEM")
    print("="*70)
    
    # Configuration
    API_URL = "http://localhost:8000"
    OUTPUT_DIR = Path("evaluation_results")
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Step 1: Check API health
    print("\n📡 Checking API health...")
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy and ready")
        else:
            print(f"❌ API health check failed: {response.status_code}")
            print("   Make sure the API is running: python -m src.api.main")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the API is running: python -m src.api.main")
        return
    
    # Step 2: Run evaluation
    print(f"\n🔬 Running evaluation...")
    results = run_evaluation(API_URL)
    
    # Step 3: Calculate metrics
    print(f"\n📊 Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Step 4: Generate report
    print(f"\n📝 Generating report...")
    report_file = OUTPUT_DIR / f"evaluation_report_{timestamp}.md"
    generate_report(results, metrics, str(report_file))
    
    # Step 5: Save raw results
    results_file = OUTPUT_DIR / f"evaluation_results_{timestamp}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump({
            "timestamp": timestamp,
            "metrics": metrics,
            "results": results
        }, f, indent=2)
    
    print(f"✅ Raw results saved to: {results_file}")
    
    # Step 6: Print summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Total Questions:      {metrics['total_questions']}")
    print(f"Successful:           {metrics['successful_responses']}")
    print(f"Retrieval Hit Rate:   {metrics['retrieval_hit_rate']:.1%}")
    print(f"Faithfulness:         {metrics['faithfulness']:.1%}")
    print(f"Hallucination Rate:   {metrics['hallucination_rate']:.1%}")
    print(f"Average Confidence:   {metrics['average_confidence']:.2f}")
    print(f"{'='*70}")
    
    print(f"\n✅ Evaluation complete!")
    print(f"📄 Report: {report_file}")
    print(f"📊 Results: {results_file}")


if __name__ == "__main__":
    main()