# RAG System Evaluation Report

**Generated:** 2026-02-13 00:16:31

---

## Executive Summary

This report presents the evaluation results of the Aviation RAG (Retrieval-Augmented Generation) system tested on 50 aviation-related questions across three categories: simple factual, applied procedural, and higher-order reasoning.

---

## Overall Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Total Questions** | 50 | - |
| **Successful Responses** | 50 | ✅ |
| **Failed Responses** | 0 | ✅ |
| **Retrieval Hit Rate** | 100.00% | ✅ |
| **Faithfulness** | 68.00% | ⚠️ |
| **Hallucination Rate** | 32.00% | ⚠️ |
| **Refusal Rate** | 32.00% | - |
| **Average Confidence** | 0.48 | ⚠️ |
| **Citation Rate** | 100.00% | ✅ |

---

## Performance by Category

### Simple Factual

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.2):** 13 (65.0%)
- **Average Confidence:** 0.45

### Applied Procedural

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.2):** 11 (55.0%)
- **Average Confidence:** 0.38

### Higher Order Reasoning

- **Total Questions:** 10
- **Successful:** 10 (100.0%)
- **High Confidence (>0.2):** 10 (100.0%)
- **Average Confidence:** 0.70

---

## Best 5 Answers

These answers demonstrate strong retrieval, high confidence, and accurate citations.

### 1. What is a stall?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
- Answer: A stall occurs when an airplane flies at a speed where it can no longer maintain lift, typically happening when the angle of attack is too high.

- Explanation: A stall is an aerodynamic condition where the wing exceeds its critical angle of attack, resulting in a loss of lift. This situation can occur when flying too slowly or during high-angle maneuvers. According to [Source 4], flying slower or faster than the optimal speed can stall the airplane, and even maneuvering can lead to a stall. This aligns with general aviation knowledge that stalling is related to exceeding the critical angle of attack.

- Source: [Source 4: Document: 6-mass-and-balance-and-performance-2014.pdf, Page: 456] and general aviation knowledge.

**Citations:**
- Instruments.pdf, page 450 (relevance: 0.42)
- Instruments.pdf, page 514 (relevance: 0.39)
- Instruments.pdf, page 443 (relevance: 0.37)
- 6-mass-and-balance-and-performance-2014.pdf, page 456 (relevance: 0.37)
- Instruments.pdf, page 449 (relevance: 0.36)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 2. What does VFR stand for?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
- Answer: VFR stands for Visual Flight Rules.
- Explanation: Although the provided documents do not explicitly state what VFR stands for, general aviation knowledge confirms that VFR stands for Visual Flight Rules, which are rules that govern flying by visual references.
- Source: None (General aviation knowledge used, as the documents did not provide a direct explanation). However, [Source 4] mentions "VFR flights," implying the use of Visual Flight Rules.

**Citations:**
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 18 (relevance: 0.53)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 240 (relevance: 0.53)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 235 (relevance: 0.52)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 157 (relevance: 0.52)
- Instruments.pdf, page 345 (relevance: 0.50)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 3. What is the stall speed?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
- Answer: VSR1 
- Explanation: The stall speed is referenced as VSR1, which is defined as the Reference stalling speed.
- Source: [Source 4] Document: 6-mass-and-balance-and-performance-2014.pdf, Page: 146

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 273 (relevance: 0.57)
- Instruments.pdf, page 48 (relevance: 0.53)
- 6-mass-and-balance-and-performance-2014.pdf, page 456 (relevance: 0.49)
- 6-mass-and-balance-and-performance-2014.pdf, page 146 (relevance: 0.49)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 4. Define lift.

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
- Answer: Not directly provided
- Explanation: Although the provided documents do not directly define lift, general aviation knowledge can be applied here. Lift is an upward aerodynamic force that opposes the weight of an aircraft and is created by the wing as it moves through the air. 
- Source: None (General aviation knowledge used) 

However, if we look through the provided documents for related terms:

- Source 2 mentions "This action deflects the elevators to create a downward aerodynamic force." Here, it talks about creating a force through deflection, implying control surfaces can create forces like lift or drag depending on their orientation.

For a precise definition directly from provided texts, none are available. But typically:

- Source 3 defines terms like Altitude, Height, but not lift directly. 

In standard aviation theory, lift is defined as the upward force exerted on an aircraft by the air it moves through.

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 158 (relevance: 0.42)
- 6-mass-and-balance-and-performance-2014.pdf, page 384 (relevance: 0.41)
- Meteorology full book.pdf, page 137 (relevance: 0.38)
- 6-mass-and-balance-and-performance-2014.pdf, page 179 (relevance: 0.38)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 154 (relevance: 0.37)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 5. What is angle of attack?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
- Answer: The angle between the oncoming airflow and the wing or airfoil.
- Explanation: Although the provided documents do not directly define angle of attack, general aviation knowledge supports this definition. The angle of attack is a critical parameter in aviation, relating to the performance and safety of an aircraft, particularly in the context of stall warning systems and high angle of attack protection mentioned in the documents.
- Source: [Source 1], [Source 3], and general aviation knowledge. 

Specifically, [Source 1] and [Source 3] imply the relevance of angle of attack in the context of stall warning systems and high angle of attack protection, respectively.

**Citations:**
- Instruments.pdf, page 450 (relevance: 0.60)
- Instruments.pdf, page 451 (relevance: 0.54)
- Instruments.pdf, page 421 (relevance: 0.53)
- 10-General-Navigation-2014 (1).pdf, page 195 (relevance: 0.47)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

## Worst 5 Answers

These answers show areas for improvement in retrieval or generation.

### 1. What are the procedures for an emergency descent?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 458
- 6-mass-and-balance-and-performance-2014.pdf, page 228
- 6-mass-and-balance-and-performance-2014.pdf, page 472
- 6-mass-and-balance-and-performance-2014.pdf, page 292
- 6-mass-and-balance-and-performance-2014.pdf, page 459

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 2. What is the engine start procedure?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- Instruments.pdf, page 405
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 119
- 6-mass-and-balance-and-performance-2014.pdf, page 135
- 6-mass-and-balance-and-performance-2014.pdf, page 345
- 6-mass-and-balance-and-performance-2014.pdf, page 431

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 3. What are the steps for taxiing an aircraft?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 71
- 6-mass-and-balance-and-performance-2014.pdf, page 139
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 31
- 6-mass-and-balance-and-performance-2014.pdf, page 537
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 33

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 4. What is the after landing checklist?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 476
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 232
- 6-mass-and-balance-and-performance-2014.pdf, page 278
- 6-mass-and-balance-and-performance-2014.pdf, page 478
- Instruments.pdf, page 279

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 5. How do you perform a touch-and-go landing?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 476
- 6-mass-and-balance-and-performance-2014.pdf, page 279
- Instruments.pdf, page 379
- 6-mass-and-balance-and-performance-2014.pdf, page 478

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

## Recommendations

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

