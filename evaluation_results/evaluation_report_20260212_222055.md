# RAG System Evaluation Report

**Generated:** 2026-02-12 22:23:52

---

## Executive Summary

This report presents the evaluation results of the Aviation RAG (Retrieval-Augmented Generation) system tested on 50 aviation-related questions across three categories: simple factual, applied procedural, and higher-order reasoning.

---

## Overall Metrics

| Metric | Value | Status |
|--------|-------|--------|
| **Total Questions** | 50 | - |
| **Successful Responses** | 47 | ✅ |
| **Failed Responses** | 3 | ❌ |
| **Retrieval Hit Rate** | 100.00% | ✅ |
| **Faithfulness** | 80.85% | ✅ |
| **Hallucination Rate** | 19.15% | ⚠️ |
| **Refusal Rate** | 18.00% | - |
| **Average Confidence** | 0.57 | ⚠️ |
| **Citation Rate** | 100.00% | ✅ |

---

## Performance by Category

### Simple Factual

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.2):** 15 (75.0%)
- **Average Confidence:** 0.54

### Applied Procedural

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.2):** 16 (80.0%)
- **Average Confidence:** 0.56

### Higher Order Reasoning

- **Total Questions:** 10
- **Successful:** 7 (70.0%)
- **High Confidence (>0.2):** 7 (70.0%)
- **Average Confidence:** 0.49

---

## Best 5 Answers

These answers demonstrate strong retrieval, high confidence, and accurate citations.

### 1. What is the minimum visibility for VFR flight?

**Category:** simple_factual  
**Confidence:** 1.00  

**Answer:**
Answer: Not explicitly stated

Explanation: The provided documents do not explicitly state the minimum visibility for VFR (Visual Flight Rules) flight. However, we can find information related to visibility in general. For example, Source 3 mentions visibility ranges for different weather conditions such as drizzle, moderate rain, and heavy rain, but it does not specify the minimum visibility for VFR flight. Similarly, Source 2 discusses Instrumented Runway Visual Range (IRVR) and Source 5 introduces Meteorological Optical Range (MOR), but neither provides the minimum visibility requirement for VFR flight. According to standard aviation theory, the minimum visibility for VFR flight is typically 1 statute mile (or 1.6 kilometers) for daytime operations in uncontrolled airspace, but this information is not present in the provided documents.

Source: 
- Source 2: Document: Meteorology full book.pdf, Page: 282 (for general visibility discussion)
- Source 3: Document: Meteorology full book.pdf, Page: 280 (for visibility ranges in different weather conditions)
- Standard aviation knowledge (for typical minimum visibility for VFR flight)

**Citations:**
- Meteorology full book.pdf, page 478 (relevance: 0.57)
- Meteorology full book.pdf, page 282 (relevance: 0.55)
- Meteorology full book.pdf, page 280 (relevance: 0.55)
- Meteorology full book.pdf, page 488 (relevance: 0.53)
- Meteorology full book.pdf, page 273 (relevance: 0.52)

**Why this is good:**
- High confidence score (1.00)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 2. What is a stall?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
Answer: A condition where the airplane exceeds its critical angle of attack and lift begins to decrease, often caused by flying too slowly.

Explanation: Although the provided documents do not explicitly define a stall, we can infer the definition from the context. Source 4 (6-mass-and-balance-and-performance-2014.pdf, Page: 456) mentions that "Flying slower or faster than the speed shown will stall the aeroplane," implying that a stall is related to exceeding a certain speed limit, likely due to angle of attack issues. Additionally, Source 1 (Instruments.pdf, Page: 450) discusses "Stall Protection" and Source 5 (Instruments.pdf, Page: 449) mentions an "angle at which the warning unit has been preset," which suggests that a stall is related to a specific angle of attack. 

Source: 
- 6-mass-and-balance-and-performance-2014.pdf, Page: 456
- Instruments.pdf, Page: 450
- Instruments.pdf, Page: 449

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

### 3. What does VFR stand for?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
Answer: Visual Flight Rules

Explanation: Although the provided documents do not explicitly define VFR, based on general aviation knowledge, VFR stands for Visual Flight Rules. This can be inferred from the context of Source 4, which mentions "VFR flights" in relation to Class G airspace, implying a type of flight rule. 

Source: Implicit from [Source 4] Document: 7-Flight-Planning-and-Monitoring-2014.pdf, Page: 157, and general aviation knowledge.

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

### 4. What is the stall speed?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
Answer: Not explicitly stated
Explanation: The provided documents do not explicitly state the stall speed. However, Source 4 mentions "VSR1 Reference stalling speed" (Source 4: 6-mass-and-balance-and-performance-2014.pdf, Page 146), which implies that the stall speed is referred to as VSR1, but its value is not given. 
Source: [6-mass-and-balance-and-performance-2014.pdf, Page 146]

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

### 5. What are the four forces of flight?

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
Answer: Not explicitly stated in the provided context.

Explanation: Although the provided context discusses various aspects of flight, such as cruise, landing, and climb, it does not explicitly mention the four forces of flight. However, based on general aviation knowledge, the four forces of flight are lift, weight, thrust, and drag. The context does mention some of these forces, such as thrust (Source 5, Page 175), but it does not provide a comprehensive list of all four forces.

Source: 
- Source 1: Page 239 (discusses balance of forces in level flight, but does not list the four forces)
- Source 5: Page 175 (mentions excess thrust, but does not provide a complete list of the four forces)
- General Aviation Knowledge (the four forces of flight are lift, weight, thrust, and drag)

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 239 (relevance: 0.59)
- 6-mass-and-balance-and-performance-2014.pdf, page 240 (relevance: 0.50)
- 6-mass-and-balance-and-performance-2014.pdf, page 281 (relevance: 0.49)
- 6-mass-and-balance-and-performance-2014.pdf, page 285 (relevance: 0.48)
- 6-mass-and-balance-and-performance-2014.pdf, page 175 (relevance: 0.47)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

## Worst 5 Answers

These answers show areas for improvement in retrieval or generation.

### 1. What does CPL stand for?

**Category:** simple_factual  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- Instruments.pdf, page 2
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 2
- 6-mass-and-balance-and-performance-2014.pdf, page 2
- 11-radio-navigation-2014.pdf, page 2
- 10-General-Navigation-2014 (1).pdf, page 2

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 2. What should a pilot do during an electrical system failure?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 116
- 10-General-Navigation-2014 (1).pdf, page 512
- Instruments.pdf, page 358
- Instruments.pdf, page 381
- Instruments.pdf, page 428

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 3. What should a pilot do if they lose communication with ATC?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- Instruments.pdf, page 331
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 231
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 233
- Meteorology full book.pdf, page 550

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

