# RAG System Evaluation Report

**Generated:** 2026-02-12 22:16:08

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
| **Faithfulness** | 80.00% | ✅ |
| **Hallucination Rate** | 20.00% | ⚠️ |
| **Refusal Rate** | 22.00% | - |
| **Average Confidence** | 0.57 | ⚠️ |
| **Citation Rate** | 100.00% | ✅ |

---

## Performance by Category

### Simple Factual

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.2):** 16 (80.0%)
- **Average Confidence:** 0.57

### Applied Procedural

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.2):** 14 (70.0%)
- **Average Confidence:** 0.49

### Higher Order Reasoning

- **Total Questions:** 10
- **Successful:** 10 (100.0%)
- **High Confidence (>0.2):** 10 (100.0%)
- **Average Confidence:** 0.70

---

## Best 5 Answers

These answers demonstrate strong retrieval, high confidence, and accurate citations.

### 1. What is the minimum visibility for VFR flight?

**Category:** simple_factual  
**Confidence:** 1.00  

**Answer:**
Answer: Not explicitly stated

Explanation: The provided documents do not explicitly state the minimum visibility for VFR (Visual Flight Rules) flight. However, they do discuss various aspects of visibility, such as instrumented runway visual range (IRVR) (Source 2: Meteorology full book.pdf, Page 282) and different types of precipitation and their associated visibility ranges (Source 3: Meteorology full book.pdf, Page 280). Unfortunately, none of the documents provide a clear answer to the question about the minimum visibility for VFR flight.

Source: 
- Source 2: Meteorology full book.pdf, Page 282
- Source 3: Meteorology full book.pdf, Page 280 
Note: Standard aviation rules typically state that for VFR flight, the minimum visibility is 5 km (or 3 SM) for day operations and 5 km (or 3 SM) for night operations in some countries, but this information is not present in the provided context.

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

Explanation: Although none of the provided sources directly define what a stall is, we can infer the definition from the context. Source 4 (6-mass-and-balance-and-performance-2014.pdf, Page: 456) mentions that "Flying slower or faster than the speed shown will stall the aeroplane," implying that a stall is related to exceeding a certain speed limit, likely due to angle of attack issues. Additionally, Source 1 (Instruments.pdf, Page: 450) discusses "Stall Protection" and Source 5 (Instruments.pdf, Page: 449) mentions an "angle at which the warning unit has been preset," which suggests that a stall is related to a specific angle of attack.

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

Source: Implicitly from [Source 4] Document: 7-Flight-Planning-and-Monitoring-2014.pdf, Page: 157, and general aviation knowledge.

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

### 5. Define lift.

**Category:** simple_factual  
**Confidence:** 0.70  

**Answer:**
Answer: Not explicitly defined in the provided context.

Explanation: The term "lift" is not explicitly defined in the provided documents. However, based on general aviation knowledge, lift can be defined as the upward force exerted on an aircraft wing or airfoil, opposing the weight of the aircraft and keeping it flying. Although the documents do not provide a direct definition, they discuss related concepts such as aerodynamic forces (Source 2) and altitude (Source 3), which are relevant to the concept of lift.

Source: 
- Source 1: 6-mass-and-balance-and-performance-2014.pdf (no mention of lift)
- Source 2: 6-mass-and-balance-and-performance-2014.pdf (discusses aerodynamic forces, but not lift specifically)
- Source 3: Meteorology full book.pdf (discusses altitude, but not lift)
- Source 4: 6-mass-and-balance-and-performance-2014.pdf (discusses climb, but not lift)
- Source 5: 7-Flight-Planning-and-Monitoring-2014.pdf (discusses flight levels, but not lift)

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

## Worst 5 Answers

These answers show areas for improvement in retrieval or generation.

### 1. How do you perform a crosswind landing?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 10-General-Navigation-2014 (1).pdf, page 181
- 10-General-Navigation-2014 (1).pdf, page 187
- 10-General-Navigation-2014 (1).pdf, page 184
- 10-General-Navigation-2014 (1).pdf, page 186
- 6-mass-and-balance-and-performance-2014.pdf, page 286

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 2. What should a pilot do if they lose communication with ATC?

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

### 3. What is the after landing checklist?

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

### 4. How do you perform a touch-and-go landing?

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

### 5. How do you perform a short field takeoff?

**Category:** applied_procedural  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 493
- 6-mass-and-balance-and-performance-2014.pdf, page 427
- 6-mass-and-balance-and-performance-2014.pdf, page 139
- 6-mass-and-balance-and-performance-2014.pdf, page 300
- 6-mass-and-balance-and-performance-2014.pdf, page 522

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

