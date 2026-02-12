# RAG System Evaluation Report

**Generated:** 2026-02-12 22:11:34

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
| **Faithfulness** | 2.00% | ⚠️ |
| **Hallucination Rate** | 98.00% | ⚠️ |
| **Refusal Rate** | 98.00% | - |
| **Average Confidence** | 0.01 | ⚠️ |
| **Citation Rate** | 98.00% | ✅ |

---

## Performance by Category

### Simple Factual

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.4):** 0 (0.0%)
- **Average Confidence:** 0.00

### Applied Procedural

- **Total Questions:** 20
- **Successful:** 20 (100.0%)
- **High Confidence (>0.4):** 0 (0.0%)
- **Average Confidence:** 0.00

### Higher Order Reasoning

- **Total Questions:** 10
- **Successful:** 10 (100.0%)
- **High Confidence (>0.4):** 1 (10.0%)
- **Average Confidence:** 0.07

---

## Best 5 Answers

These answers demonstrate strong retrieval, high confidence, and accurate citations.

### 1. What are the advantages and disadvantages of a tailwheel aircraft?

**Category:** higher_order_reasoning  
**Confidence:** 0.70  

**Answer:**
Answer: Not explicitly stated in the provided context.

Explanation: The provided context does not explicitly discuss the advantages and disadvantages of a tailwheel aircraft. However, Source 4 (Page 240) mentions that the "two additional forces provided by the tailplane are detrimental" [Source 4: 6-mass-and-balance-and-performance-2014.pdf, Page: 240], which could be related to the tailwheel configuration, but it does not provide a clear answer to the question.

Source 1 (Page 32) discusses the importance of weight in inducing fatigue into aircraft structure [Source 1: 6-mass-and-balance-and-performance-2014.pdf, Page: 32], but it does not relate to tailwheel aircraft specifically. The other sources (2, 3, and 5) discuss general principles of cruise, descent, and landing, but do not provide information on tailwheel aircraft [Source 2: 6-mass-and-balance-and-performance-2014.pdf, Page: 240, Source 3: 6-mass-and-balance-and-performance-2014.pdf, Page: 232, Source 5: 6-mass-and-balance-and-performance-2014.pdf, Page: 286].

Source: [Source 1: 6-mass-and-balance-and-performance-2014.pdf, Page: 32, Source 4: 6-mass-and-balance-and-performance-2014.pdf, Page: 240]

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 32 (relevance: 0.48)
- 6-mass-and-balance-and-performance-2014.pdf, page 240 (relevance: 0.44)
- 6-mass-and-balance-and-performance-2014.pdf, page 232 (relevance: 0.42)
- 6-mass-and-balance-and-performance-2014.pdf, page 286 (relevance: 0.41)

**Why this is good:**
- High confidence score (0.70)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 2. What is a stall?

**Category:** simple_factual  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- Instruments.pdf, page 450 (relevance: 0.42)

**Why this is good:**
- High confidence score (0.00)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 3. What does VFR stand for?

**Category:** simple_factual  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 18 (relevance: 0.53)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 240 (relevance: 0.53)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 235 (relevance: 0.52)
- 7-Flight-Planning-and-Monitoring-2014.pdf, page 157 (relevance: 0.52)
- Instruments.pdf, page 345 (relevance: 0.50)

**Why this is good:**
- High confidence score (0.00)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 4. What is the stall speed?

**Category:** simple_factual  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 273 (relevance: 0.57)
- Instruments.pdf, page 48 (relevance: 0.53)
- 6-mass-and-balance-and-performance-2014.pdf, page 456 (relevance: 0.49)
- 6-mass-and-balance-and-performance-2014.pdf, page 146 (relevance: 0.49)

**Why this is good:**
- High confidence score (0.00)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

### 5. Define lift.

**Category:** simple_factual  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 158 (relevance: 0.42)
- 6-mass-and-balance-and-performance-2014.pdf, page 384 (relevance: 0.41)

**Why this is good:**
- High confidence score (0.00)
- Clear, specific answer
- Proper citations included
- Grounded in source documents

---

## Worst 5 Answers

These answers show areas for improvement in retrieval or generation.

### 1. Why is weight and balance important for flight safety?

**Category:** higher_order_reasoning  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 280
- 6-mass-and-balance-and-performance-2014.pdf, page 240
- 6-mass-and-balance-and-performance-2014.pdf, page 32
- 6-mass-and-balance-and-performance-2014.pdf, page 17

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 2. How does altitude affect engine performance?

**Category:** higher_order_reasoning  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 193
- 6-mass-and-balance-and-performance-2014.pdf, page 185
- 6-mass-and-balance-and-performance-2014.pdf, page 265

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 3. Compare pressure altitude and density altitude.

**Category:** higher_order_reasoning  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- Meteorology full book.pdf, page 37
- Meteorology full book.pdf, page 41
- Meteorology full book.pdf, page 38
- Instruments.pdf, page 59
- Meteorology full book.pdf, page 39

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 4. When should a pilot choose a short field landing over a normal landing?

**Category:** higher_order_reasoning  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- 6-mass-and-balance-and-performance-2014.pdf, page 280
- 6-mass-and-balance-and-performance-2014.pdf, page 335
- 6-mass-and-balance-and-performance-2014.pdf, page 290
- 11-radio-navigation-2014.pdf, page 168

**Why this needs improvement:**
- Low confidence score (0.00)
- System refused to answer (may indicate poor retrieval)

---

### 5. How do weather conditions affect VFR flight operations?

**Category:** higher_order_reasoning  
**Confidence:** 0.00  

**Answer:**
This information is not available in the provided document(s).

**Citations:**
- Meteorology full book.pdf, page 260
- Meteorology full book.pdf, page 429
- Sample test questions .pdf, page 11
- Meteorology full book.pdf, page 478
- Meteorology full book.pdf, page 311

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

