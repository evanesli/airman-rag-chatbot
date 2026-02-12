"""
RAG Pipeline Module
-------------------
Complete Retrieval-Augmented Generation pipeline.

Key Features:
1. Query enhancement and preprocessing
2. Vector store retrieval
3. Relevance filtering
4. LLM-based answer generation
5. Citation formatting
6. Hallucination detection
7. Confidence scoring
"""

import re
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline"""
    top_k: int = 5  # Number of chunks to retrieve
    min_similarity: float = 0.2  # Minimum similarity threshold
    max_context_length: int = 3000  # Max chars for LLM context
    enable_query_enhancement: bool = True
    enable_hallucination_check: bool = True
    enforce_grounding: bool = True


class QueryEnhancer:
    """
    Enhance user queries for better retrieval.
    
    Features:
    - Aviation acronym expansion
    - Spelling correction
    - Query reformulation
    """
    
    # Common aviation acronyms
    ACRONYMS = {
        "ppl": "private pilot license",
        "cpl": "commercial pilot license",
        "atpl": "airline transport pilot license",
        "vfr": "visual flight rules",
        "ifr": "instrument flight rules",
        "atc": "air traffic control",
        "agl": "above ground level",
        "msl": "mean sea level",
        "tas": "true airspeed",
        "ias": "indicated airspeed",
        "kias": "knots indicated airspeed",
        "rpm": "revolutions per minute",
        "vor": "vhf omnidirectional range",
        "ils": "instrument landing system",
        "tcas": "traffic collision avoidance system",
    }
    
    def enhance_query(self, query: str) -> str:
        """
        Enhance query for better retrieval
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query
        """
        enhanced = query.lower()
        
        # Expand acronyms
        for acronym, expansion in self.ACRONYMS.items():
            # Match whole words only
            pattern = r'\b' + acronym + r'\b'
            enhanced = re.sub(pattern, f"{acronym} {expansion}", enhanced, flags=re.IGNORECASE)
        
        # Clean up extra spaces
        enhanced = re.sub(r'\s+', ' ', enhanced).strip()
        
        return enhanced


class HallucinationDetector:
    """
    Detect potential hallucinations in generated answers.
    
    Methods:
    - Check if answer contains information not in context
    - Verify citations are accurate
    """
    
    def check_answer(self, 
                     answer: str,
                     context_chunks: List[Dict],
                     strict: bool = True) -> Tuple[bool, float, str]:
        """
        Check if answer is grounded in context
        
        Args:
            answer: Generated answer
            context_chunks: Retrieved chunks
            strict: Whether to use strict checking
            
        Returns:
            Tuple of (is_grounded, confidence, reason)
        """
        # Check for explicit refusal
        refusal_phrases = [
            "not available in the provided document",
            "cannot find this information",
            "not mentioned in the documents",
            "information is not present"
        ]
        
        answer_lower = answer.lower()
        
        for phrase in refusal_phrases:
            if phrase in answer_lower:
                return True, 1.0, "Correctly refused to answer"
        
        # Extract text from chunks
        context_text = " ".join([
            chunk.get("text_preview", "") for chunk in context_chunks
        ]).lower()
        
        # Simple check: are key terms from answer in context?
        # Extract words from answer (excluding common words)
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by"}
        
        answer_words = set(re.findall(r'\b\w+\b', answer_lower))
        answer_words = answer_words - stop_words
        
        # Check how many answer words appear in context
        words_in_context = sum(1 for word in answer_words if word in context_text)
        
        if len(answer_words) == 0:
            return True, 0.5, "Answer too short to verify"
        
        coverage = words_in_context / len(answer_words)
        
        # Confidence scoring
        if coverage > 0.8:
            is_grounded = True
            confidence = 0.9
            reason = "Most answer content found in context"
        elif coverage > 0.2:
            is_grounded = True
            confidence = 0.7
            reason = "Moderate overlap with context"
        elif coverage > 0.2:
            is_grounded = not strict
            confidence = 0.5
            reason = "Limited overlap with context"
        else:
            is_grounded = False
            confidence = 0.3
            reason = "Answer may contain hallucinated information"
        
        return is_grounded, confidence, reason


class RAGPipeline:
    """
    Complete RAG pipeline combining retrieval and generation.
    
    Architecture:
    Query → Enhancement → Retrieval → Relevance Check → Generation → Verification
    """
    
    def __init__(self,
                 vector_store,
                 embedding_generator,
                 llm_client,
                 config: Optional[RAGConfig] = None):
        """
        Initialize RAG pipeline
        
        Args:
            vector_store: FAISSVectorStore instance
            embedding_generator: EmbeddingGenerator instance
            llm_client: LLMClient instance
            config: RAGConfig or None for defaults
        """
        self.vector_store = vector_store
        self.embedding_generator = embedding_generator
        self.llm_client = llm_client
        self.config = config or RAGConfig()
        
        self.query_enhancer = QueryEnhancer()
        self.hallucination_detector = HallucinationDetector()
        
        logger.info("Initialized RAG pipeline")
    
    def answer_question(self, 
                       query: str,
                       return_debug: bool = False) -> Dict:
        """
        Answer a question using RAG pipeline
        
        Args:
            query: User question
            return_debug: Include debug information in response
            
        Returns:
            Dict with answer, citations, and metadata
        """
        logger.info(f"Processing query: {query}")
        
        # Step 1: Enhance query
        if self.config.enable_query_enhancement:
            enhanced_query = self.query_enhancer.enhance_query(query)
            logger.info(f"Enhanced query: {enhanced_query}")
        else:
            enhanced_query = query
        
        # Step 2: Generate query embedding
        query_embedding = self.embedding_generator.embed_text(enhanced_query)
        
        # Step 3: Retrieve relevant chunks
        retrieved_chunks = self.vector_store.search(
            query_embedding,
            k=self.config.top_k
        )
        
        logger.info(f"Retrieved {len(retrieved_chunks)} chunks")
        
        # Step 4: Filter by similarity threshold
        filtered_chunks = [
            chunk for chunk in retrieved_chunks
            if chunk['score'] >= self.config.min_similarity
        ]
        
        logger.info(f"Filtered to {len(filtered_chunks)} chunks above threshold {self.config.min_similarity}")
        
        # Step 5: Check if we have relevant context
        if not filtered_chunks:
            logger.warning("No relevant context found")
            return {
                "answer": "This information is not available in the provided document(s).",
                "citations": [],
                "confidence": 0.0,
                "debug_info": {
                    "query": query,
                    "enhanced_query": enhanced_query,
                    "retrieved_chunks": 0,
                    "reason": "No chunks above similarity threshold"
                } if return_debug else None
            }
        
        # Step 6: Generate answer using LLM
        context_chunks_for_llm = [chunk['metadata'] for chunk in filtered_chunks]
        
        answer, llm_metadata = self.llm_client.generate_with_context(
            query=query,
            context_chunks=context_chunks_for_llm,
            enforce_grounding=self.config.enforce_grounding
        )
        
        logger.info("Generated answer from LLM")
        
        # Step 7: Hallucination check
        if self.config.enable_hallucination_check:
            is_grounded, confidence, check_reason = self.hallucination_detector.check_answer(
                answer,
                context_chunks_for_llm,
                strict=self.config.enforce_grounding
            )
            
            logger.info(f"Hallucination check: grounded={is_grounded}, confidence={confidence:.2f}")
            
            if not is_grounded:
                logger.warning(f"Potential hallucination detected: {check_reason}")
                # Override with refusal
                answer = "This information is not available in the provided document(s)."
                confidence = 0.0
        else:
            confidence = 0.8  # Default confidence if no checking
        
        # Step 8: Format citations
        citations = self._format_citations(filtered_chunks)
        
        # Step 9: Build response
        response = {
            "answer": answer,
            "citations": citations,
            "confidence": confidence
        }
        
        # Add debug info if requested
        if return_debug:
            response["debug_info"] = {
                "original_query": query,
                "enhanced_query": enhanced_query,
                "retrieved_chunks": len(retrieved_chunks),
                "filtered_chunks": len(filtered_chunks),
                "similarity_scores": [chunk['score'] for chunk in filtered_chunks],
                "llm_metadata": llm_metadata,
                "grounding_check": {
                    "is_grounded": is_grounded if self.config.enable_hallucination_check else None,
                    "confidence": confidence,
                    "reason": check_reason if self.config.enable_hallucination_check else None
                },
                "chunks": [
                    {
                        "chunk_id": chunk['metadata'].get('chunk_id'),
                        "document": chunk['metadata'].get('document_name'),
                        "page": chunk['metadata'].get('page_number'),
                        "score": chunk['score'],
                        "preview": chunk['metadata'].get('text_preview', '')[:100]
                    }
                    for chunk in filtered_chunks
                ]
            }
        
        return response
    
    def _format_citations(self, chunks: List[Dict]) -> List[Dict]:
        """
        Format citations from retrieved chunks
        
        Args:
            chunks: Retrieved chunks with metadata
            
        Returns:
            List of formatted citation dicts
        """
        citations = []
        seen = set()  # Avoid duplicate citations
        
        for chunk in chunks:
            metadata = chunk['metadata']
            doc_name = metadata.get('document_name', 'Unknown')
            page_num = metadata.get('page_number', '?')
            
            # Create unique key for this citation
            citation_key = f"{doc_name}::{page_num}"
            
            if citation_key not in seen:
                citations.append({
                    "document": doc_name,
                    "page": page_num,
                    "chunk_id": metadata.get('chunk_id'),
                    "relevance_score": chunk['score']
                })
                seen.add(citation_key)
        
        return citations
    
    def batch_answer(self, queries: List[str]) -> List[Dict]:
        """
        Answer multiple questions in batch
        
        Args:
            queries: List of questions
            
        Returns:
            List of response dicts
        """
        return [self.answer_question(query) for query in queries]


if __name__ == "__main__":
    """Test RAG pipeline"""
    import argparse
    import sys
    import os
    
    # 1. Parse command line arguments
    parser = argparse.ArgumentParser(description="Test RAG Pipeline")
    parser.add_argument("--provider", type=str, default="openai", 
                      help="LLM Provider (openai, anthropic, groq)")
    args = parser.parse_args()

    print("="*70)
    print(f"Testing RAG Pipeline with {args.provider.upper()}")
    print("="*70)
    
    # 2. Add src to path if needed
    if 'src' not in sys.path:
        sys.path.insert(0, 'src')

    try:
        from src.ingestion.vector_store import FAISSVectorStore
        from src.ingestion.embeddings import EmbeddingGenerator, EmbeddingConfig
        from src.generation.llm_client import create_llm_client, LLMProvider
        
        # 3. Check API Keys based on provider
        if args.provider == "groq" and not os.getenv("GROQ_API_KEY"):
            print("\n❌ Error: GROQ_API_KEY not found!")
            print("  Run: $env:GROQ_API_KEY='your_key_here'")
            sys.exit(1)
        elif args.provider == "openai" and not os.getenv("OPENAI_API_KEY"):
            print("\n❌ Error: OPENAI_API_KEY not found!")
            sys.exit(1)
        elif args.provider == "anthropic" and not os.getenv("ANTHROPIC_API_KEY"):
            print("\n❌ Error: ANTHROPIC_API_KEY not found!")
            sys.exit(1)

        # 4. Load vector store
        print("\n📂 Loading vector store...")
        store = FAISSVectorStore()
        try:
            store.load("data/vector_store", "aviation_vectors")
            print(f"✅ Loaded {len(store.metadata)} vectors")
        except Exception as e:
            print(f"❌ Failed to load vector store: {e}")
            print("  Did you run 'python build_vector_store.py'?")
            sys.exit(1)
        
        # 5. Initialize embedding generator
        print("\n🔧 Initializing embedding generator...")
        emb_config = EmbeddingConfig()
        generator = EmbeddingGenerator(emb_config)
        print("✅ Generator ready")
        
        # 6. Initialize LLM client (USING THE ARGUMENT)
        print(f"\n🤖 Initializing LLM client ({args.provider})...")
        llm = create_llm_client(provider=args.provider)
        print(f"✅ Using {args.provider}")
        
        # 7. Create RAG pipeline
        print("\n🔗 Creating RAG pipeline...")
        config = RAGConfig(
            top_k=5,
            min_similarity=0.35,  # Lowered slightly for better recall
            enable_query_enhancement=True,
            enable_hallucination_check=True,
            enforce_grounding=True
        )
        
        rag = RAGPipeline(store, generator, llm, config)
        print("✅ Pipeline ready")
        
        # 8. Test queries
        test_queries = [
            "What is a stall?",  # Broader question, easier to find
            "What are the emergency procedures for engine failure?",
        ]
        
        for query in test_queries:
            print("\n" + "="*70)
            print(f"🔍 Query: {query}")
            print("="*70)
            
            result = rag.answer_question(query, return_debug=True)
            
            print(f"\n✅ Answer:")
            print("-" * 70)
            print(result['answer'])
            print("-" * 70)
            
            # Safe printing of debug info (prevents crashes if empty)
            if result.get('debug_info'):
                debug = result['debug_info']
                print(f"\n🔍 Debug Info:")
                
                # FIX: Handle both Lists (len) and Integers (counts)
                retrieved = debug.get('retrieved_chunks', 0)
                r_count = len(retrieved) if isinstance(retrieved, list) else retrieved
                print(f"  Retrieved: {r_count} chunks")
                
                filtered = debug.get('filtered_chunks', 0)
                f_count = len(filtered) if isinstance(filtered, list) else filtered
                print(f"  Filtered: {f_count} chunks")
                
                scores = debug.get('similarity_scores', [])
                if scores:
                    print(f"  Top score: {max(scores):.3f}")
                else:
                    print(f"  Top score: 0.000")
        
        print("\n" + "="*70)
        print("✅ RAG Pipeline Test Complete!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()