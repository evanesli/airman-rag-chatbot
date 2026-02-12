"""
LLM Client Module
------------------
Handles communication with LLM providers (OpenAI, Anthropic, local models).

Key Features:
1. Multiple provider support
2. Structured prompts for grounded answers
3. Retry logic and error handling
4. Token counting and cost estimation
5. Streaming support
"""

import os
import logging
from typing import List, Dict, Optional, Tuple,Any
from dataclasses import dataclass
from enum import Enum
import time

# LLM Providers
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    LOCAL = "local"


@dataclass
class LLMConfig:
    """Configuration for LLM"""
    provider: LLMProvider = LLMProvider.OPENAI
    model: str = "gpt-3.5-turbo"
    temperature: float = 0.1  # Low temperature for factual answers
    max_tokens: int = 1000
    api_key: Optional[str] = None
    
    def __post_init__(self):
        # Load API key from environment if not provided
        if self.api_key is None:
            if self.provider == LLMProvider.OPENAI:
                self.api_key = os.getenv("OPENAI_API_KEY")
            elif self.provider == LLMProvider.ANTHROPIC:
                self.api_key = os.getenv("ANTHROPIC_API_KEY")
            elif self.provider == LLMProvider.GROQ:   # <--- Add this
                self.api_key = os.getenv("GROQ_API_KEY")    


class LLMClient:
    """
    Universal LLM client supporting multiple providers.
    
    Features:
    - OpenAI (GPT-3.5, GPT-4)
    - Anthropic (Claude)
    - Local models (via Ollama/LM Studio)
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """
        Initialize LLM client
        
        Args:
            config: LLMConfig object or None for defaults
        """
        self.config = config or LLMConfig()
        self.client: Any = None
        
        # Initialize provider-specific client
        if self.config.provider == LLMProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI not installed. Run: pip install openai")
            if not self.config.api_key:
                raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable")
            self.client = openai.OpenAI(api_key=self.config.api_key)
            logger.info(f"Initialized OpenAI client with model: {self.config.model}")
            
        elif self.config.provider == LLMProvider.ANTHROPIC:
            if not ANTHROPIC_AVAILABLE:
                raise ImportError("Anthropic not installed. Run: pip install anthropic")
            if not self.config.api_key:
                raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable")
            self.client = anthropic.Anthropic(api_key=self.config.api_key)
            logger.info(f"Initialized Anthropic client with model: {self.config.model}")

        # ... inside __init__ ...
        elif self.config.provider == LLMProvider.GROQ:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI library required for Groq. Run: pip install openai")
            if not self.config.api_key:
                raise ValueError("Groq API key not found. Set GROQ_API_KEY environment variable")
            
            # Groq uses the OpenAI client but with a custom base_url
            self.client = openai.OpenAI(
                api_key=self.config.api_key,
                base_url="https://api.groq.com/openai/v1"
            )
            logger.info(f"Initialized Groq client with model: {self.config.model}")    
            
        else:
            raise ValueError(f"Provider {self.config.provider} not yet implemented")
    
    def generate(self, 
                 system_prompt: str,
                 user_message: str,
                 max_retries: int = 3) -> str:
        """
        Generate response from LLM
        
        Args:
            system_prompt: System instructions
            user_message: User query
            max_retries: Number of retries on failure
            
        Returns:
            Generated text response
        """
        for attempt in range(max_retries):
            try:
                if self.config.provider == LLMProvider.OPENAI:
                    return self._generate_openai(system_prompt, user_message)
                elif self.config.provider == LLMProvider.ANTHROPIC:
                    return self._generate_anthropic(system_prompt, user_message)
                if self.config.provider == LLMProvider.OPENAI or self.config.provider == LLMProvider.GROQ:
                    return self._generate_openai(system_prompt, user_message)
            except Exception as e:
                logger.error(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
        raise RuntimeError("Failed to generate response after retries")
    def _generate_openai(self, system_prompt: str, user_message: str) -> str:
        """Generate using OpenAI"""
        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens
        )
        
        return response.choices[0].message.content
    
    def _generate_anthropic(self, system_prompt: str, user_message: str) -> str:
        """Generate using Anthropic Claude"""
        response = self.client.messages.create(
            model=self.config.model,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": user_message}
            ]
        )
        
        return response.content[0].text
    
    def generate_with_context(self,
                             query: str,
                             context_chunks: List[Dict],
                             enforce_grounding: bool = True) -> Tuple[str, Dict]:
        """
        Generate answer using retrieved context chunks
        
        Args:
            query: User question
            context_chunks: Retrieved chunks with metadata
            enforce_grounding: Whether to strictly enforce answer grounding
            
        Returns:
            Tuple of (answer text, metadata dict)
        """
        # Build context from chunks
        context_text = self._format_context(context_chunks)
        
        # Create system prompt
        system_prompt = self._create_system_prompt(enforce_grounding)
        
        # Create user message
        user_message = self._create_user_message(query, context_text)
        
        # Generate answer
        answer = self.generate(system_prompt, user_message)
        
        # Create metadata
        metadata = {
            "model": self.config.model,
            "provider": self.config.provider.value,
            "num_chunks": len(context_chunks),
            "sources": [
                {
                    "document": chunk.get("document_name", "Unknown"),
                    "page": chunk.get("page_number", "?"),
                    "chunk_id": chunk.get("chunk_id", "?")
                }
                for chunk in context_chunks
            ]
        }
        
        return answer, metadata
    
    def _format_context(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks into context string"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("document_name", "Unknown")
            page_num = chunk.get("page_number", "?")
            text = chunk.get("text_preview", "")
            
            # Format: [Source 1] Document: PPL_Manual.pdf, Page: 23
            # Content: ...
            context_parts.append(
                f"[Source {i}] Document: {doc_name}, Page: {page_num}\n"
                f"Content: {text}\n"
            )
        
        return "\n".join(context_parts)
    
    def _create_system_prompt(self, enforce_grounding: bool) -> str:
        base_prompt = """You are an aviation expert.

Rules:
1. PRIORITIZE the provided documents.
2. If the documents provide facts (e.g., "Lapse rate is 2 degrees") but lack the explanation (e.g., "Why?"), YOU MAY USE GENERAL AVIATION KNOWLEDGE to fill in the gap.
3. Answer Multiple Choice Questions (MCQ) directly. If the exact option isn't in the text, choose the one that aligns with standard aviation theory and explain why.
4. Always cite the document you used for the core facts.

Format:
- Answer: [Option]
- Explanation: [Your explanation]
- Source: [Citation]
"""
        return base_prompt
    
    def _create_user_message(self, query: str, context: str) -> str:
        """Create user message with query and context"""
        return f"""Context from aviation documents:

{context}

Question: {query}

Please provide an answer based strictly on the context above. Include citations."""


def create_llm_client(provider: str = "openai", 
                      model: Optional[str] = None) -> LLMClient:
    """
    Convenience function to create LLM client
    """
    # 1. Map the string to the correct Enum
    if provider.lower() == "groq":
        provider_enum = LLMProvider.GROQ
        default_model = "meta-llama/llama-4-scout-17b-16e-instruct" 
    elif provider.lower() == "anthropic":
        provider_enum = LLMProvider.ANTHROPIC
        default_model = "claude-3-haiku-20240307"
    else:
        # Default to OpenAI if unknown
        provider_enum = LLMProvider.OPENAI
        default_model = "gpt-3.5-turbo"
    
    # 2. Set default model if none provided
    if model is None:
        model = default_model
    
    # 3. Create config
    config = LLMConfig(
        provider=provider_enum,
        model=model,
        temperature=0.1,
        max_tokens=1000
    )
    
    return LLMClient(config)


if __name__ == "__main__":
    """Test LLM client"""
    
    print("="*70)
    print("Testing LLM Client")
    print("="*70)
    
    # Check if API keys are available
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    
    print(f"\nAPI Keys:")
    print(f"  OpenAI: {'✅ Found' if openai_key else '❌ Not found'}")
    print(f"  Anthropic: {'✅ Found' if anthropic_key else '❌ Not found'}")
    
    if not openai_key and not anthropic_key:
        print("\n⚠️  No API keys found!")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        exit(1)
    
    # Test with available provider
    provider = "openai" if openai_key else "anthropic"
    
    print(f"\n🔧 Initializing {provider.upper()} client...")
    client = create_llm_client(provider=provider)
    
    # Sample context chunks
    sample_chunks = [
        {
            "document_name": "PPL_Manual.pdf",
            "page_number": 23,
            "chunk_id": "chunk_45",
            "text_preview": "The stall speed (Vs) for this aircraft is 45 KIAS in the landing configuration. This represents the minimum speed at which the aircraft can maintain controlled flight."
        },
        {
            "document_name": "PPL_Manual.pdf",
            "page_number": 24,
            "chunk_id": "chunk_46",
            "text_preview": "During stall recovery, the pilot must: 1) Lower the nose to reduce angle of attack, 2) Add full power, 3) Level the wings, 4) Return to normal flight attitude."
        }
    ]
    
    # Test query
    query = "What is the stall speed and how do you recover from a stall?"
    
    print(f"\n📝 Query: {query}")
    print(f"\n🤖 Generating answer...")
    
    answer, metadata = client.generate_with_context(
        query=query,
        context_chunks=sample_chunks,
        enforce_grounding=True
    )
    
    print(f"\n✅ Answer:")
    print("-" * 70)
    print(answer)
    print("-" * 70)
    
    print(f"\n📊 Metadata:")
    print(f"  Model: {metadata['model']}")
    print(f"  Provider: {metadata['provider']}")
    print(f"  Chunks used: {metadata['num_chunks']}")
    print(f"  Sources:")
    for source in metadata['sources']:
        print(f"    - {source['document']}, page {source['page']}")
    
    # Test refusal (no relevant context)
    print("\n" + "="*70)
    print("Testing Refusal (No Relevant Context)")
    print("="*70)
    
    irrelevant_chunks = [
        {
            "document_name": "PPL_Manual.pdf",
            "page_number": 50,
            "chunk_id": "chunk_100",
            "text_preview": "Weather minimums for VFR flight require 3 statute miles visibility and specific cloud clearances."
        }
    ]
    
    query_no_answer = "What is the maximum takeoff weight?"
    
    print(f"\n📝 Query: {query_no_answer}")
    print(f"🤖 Generating answer...")
    
    answer_refused, _ = client.generate_with_context(
        query=query_no_answer,
        context_chunks=irrelevant_chunks,
        enforce_grounding=True
    )
    
    print(f"\n✅ Answer:")
    print("-" * 70)
    print(answer_refused)
    print("-" * 70)
    
    if "not available" in answer_refused.lower():
        print("\n✅ Correctly refused to answer!")
    else:
        print("\n⚠️  May have hallucinated - check answer carefully")