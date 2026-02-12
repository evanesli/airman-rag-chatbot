"""
FastAPI Application
-------------------
Main API server for Aviation RAG system.

Endpoints:
- POST /ask - Answer questions
- POST /ingest - Ingest new documents
- GET /health - Health check
- GET /stats - System statistics
"""

import os
import sys
import shutil
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import logging
import uvicorn

from ingestion.vector_store import FAISSVectorStore
from ingestion.embeddings import EmbeddingGenerator, EmbeddingConfig
from generation.llm_client import create_llm_client, LLMClient
from generation.rag_pipeline import RAGPipeline, RAGConfig

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Aviation RAG API",
    description="AI-powered aviation document question answering system",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Request/Response Models
# ============================================================================

class AskRequest(BaseModel):
    """Request model for /ask endpoint"""
    question: str = Field(..., description="Question to answer")
    top_k: int = Field(5, description="Number of chunks to retrieve")
    min_similarity: float = Field(0.2, description="Minimum similarity threshold")
    include_debug: bool = Field(False, description="Include debug information")


class Citation(BaseModel):
    """Citation information"""
    document: str
    page: int | str
    chunk_id: Optional[str] = None
    relevance_score: Optional[float] = None


class AskResponse(BaseModel):
    """Response model for /ask endpoint"""
    answer: str
    citations: List[Citation]
    confidence: float
    debug_info: Optional[Dict] = None


class HealthResponse(BaseModel):
    """Response model for /health endpoint"""
    status: str
    vector_store_loaded: bool
    total_vectors: int
    embedding_model: str
    llm_provider: str


class StatsResponse(BaseModel):
    """Response model for /stats endpoint"""
    total_documents: int
    total_chunks: int
    embedding_dimension: int
    index_type: str


# ============================================================================
# Global state
# ============================================================================

class AppState:
    """Application state holder"""
    vector_store: Optional[FAISSVectorStore] = None
    embedding_generator: Optional[EmbeddingGenerator] = None
    llm_client: Optional[LLMClient] = None  # <--- Added explicit type hint
    rag_pipeline: Optional[RAGPipeline] = None
    is_ready: bool = False


state = AppState()


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize all components on startup"""
    logger.info("Starting Aviation RAG API...")
    
    try:
        # Load vector store
        logger.info("Loading vector store...")
        state.vector_store = FAISSVectorStore()
        
        vector_store_path = "data/vector_store"
        if os.path.exists(os.path.join(vector_store_path, "aviation_vectors.index")):
            state.vector_store.load(vector_store_path, "aviation_vectors")
            logger.info(f"✅ Loaded {len(state.vector_store.metadata)} vectors")
        else:
            logger.warning(f"⚠️  Vector store not found at {vector_store_path}")
            logger.warning("   Run build_vector_store.py first!")
            return
        
        # Initialize embedding generator
        logger.info("Initializing embedding generator...")
        emb_config = EmbeddingConfig(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="auto"
        )
        state.embedding_generator = EmbeddingGenerator(emb_config)
        logger.info("✅ Embedding generator ready")
        
        # Initialize LLM client
        # Initialize LLM client
        logger.info("Initializing LLM client...")
        openai_key = os.getenv("OPENAI_API_KEY")
        anthropic_key = os.getenv("ANTHROPIC_API_KEY")
        groq_key = os.getenv("GROQ_API_KEY")  # <--- Check for Groq
        
        if groq_key:
            state.llm_client = create_llm_client(provider="groq")
            logger.info("✅ Using Groq")
        elif openai_key:
            state.llm_client = create_llm_client(provider="openai")
            logger.info("✅ Using OpenAI")
        elif anthropic_key:
            state.llm_client = create_llm_client(provider="anthropic")
            logger.info("✅ Using Anthropic")
        else:
            logger.error("❌ No API key found!")
            logger.error("   Set GROQ_API_KEY, OPENAI_API_KEY or ANTHROPIC_API_KEY")
            return
        
        # Initialize RAG pipeline
        logger.info("Initializing RAG pipeline...")
        rag_config = RAGConfig(
            top_k=5,
            min_similarity=0.2,
            enable_query_enhancement=True,
            enable_hallucination_check=True,
            enforce_grounding=True
        )
        
        state.rag_pipeline = RAGPipeline(
            state.vector_store,
            state.embedding_generator,
            state.llm_client,
            rag_config
        )
        logger.info("✅ RAG pipeline ready")
        
        state.is_ready = True
        logger.info("🚀 Aviation RAG API is ready!")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        import traceback
        traceback.print_exc()


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Aviation RAG API...")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint"""
    return {
        "message": "Aviation RAG API",
        "version": "1.0.0",
        "status": "running" if state.is_ready else "initializing",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """Health check endpoint"""
    
    if not state.is_ready:
        raise HTTPException(
            status_code=503,
            detail="System not ready. Check logs for errors."
        )
    
    return HealthResponse(
        status="healthy",
        vector_store_loaded=state.vector_store is not None,
        total_vectors=len(state.vector_store.metadata) if state.vector_store else 0,
        embedding_model=state.embedding_generator.config.model_name if state.embedding_generator else "unknown",
        llm_provider=state.llm_client.config.provider.value if state.llm_client else "unknown"
    )


@app.get("/stats", response_model=StatsResponse, tags=["General"])
async def get_stats():
    """Get system statistics"""
    
    # FIX: Explicitly check if vector_store is None to satisfy Pylance
    if not state.is_ready or state.vector_store is None:
        raise HTTPException(
            status_code=503,
            detail="System not ready or vector store not loaded"
        )

    # Count unique documents
    
    # Count unique documents
    documents = set()
    for meta in state.vector_store.metadata:
        doc_name = meta.get('document_name')
        if doc_name:
            documents.add(doc_name)
    
    stats = state.vector_store.get_stats()
    
    return StatsResponse(
        total_documents=len(documents),
        total_chunks=stats['total_vectors'],
        embedding_dimension=stats['dimension'],
        index_type=stats['index_type']
    )


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
async def ask_question(request: AskRequest):
    """
    Answer a question using RAG pipeline
    
    Example:
    ```json
    {
        "question": "What are the emergency procedures for engine failure?",
        "top_k": 5,
        "min_similarity": 0.1,
        "include_debug": false
    }
    ```
    """
    
    # FIX: Explicitly check if rag_pipeline is None
    if not state.is_ready or state.rag_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="System not ready. Pipeline not initialized."
        )
    
    if not request.question or len(request.question.strip()) == 0:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty"
        )
    
    try:
        # Update pipeline config for this request
        state.rag_pipeline.config.top_k = request.top_k
        state.rag_pipeline.config.min_similarity = request.min_similarity
        
        # Get answer
        result = state.rag_pipeline.answer_question(
            query=request.question,
            return_debug=request.include_debug
        )
        
        # Format response
        citations = [Citation(**c) for c in result['citations']]
        
        return AskResponse(
            answer=result['answer'],
            citations=citations,
            confidence=result['confidence'],
            debug_info=result.get('debug_info')
        )
        
    except Exception as e:
        logger.error(f"Error processing question: {e}")
        import traceback
        traceback.print_exc()
        
        raise HTTPException(
            status_code=500,
            detail=f"Error processing question: {str(e)}"
        )


@app.post("/ingest", tags=["Ingestion"])
async def ingest_document(file: UploadFile = File(...)):
    """
    Ingest a new PDF document
    """
    
    # 1. Validate file type
    if not file.filename or not file.filename.endswith('.pdf'):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported"
        )
    
    try:
        # 2. Save the file to data/raw/
        save_dir = "data/raw"
        os.makedirs(save_dir, exist_ok=True)
        file_path = os.path.join(save_dir, file.filename)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        logger.info(f"✅ Saved file to {file_path}")
        
        # 3. (Optional) Trigger a rebuild or just notify
        # For a production app, we would process chunks here live.
        # For now, we save it and tell the user to rebuild.
        
        return {
            "message": f"Successfully uploaded {file.filename}",
            "file_path": file_path,
            "next_step": "Run 'python build_vector_store.py' to index this file."
        }

    except Exception as e:
        logger.error(f"Error saving file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main
# ============================================================================

def main():
    """Run the API server"""
    
    # Configuration
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )


if __name__ == "__main__":
    main()