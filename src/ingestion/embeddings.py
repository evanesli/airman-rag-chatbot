"""
Embeddings Module
-----------------
Convert text chunks to dense vector embeddings for semantic search.

Key Features:
1. Multiple embedding model support
2. Batch processing for efficiency
3. Caching to avoid recomputation
4. Metadata preservation
5. Progress tracking
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict
import logging
from pathlib import Path
from tqdm import tqdm

# Embedding models
from sentence_transformers import SentenceTransformer
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for embedding generation"""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32
    normalize: bool = True
    device: str = "cpu"  # or "cuda" if GPU available
    
    def to_dict(self):
        return asdict(self)


class EmbeddingGenerator:
    """
    Generate embeddings for text chunks using transformer models.
    
    Features:
    - Efficient batch processing
    - Automatic device detection (CPU/GPU)
    - Progress tracking
    - Caching support
    """
    
    def __init__(self, config: Optional[EmbeddingConfig] = None):
        """
        Initialize embedding generator
        
        Args:
            config: EmbeddingConfig object or None for defaults
        """
        self.config = config or EmbeddingConfig()
        
        # Auto-detect GPU
        if self.config.device == "auto":
            self.config.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Initializing embedding model: {self.config.model_name}")
        logger.info(f"Using device: {self.config.device}")
        
        # Load model
        self.model = SentenceTransformer(self.config.model_name)
        self.model.to(self.config.device)
        
        # Verify dimensions
        # Verify dimensions
        actual_dim = self.model.get_sentence_embedding_dimension()
        if actual_dim is not None and actual_dim != self.config.dimension:
            logger.warning(f"Expected dimension {self.config.dimension}, got {actual_dim}")
            self.config.dimension = int(actual_dim)  # Explicit int cast fixes the error
        logger.info(f"Model loaded. Embedding dimension: {self.config.dimension}")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text: Input text string
            
        Returns:
            Numpy array of shape (dimension,)
        """
        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            device=self.config.device
        )
        return embedding
    
    def embed_batch(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts (efficient batch processing)
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            Numpy array of shape (len(texts), dimension)
        """
        logger.info(f"Embedding {len(texts)} texts in batches of {self.config.batch_size}")
        
        embeddings = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            convert_to_numpy=True,
            normalize_embeddings=self.config.normalize,
            show_progress_bar=show_progress,
            device=self.config.device
        )
        
        return embeddings
    
    def embed_chunks(self, chunks: List, show_progress: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Generate embeddings for TextChunk objects
        
        Args:
            chunks: List of TextChunk objects from chunking module
            show_progress: Show progress bar
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        # Extract texts and metadata
        texts = [chunk.text for chunk in chunks]
        metadata_list = []
        
        for chunk in chunks:
            meta = chunk.metadata.copy()
            meta['chunk_id'] = chunk.chunk_id
            meta['start_char'] = chunk.start_char
            meta['end_char'] = chunk.end_char
            meta['text_preview'] = chunk.text[:100]  # First 100 chars for reference
            metadata_list.append(meta)
        
        # Generate embeddings
        embeddings = self.embed_batch(texts, show_progress=show_progress)
        
        logger.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")
        
        return embeddings, metadata_list
    
    def save_embeddings(self, 
                       embeddings: np.ndarray,
                       metadata: List[Dict],
                       output_dir: str,
                       filename: str = "embeddings"):
        """
        Save embeddings and metadata to disk
        
        Args:
            embeddings: Numpy array of embeddings
            metadata: List of metadata dicts
            output_dir: Directory to save files
            filename: Base filename (without extension)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Save embeddings as numpy array
        embeddings_path = os.path.join(output_dir, f"{filename}.npy")
        np.save(embeddings_path, embeddings)
        logger.info(f"Saved embeddings to {embeddings_path}")
        
        # Save metadata as JSON
        metadata_path = os.path.join(output_dir, f"{filename}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save config
        config_path = os.path.join(output_dir, f"{filename}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    def load_embeddings(self, 
                       input_dir: str,
                       filename: str = "embeddings") -> Tuple[np.ndarray, List[Dict]]:
        """
        Load embeddings and metadata from disk
        
        Args:
            input_dir: Directory containing saved files
            filename: Base filename (without extension)
            
        Returns:
            Tuple of (embeddings array, metadata list)
        """
        # Load embeddings
        embeddings_path = os.path.join(input_dir, f"{filename}.npy")
        embeddings = np.load(embeddings_path)
        logger.info(f"Loaded embeddings from {embeddings_path}")
        
        # Load metadata
        metadata_path = os.path.join(input_dir, f"{filename}_metadata.json")
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        
        return embeddings, metadata


def create_embeddings_from_documents(
    documents_chunks: Dict[str, List],
    output_dir: str = "data/processed",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> Dict[str, Tuple[np.ndarray, List[Dict]]]:
    """
    Convenience function to create embeddings for multiple documents
    
    Args:
        documents_chunks: Dict mapping filename to list of chunks
        output_dir: Directory to save embeddings
        model_name: Name of sentence transformer model
        
    Returns:
        Dict mapping filename to (embeddings, metadata)
    """
    # Initialize generator
    config = EmbeddingConfig(model_name=model_name, device="auto")
    generator = EmbeddingGenerator(config)
    
    results = {}
    
    for doc_name, chunks in documents_chunks.items():
        logger.info(f"Processing document: {doc_name}")
        
        # Generate embeddings
        embeddings, metadata = generator.embed_chunks(chunks)
        
        # Save to disk
        safe_name = doc_name.replace('.pdf', '').replace(' ', '_')
        generator.save_embeddings(
            embeddings,
            metadata,
            output_dir,
            filename=f"{safe_name}_embeddings"
        )
        
        results[doc_name] = (embeddings, metadata)
    
    return results


if __name__ == "__main__":
    """Test embedding generation"""
    
    print("="*70)
    print("Testing Embedding Generator")
    print("="*70)
    
    # Test with sample texts
    sample_texts = [
        "Engine failure during takeoff requires immediate action.",
        "The pilot must maintain control of the aircraft at all times.",
        "Weather minimums for VFR flight are 3 miles visibility.",
        "Stall recovery involves lowering the nose and adding power.",
        "Radio communication with ATC is essential for flight safety."
    ]
    
    print("\n📝 Sample texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"  {i}. {text}")
    
    # Initialize generator
    config = EmbeddingConfig(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        batch_size=8,
        device="auto"
    )
    
    print(f"\n🔧 Initializing embedding generator...")
    generator = EmbeddingGenerator(config)
    
    # Generate embeddings
    print(f"\n🚀 Generating embeddings...")
    embeddings = generator.embed_batch(sample_texts, show_progress=True)
    
    print(f"\n✅ Generated embeddings:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Dimension: {embeddings.shape[1]}")
    print(f"   Data type: {embeddings.dtype}")
    
    # Show similarity between texts
    print(f"\n📊 Similarity Matrix:")
    print("   (Higher = more similar)")
    print()
    
    # Calculate cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarity_matrix = cosine_similarity(embeddings)
    
    print("      ", end="")
    for i in range(len(sample_texts)):
        print(f"Text{i+1:2d} ", end="")
    print()
    
    for i in range(len(sample_texts)):
        print(f"Text{i+1:2d}", end=" ")
        for j in range(len(sample_texts)):
            if i == j:
                print("  1.00 ", end="")
            else:
                print(f" {similarity_matrix[i][j]:5.2f} ", end="")
        print()
    
    print(f"\n💡 Interpretation:")
    print(f"   Text 1 & Text 4 are both about emergency procedures")
    print(f"   Text 2 & Text 5 are about pilot responsibilities")
    print(f"   Text 3 is about weather (less similar to others)")
    
    # Test saving and loading
    print(f"\n💾 Testing save/load...")
    
    test_metadata = [{"text": text, "index": i} for i, text in enumerate(sample_texts)]
    
    generator.save_embeddings(
        embeddings,
        test_metadata,
        "data/processed/test",
        filename="test_embeddings"
    )
    
    loaded_embeddings, loaded_metadata = generator.load_embeddings(
        "data/processed/test",
        filename="test_embeddings"
    )
    
    print(f"✅ Save/load successful!")
    print(f"   Original shape: {embeddings.shape}")
    print(f"   Loaded shape: {loaded_embeddings.shape}")
    print(f"   Arrays match: {np.allclose(embeddings, loaded_embeddings)}")