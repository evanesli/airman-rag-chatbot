"""
Vector Store Module
-------------------
FAISS-based vector store for efficient similarity search.

Key Features:
1. Fast similarity search using FAISS
2. Metadata tracking for citations
3. Multiple index types (flat, IVF, HNSW)
4. Save/load functionality
5. Batch search support
"""

import os
import json
import pickle
from typing import List, Dict, Optional, Tuple,Any
import numpy as np
import logging
from pathlib import Path

import faiss
from dataclasses import dataclass, asdict

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class VectorStoreConfig:
    """Configuration for vector store"""
    index_type: str = "Flat"  # "Flat", "IVF", "HNSW"
    dimension: int = 384
    metric: str = "cosine"  # "cosine" or "l2"
    nlist: int = 100  # For IVF index
    nprobe: int = 10  # Search in top 10 clusters
    
    def to_dict(self):
        return asdict(self)


class FAISSVectorStore:
    """
    FAISS-based vector store for semantic search.
    
    Supports:
    - Fast similarity search
    - Metadata storage and retrieval
    - Multiple index types
    - Persistent storage
    """
    
    def __init__(self, config: Optional[VectorStoreConfig] = None):
        """
        Initialize vector store
        
        Args:
            config: VectorStoreConfig or None for defaults
        """
        self.config = config or VectorStoreConfig()
        self.index: Any = None
        self.metadata = []
        self.id_to_index = {}  # Map chunk_id to FAISS index
        
        logger.info(f"Initialized FAISSVectorStore with {self.config.index_type} index")
    
    def _create_index(self, dimension: int) -> faiss.Index:
        """
        Create FAISS index based on configuration
        
        Args:
            dimension: Embedding dimension
            
        Returns:
            FAISS index object
        """
        if self.config.metric == "cosine":
            # Normalize vectors for cosine similarity
            logger.info("Using cosine similarity (normalized vectors)")
            measure = faiss.METRIC_INNER_PRODUCT
        else:
            logger.info("Using L2 distance")
            measure = faiss.METRIC_L2
        
        if self.config.index_type == "Flat":
            # Exact search (best quality, slower for large datasets)
            index = faiss.IndexFlatIP(dimension) if measure == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
            logger.info("Created Flat index (exact search)")
            
        elif self.config.index_type == "IVF":
            # Inverted file index (faster, approximate)
            quantizer = faiss.IndexFlatIP(dimension) if measure == faiss.METRIC_INNER_PRODUCT else faiss.IndexFlatL2(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, self.config.nlist, measure)
            logger.info(f"Created IVF index with {self.config.nlist} clusters")
            
        elif self.config.index_type == "HNSW":
            # Hierarchical Navigable Small World (very fast, approximate)
            index = faiss.IndexHNSWFlat(dimension, 32)
            logger.info("Created HNSW index")
            
        else:
            raise ValueError(f"Unknown index type: {self.config.index_type}")
        
        return index
    
    def add_embeddings(self, 
                      embeddings: np.ndarray,
                      metadata: List[Dict],
                      normalize: bool = True):
        """
        Add embeddings to the vector store
        
        Args:
            embeddings: Numpy array of shape (n, dimension)
            metadata: List of metadata dicts (one per embedding)
            normalize: Normalize embeddings for cosine similarity
        """
        if len(embeddings) != len(metadata):
            raise ValueError("Number of embeddings must match number of metadata entries")
        
        # Create index if not exists
        if self.index is None:
            self.config.dimension = embeddings.shape[1]
            self.index = self._create_index(self.config.dimension)
        
        # Normalize if using cosine similarity
        if normalize and self.config.metric == "cosine":
            faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if self.config.index_type == "IVF" and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("Index trained")
        
        # Get current size before adding
        start_idx = len(self.metadata)
        
        # Add to index
        self.index.add(embeddings)
        
        # Add metadata
        for i, meta in enumerate(metadata):
            self.metadata.append(meta)
            chunk_id = meta.get('chunk_id', f'chunk_{start_idx + i}')
            self.id_to_index[chunk_id] = start_idx + i
        
        logger.info(f"Added {len(embeddings)} embeddings. Total: {len(self.metadata)}")
    
    def search(self, 
              query_embedding: np.ndarray,
              k: int = 5,
              filter_metadata: Optional[Dict] = None) -> List[Dict]:
        """
        Search for similar vectors
        
        Args:
            query_embedding: Query vector of shape (dimension,)
            k: Number of results to return
            filter_metadata: Optional dict to filter results
            
        Returns:
            List of dicts with 'score', 'metadata', and 'index'
        """
        if self.index is None:
            raise ValueError("No embeddings added to the index yet")
        
        # Reshape query to (1, dimension)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize if using cosine similarity
        if self.config.metric == "cosine":
            faiss.normalize_L2(query_embedding)
        
        # Set nprobe for IVF index
        if self.config.index_type == "IVF":
            self.index.nprobe = self.config.nprobe
        
        # Search
        scores, indices = self.index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for missing results
                continue
            
            result = {
                'score': float(score),
                'metadata': self.metadata[idx],
                'index': int(idx)
            }
            
            # Apply metadata filter if provided
            if filter_metadata:
                if all(result['metadata'].get(k) == v for k, v in filter_metadata.items()):
                    results.append(result)
            else:
                results.append(result)
        
        return results
    
    def search_batch(self,
                    query_embeddings: np.ndarray,
                    k: int = 5) -> List[List[Dict]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Array of shape (n_queries, dimension)
            k: Number of results per query
            
        Returns:
            List of result lists (one per query)
        """
        if self.index is None:
            raise ValueError("No embeddings added to the index yet")
        
        # Normalize if using cosine similarity
        if self.config.metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        # Set nprobe for IVF index
        if self.config.index_type == "IVF":
            self.index.nprobe = self.config.nprobe
        
        # Search
        scores, indices = self.index.search(query_embeddings, k)
        
        # Format results for each query
        all_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx == -1:
                    continue
                
                results.append({
                    'score': float(score),
                    'metadata': self.metadata[idx],
                    'index': int(idx)
                })
            all_results.append(results)
        
        return all_results
    
    def get_by_id(self, chunk_id: str) -> Optional[Dict]:
        """
        Get metadata for a specific chunk ID
        
        Args:
            chunk_id: Chunk identifier
            
        Returns:
            Metadata dict or None if not found
        """
        idx = self.id_to_index.get(chunk_id)
        if idx is not None:
            return self.metadata[idx]
        return None
    
    def save(self, directory: str, name: str = "vector_store"):
        """
        Save vector store to disk
        
        Args:
            directory: Directory to save files
            name: Base name for files
        """
        os.makedirs(directory, exist_ok=True)
        
        # Save FAISS index
        index_path = os.path.join(directory, f"{name}.index")
        faiss.write_index(self.index, index_path)
        logger.info(f"Saved FAISS index to {index_path}")
        
        # Save metadata
        metadata_path = os.path.join(directory, f"{name}_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")
        
        # Save ID mapping
        mapping_path = os.path.join(directory, f"{name}_id_mapping.pkl")
        with open(mapping_path, 'wb') as f:
            pickle.dump(self.id_to_index, f)
        logger.info(f"Saved ID mapping to {mapping_path}")
        
        # Save config
        config_path = os.path.join(directory, f"{name}_config.json")
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        logger.info(f"Saved config to {config_path}")
    
    def load(self, directory: str, name: str = "vector_store"):
        """
        Load vector store from disk
        
        Args:
            directory: Directory containing saved files
            name: Base name for files
        """
        # Load FAISS index
        index_path = os.path.join(directory, f"{name}.index")
        self.index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index from {index_path}")
        
        # Load metadata
        metadata_path = os.path.join(directory, f"{name}_metadata.json")
        with open(metadata_path, 'rb') as f:
            self.metadata = json.load(f)
        logger.info(f"Loaded metadata from {metadata_path}")
        
        # Load ID mapping
        mapping_path = os.path.join(directory, f"{name}_id_mapping.pkl")
        with open(mapping_path, 'rb') as f:
            self.id_to_index = pickle.load(f)
        logger.info(f"Loaded ID mapping from {mapping_path}")
        
        # Load config
        config_path = os.path.join(directory, f"{name}_config.json")
        with open(config_path, 'rb') as f:
            config_dict = json.load(f)
            self.config = VectorStoreConfig(**config_dict)
        logger.info(f"Loaded config from {config_path}")
    
    def get_stats(self) -> Dict:
        """Get statistics about the vector store"""
        return {
            "total_vectors": len(self.metadata),
            "dimension": self.config.dimension,
            "index_type": self.config.index_type,
            "metric": self.config.metric,
            "is_trained": self.index.is_trained if hasattr(self.index, 'is_trained') else True
        }


def build_vector_store_from_embeddings(
    embeddings_dict: Dict[str, Tuple[np.ndarray, List[Dict]]],
    output_dir: str = "data/vector_store",
    index_type: str = "Flat"
) -> FAISSVectorStore:
    """
    Convenience function to build vector store from embeddings
    
    Args:
        embeddings_dict: Dict mapping doc name to (embeddings, metadata)
        output_dir: Directory to save vector store
        index_type: Type of FAISS index
        
    Returns:
        FAISSVectorStore object
    """
    config = VectorStoreConfig(index_type=index_type, metric="cosine")
    store = FAISSVectorStore(config)
    
    # Add all embeddings
    for doc_name, (embeddings, metadata) in embeddings_dict.items():
        logger.info(f"Adding embeddings from {doc_name}")
        
        # Add document name to metadata
        for meta in metadata:
            meta['document_name'] = doc_name
        
        store.add_embeddings(embeddings, metadata)
    
    # Save to disk
    store.save(output_dir)
    
    logger.info(f"Vector store built with {len(store.metadata)} vectors")
    
    return store


if __name__ == "__main__":
    """Test vector store"""
    
    print("="*70)
    print("Testing FAISS Vector Store")
    print("="*70)
    
    # Create sample embeddings
    np.random.seed(42)
    n_vectors = 100
    dimension = 384
    
    embeddings = np.random.randn(n_vectors, dimension).astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(embeddings)
    
    # Create metadata
    metadata = [
        {
            'chunk_id': f'chunk_{i}',
            'text_preview': f'Sample text chunk {i}',
            'page_number': (i // 5) + 1,
            'document_name': 'test_doc.pdf'
        }
        for i in range(n_vectors)
    ]
    
    print(f"\n📊 Sample data:")
    print(f"   Vectors: {n_vectors}")
    print(f"   Dimension: {dimension}")
    
    # Create vector store
    print(f"\n🔧 Creating vector store...")
    config = VectorStoreConfig(index_type="Flat", metric="cosine")
    store = FAISSVectorStore(config)
    
    # Add embeddings
    print(f"\n📥 Adding embeddings...")
    store.add_embeddings(embeddings, metadata)
    
    # Test search
    print(f"\n🔍 Testing search...")
    query = embeddings[0]  # Use first vector as query
    results = store.search(query, k=5)
    
    print(f"\n✅ Top 5 results for query (should include original vector):")
    for i, result in enumerate(results, 1):
        print(f"   {i}. Score: {result['score']:.4f} | "
              f"Chunk: {result['metadata']['chunk_id']} | "
              f"Page: {result['metadata']['page_number']}")
    
    # Test save/load
    print(f"\n💾 Testing save/load...")
    save_dir = "data/vector_store/test"
    store.save(save_dir, "test_store")
    
    # Create new store and load
    new_store = FAISSVectorStore()
    new_store.load(save_dir, "test_store")
    
    # Test search on loaded store
    new_results = new_store.search(query, k=5)
    
    print(f"✅ Loaded store search results match: {results[0]['score'] == new_results[0]['score']}")
    
    # Show stats
    print(f"\n📈 Vector store statistics:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"   {key}: {value}")