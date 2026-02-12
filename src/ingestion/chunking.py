"""
Chunking Module
---------------
Intelligent text chunking for aviation documents with semantic awareness.

Key Features:
1. Multiple chunking strategies (fixed, semantic, sliding window)
2. Aviation-specific boundary detection
3. Metadata preservation for citations
4. Configurable chunk size and overlap
5. Section header preservation
"""

import re
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from enum import Enum

# For sentence splitting
import nltk
nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

from nltk.tokenize import sent_tokenize

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkingStrategy(Enum):
    """Available chunking strategies"""
    FIXED = "fixed"              # Fixed character count
    SEMANTIC = "semantic"        # Split at semantic boundaries
    SLIDING = "sliding"          # Sliding window with overlap
    AVIATION = "aviation"        # Aviation-optimized (recommended)


@dataclass
class TextChunk:
    """Represents a single chunk of text"""
    chunk_id: str                    # Unique identifier
    text: str                        # The actual text content
    metadata: Dict[str, Any]         # Source info, page number, etc.
    
    # Position tracking
    start_char: int                  # Start position in original text
    end_char: int                    # End position in original text
    
    # Context
    previous_chunk_id: Optional[str] = None
    next_chunk_id: Optional[str] = None
    
    def __repr__(self):
        return f"Chunk {self.chunk_id}: {len(self.text)} chars (page {self.metadata.get('page_number', '?')})"
    
    def to_dict(self):
        """Convert to dictionary for storage"""
        return {
            "chunk_id": self.chunk_id,
            "text": self.text,
            "metadata": self.metadata,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "previous_chunk_id": self.previous_chunk_id,
            "next_chunk_id": self.next_chunk_id
        }


class AviationChunker:
    """
    Advanced text chunker optimized for aviation documents.
    
    Features:
    - Smart boundary detection (sentences, paragraphs)
    - Aviation-specific splitting (keeps procedures together)
    - Section header preservation
    - Metadata tracking for citations
    """
    
    def __init__(self,
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 strategy: ChunkingStrategy = ChunkingStrategy.AVIATION,
                 preserve_sections: bool = True):
        """
        Initialize chunker
        
        Args:
            chunk_size: Target size in characters (default: 512)
            chunk_overlap: Overlap between chunks in characters (default: 128)
            strategy: Chunking strategy to use
            preserve_sections: Whether to keep section headers with chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.strategy = strategy
        self.preserve_sections = preserve_sections
        
        # Aviation-specific patterns
        self.section_patterns = [
            r'^(Chapter|Section|Part)\s+\d+',
            r'^\d+\.\d+\s+[A-Z]',  # 1.1 HEADING
            r'^[A-Z][A-Z\s]{5,}$',  # ALL CAPS HEADINGS
        ]
        
        self.procedure_patterns = [
            r'^\s*\d+\.\s+',  # 1. Step one
            r'^\s*[a-z]\)\s+',  # a) Substep
            r'CHECKLIST:',
            r'PROCEDURE:',
            r'WARNING:',
            r'CAUTION:',
            r'NOTE:',
        ]
        
        logger.info(f"Initialized AviationChunker: size={chunk_size}, overlap={chunk_overlap}, strategy={strategy.value}")
    
    def chunk_pages(self, pages: List) -> List[TextChunk]:
        """
        Chunk a list of PageContent objects
        
        Args:
            pages: List of PageContent objects from pdf_loader
            
        Returns:
            List of TextChunk objects
        """
        all_chunks = []
        chunk_counter = 0
        
        for page in pages:
            # Add page-level metadata
            page_metadata = {
                "page_number": page.page_number,
                "source_metadata": page.metadata
            }
            
            # Chunk this page
            if self.strategy == ChunkingStrategy.AVIATION:
                chunks = self._chunk_aviation(page.text, page_metadata, chunk_counter)
            elif self.strategy == ChunkingStrategy.SEMANTIC:
                chunks = self._chunk_semantic(page.text, page_metadata, chunk_counter)
            elif self.strategy == ChunkingStrategy.SLIDING:
                chunks = self._chunk_sliding(page.text, page_metadata, chunk_counter)
            else:  # FIXED
                chunks = self._chunk_fixed(page.text, page_metadata, chunk_counter)
            
            all_chunks.extend(chunks)
            chunk_counter += len(chunks)
        
        # Link chunks together (previous/next)
        for i in range(len(all_chunks)):
            if i > 0:
                all_chunks[i].previous_chunk_id = all_chunks[i-1].chunk_id
            if i < len(all_chunks) - 1:
                all_chunks[i].next_chunk_id = all_chunks[i+1].chunk_id
        
        logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
        return all_chunks
    
    def _chunk_aviation(self, text: str, metadata: Dict, start_id: int) -> List[TextChunk]:
        """
        Aviation-optimized chunking strategy
        
        Priority:
        1. Keep procedures/checklists together
        2. Preserve section headers with content
        3. Split at sentence boundaries
        4. Maintain context with overlap
        """
        chunks = []
        
        # First, identify special sections (procedures, warnings, etc.)
        special_sections = self._identify_special_sections(text)
        
        if special_sections:
            # Chunk with special section awareness
            chunks = self._chunk_with_sections(text, metadata, start_id, special_sections)
        else:
            # Fall back to semantic chunking
            chunks = self._chunk_semantic(text, metadata, start_id)
        
        return chunks
    
    def _identify_special_sections(self, text: str) -> List[Tuple[int, int, str]]:
        """
        Identify special sections that should stay together
        
        Returns:
            List of (start_pos, end_pos, section_type)
        """
        sections = []
        
        # Find procedures (numbered lists)
        procedure_pattern = r'((?:^\s*\d+\.\s+.+\n)+)'
        for match in re.finditer(procedure_pattern, text, re.MULTILINE):
            sections.append((match.start(), match.end(), "procedure"))
        
        # Find warnings/cautions
        warning_pattern = r'((?:WARNING|CAUTION|NOTE):.*?(?=\n\n|\n[A-Z]|$))'
        for match in re.finditer(warning_pattern, text, re.DOTALL):
            sections.append((match.start(), match.end(), "warning"))
        
        # Sort by position
        sections.sort(key=lambda x: x[0])
        
        return sections
    
    def _chunk_with_sections(self, text: str, metadata: Dict, start_id: int, 
                            sections: List[Tuple[int, int, str]]) -> List[TextChunk]:
        """Chunk text while preserving special sections"""
        chunks = []
        current_pos = 0
        chunk_id = start_id
        
        for section_start, section_end, section_type in sections:
            # Chunk text before this section
            if section_start > current_pos:
                before_text = text[current_pos:section_start]
                before_chunks = self._chunk_semantic(before_text, metadata, chunk_id)
                chunks.extend(before_chunks)
                chunk_id += len(before_chunks)
            
            # Add the special section as its own chunk (or chunks if too large)
            section_text = text[section_start:section_end]
            
            if len(section_text) <= self.chunk_size * 1.5:
                # Keep as single chunk
                chunk = TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=section_text,
                    metadata={**metadata, "section_type": section_type},
                    start_char=section_start,
                    end_char=section_end
                )
                chunks.append(chunk)
                chunk_id += 1
            else:
                # Split large section
                section_chunks = self._chunk_semantic(section_text, 
                                                     {**metadata, "section_type": section_type},
                                                     chunk_id)
                chunks.extend(section_chunks)
                chunk_id += len(section_chunks)
            
            current_pos = section_end
        
        # Chunk remaining text
        if current_pos < len(text):
            remaining_text = text[current_pos:]
            remaining_chunks = self._chunk_semantic(remaining_text, metadata, chunk_id)
            chunks.extend(remaining_chunks)
        
        return chunks
    
    def _chunk_semantic(self, text: str, metadata: Dict, start_id: int) -> List[TextChunk]:
        """
        Semantic chunking - split at sentence boundaries
        
        Strategy:
        1. Split into sentences
        2. Group sentences until reaching chunk_size
        3. Add overlap from previous chunk
        """
        if not text.strip():
            return []
        
        chunks = []
        
        # Split into sentences
        sentences = sent_tokenize(text)
        
        current_chunk = []
        current_length = 0
        chunk_id = start_id
        start_pos = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # Check if adding this sentence exceeds chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk from current sentences
                chunk_text = " ".join(current_chunk)
                
                chunk = TextChunk(
                    chunk_id=f"chunk_{chunk_id}",
                    text=chunk_text,
                    metadata=metadata.copy(),
                    start_char=start_pos,
                    end_char=start_pos + len(chunk_text)
                )
                chunks.append(chunk)
                
                # Calculate overlap
                overlap_sentences = []
                overlap_length = 0
                
                # Take sentences from end for overlap
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences + [sentence]
                current_length = sum(len(s) for s in current_chunk)
                start_pos = start_pos + len(chunk_text) - overlap_length
                chunk_id += 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = " ".join(current_chunk)
            chunk = TextChunk(
                chunk_id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata=metadata.copy(),
                start_char=start_pos,
                end_char=start_pos + len(chunk_text)
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sliding(self, text: str, metadata: Dict, start_id: int) -> List[TextChunk]:
        """
        Sliding window chunking
        
        Creates overlapping chunks with fixed stride
        """
        chunks = []
        stride = self.chunk_size - self.chunk_overlap
        chunk_id = start_id
        
        for i in range(0, len(text), stride):
            chunk_text = text[i:i + self.chunk_size]
            
            # Skip if too small
            if len(chunk_text) < 50:
                continue
            
            chunk = TextChunk(
                chunk_id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata=metadata.copy(),
                start_char=i,
                end_char=i + len(chunk_text)
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    def _chunk_fixed(self, text: str, metadata: Dict, start_id: int) -> List[TextChunk]:
        """
        Simple fixed-size chunking (least sophisticated)
        """
        chunks = []
        chunk_id = start_id
        
        for i in range(0, len(text), self.chunk_size):
            chunk_text = text[i:i + self.chunk_size]
            
            if len(chunk_text) < 50:
                continue
            
            chunk = TextChunk(
                chunk_id=f"chunk_{chunk_id}",
                text=chunk_text,
                metadata=metadata.copy(),
                start_char=i,
                end_char=i + len(chunk_text)
            )
            chunks.append(chunk)
            chunk_id += 1
        
        return chunks
    
    def add_section_headers(self, chunks: List[TextChunk], text: str) -> List[TextChunk]:
        """
        Add relevant section headers to chunks for better context
        
        Example: If chunk is from "Chapter 3: Navigation", 
                 prepend "Chapter 3: Navigation\n\n" to chunk text
        """
        if not self.preserve_sections:
            return chunks
        
        # Find all section headers in text
        headers = []
        for pattern in self.section_patterns:
            for match in re.finditer(pattern, text, re.MULTILINE):
                headers.append((match.start(), match.group()))
        
        headers.sort(key=lambda x: x[0])
        
        # Add headers to chunks
        enhanced_chunks = []
        for chunk in chunks:
            # Find most recent header before this chunk
            current_header = None
            for header_pos, header_text in headers:
                if header_pos < chunk.start_char:
                    current_header = header_text
                else:
                    break
            
            # Add header to chunk
            if current_header:
                enhanced_text = f"{current_header}\n\n{chunk.text}"
                chunk.text = enhanced_text
                chunk.metadata["section_header"] = current_header
            
            enhanced_chunks.append(chunk)
        
        return enhanced_chunks
    
    def get_chunk_statistics(self, chunks: List[TextChunk]) -> Dict:
        """Calculate statistics about chunks"""
        if not chunks:
            return {}
        
        lengths = [len(c.text) for c in chunks]
        
        return {
            "total_chunks": len(chunks),
            "avg_length": sum(lengths) / len(lengths),
            "min_length": min(lengths),
            "max_length": max(lengths),
            "total_chars": sum(lengths),
            "chunks_with_sections": sum(1 for c in chunks if "section_header" in c.metadata)
        }


def chunk_documents(pages_dict: Dict, 
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> Dict[str, List[TextChunk]]:
    """
    Convenience function to chunk multiple documents
    
    Args:
        pages_dict: Dict mapping filename to (pages, metadata) from pdf_loader
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        Dict mapping filename to list of chunks
    """
    chunker = AviationChunker(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        strategy=ChunkingStrategy.AVIATION
    )
    
    results = {}
    
    for filename, (pages, metadata) in pages_dict.items():
        logger.info(f"Chunking document: {filename}")
        chunks = chunker.chunk_pages(pages)
        
        # Add document-level metadata
        for chunk in chunks:
            chunk.metadata["document_name"] = filename
            chunk.metadata["document_metadata"] = metadata.to_dict()
        
        results[filename] = chunks
        
        # Log statistics
        stats = chunker.get_chunk_statistics(chunks)
        logger.info(f"  Created {stats['total_chunks']} chunks, avg length: {stats['avg_length']:.0f}")
    
    return results


if __name__ == "__main__":
    """Test the chunker"""
    
    print("="*70)
    print("Testing Aviation Chunker")
    print("="*70)
    
    # Sample aviation text
    sample_text = """
    Chapter 3: Emergency Procedures
    
    Engine Failure During Takeoff
    
    If engine failure occurs during the takeoff roll before reaching VR (rotation speed):
    
    1. Close the throttle immediately
    2. Apply brakes firmly but smoothly
    3. Maintain directional control using rudder
    4. If remaining runway is insufficient, consider:
       a) Continuing straight ahead if clear
       b) Turning to avoid obstacles
    
    WARNING: Do not attempt to return to the runway after becoming airborne 
    with insufficient altitude or airspeed.
    
    After becoming airborne with engine failure:
    
    1. Establish best glide speed (Vg)
    2. Select a suitable landing area
    3. Complete emergency checklist
    4. Transmit MAYDAY on 121.5 MHz
    
    NOTE: Best glide speed for this aircraft is 68 KIAS.
    """
    
    # Create mock PageContent
    from dataclasses import dataclass
    
    @dataclass
    class MockPage:
        page_number: int
        text: str
        metadata: dict
    
    pages = [MockPage(page_number=15, text=sample_text, metadata={})]
    
    # Test different strategies
    strategies = [
        ChunkingStrategy.FIXED,
        ChunkingStrategy.SEMANTIC,
        ChunkingStrategy.AVIATION
    ]
    
    for strategy in strategies:
        print(f"\n{'='*70}")
        print(f"Strategy: {strategy.value}")
        print('='*70)
        
        chunker = AviationChunker(
            chunk_size=1000,
            chunk_overlap=200,
            strategy=strategy
        )
        
        chunks = chunker.chunk_pages(pages)
        
        print(f"\nCreated {len(chunks)} chunks:\n")
        
        for i, chunk in enumerate(chunks):
            print(f"Chunk {i+1}:")
            print(f"  ID: {chunk.chunk_id}")
            print(f"  Length: {len(chunk.text)} chars")
            print(f"  Preview: {chunk.text[:100]}...")
            if "section_type" in chunk.metadata:
                print(f"  Type: {chunk.metadata['section_type']}")
            print()
        
        stats = chunker.get_chunk_statistics(chunks)
        print(f"Statistics: {stats}")