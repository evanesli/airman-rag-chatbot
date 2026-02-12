"""
PDF Loader Module
-----------------
Extracts text from aviation PDF documents while preserving structure and metadata.

Key Features:
1. Multiple extraction methods (PyPDF, PyMuPDF, pdfplumber)
2. Page-level text extraction with metadata
3. Aviation-specific text cleaning
4. Table and figure detection
5. Header/footer removal
"""

import os
import re
from typing import List, Dict, Optional, Tuple , Any
from pathlib import Path
import logging
from dataclasses import dataclass
from tqdm import tqdm  # Progress bar

# PDF libraries
import pypdf
import fitz  # PyMuPDF
import pdfplumber

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class PageContent:
    """Represents content from a single PDF page"""
    page_number: int
    text: str
    metadata: Dict[str, Any]
    
    def __repr__(self):
        return f"Page {self.page_number}: {len(self.text)} chars"


@dataclass
class DocumentMetadata:
    """Metadata about the entire document"""
    filename: str
    filepath: str
    total_pages: int
    file_size_mb: float
    creation_date: Optional[str]
    author: Optional[str]
    title: Optional[str]
    
    def to_dict(self):
        return {
            "filename": self.filename,
            "filepath": self.filepath,
            "total_pages": self.total_pages,
            "file_size_mb": round(self.file_size_mb, 2),
            "creation_date": self.creation_date,
            "author": self.author,
            "title": self.title
        }


class AviationPDFLoader:
    """
    Advanced PDF loader optimized for aviation documents.
    
    Uses multiple extraction methods and chooses the best result.
    Preserves document structure and metadata for accurate citations.
    """
    
    def __init__(self, 
                 extraction_method: str = "auto",
                 remove_headers_footers: bool = True,
                 clean_text: bool = True):
        """
        Initialize PDF Loader
        
        Args:
            extraction_method: "auto", "pypdf", "pymupdf", or "pdfplumber"
            remove_headers_footers: Whether to remove repeating headers/footers
            clean_text: Whether to clean and normalize extracted text
        """
        self.extraction_method = extraction_method
        self.remove_headers_footers = remove_headers_footers
        self.clean_text = clean_text
        
        logger.info(f"Initialized AviationPDFLoader with method: {extraction_method}")
    
    def load_pdf(self, pdf_path: str) -> Tuple[List[PageContent], DocumentMetadata]:
        """
        Load a PDF file and extract all content
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (list of PageContent objects, DocumentMetadata)
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Loading PDF: {pdf_path}")
        
        # Extract metadata first
        metadata = self._extract_metadata(pdf_path)
        logger.info(f"Document has {metadata.total_pages} pages")
        
        # Extract content based on method
        if self.extraction_method == "auto":
            pages = self._auto_extract(pdf_path)
        elif self.extraction_method == "pypdf":
            pages = self._extract_with_pypdf(pdf_path)
        elif self.extraction_method == "pymupdf":
            pages = self._extract_with_pymupdf(pdf_path)
        elif self.extraction_method == "pdfplumber":
            pages = self._extract_with_pdfplumber(pdf_path)
        else:
            raise ValueError(f"Unknown extraction method: {self.extraction_method}")
        
        # Post-processing
        if self.remove_headers_footers:
            pages = self._remove_headers_footers(pages)
        
        if self.clean_text:
            pages = self._clean_pages(pages)
        
        logger.info(f"Successfully extracted {len(pages)} pages")
        
        return pages, metadata
    
    def _extract_metadata(self, pdf_path: str) -> DocumentMetadata:
        """Extract metadata from PDF"""
        try:
            # Use PyPDF for metadata
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Get basic info
                num_pages = len(pdf_reader.pages)
                file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # MB
                
                # Get metadata if available
                info = pdf_reader.metadata
                
                return DocumentMetadata(
                    filename=os.path.basename(pdf_path),
                    filepath=pdf_path,
                    total_pages=num_pages,
                    file_size_mb=file_size,
                    creation_date=info.get('/CreationDate', None) if info else None,
                    author=info.get('/Author', None) if info else None,
                    title=info.get('/Title', None) if info else None
                )
        except Exception as e:
            logger.error(f"Error extracting metadata: {e}")
            # Return minimal metadata
            return DocumentMetadata(
                filename=os.path.basename(pdf_path),
                filepath=pdf_path,
                total_pages=0,
                file_size_mb=0,
                creation_date=None,
                author=None,
                title=None
            )
    
    def _auto_extract(self, pdf_path: str) -> List[PageContent]:
        """
        Try multiple extraction methods and choose the best result
        
        Strategy:
        1. Try PyMuPDF (fastest, good quality)
        2. If text is sparse, try pdfplumber (better for complex layouts)
        3. Fallback to PyPDF
        """
        logger.info("Using auto extraction - trying multiple methods...")
        
        # Try PyMuPDF first
        try:
            pages = self._extract_with_pymupdf(pdf_path)
            avg_chars = sum(len(p.text) for p in pages) / len(pages) if pages else 0
            
            if avg_chars > 100:  # Good extraction
                logger.info(f"PyMuPDF successful (avg {avg_chars:.0f} chars/page)")
                return pages
        except Exception as e:
            logger.warning(f"PyMuPDF failed: {e}")
        
        # Try pdfplumber for complex layouts
        try:
            pages = self._extract_with_pdfplumber(pdf_path)
            avg_chars = sum(len(p.text) for p in pages) / len(pages) if pages else 0
            
            if avg_chars > 100:
                logger.info(f"pdfplumber successful (avg {avg_chars:.0f} chars/page)")
                return pages
        except Exception as e:
            logger.warning(f"pdfplumber failed: {e}")
        
        # Fallback to PyPDF
        logger.info("Falling back to PyPDF...")
        return self._extract_with_pypdf(pdf_path)
    
    def _extract_with_pypdf(self, pdf_path: str) -> List[PageContent]:
        """Extract using PyPDF (basic but reliable)"""
        pages = []
        
        with open(pdf_path, 'rb') as file:
            pdf_reader = pypdf.PdfReader(file)
            
            for page_num, page in enumerate(pdf_reader.pages, start=1):
                text = page.extract_text()
                
                pages.append(PageContent(
                    page_number=page_num,
                    text=text,
                    metadata={
                        "extraction_method": "pypdf",
                        "char_count": len(text)
                    }
                ))
        
        return pages
    
    def _extract_with_pymupdf(self, pdf_path: str) -> List[PageContent]:
        """Extract using PyMuPDF with progress tracking and safety checks"""
        pages = []
        
        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            logger.error(f"Could not open {pdf_path}: {e}")
            return []
        
        # Use tqdm for a progress bar
        print(f"   ⏳ Processing {len(doc)} pages...")
        
        for page_num in tqdm(range(len(doc)), desc="   Extracting", unit="page"):
            try:
                page = doc[page_num]
                
                # FIX 1: Pylance thinks this might not be a string. We force it to be one.
                raw_text = page.get_text("text")
                text = str(raw_text) if raw_text else ""
                
                # FIX 2: Pylance gets confused by .get_text("dict").
                # We declare 'page_data' as Any to tell Pylance: "Trust us, it's a dict"
                # This fixes the "slice" and "__getitem__" errors.
                try:
                    page_data: Any = page.get_text("dict")
                    blocks = page_data.get("blocks", [])
                    num_blocks = len(blocks)
                except Exception:
                    num_blocks = 0
                
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text=text,
                    metadata={
                        "extraction_method": "pymupdf",
                        "char_count": len(text),
                        "num_blocks": num_blocks
                    }
                ))
            except Exception as e:
                logger.warning(f"Skipping page {page_num + 1} due to error: {e}")
                continue
        
        doc.close()
        return pages
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> List[PageContent]:
        """Extract using pdfplumber (best for tables and complex layouts)"""
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text()
                
                # Try to extract tables
                tables = page.extract_tables()
                
                pages.append(PageContent(
                    page_number=page_num,
                    text=text or "",
                    metadata={
                        "extraction_method": "pdfplumber",
                        "char_count": len(text) if text else 0,
                        "num_tables": len(tables) if tables else 0
                    }
                ))
        
        return pages
    
    def _remove_headers_footers(self, pages: List[PageContent]) -> List[PageContent]:
        """
        Remove repeating headers and footers
        
        Strategy:
        1. Find lines that appear on most pages in same position
        2. Remove them from all pages
        """
        if len(pages) < 3:
            return pages  # Need multiple pages to detect patterns
        
        # Split each page into lines
        page_lines = [page.text.split('\n') for page in pages]
        
        # Find common first lines (headers)
        first_lines = [lines[0] if lines else "" for lines in page_lines]
        most_common_header = max(set(first_lines), key=first_lines.count)
        header_frequency = first_lines.count(most_common_header) / len(first_lines)
        
        # Find common last lines (footers)
        last_lines = [lines[-1] if lines else "" for lines in page_lines]
        most_common_footer = max(set(last_lines), key=last_lines.count)
        footer_frequency = last_lines.count(most_common_footer) / len(last_lines)
        
        # Remove if appears on >70% of pages
        cleaned_pages = []
        for page in pages:
            text = page.text
            
            if header_frequency > 0.7 and most_common_header:
                text = text.replace(most_common_header, "", 1)
            
            if footer_frequency > 0.7 and most_common_footer:
                # Remove from end
                if text.endswith(most_common_footer):
                    text = text[:-len(most_common_footer)]
            
            cleaned_pages.append(PageContent(
                page_number=page.page_number,
                text=text,
                metadata=page.metadata
            ))
        
        logger.info(f"Removed headers/footers (header freq: {header_frequency:.2f}, footer freq: {footer_frequency:.2f})")
        return cleaned_pages
    
    def _clean_pages(self, pages: List[PageContent]) -> List[PageContent]:
        """Clean and normalize text"""
        cleaned_pages = []
        
        for page in pages:
            text = page.text
            
            # Apply cleaning
            text = self._clean_text(text)
            
            cleaned_pages.append(PageContent(
                page_number=page.page_number,
                text=text,
                metadata=page.metadata
            ))
        
        return cleaned_pages
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text
        
        Aviation-specific cleaning:
        - Normalize whitespace
        - Fix common OCR errors
        - Preserve aviation acronyms
        - Fix hyphenation
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix hyphenation at line breaks
        text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
        
        # Normalize quotes
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(''', "'").replace(''', "'")
        
        # Remove multiple spaces
        text = re.sub(r' {2,}', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def load_directory(self, directory_path: str) -> Dict[str, Tuple[List[PageContent], DocumentMetadata]]:
        """
        Load all PDF files from a directory
        
        Args:
            directory_path: Path to directory containing PDFs
            
        Returns:
            Dictionary mapping filename to (pages, metadata)
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        pdf_files = list(Path(directory_path).glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {directory_path}")
            return {}
        
        logger.info(f"Found {len(pdf_files)} PDF files")
        
        results = {}
        for pdf_file in pdf_files:
            try:
                pages, metadata = self.load_pdf(str(pdf_file))
                results[metadata.filename] = (pages, metadata)
            except Exception as e:
                logger.error(f"Error loading {pdf_file}: {e}")
                continue
        
        return results


# Utility function for quick loading
def load_aviation_pdfs(directory: str = "data/raw") -> Dict:
    """
    Quick function to load all aviation PDFs
    
    Args:
        directory: Path to PDF directory
        
    Returns:
        Dictionary of loaded documents
    """
    loader = AviationPDFLoader(
        extraction_method="auto",
        remove_headers_footers=True,
        clean_text=True
    )
    
    return loader.load_directory(directory)


if __name__ == "__main__":
    """Test the PDF loader"""
    
    print("="*60)
    print("Testing Aviation PDF Loader")
    print("="*60)
    
    # Test loading a single PDF
    pdf_path = "data/raw/sample.pdf"  # Replace with your PDF
    
    if os.path.exists(pdf_path):
        loader = AviationPDFLoader()
        pages, metadata = loader.load_pdf(pdf_path)
        
        print(f"\n✅ Loaded: {metadata.filename}")
        print(f"   Pages: {metadata.total_pages}")
        print(f"   Size: {metadata.file_size_mb} MB")
        print(f"   Author: {metadata.author}")
        print(f"\nFirst page preview:")
        print(pages[0].text[:500] + "...")
    else:
        print(f"\n⚠️  Sample PDF not found at {pdf_path}")
        print("   Place a PDF in data/raw/ and update the path")
    
    # Test loading directory
    print("\n" + "="*60)
    print("Loading all PDFs from data/raw/")
    print("="*60)
    
    if os.path.exists("data/raw"):
        docs = load_aviation_pdfs("data/raw")
        
        print(f"\n✅ Loaded {len(docs)} documents:")
        for filename, (pages, metadata) in docs.items():
            print(f"\n📄 {filename}")
            print(f"   Pages: {len(pages)}")
            print(f"   Avg chars/page: {sum(len(p.text) for p in pages) / len(pages):.0f}")
    else:
        print("\n⚠️  data/raw/ directory not found")