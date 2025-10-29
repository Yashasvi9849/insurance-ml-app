"""
PDF Extractor Module
Handles text extraction from PDF documents with OCR fallback
"""

import logging
from pathlib import Path
from typing import Dict, Optional, List
import re

# PDF Processing libraries
import pdfplumber
import PyPDF2
from pdf2image import convert_from_path
import pytesseract

# Data structures
from dataclasses import dataclass, asdict
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExtractedDocument:
    """Data structure for extracted PDF content"""
    filename: str
    extraction_method: str  # 'pdfplumber', 'pypdf2', or 'ocr'
    full_text: str
    page_count: int
    extracted_at: str
    file_size_kb: float
    tables: List[Dict] = None  # For tabular data
    metadata: Dict = None
    extraction_confidence: float = 1.0  # Lower for OCR
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)


class PDFExtractor:
    """
    Extracts text content from PDF documents
    Tries multiple methods: pdfplumber → PyPDF2 → OCR
    """
    
    def __init__(self, use_ocr: bool = True, ocr_language: str = 'eng'):
        """
        Initialize PDF extractor
        
        Args:
            use_ocr: Whether to use OCR for image-based PDFs
            ocr_language: Tesseract language code (eng, spa, etc.)
        """
        self.use_ocr = use_ocr
        self.ocr_language = ocr_language
        self.stats = {
            'total_processed': 0,
            'pdfplumber_success': 0,
            'pypdf2_success': 0,
            'ocr_used': 0,
            'failed': 0
        }
        
    def extract_pdf(self, pdf_path: str) -> Optional[ExtractedDocument]:
        """
        Main extraction method - tries multiple approaches
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            ExtractedDocument object or None if failed
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            logger.error(f" File not found: {pdf_path}")
            return None
        
        logger.info(f"Processing: {pdf_path.name}")
        
        # Try extraction methods in order of preference
        result = None
        
        # Method 1: pdfplumber (best for structured PDFs)
        result = self._extract_with_pdfplumber(pdf_path)
        if result and len(result.full_text.strip()) > 100:
            self.stats['pdfplumber_success'] += 1
            logger.info(f" Extracted with pdfplumber: {len(result.full_text)} chars")
            return result
        
        # Method 2: PyPDF2 (fallback)
        result = self._extract_with_pypdf2(pdf_path)
        if result and len(result.full_text.strip()) > 100:
            self.stats['pypdf2_success'] += 1
            logger.info(f"Extracted with PyPDF2: {len(result.full_text)} chars")
            return result
        
        # Method 3: OCR (for scanned/handwritten documents)
        if self.use_ocr:
            logger.warning(f" Attempting OCR for: {pdf_path.name}")
            result = self._extract_with_ocr(pdf_path)
            if result and len(result.full_text.strip()) > 50:
                self.stats['ocr_used'] += 1
                logger.info(f" Extracted with OCR: {len(result.full_text)} chars")
                return result
        
        # All methods failed
        self.stats['failed'] += 1
        logger.error(f"Failed to extract text from: {pdf_path.name}")
        return None
    
    def _extract_with_pdfplumber(self, pdf_path: Path) -> Optional[ExtractedDocument]:
        """
        Extract text using pdfplumber (best for modern PDFs)
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            ExtractedDocument or None
        """
        try:
            text_content = []
            tables = []
            page_count = 0
            
            with pdfplumber.open(pdf_path) as pdf:
                page_count = len(pdf.pages)
                
                for page_num, page in enumerate(pdf.pages, 1):
                    # Extract text
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"\n--- Page {page_num} ---\n")
                        text_content.append(page_text)
                    
                    # Extract tables
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table_idx, table in enumerate(page_tables):
                            tables.append({
                                'page': page_num,
                                'table_index': table_idx,
                                'data': table
                            })
            
            full_text = "\n".join(text_content)
            
            if not full_text.strip():
                return None
            
            return ExtractedDocument(
                filename=pdf_path.name,
                extraction_method='pdfplumber',
                full_text=full_text,
                page_count=page_count,
                extracted_at=datetime.now().isoformat(),
                file_size_kb=pdf_path.stat().st_size / 1024,
                tables=tables if tables else None,
                extraction_confidence=1.0
            )
            
        except Exception as e:
            logger.debug(f"pdfplumber failed for {pdf_path.name}: {e}")
            return None
    
    def _extract_with_pypdf2(self, pdf_path: Path) -> Optional[ExtractedDocument]:
        """
        Extract text using PyPDF2 (fallback method)
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            ExtractedDocument or None
        """
        try:
            text_content = []
            
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    page_text = page.extract_text()
                    if page_text:
                        text_content.append(f"\n--- Page {page_num} ---\n")
                        text_content.append(page_text)
            
            full_text = "\n".join(text_content)
            
            if not full_text.strip():
                return None
            
            return ExtractedDocument(
                filename=pdf_path.name,
                extraction_method='pypdf2',
                full_text=full_text,
                page_count=page_count,
                extracted_at=datetime.now().isoformat(),
                file_size_kb=pdf_path.stat().st_size / 1024,
                extraction_confidence=0.9
            )
            
        except Exception as e:
            logger.debug(f"PyPDF2 failed for {pdf_path.name}: {e}")
            return None
    
    def _extract_with_ocr(self, pdf_path: Path) -> Optional[ExtractedDocument]:
        """
        Extract text using OCR (for scanned/image-based PDFs)
        
        Args:
            pdf_path: Path to PDF
            
        Returns:
            ExtractedDocument or None
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path, dpi=300)
            page_count = len(images)
            
            text_content = []
            
            for page_num, image in enumerate(images, 1):
                logger.debug(f"Running OCR on page {page_num}/{page_count}")
                
                # Run OCR
                page_text = pytesseract.image_to_string(
                    image, 
                    lang=self.ocr_language,
                    config='--psm 6'  # Assume uniform text block
                )
                
                if page_text.strip():
                    text_content.append(f"\n--- Page {page_num} ---\n")
                    text_content.append(page_text)
            
            full_text = "\n".join(text_content)
            
            if not full_text.strip():
                return None
            
            return ExtractedDocument(
                filename=pdf_path.name,
                extraction_method='ocr',
                full_text=full_text,
                page_count=page_count,
                extracted_at=datetime.now().isoformat(),
                file_size_kb=pdf_path.stat().st_size / 1024,
                extraction_confidence=0.7  # Lower confidence for OCR
            )
            
        except Exception as e:
            logger.error(f"OCR failed for {pdf_path.name}: {e}")
            return None
    
    def batch_extract(self, pdf_directory: str, output_directory: str = None) -> List[ExtractedDocument]:
        """
        Extract text from all PDFs in a directory
        
        Args:
            pdf_directory: Directory containing PDFs
            output_directory: Optional directory to save extracted text
            
        Returns:
            List of ExtractedDocument objects
        """
        pdf_dir = Path(pdf_directory)
        pdf_files = list(pdf_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {pdf_directory}")
            return []
        
        logger.info(f"📚 Found {len(pdf_files)} PDF files")
        logger.info("=" * 60)
        
        results = []
        
        for pdf_file in pdf_files:
            result = self.extract_pdf(pdf_file)
            if result:
                results.append(result)
                self.stats['total_processed'] += 1
                
                # Optionally save extracted text
                if output_directory:
                    self._save_extracted_text(result, output_directory)
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _save_extracted_text(self, document: ExtractedDocument, output_dir: str):
        """Save extracted text to file"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        txt_filename = Path(document.filename).stem + ".txt"
        txt_path = output_path / txt_filename
        
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"Filename: {document.filename}\n")
            f.write(f"Method: {document.extraction_method}\n")
            f.write(f"Pages: {document.page_count}\n")
            f.write(f"Extracted: {document.extracted_at}\n")
            f.write("=" * 60 + "\n\n")
            f.write(document.full_text)
    
    def _print_summary(self):
        """Print extraction statistics"""
        logger.info("=" * 60)
        logger.info("📊 EXTRACTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total processed: {self.stats['total_processed']}")
        logger.info(f" pdfplumber: {self.stats['pdfplumber_success']}")
        logger.info(f"PyPDF2: {self.stats['pypdf2_success']}")
        logger.info(f"  OCR used: {self.stats['ocr_used']}")
        logger.info(f" Failed: {self.stats['failed']}")
        logger.info("=" * 60)


# Example usage
if __name__ == "__main__":
    # Test extraction
    extractor = PDFExtractor(use_ocr=True)
    
    # Single file
    # result = extractor.extract_pdf("data/raw_pdfs/sample.pdf")
    # if result:
    #     print(result.full_text[:500])
    
    # Batch extraction
    documents = extractor.batch_extract(
        pdf_directory="data/raw_pdfs/my_docs",
        output_directory="data/processed/extracted_text"
    )
    
    print(f"\n Extracted {len(documents)} documents")