"""
Advanced Document Processor for Scientific Papers
Handles intelligent chunking, section detection, and metadata extraction
"""

import os
import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging

import PyPDF2
from docx import Document as DocxDocument
import tiktoken
from dataclasses import dataclass

# Optional: better PDF text extraction fallback
try:
    import fitz  # PyMuPDF
    HAS_FITZ = True
except Exception:
    HAS_FITZ = False

@dataclass
class DocumentChunk:
    """Represents a chunk of document content with metadata"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: int = 0
    token_count: int = 0

class DocumentProcessor:
    """Advanced document processor with intelligent chunking and section detection"""
    
    def __init__(self, config):
        self.config = config
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Section patterns for scientific papers
        self.section_patterns = {
            'abstract': r'\b(abstract|summary)\b',
            'introduction': r'\b(introduction|background)\b',
            'methods': r'\b(methods?|methodology|experimental|materials)\b',
            'results': r'\b(results?|findings)\b',
            'discussion': r'\b(discussion|analysis)\b',
            'conclusion': r'\b(conclusions?|summary|future work)\b',
            'references': r'\b(references?|bibliography|citations?)\b',
            'acknowledgments': r'\b(acknowledgments?|acknowledgements?)\b'
        }
        
        # Author extraction patterns
        self.author_patterns = [
            r'([A-Z][a-z]+(?:\s+[A-Z]\.)?\s+[A-Z][a-z]+)',  # First Last or First M. Last
            r'([A-Z][a-z]+,\s*[A-Z]\.(?:\s*[A-Z]\.)?)',     # Last, F. or Last, F.M.
            r'([A-Z]\.(?:\s*[A-Z]\.)*\s+[A-Z][a-z]+)',      # F. Last or F.M. Last
        ]
        
        # Citation patterns
        self.citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023) or (2023)
            r'\[[^\]]*\d+[^\]]*\]',  # [1] or [Author, 2023]
        ]
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process a document and return intelligent chunks"""
        try:
            # Extract text and basic metadata
            text, metadata = self._extract_text_and_metadata(file_path)
            
            if not text.strip():
                self.logger.warning(f"No text extracted from {file_path}")
                return []
            
            # Generate document ID
            doc_id = self._generate_document_id(file_path, text)
            
            # Detect sections
            sections = self._detect_sections(text)
            
            # Extract additional metadata
            enhanced_metadata = self._extract_metadata(text, metadata, file_path)
            
            # Create intelligent chunks
            chunks = self._create_intelligent_chunks(
                text, doc_id, sections, enhanced_metadata
            )
            
            self.logger.info(f"Processed {file_path}: {len(chunks)} chunks, {len(sections)} sections")
            return chunks
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def _extract_text_and_metadata(self, file_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and basic metadata from various file formats"""
        file_path = Path(file_path)
        metadata = {
            'filename': file_path.name,
            'file_path': str(file_path),
            'file_size': file_path.stat().st_size if file_path.exists() else 0
        }
        
        if file_path.suffix.lower() == '.pdf':
            return self._extract_from_pdf(file_path, metadata)
        elif file_path.suffix.lower() == '.docx':
            return self._extract_from_docx(file_path, metadata)
        elif file_path.suffix.lower() in ['.txt', '.md']:
            return self._extract_from_text(file_path, metadata)
        else:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

    def _extract_from_pdf(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extract text from PDF with page tracking; fallback to PyMuPDF if needed"""
        text = ""
        page_texts = []
        extracted_pages = 0
        
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                metadata['total_pages'] = len(reader.pages)
                
                for page_num, page in enumerate(reader.pages, 1):
                    try:
                        page_text = page.extract_text() or ""
                    except Exception:
                        page_text = ""
                    if page_text.strip():
                        extracted_pages += 1
                    text += f"\n[PAGE {page_num}]\n{page_text}\n"
                    page_texts.append({
                        'page_number': page_num,
                        'text': page_text,
                        'char_start': len(text) - len(page_text) - len(f"\n[PAGE {page_num}]\n") - 1,
                        'char_end': len(text) - 1
                    })
                
                metadata['page_texts'] = page_texts
                
                # Extract PDF metadata
                if reader.metadata:
                    metadata.update({
                        'title': reader.metadata.get('/Title', ''),
                        'author': reader.metadata.get('/Author', ''),
                        'subject': reader.metadata.get('/Subject', ''),
                        'creator': reader.metadata.get('/Creator', ''),
                        'creation_date': reader.metadata.get('/CreationDate', '')
                    })
        except Exception as e:
            self.logger.error(f"Error extracting PDF {file_path} with PyPDF2: {str(e)}")
        
        # Fallback to PyMuPDF if PyPDF2 yielded little text
        try:
            if HAS_FITZ:
                total_pages = metadata.get('total_pages', 0) or 0
                # If less than 30% of pages yielded text, try fitz
                if total_pages > 0 and extracted_pages / total_pages < 0.3:
                    self.logger.info("PyPDF2 yielded little text; trying PyMuPDF fallback...")
                    return self._extract_from_pdf_with_pymupdf(file_path, metadata)
        except Exception as e:
            self.logger.warning(f"PyMuPDF fallback failed: {e}")
        
        return text, metadata

    def _extract_from_docx(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extract text from Word document"""
        text = ""
        
        try:
            doc = DocxDocument(file_path)
            
            for para in doc.paragraphs:
                text += para.text + "\n"
            
            # Extract document properties
            props = doc.core_properties
            metadata.update({
                'title': props.title or '',
                'author': props.author or '',
                'subject': props.subject or '',
                'created': props.created,
                'modified': props.modified
            })
            
        except Exception as e:
            self.logger.error(f"Error extracting DOCX {file_path}: {str(e)}")
            
        return text, metadata

    def _extract_from_text(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as file:
                text = file.read()
                
        return text, metadata

    def _extract_from_pdf_with_pymupdf(self, file_path: Path, metadata: Dict) -> Tuple[str, Dict]:
        """Fallback extractor using PyMuPDF for better OCR-less text extraction"""
        if not HAS_FITZ:
            return "", metadata
        text = ""
        page_texts = []
        try:
            doc = fitz.open(str(file_path))
            metadata['total_pages'] = len(doc)
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_text = page.get_text("text") or ""
                text += f"\n[PAGE {page_num+1}]\n{page_text}\n"
                page_texts.append({
                    'page_number': page_num + 1,
                    'text': page_text,
                    'char_start': len(text) - len(page_text) - len(f"\n[PAGE {page_num+1}]\n") - 1,
                    'char_end': len(text) - 1
                })
        except Exception as e:
            self.logger.error(f"Error extracting PDF {file_path} with PyMuPDF: {e}")
        finally:
            try:
                doc.close()
            except Exception:
                pass
        metadata['page_texts'] = page_texts
        return text, metadata

    def _detect_sections(self, text: str) -> Dict[str, List[Tuple[int, int]]]:
        """Detect sections in scientific papers using multiple strategies"""
        sections = {}
        text_lower = text.lower()
        
        # Strategy 1: Header patterns
        for section_name, pattern in self.section_patterns.items():
            matches = []
            for match in re.finditer(pattern, text_lower, re.IGNORECASE | re.MULTILINE):
                # Look for the end of this section (next section or end of text)
                start_pos = match.start()
                
                # Find the end by looking for the next section
                end_pos = len(text)
                for other_pattern in self.section_patterns.values():
                    if other_pattern != pattern:
                        next_match = re.search(other_pattern, text_lower[start_pos + 100:], re.IGNORECASE)
                        if next_match:
                            potential_end = start_pos + 100 + next_match.start()
                            end_pos = min(end_pos, potential_end)
                
                matches.append((start_pos, end_pos))
            
            if matches:
                sections[section_name] = matches
        
        # Strategy 2: Numbered sections (1. Introduction, 2. Methods, etc.)
        numbered_pattern = r'^(\d+\.?\d*)\s*([A-Z][A-Za-z\s]+)$'
        for match in re.finditer(numbered_pattern, text, re.MULTILINE):
            section_title = match.group(2).strip().lower()
            for section_name, pattern in self.section_patterns.items():
                if re.search(pattern, section_title):
                    start_pos = match.start()
                    # Find next numbered section or end
                    next_numbered = re.search(numbered_pattern, text[start_pos + 10:], re.MULTILINE)
                    end_pos = start_pos + 10 + next_numbered.start() if next_numbered else len(text)
                    
                    if section_name not in sections:
                        sections[section_name] = []
                    sections[section_name].append((start_pos, end_pos))
        
        return sections

    def _extract_metadata(self, text: str, base_metadata: Dict, file_path: str) -> Dict[str, Any]:
        """Extract enhanced metadata from document content"""
        metadata = base_metadata.copy()
        
        # Extract title (first significant line or from metadata)
        if not metadata.get('title'):
            lines = text.split('\n')
            for line in lines[:10]:  # Check first 10 lines
                line = line.strip()
                if len(line) > 10 and not line.isupper() and not line.islower():
                    metadata['title'] = line
                    break
        
        # Extract authors using patterns
        authors = set()
        for pattern in self.author_patterns:
            matches = re.findall(pattern, text[:2000])  # Check first 2000 chars
            authors.update(matches)
        
        metadata['authors'] = list(authors)
        
        # Extract year (fixed regex to capture full year)
        year_pattern = r'\b(?:19|20)\d{2}\b'  # Non-capturing group to get full year
        years = re.findall(year_pattern, text[:3000])
        if years:
            # Get the most recent year that's not in the future
            current_year = 2025  # Update this as needed
            try:
                valid_years = [int(y) for y in years if isinstance(y, str) and y.isdigit() and int(y) <= current_year]
                metadata['year'] = max(valid_years) if valid_years else None
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Error processing years {years}: {e}")
                metadata['year'] = None
        
        # Extract journal/conference (look for common patterns)
        journal_patterns = [
            r'(?:published in|appears in|journal of|proceedings of)\s+([^.\n]+)',
            r'([A-Z][a-z\s]+Journal[^.\n]*)',
            r'([A-Z][a-z\s]+Conference[^.\n]*)'
        ]
        
        for pattern in journal_patterns:
            match = re.search(pattern, text[:2000], re.IGNORECASE)
            if match:
                metadata['journal'] = match.group(1).strip()
                break
        
        # Count citations
        citation_count = 0
        for pattern in self.citation_patterns:
            citation_count += len(re.findall(pattern, text))
        metadata['citation_count'] = citation_count
        
        # Extract keywords (simple approach)
        # Look for explicit keywords section
        keywords_match = re.search(r'keywords?:\s*([^.\n]+)', text[:3000], re.IGNORECASE)
        if keywords_match:
            keywords = [k.strip() for k in keywords_match.group(1).split(',')]
            metadata['keywords'] = keywords
        
        return metadata

    def _create_intelligent_chunks(
        self, 
        text: str, 
        doc_id: str, 
        sections: Dict[str, List[Tuple[int, int]]], 
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Create intelligent chunks that preserve context and meaning"""
        chunks = []
        chunk_index = 0
        
        # Strategy 1: Section-based chunking (preferred)
        if sections:
            for section_name, section_ranges in sections.items():
                for start_pos, end_pos in section_ranges:
                    section_text = text[start_pos:end_pos].strip()
                    if len(section_text) < 50:  # Skip tiny sections
                        continue
                    
                    # Split large sections into smaller chunks
                    section_chunks = self._split_text_intelligently(
                        section_text, 
                        max_tokens=self.config.max_tokens_per_chunk
                    )
                    
                    for chunk_text in section_chunks:
                        page_num = self._get_page_number(start_pos, metadata.get('page_texts', []))
                        
                        chunk = DocumentChunk(
                            content=chunk_text,
                            metadata=metadata.copy(),
                            chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                            document_id=doc_id,
                            section=section_name,
                            page_number=page_num,
                            chunk_index=chunk_index,
                            token_count=len(self.encoding.encode(chunk_text))
                        )
                        chunks.append(chunk)
                        chunk_index += 1
        
        # Strategy 2: Fallback to intelligent text splitting
        if not chunks:
            text_chunks = self._split_text_intelligently(text, self.config.max_tokens_per_chunk)
            
            for chunk_text in text_chunks:
                chunk = DocumentChunk(
                    content=chunk_text,
                    metadata=metadata.copy(),
                    chunk_id=self._generate_chunk_id(doc_id, chunk_index),
                    document_id=doc_id,
                    chunk_index=chunk_index,
                    token_count=len(self.encoding.encode(chunk_text))
                )
                chunks.append(chunk)
                chunk_index += 1
        
        return chunks

    def _split_text_intelligently(self, text: str, max_tokens: int = 500) -> List[str]:
        """Split text into chunks while preserving semantic boundaries"""
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # Check if adding this paragraph would exceed token limit
            test_chunk = current_chunk + "\n\n" + paragraph if current_chunk else paragraph
            token_count = len(self.encoding.encode(test_chunk))
            
            if token_count <= max_tokens:
                current_chunk = test_chunk
            else:
                # Save current chunk if it has content
                if current_chunk:
                    chunks.append(current_chunk.strip())
                
                # Handle oversized paragraphs
                if len(self.encoding.encode(paragraph)) > max_tokens:
                    # Split by sentences
                    sentences = re.split(r'[.!?]+', paragraph)
                    temp_chunk = ""
                    
                    for sentence in sentences:
                        sentence = sentence.strip()
                        if not sentence:
                            continue
                        
                        test_sentence = temp_chunk + ". " + sentence if temp_chunk else sentence
                        if len(self.encoding.encode(test_sentence)) <= max_tokens:
                            temp_chunk = test_sentence
                        else:
                            if temp_chunk:
                                chunks.append(temp_chunk.strip())
                            temp_chunk = sentence
                    
                    if temp_chunk:
                        current_chunk = temp_chunk
                    else:
                        current_chunk = ""
                else:
                    current_chunk = paragraph
        
        # Add the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks

    def _get_page_number(self, char_position: int, page_texts: List[Dict]) -> Optional[int]:
        """Get page number for a character position"""
        for page_info in page_texts:
            if page_info['char_start'] <= char_position <= page_info['char_end']:
                return page_info['page_number']
        return None

    def _generate_document_id(self, file_path: str, text: str) -> str:
        """Generate unique document ID"""
        content_hash = hashlib.md5(text.encode()).hexdigest()[:8]
        filename = Path(file_path).stem
        return f"{filename}_{content_hash}"

    def _generate_chunk_id(self, doc_id: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        return f"{doc_id}_chunk_{chunk_index}"
