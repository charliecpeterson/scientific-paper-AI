"""
Enhanced Document Processor with AI-Powered Metadata Extraction
Intelligently extracts and verifies paper titles, authors, and content structure
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
from dataclasses import dataclass

from langchain_ollama import OllamaLLM
import tiktoken

from .document_processor import DocumentProcessor, DocumentChunk
from .multimodal_processor import MultimodalProcessor


@dataclass
class EnhancedDocumentMetadata:
    """Enhanced metadata with AI-verified information"""
    # Original file info
    filename: str
    file_path: str
    file_size: int
    
    # AI-extracted paper information
    paper_title: str  # AI-verified actual paper title
    paper_title_confidence: float  # 0-1 confidence score
    authors: List[str]  # Properly parsed author list
    publication_year: Optional[int]
    journal_or_venue: Optional[str]
    
    # Document structure
    abstract: Optional[str]
    sections: List[str]  # List of detected section names
    total_pages: int
    total_chunks: int
    
    # Content analysis
    research_domain: Optional[str]  # AI-identified research area
    key_topics: List[str]  # AI-extracted key topics
    document_type: str  # journal_article, conference_paper, thesis, etc.
    
    # Quality metrics
    processing_quality: str  # high, medium, low
    extraction_errors: List[str]


class EnhancedDocumentProcessor:
    """Advanced document processor with AI-powered metadata extraction"""
    
    def __init__(self, config):
        self.config = config
        self.basic_processor = DocumentProcessor(config)
        self.multimodal_processor = MultimodalProcessor(config)
        self.encoding = tiktoken.get_encoding("cl100k_base")
        
        # Initialize AI model for metadata extraction
        try:
            self.metadata_extractor = OllamaLLM(
                model=config.llm_model,  # Use the main model
                base_url="http://localhost:11434",
                temperature=0.1  # Low temperature for consistent extraction
            )
            self.has_ai = True
        except Exception as e:
            logging.warning(f"AI model not available for metadata extraction: {e}")
            self.has_ai = False
            
        self.logger = logging.getLogger(__name__)

    def process_document_enhanced(self, file_path: str) -> Tuple[List[DocumentChunk], EnhancedDocumentMetadata]:
        """Process document with enhanced AI-powered metadata extraction"""
        
        file_path = Path(file_path)
        
        try:
            # Step 1: Basic document processing
            basic_chunks = self.basic_processor.process_document(str(file_path))
            
            if not basic_chunks:
                raise ValueError("No content extracted from document")
            
            # Step 2: Extract full text for AI analysis
            full_text = self._extract_full_text_for_analysis(basic_chunks)
            
            # Step 3: AI-powered metadata extraction
            enhanced_metadata = self._extract_enhanced_metadata(file_path, full_text, basic_chunks)
            
            # Step 4: Update chunks with enhanced metadata
            enhanced_chunks = self._update_chunks_with_metadata(basic_chunks, enhanced_metadata)
            
            # Step 5: Try multimodal processing if available
            if file_path.suffix.lower() == '.pdf':
                try:
                    enhanced_metadata = self._add_multimodal_insights(file_path, enhanced_metadata)
                except Exception as e:
                    self.logger.warning(f"Multimodal processing failed: {e}")
            
            self.logger.info(f"Enhanced processing complete for {file_path.name}")
            self.logger.info(f"Extracted title: '{enhanced_metadata.paper_title}' (confidence: {enhanced_metadata.paper_title_confidence:.2f})")
            
            return enhanced_chunks, enhanced_metadata
            
        except Exception as e:
            self.logger.error(f"Enhanced processing failed for {file_path}: {e}")
            # Fallback to basic processing
            return self._fallback_processing(file_path)

    def _extract_full_text_for_analysis(self, chunks: List[DocumentChunk]) -> str:
        """Extract and prepare full text for AI analysis"""
        # Combine all chunks but limit to first ~8000 tokens for analysis
        full_text = ""
        token_count = 0
        max_tokens = 8000
        
        for chunk in chunks:
            chunk_tokens = len(self.encoding.encode(chunk.content))
            if token_count + chunk_tokens > max_tokens:
                break
            full_text += chunk.content + "\n\n"
            token_count += chunk_tokens
            
        return full_text

    def _extract_enhanced_metadata(self, file_path: Path, full_text: str, chunks: List[DocumentChunk]) -> EnhancedDocumentMetadata:
        """Use AI to extract and verify comprehensive metadata"""
        
        # Get basic metadata from first chunk
        basic_meta = chunks[0].metadata if chunks else {}
        
        # Initialize enhanced metadata with defaults
        enhanced_meta = EnhancedDocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            paper_title=basic_meta.get('title', file_path.stem),
            paper_title_confidence=0.5,  # Default confidence
            authors=basic_meta.get('authors', []),
            publication_year=basic_meta.get('year'),
            journal_or_venue=basic_meta.get('journal'),
            abstract=None,
            sections=[],
            total_pages=basic_meta.get('total_pages', 1),
            total_chunks=len(chunks),
            research_domain=None,
            key_topics=[],
            document_type="unknown",
            processing_quality="medium",
            extraction_errors=[]
        )
        
        if not self.has_ai:
            enhanced_meta.processing_quality = "basic"
            return enhanced_meta
        
        try:
            # AI-powered metadata extraction
            ai_metadata = self._ai_extract_metadata(full_text, file_path.name)
            
            if ai_metadata:
                # Update with AI-extracted information
                enhanced_meta.paper_title = ai_metadata.get('title', enhanced_meta.paper_title)
                enhanced_meta.paper_title_confidence = ai_metadata.get('title_confidence', 0.8)
                enhanced_meta.authors = ai_metadata.get('authors', enhanced_meta.authors)
                
                # Handle year with proper type validation
                ai_year = ai_metadata.get('year')
                if ai_year is not None:
                    try:
                        enhanced_meta.publication_year = int(ai_year) if isinstance(ai_year, (str, int, float)) else enhanced_meta.publication_year
                    except (ValueError, TypeError):
                        enhanced_meta.publication_year = enhanced_meta.publication_year
                
                enhanced_meta.journal_or_venue = ai_metadata.get('venue', enhanced_meta.journal_or_venue)
                enhanced_meta.abstract = ai_metadata.get('abstract')
                enhanced_meta.research_domain = ai_metadata.get('research_domain')
                enhanced_meta.key_topics = ai_metadata.get('key_topics', [])
                enhanced_meta.document_type = ai_metadata.get('document_type', 'journal_article')
                enhanced_meta.processing_quality = "high"
                
        except Exception as e:
            self.logger.error(f"AI metadata extraction failed: {e}")
            enhanced_meta.extraction_errors.append(f"AI extraction failed: {str(e)}")
            enhanced_meta.processing_quality = "medium"
        
        # Extract sections from chunks
        sections = set()
        for chunk in chunks:
            if chunk.section:
                sections.add(chunk.section)
        enhanced_meta.sections = list(sections)
        
        return enhanced_meta

    def _ai_extract_metadata(self, text: str, filename: str) -> Optional[Dict[str, Any]]:
        """Use AI to intelligently extract paper metadata"""
        
        prompt = f"""
You are an expert scientific paper analyzer. Analyze the following document text and extract precise metadata.

FILENAME: {filename}

DOCUMENT TEXT:
{text[:6000]}  

Please extract the following information in JSON format:

1. "title": The actual paper title (NOT the filename). Look for the main title at the beginning of the document.
2. "title_confidence": Float 0-1 indicating how confident you are in the title extraction
3. "authors": List of author names in proper format (e.g., ["John Smith", "Jane Doe"])
4. "year": Publication year as integer (if found)
5. "venue": Journal name or conference name (if found)
6. "abstract": The paper's abstract or summary (if found)
7. "research_domain": Primary research area (e.g., "Machine Learning", "Chemistry", "Physics")
8. "key_topics": List of 3-5 key topics/keywords from the paper
9. "document_type": One of ["journal_article", "conference_paper", "thesis", "preprint", "technical_report", "other"]

IMPORTANT RULES:
- The title should be the ACTUAL paper title, not the filename
- If you can't find a clear title, use the most prominent heading
- Only include information you can confidently extract
- Use null for missing information
- Be precise with author names (avoid titles like "Dr." or "Prof.")

Return ONLY the JSON object, no other text:
"""
        
        try:
            response = self.metadata_extractor.invoke(prompt)
            
            # Parse JSON response
            # Clean up response in case there's extra text
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                return json.loads(json_str)
            else:
                self.logger.warning("AI response not in JSON format")
                return None
                
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse AI response as JSON: {e}")
            return None
        except Exception as e:
            self.logger.error(f"AI metadata extraction error: {e}")
            return None

    def _update_chunks_with_metadata(self, chunks: List[DocumentChunk], enhanced_meta: EnhancedDocumentMetadata) -> List[DocumentChunk]:
        """Update chunk metadata with enhanced information"""
        
        for chunk in chunks:
            # Update with enhanced metadata
            chunk.metadata.update({
                'paper_title': enhanced_meta.paper_title,
                'title_confidence': enhanced_meta.paper_title_confidence,
                'enhanced_authors': enhanced_meta.authors,
                'research_domain': enhanced_meta.research_domain,
                'key_topics': enhanced_meta.key_topics,
                'document_type': enhanced_meta.document_type,
                'processing_quality': enhanced_meta.processing_quality,
                'publication_year': enhanced_meta.publication_year,
                'venue': enhanced_meta.journal_or_venue
            })
            
            # Keep original title as fallback
            if 'original_title' not in chunk.metadata:
                chunk.metadata['original_title'] = chunk.metadata.get('title', enhanced_meta.filename)
            
            # Use enhanced title as primary
            chunk.metadata['title'] = enhanced_meta.paper_title
            
        return chunks

    def _add_multimodal_insights(self, file_path: Path, enhanced_meta: EnhancedDocumentMetadata) -> EnhancedDocumentMetadata:
        """Add insights from multimodal processing if available"""
        
        try:
            _, mm_metadata, mm_chunks = self.multimodal_processor.process_document_multimodal(file_path)
            
            # Add multimodal statistics
            enhanced_meta.extraction_errors.extend(mm_metadata.get('processing_errors', []))
            
            # Update processing quality based on multimodal success
            if mm_metadata.get('images_extracted', 0) > 0 or mm_metadata.get('tables_extracted', 0) > 0:
                enhanced_meta.processing_quality = "high_multimodal"
                
        except Exception as e:
            enhanced_meta.extraction_errors.append(f"Multimodal processing failed: {str(e)}")
            
        return enhanced_meta

    def _fallback_processing(self, file_path: Path) -> Tuple[List[DocumentChunk], EnhancedDocumentMetadata]:
        """Fallback to basic processing when enhanced processing fails"""
        
        chunks = self.basic_processor.process_document(str(file_path))
        
        basic_meta = chunks[0].metadata if chunks else {}
        enhanced_meta = EnhancedDocumentMetadata(
            filename=file_path.name,
            file_path=str(file_path),
            file_size=file_path.stat().st_size if file_path.exists() else 0,
            paper_title=basic_meta.get('title', file_path.stem),
            paper_title_confidence=0.3,
            authors=basic_meta.get('authors', []),
            publication_year=basic_meta.get('year'),
            journal_or_venue=basic_meta.get('journal'),
            abstract=None,
            sections=[],
            total_pages=basic_meta.get('total_pages', 1),
            total_chunks=len(chunks),
            research_domain=None,
            key_topics=[],
            document_type="unknown",
            processing_quality="basic_fallback",
            extraction_errors=["Enhanced processing failed, using basic fallback"]
        )
        
        return chunks, enhanced_meta

    def get_document_display_name(self, metadata: Dict[str, Any]) -> str:
        """Generate a user-friendly display name for a document"""
        
        paper_title = metadata.get('paper_title', metadata.get('title', ''))
        filename = metadata.get('filename', 'Unknown Document')
        confidence = metadata.get('title_confidence', 0.0)
        
        if paper_title and confidence > 0.6:
            return paper_title
        elif paper_title and confidence > 0.3:
            return f"{paper_title} [?]"  # Indicate uncertainty
        else:
            return filename  # Fallback to filename
    
    def get_document_summary(self, metadata: Dict[str, Any]) -> str:
        """Generate a summary description for a document"""
        
        parts = []
        
        # Title and confidence indicator
        title = self.get_document_display_name(metadata)
        confidence = metadata.get('title_confidence', 0.0)
        if confidence < 0.6:
            parts.append(f"📄 **{title}**")
        else:
            parts.append(f"📑 **{title}**")
        
        # Authors
        authors = metadata.get('enhanced_authors', metadata.get('authors', []))
        if authors:
            if len(authors) <= 3:
                parts.append(f"*by {', '.join(authors)}*")
            else:
                parts.append(f"*by {', '.join(authors[:2])} et al.*")
        
        # Year and venue
        year = metadata.get('publication_year')
        venue = metadata.get('venue')
        if year and venue:
            parts.append(f"({year}, {venue})")
        elif year:
            parts.append(f"({year})")
        elif venue:
            parts.append(f"({venue})")
        
        # Research domain
        domain = metadata.get('research_domain')
        if domain:
            parts.append(f"🔬 *{domain}*")
        
        return " ".join(parts)
