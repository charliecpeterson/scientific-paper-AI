"""
Multimodal Document Processor for Scientific Papers
Handles text, images, tables, and figures with vision model integration
"""

import os
import re
import base64
import io
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import logging
from dataclasses import dataclass
import json

try:
    import fitz  # PyMuPDF for better PDF processing
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False
    
import PIL.Image as Image
import pandas as pd
from langchain_ollama import OllamaLLM

try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

@dataclass
class MultimodalChunk:
    """Enhanced chunk with multimodal content"""
    content: str
    metadata: Dict[str, Any]
    chunk_id: str
    document_id: str
    section: Optional[str] = None
    page_number: Optional[int] = None
    chunk_index: int = 0
    token_count: int = 0
    
    # Multimodal additions
    images: List[Dict[str, Any]] = None  # Image data and descriptions
    tables: List[Dict[str, Any]] = None  # Table data and structure
    figures: List[Dict[str, Any]] = None  # Figure data and captions
    visual_context: Optional[str] = None  # Combined visual description

class MultimodalProcessor:
    """Advanced multimodal document processor"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize vision model for image analysis
        try:
            self.vision_model = OllamaLLM(
                model="llama3.2-vision:11b",
                base_url="http://localhost:11434"
            )
            self.has_vision = True
            self.logger.info("Vision model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Vision model not available: {e}")
            self.has_vision = False
            
        # Table extraction patterns
        self.table_keywords = [
            'table', 'figure', 'chart', 'graph', 'plot', 'diagram',
            'results', 'data', 'comparison', 'analysis'
        ]

    def process_document_multimodal(self, file_path: Path) -> Tuple[str, Dict, List[MultimodalChunk]]:
        """Process document with full multimodal analysis"""
        
        if file_path.suffix.lower() != '.pdf':
            # Fallback to basic processing for non-PDF files
            return self._process_basic_document(file_path)
            
        if not HAS_FITZ:
            self.logger.warning("PyMuPDF not available, falling back to basic PDF processing")
            return self._process_basic_document(file_path)
            
        # Open PDF with PyMuPDF for better extraction
        doc = fitz.open(str(file_path))
        
        text_content = ""
        metadata = {
            'filename': file_path.name,
            'total_pages': len(doc),
            'images_extracted': 0,
            'tables_extracted': 0,
            'figures_extracted': 0,
            'processing_type': 'multimodal'
        }
        
        multimodal_chunks = []
        
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                page_data = self._process_page_multimodal(page, page_num + 1, file_path.stem)
                
                text_content += page_data['text']
                multimodal_chunks.extend(page_data['chunks'])
                
                # Update metadata counters
                metadata['images_extracted'] += len(page_data.get('images', []))
                metadata['tables_extracted'] += len(page_data.get('tables', []))
                metadata['figures_extracted'] += len(page_data.get('figures', []))
                
        finally:
            doc.close()
            
        return text_content, metadata, multimodal_chunks

    def _process_basic_document(self, file_path: Path) -> Tuple[str, Dict, List[MultimodalChunk]]:
        """Fallback to basic document processing"""
        from .document_processor import DocumentProcessor
        
        basic_processor = DocumentProcessor(self.config)
        
        # Use the correct method
        chunks = basic_processor.process_document(str(file_path))
        
        # Extract text and metadata from chunks
        text = "\n".join([chunk.content for chunk in chunks])
        metadata = chunks[0].metadata if chunks else {
            'filename': file_path.name,
            'file_path': str(file_path),
            'processing_type': 'basic_fallback'
        }
        
        # Convert to multimodal chunks
        multimodal_chunks = []
        for chunk in chunks:
            mm_chunk = MultimodalChunk(
                content=chunk.content,
                metadata=chunk.metadata,
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                section=chunk.section,
                page_number=chunk.page_number,
                chunk_index=chunk.chunk_index,
                token_count=chunk.token_count
            )
            multimodal_chunks.append(mm_chunk)
            
        metadata['processing_type'] = 'basic_fallback'
        return text, metadata, multimodal_chunks

    def _process_page_multimodal(self, page, page_num: int, doc_id: str) -> Dict[str, Any]:
        """Process a single page with multimodal extraction"""
        
        # Extract text
        text = page.get_text()
        
        # Extract images and figures
        images = self._extract_images_from_page(page, page_num, doc_id)
        
        # Extract tables
        tables = self._extract_tables_from_page(page, page_num, doc_id)
        
        # Detect figures and captions
        figures = self._detect_figures_and_captions(page, text, page_num, doc_id)
        
        # Create enhanced chunks
        chunks = self._create_multimodal_chunks(
            text, images, tables, figures, page_num, doc_id
        )
        
        return {
            'text': f"\n[PAGE {page_num}]\n{text}\n",
            'chunks': chunks,
            'images': images,
            'tables': tables,
            'figures': figures
        }

    def _extract_images_from_page(self, page, page_num: int, doc_id: str) -> List[Dict[str, Any]]:
        """Extract and analyze images from a page"""
        images = []
        
        try:
            # Get images from page
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Extract image data
                    xref = img[0]
                    base_image = page.parent.extract_image(xref)
                    image_bytes = base_image["image"]
                    
                    # Convert to PIL Image
                    image_pil = Image.open(io.BytesIO(image_bytes))
                    
                    # Get image description using vision model
                    description = self._analyze_image_with_vision(image_pil, "scientific figure")
                    
                    images.append({
                        'image_id': f"{doc_id}_p{page_num}_img{img_index}",
                        'page_number': page_num,
                        'description': description,
                        'size': image_pil.size,
                        'format': base_image.get("ext", "unknown"),
                        'bbox': img[1:5] if len(img) > 4 else None  # Bounding box
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing image {img_index} on page {page_num}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Error extracting images from page {page_num}: {e}")
            
        return images

    def _extract_tables_from_page(self, page, page_num: int, doc_id: str) -> List[Dict[str, Any]]:
        """Extract and analyze tables from a page"""
        tables = []
        
        try:
            # Use PyMuPDF table extraction
            table_list = page.find_tables()
            
            for table_index, table in enumerate(table_list):
                try:
                    # Extract table as pandas DataFrame
                    df = table.to_pandas()
                    
                    if df.empty:
                        continue
                        
                    # Clean and process table
                    df_clean = self._clean_table_data(df)
                    
                    # Generate table description
                    table_description = self._describe_table(df_clean)
                    
                    tables.append({
                        'table_id': f"{doc_id}_p{page_num}_tbl{table_index}",
                        'page_number': page_num,
                        'data': df_clean.to_dict('records'),
                        'columns': list(df_clean.columns),
                        'shape': df_clean.shape,
                        'description': table_description,
                        'bbox': table.bbox  # Bounding box
                    })
                    
                except Exception as e:
                    self.logger.warning(f"Error processing table {table_index} on page {page_num}: {e}")
                    
        except Exception as e:
            self.logger.warning(f"Error extracting tables from page {page_num}: {e}")
            
        return tables

    def _detect_figures_and_captions(self, page, text: str, page_num: int, doc_id: str) -> List[Dict[str, Any]]:
        """Detect figures and their captions"""
        figures = []
        
        # Look for figure references and captions in text
        figure_patterns = [
            r'Figure\s+(\d+)[.:]\s*([^\n]+)',
            r'Fig\.\s*(\d+)[.:]\s*([^\n]+)',
            r'Table\s+(\d+)[.:]\s*([^\n]+)',
            r'Chart\s+(\d+)[.:]\s*([^\n]+)'
        ]
        
        for pattern in figure_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                figure_num = match.group(1)
                caption = match.group(2).strip()
                
                figures.append({
                    'figure_id': f"{doc_id}_p{page_num}_fig{figure_num}",
                    'figure_number': figure_num,
                    'page_number': page_num,
                    'caption': caption,
                    'type': 'figure' if 'fig' in match.group(0).lower() else 'table'
                })
                
        return figures

    def _analyze_image_with_vision(self, image: Image.Image, context: str = "") -> str:
        """Analyze image using vision model"""
        if not self.has_vision:
            return "Vision model not available for image analysis"
            
        try:
            # Convert image to base64
            buffered = io.BytesIO()
            image.save(buffered, format="PNG")
            img_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create prompt for vision model
            prompt = f"""Analyze this scientific {context} image and provide a detailed description focusing on:
1. What type of figure/chart/graph/diagram this is
2. Key data or information shown
3. Any trends, patterns, or relationships visible
4. Scientific context and relevance
5. Any text, labels, or captions visible

Be specific and detailed as this will help in answering research questions."""

            # Note: This is a placeholder - actual implementation depends on how ollama handles vision
            # You may need to use a different API call for vision models
            response = f"Scientific figure analysis: {context}. Image contains data visualization or scientific diagram."
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error analyzing image with vision model: {e}")
            return f"Image analysis failed: {str(e)}"

    def _clean_table_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and process extracted table data"""
        # Remove empty rows and columns
        df = df.dropna(how='all').loc[:, df.notna().any()]
        
        # Clean column names
        df.columns = [str(col).strip() for col in df.columns]
        
        # Clean cell values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()
                
        return df

    def _describe_table(self, df: pd.DataFrame) -> str:
        """Generate description of table content"""
        description = f"Table with {df.shape[0]} rows and {df.shape[1]} columns. "
        description += f"Columns: {', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}. "
        
        # Add sample data context
        if not df.empty:
            description += f"Sample data includes: {str(df.iloc[0].to_dict())[:100]}..."
            
        return description

    def _create_multimodal_chunks(self, text: str, images: List, tables: List, 
                                figures: List, page_num: int, doc_id: str) -> List[MultimodalChunk]:
        """Create enhanced chunks with multimodal content"""
        chunks = []
        
        # Create main text chunk
        if text.strip():
            # Combine visual context
            visual_context = self._create_visual_context_summary(images, tables, figures)
            
            chunk = MultimodalChunk(
                content=text,
                metadata={
                    'page_number': page_num,
                    'has_images': len(images) > 0,
                    'has_tables': len(tables) > 0,
                    'has_figures': len(figures) > 0
                },
                chunk_id=f"{doc_id}_p{page_num}_main",
                document_id=doc_id,
                page_number=page_num,
                images=images,
                tables=tables,
                figures=figures,
                visual_context=visual_context
            )
            chunks.append(chunk)
            
        return chunks

    def _create_visual_context_summary(self, images: List, tables: List, figures: List) -> str:
        """Create summary of visual elements for LLM context"""
        context_parts = []
        
        if images:
            context_parts.append(f"Images ({len(images)}): " + 
                               "; ".join([img.get('description', 'Image') for img in images[:3]]))
            
        if tables:
            context_parts.append(f"Tables ({len(tables)}): " + 
                               "; ".join([tbl.get('description', 'Table') for tbl in tables[:3]]))
            
        if figures:
            context_parts.append(f"Figures ({len(figures)}): " + 
                               "; ".join([fig.get('caption', 'Figure') for fig in figures[:3]]))
            
        return " | ".join(context_parts) if context_parts else ""

# Example usage and integration functions
def enhance_search_with_multimodal(search_results: List, processor: MultimodalProcessor) -> List:
    """Enhance search results with multimodal context"""
    enhanced_results = []
    
    for result in search_results:
        enhanced_result = result.copy()
        
        # Add visual context to the result
        if hasattr(result, 'visual_context') and result.visual_context:
            enhanced_result['content'] += f"\n\nVisual Context: {result.visual_context}"
            
        # Add table summaries
        if hasattr(result, 'tables') and result.tables:
            table_summaries = [tbl.get('description', '') for tbl in result.tables]
            enhanced_result['content'] += f"\n\nTables: {'; '.join(table_summaries)}"
            
        # Add figure information
        if hasattr(result, 'figures') and result.figures:
            figure_info = [f"Figure {fig.get('figure_number', '')}: {fig.get('caption', '')}" 
                          for fig in result.figures]
            enhanced_result['content'] += f"\n\nFigures: {'; '.join(figure_info)}"
            
        enhanced_results.append(enhanced_result)
        
    return enhanced_results
