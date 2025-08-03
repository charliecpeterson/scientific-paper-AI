"""
Integration script for enhanced multimodal document processing
Upgrades existing system with image, table, and figure analysis
"""

import os
import logging
from pathlib import Path
from typing import List, Dict, Any
import streamlit as st

from .multimodal_processor import MultimodalProcessor, MultimodalChunk
from .intelligent_search import IntelligentSearch

class EnhancedDocumentSystem:
    """Enhanced system that integrates multimodal processing with existing search"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize processors
        self.multimodal_processor = MultimodalProcessor(config)
        self.search_engine = None
        
        # Check if vision model is available
        self.has_vision_model = self._check_vision_model()
        
    def _check_vision_model(self) -> bool:
        """Check if vision model is available in Ollama"""
        try:
            import subprocess
            result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
            available_models = result.stdout
            
            vision_models = ['llama3.2-vision', 'llava', 'moondream', 'bakllava']
            for model in vision_models:
                if model in available_models:
                    self.logger.info(f"Found vision model: {model}")
                    return True
                    
            self.logger.warning("No vision models found in Ollama")
            return False
        except Exception as e:
            self.logger.warning(f"Could not check Ollama models: {e}")
            return False
    
    def install_vision_model(self):
        """Install recommended vision model"""
        if self.has_vision_model:
            return True
            
        try:
            import subprocess
            st.info("Installing Llama 3.2 Vision model... This may take several minutes.")
            
            # Show progress
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Install the model
            process = subprocess.Popen(
                ['ollama', 'pull', 'llama3.2-vision:11b'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor progress (simplified)
            while process.poll() is None:
                status_text.text("Downloading vision model...")
                progress_bar.progress(50)  # Simplified progress
                
            if process.returncode == 0:
                progress_bar.progress(100)
                status_text.text("Vision model installed successfully!")
                self.has_vision_model = True
                return True
            else:
                st.error(f"Failed to install vision model: {process.stderr.read()}")
                return False
                
        except Exception as e:
            st.error(f"Error installing vision model: {e}")
            return False
    
    def upgrade_document_processing(self):
        """Upgrade existing documents with multimodal processing"""
        if not hasattr(self, 'search_engine') or self.search_engine is None:
            self.search_engine = IntelligentSearch(self.config)
            
        # Check if database exists
        if not self.search_engine._has_existing_data():
            st.warning("No existing documents found. Please process documents first.")
            return
            
        # Get list of documents that need upgrading
        documents_to_upgrade = self._get_documents_needing_upgrade()
        
        if not documents_to_upgrade:
            st.success("All documents are already processed with multimodal features!")
            return
            
        st.info(f"Found {len(documents_to_upgrade)} documents that can be enhanced with multimodal processing.")
        
        if st.button("Upgrade Documents with Multimodal Processing"):
            self._process_upgrade(documents_to_upgrade)
    
    def _get_documents_needing_upgrade(self) -> List[str]:
        """Get list of documents that need multimodal upgrade"""
        try:
            # Query the collection to see which documents lack multimodal metadata
            collection = self.search_engine.collection
            results = collection.get()
            
            documents_needing_upgrade = []
            processed_files = set()
            
            for metadata in results.get('metadatas', []):
                if metadata and 'file_path' in metadata:
                    file_path = metadata['file_path']
                    if file_path not in processed_files:
                        processed_files.add(file_path)
                        
                        # Check if it has multimodal metadata
                        has_multimodal = (
                            metadata.get('processing_type') == 'multimodal' or
                            'images_extracted' in metadata or
                            'visual_context' in metadata
                        )
                        
                        if not has_multimodal and Path(file_path).exists():
                            documents_needing_upgrade.append(file_path)
                            
            return documents_needing_upgrade
            
        except Exception as e:
            self.logger.error(f"Error checking documents for upgrade: {e}")
            return []
    
    def _process_upgrade(self, documents: List[str]):
        """Process documents with multimodal features"""
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        success_count = 0
        error_count = 0
        
        for i, doc_path in enumerate(documents):
            try:
                status_text.text(f"Processing {Path(doc_path).name}...")
                progress_bar.progress((i + 1) / len(documents))
                
                # Process with multimodal processor
                file_path = Path(doc_path)
                text, metadata, chunks = self.multimodal_processor.process_document_multimodal(file_path)
                
                # Update in search engine
                self._update_document_in_database(file_path, text, metadata, chunks)
                
                success_count += 1
                
            except Exception as e:
                self.logger.error(f"Error processing {doc_path}: {e}")
                error_count += 1
                
        # Show results
        st.success(f"Upgrade complete! {success_count} documents enhanced, {error_count} errors.")
        
        if error_count > 0:
            st.warning(f"{error_count} documents could not be upgraded. Check logs for details.")
    
    def _update_document_in_database(self, file_path: Path, text: str, metadata: Dict, chunks: List[MultimodalChunk]):
        """Update document in the database with multimodal content"""
        try:
            # Remove old chunks for this document
            document_id = str(file_path)
            self.search_engine.collection.delete(where={"document_id": document_id})
            
            # Add new multimodal chunks
            texts = []
            metadatas = []
            ids = []
            
            for chunk in chunks:
                # Enhance text with visual context
                enhanced_text = chunk.content
                if chunk.visual_context:
                    enhanced_text += f"\n\nVisual Elements: {chunk.visual_context}"
                    
                texts.append(enhanced_text)
                
                # Enhanced metadata
                enhanced_metadata = chunk.metadata.copy()
                enhanced_metadata.update({
                    'document_id': document_id,
                    'processing_type': 'multimodal',
                    'has_visual_content': bool(chunk.visual_context),
                    'file_path': str(file_path)
                })
                
                metadatas.append(enhanced_metadata)
                ids.append(chunk.chunk_id)
            
            # Add to collection
            if texts:
                self.search_engine.collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                self.logger.info(f"Updated {file_path.name} with {len(chunks)} multimodal chunks")
                
        except Exception as e:
            self.logger.error(f"Error updating document {file_path} in database: {e}")
            raise

def create_enhanced_search_interface():
    """Create Streamlit interface for enhanced search with multimodal features"""
    
    st.title("🔬 Enhanced Scientific Paper AI")
    st.markdown("*Now with advanced image, table, and figure analysis*")
    
    # Initialize enhanced system
    if 'enhanced_system' not in st.session_state:
        config = {}  # Load your config here
        st.session_state.enhanced_system = EnhancedDocumentSystem(config)
    
    enhanced_system = st.session_state.enhanced_system
    
    # Vision model status
    col1, col2 = st.columns(2)
    
    with col1:
        if enhanced_system.has_vision_model:
            st.success("✅ Vision model available")
        else:
            st.warning("⚠️ No vision model detected")
            if st.button("Install Vision Model"):
                enhanced_system.install_vision_model()
                st.rerun()
    
    with col2:
        st.info(f"Multimodal Processing: {'Enabled' if enhanced_system.has_vision_model else 'Basic Mode'}")
    
    # Document upgrade section
    st.header("📚 Document Processing")
    
    tab1, tab2 = st.tabs(["Upgrade Existing", "Process New"])
    
    with tab1:
        st.subheader("Enhance Existing Documents")
        st.markdown("""
        Upgrade your existing document collection with:
        - 🖼️ **Image Analysis**: Extract and describe figures, charts, graphs
        - 📊 **Table Processing**: Parse and understand data tables
        - 🔍 **Figure Captions**: Link captions with visual content
        - 🧠 **Visual Context**: Provide richer context for AI responses
        """)
        
        enhanced_system.upgrade_document_processing()
    
    with tab2:
        st.subheader("Process New Documents")
        uploaded_files = st.file_uploader(
            "Upload scientific papers", 
            accept_multiple_files=True,
            type=['pdf']
        )
        
        if uploaded_files and st.button("Process with Multimodal Analysis"):
            process_new_documents_multimodal(uploaded_files, enhanced_system)

def process_new_documents_multimodal(uploaded_files, enhanced_system):
    """Process newly uploaded documents with multimodal features"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Processing {uploaded_file.name}...")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Save uploaded file temporarily
            temp_path = Path(f"temp_{uploaded_file.name}")
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Process with multimodal processor
            text, metadata, chunks = enhanced_system.multimodal_processor.process_document_multimodal(temp_path)
            
            # Add to search engine
            enhanced_system._update_document_in_database(temp_path, text, metadata, chunks)
            
            # Cleanup
            temp_path.unlink()
            
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {e}")
    
    st.success("Documents processed successfully with multimodal analysis!")

if __name__ == "__main__":
    create_enhanced_search_interface()
