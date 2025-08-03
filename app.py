import streamlit as st
import os
from pathlib import Path
import time
from typing import List, Dict, Any

# Import our core modules
from src.document_processor import DocumentProcessor
from src.enhanced_document_processor import EnhancedDocumentProcessor
from src.intelligent_search import IntelligentSearch
from src.ai_agent import ScientificPaperAgent
from src.utils.config import Config
from src.utils.helpers import format_sources, display_document_stats

# Page configuration
st.set_page_config(
    page_title="🔬 Scientific Paper AI Analyzer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .main {
        padding-top: 1rem;
    }
    .stAlert {
        padding: 0.5rem;
        margin: 0.5rem 0;
    }
    .source-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        color: #212529;
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        border-left: 4px solid #0066cc;
        box-shadow: 0 3px 6px rgba(0,0,0,0.15);
        border: 1px solid #e9ecef;
    }
    .source-card strong {
        color: #0066cc;
        font-weight: 600;
    }
    .source-card em {
        color: #6c757d;
        font-size: 0.9rem;
    }
    .source-content {
        color: #343a40;
        line-height: 1.5;
        margin-top: 0.5rem;
        font-size: 0.95rem;
    }
    .chat-message {
        padding: 1.2rem;
        border-radius: 12px;
        margin: 0.8rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        position: relative;
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 2rem;
        border-bottom-right-radius: 4px;
    }
    .user-message strong {
        color: #ffffff;
    }
    .assistant-message {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        margin-right: 2rem;
        border-bottom-left-radius: 4px;
    }
    .assistant-message strong {
        color: #ffffff;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #e1e5e9;
        padding: 0.5rem 1rem;
    }
    .stTextInput > div > div > input:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 2px rgba(102, 126, 234, 0.2);
    }
    .stExpander > div > div > div {
        background-color: #ffffff;
    }
    .stExpander > div > div > div > div {
        color: #212529;
    }
    .stMarkdown {
        color: #212529;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'documents_processed' not in st.session_state:
        st.session_state.documents_processed = 0
    if 'config' not in st.session_state:
        st.session_state.config = Config()

def get_enhanced_document_title(doc: Dict[str, Any]) -> str:
    """Generate enhanced display title for a document"""
    paper_title = doc.get('paper_title', doc.get('title', ''))
    filename = doc.get('filename', 'Unknown Document')
    confidence = doc.get('title_confidence', 0.0)
    
    if paper_title and confidence > 0.6:
        return f"📑 {paper_title}"
    elif paper_title and confidence > 0.3:
        return f"📄 {paper_title} [?]"
    else:
        return f"📁 {filename}"

def get_document_title_info(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Get detailed title information for a document"""
    return {
        'paper_title': doc.get('paper_title', doc.get('title', '')),
        'filename': doc.get('filename', 'Unknown'),
        'confidence': doc.get('title_confidence', 0.0),
        'processing_quality': doc.get('processing_quality', 'unknown')
    }

def generate_enhanced_citation(doc: Dict[str, Any]) -> str:
    """Generate an enhanced citation for a document"""
    # Use enhanced authors if available, otherwise fall back to basic authors
    authors = doc.get('enhanced_authors', doc.get('authors', []))
    title = doc.get('paper_title', doc.get('title', doc.get('filename', 'Unknown')))
    year = doc.get('publication_year') or doc.get('year')
    venue = doc.get('venue', doc.get('journal'))
    
    # Handle year with potential format issues
    if year:
        try:
            year_int = int(year)
            # Handle edge case where old regex captured only part of year
            if year_int < 100:
                pub_year = doc.get('publication_year')
                if pub_year and int(pub_year) > 1900:
                    year = int(pub_year)
                elif year_int >= 0 and year_int <= 25:
                    year = f"20{year_int:02d}"
                else:
                    year = year_int
            else:
                year = year_int
        except (ValueError, TypeError):
            year = "n.d."
    else:
        year = "n.d."
    
    # Format citation
    citation_parts = []
    
    # Authors
    if authors:
        if len(authors) == 1:
            citation_parts.append(authors[0])
        elif len(authors) <= 3:
            citation_parts.append(", ".join(authors))
        else:
            citation_parts.append(f"{authors[0]} et al.")
    
    # Year
    citation_parts.append(f"({year})")
    
    # Title
    citation_parts.append(f'"{title}"')
    
    # Venue
    if venue:
        citation_parts.append(venue)
    
    return ". ".join(citation_parts) + "."

def sidebar_configuration():
    """Clean sidebar with only essential information"""
    st.sidebar.header("🤖 AI Assistant")
    
    # Auto-detect best available models
    config = Config()
    
    # Show current models (read-only)
    st.sidebar.info(f"🧠 **AI Model**: {config.llm_model}")
    st.sidebar.info(f"🔍 **Search Model**: {config.embedding_model}")
    
    # Show collection stats if available
    if st.session_state.agent and st.session_state.documents_processed > 0:
        stats = st.session_state.agent.search_engine.get_document_stats()
        if stats:
            st.sidebar.subheader("📊 Your Collection")
            st.sidebar.metric("Documents", stats.get('total_documents', 0))
            st.sidebar.metric("Searchable Chunks", stats.get('total_chunks', 0))
            
            if stats.get('unique_authors', 0) > 0:
                st.sidebar.metric("Unique Authors", stats.get('unique_authors', 0))
    else:
        st.sidebar.info("📄 Upload documents to get started")
    
    return config

def document_upload_section():
    """Document upload and processing section with automatic persistent storage handling"""
    st.header("📤 Document Upload & Management")
    
    # Auto-initialize and check persistent storage
    if st.session_state.agent is None:
        with st.spinner("🔍 Checking for existing documents..."):
            try:
                # Always initialize search engine to check persistent storage
                search_engine = IntelligentSearch(st.session_state.config)
                existing_stats = search_engine.get_document_stats()
                
                if existing_stats and existing_stats.get('total_chunks', 0) > 0:
                    # Automatically load existing documents
                    st.session_state.agent = ScientificPaperAgent(
                        search_engine=search_engine,
                        config=st.session_state.config
                    )
                    st.session_state.documents_processed = existing_stats['total_chunks']
                    
                    st.success(f"✅ **Automatically loaded {existing_stats['total_documents']} documents ({existing_stats['total_chunks']} chunks) from persistent storage!**")
                    st.info("Your documents are ready for querying. Check the **Documents** tab to manage them.")
                else:
                    st.info("� No existing documents found in storage. Upload your first documents below.")
                    
            except Exception as e:
                st.warning(f"Could not check persistent storage: {str(e)}")
                st.info("📄 Upload documents to get started.")
    else:
        # Agent already initialized - show current status
        stats = st.session_state.agent.search_engine.get_document_stats()
        if stats:
            st.success(f"✅ **Currently loaded: {stats['total_documents']} documents ({stats['total_chunks']} chunks)**")
            st.info("Your documents are ready for querying. Use the area below to add more documents.")
        else:
            st.warning("⚠️ Agent loaded but no documents found. Upload documents below.")
    
    st.divider()
    
    # File upload section
    st.subheader("📁 Add New Documents")
    uploaded_files = st.file_uploader(
        "Upload scientific papers to add to your collection",
        type=['pdf', 'docx', 'txt'],
        accept_multiple_files=True,
        help="Supported formats: PDF, Word (.docx), Plain Text"
    )
    
    if uploaded_files:
        st.write(f"📋 **Ready to process {len(uploaded_files)} new files:**")
        for file in uploaded_files:
            st.write(f"• {file.name} ({file.size:,} bytes)")
        
        if st.button("🚀 Add to Collection", type="primary"):
            with st.spinner("Processing and adding new documents with AI-powered analysis..."):
                try:
                    # Initialize enhanced document processor
                    processor = EnhancedDocumentProcessor(st.session_state.config)
                    
                    # Process each file with enhanced metadata extraction
                    all_documents = []
                    all_metadata = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    for i, file in enumerate(uploaded_files):
                        status_text.text(f"🔍 AI analyzing: {file.name}")
                        
                        # Save uploaded file temporarily
                        temp_path = f"temp_{file.name}"
                        with open(temp_path, "wb") as f:
                            f.write(file.read())
                        
                        # Process document with enhanced AI analysis
                        try:
                            enhanced_chunks, enhanced_metadata = processor.process_document_enhanced(temp_path)
                            all_documents.extend(enhanced_chunks)
                            all_metadata.append(enhanced_metadata)
                            
                            # Show extraction results
                            st.info(f"✅ **{file.name}** → AI extracted: '{enhanced_metadata.paper_title}' (confidence: {enhanced_metadata.paper_title_confidence:.2f})")
                            
                        except Exception as e:
                            st.warning(f"⚠️ Enhanced processing failed for {file.name}, using basic processing: {str(e)}")
                            # Fallback to basic processing
                            basic_processor = DocumentProcessor(st.session_state.config)
                            documents = basic_processor.process_document(temp_path)
                            all_documents.extend(documents)
                        
                        # Clean up temp file
                        os.remove(temp_path)
                        
                        # Update progress
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    # Add to existing collection (or create if none exists)
                    if st.session_state.agent is None:
                        # First time setup
                        search_engine = IntelligentSearch(st.session_state.config)
                        search_engine.add_documents(all_documents)
                        
                        st.session_state.agent = ScientificPaperAgent(
                            search_engine=search_engine,
                            config=st.session_state.config
                        )
                        st.session_state.documents_processed = len(all_documents)
                    else:
                        # Add to existing collection
                        st.session_state.agent.search_engine.add_documents(all_documents)
                        st.session_state.documents_processed += len(all_documents)
                    
                    # Success feedback
                    status_text.empty()
                    progress_bar.empty()
                    st.success(f"✅ Successfully added {len(uploaded_files)} files!")
                    
                    # Auto-refresh to show new state
                    time.sleep(1)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing documents: {str(e)}")
                    st.info("💡 Make sure the files are valid PDF/DOCX/TXT format")
    
    # Show collection summary if documents exist
    if st.session_state.agent:
        st.divider()
        st.subheader("📊 Current Collection Summary")
        
        try:
            stats = st.session_state.agent.search_engine.get_document_stats()
            if stats:
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Documents", stats.get('total_documents', 0))
                with col2:
                    st.metric("🧩 Chunks", stats.get('total_chunks', 0))
                with col3:
                    st.metric("👥 Authors", stats.get('unique_authors', 0))
                with col4:
                    avg_chunks = stats.get('total_chunks', 0) // max(stats.get('total_documents', 1), 1)
                    st.metric("📈 Avg/Doc", avg_chunks)
                
                st.success("🎯 Collection ready for queries!")
            else:
                st.warning("⚠️ Collection loaded but no statistics available.")
                
        except Exception as e:
            st.error(f"Error loading collection stats: {str(e)}")

def documents_overview_section():
    """Documents overview and management section"""
    st.header("📚 Document Library & Management")
    st.markdown("*Manage your document collection - view details and permanently remove documents from both database and disk*")
    
    if st.session_state.agent is None:
        st.info("👆 Upload documents first to see your document library")
        return
    
    # Get document details with error handling
    try:
        documents = st.session_state.agent.search_engine.get_document_details()
        
        if not documents:
            st.warning("No documents found in the knowledge base.")
            return
        
        # Summary stats with error handling
        total_docs = len(documents)
        total_chunks = sum(doc.get('total_chunks', 0) for doc in documents)
        all_authors = set()
        
        for doc in documents:
            # Handle authors safely
            authors = doc.get('authors', [])
            if isinstance(authors, list):
                all_authors.update(authors)
            elif isinstance(authors, str):
                all_authors.add(authors)
        
        # Display summary
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📄 Total Documents", total_docs)
        with col2:
            st.metric("🧩 Total Chunks", total_chunks)
        with col3:
            st.metric("👥 Unique Authors", len(all_authors))
        with col4:
            avg_chunks = total_chunks // total_docs if total_docs > 0 else 0
            st.metric("📊 Avg Chunks/Doc", avg_chunks)
        
        st.divider()
        
        # Search/filter documents
        col1, col2 = st.columns([3, 1])
        with col1:
            search_term = st.text_input("🔍 Search documents by title, author, or filename", 
                                      placeholder="Type to filter documents...")
        with col2:
            sort_by = st.selectbox("Sort by", ["Title", "Authors", "Chunks", "Filename"])
        
        # Filter documents based on search
        filtered_docs = documents
        if search_term:
            search_lower = search_term.lower()
            filtered_docs = []
            for doc in documents:
                if (search_lower in doc['title'].lower() or
                    search_lower in doc['filename'].lower() or
                    any(search_lower in author.lower() for author in doc.get('authors', []))):
                    filtered_docs.append(doc)
        
        # Sort documents
        if sort_by == "Title":
            filtered_docs.sort(key=lambda x: x['title'])
        elif sort_by == "Authors":
            filtered_docs.sort(key=lambda x: ', '.join(x.get('authors', [''])))
        elif sort_by == "Chunks":
            filtered_docs.sort(key=lambda x: x['total_chunks'], reverse=True)
        elif sort_by == "Filename":
            filtered_docs.sort(key=lambda x: x['filename'])
        
        st.write(f"Showing {len(filtered_docs)} of {total_docs} documents")
        
        # Bulk operations
        if filtered_docs:
            st.divider()
            col1, col2 = st.columns([3, 1])
            with col1:
                st.write("**🔧 Bulk Operations:**")
            with col2:
                if st.button("🗑️ Clear All Documents", type="secondary"):
                    st.session_state.confirm_bulk_delete = True
            
            # Bulk delete confirmation
            if st.session_state.get("confirm_bulk_delete", False):
                st.error("⚠️ **DANGER: This will delete ALL documents and files!**")
                st.markdown("""
                **This action will:**
                • Remove all documents and chunks from the knowledge base
                • **Delete all PDF/document files from disk permanently**
                • Clear the entire document collection
                • **This action cannot be undone - files will be permanently deleted**
                
                ⚠️ **Make sure you have backups of important files before proceeding!**
                """)
                
                col_yes, col_no = st.columns(2)
                with col_yes:
                    if st.button("✅ Yes, Delete Everything", key="bulk_delete_confirm"):
                        clear_all_documents()
                        st.session_state.confirm_bulk_delete = False
                        st.rerun()
                
                with col_no:
                    if st.button("❌ Cancel", key="bulk_delete_cancel"):
                        st.session_state.confirm_bulk_delete = False
                        st.rerun()
            
            st.divider()
        
        # Display documents with enhanced labeling
        for i, doc in enumerate(filtered_docs):
            # Enhanced document display
            display_title = get_enhanced_document_title(doc)
            title_info = get_document_title_info(doc)
            
            with st.expander(display_title, expanded=False):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Enhanced document information
                    st.write("**📋 Document Information:**")
                    
                    # Title analysis section
                    st.write("**🏷️ Title Analysis:**")
                    paper_title = doc.get('paper_title', doc.get('title', 'Unknown'))
                    confidence = doc.get('title_confidence', 0.0)
                    
                    if confidence > 0.6:
                        st.success(f"✅ **AI-Extracted Title:** {paper_title}")
                        st.caption(f"Confidence: {confidence:.2f} (High)")
                    elif confidence > 0.3:
                        st.warning(f"⚠️ **AI-Extracted Title:** {paper_title}")
                        st.caption(f"Confidence: {confidence:.2f} (Medium)")
                    else:
                        st.error(f"❓ **Detected Title:** {paper_title}")
                        st.caption(f"Confidence: {confidence:.2f} (Low - using filename)")
                    
                    st.write(f"**📁 Original Filename:** `{doc['filename']}`")
                    
                    # Enhanced metadata display
                    if doc.get('enhanced_authors'):
                        authors_str = ', '.join(doc['enhanced_authors'])
                        st.write(f"**👥 Authors:** {authors_str}")
                    elif doc.get('authors'):
                        authors_str = ', '.join(doc['authors'])
                        st.write(f"**👥 Authors:** {authors_str} *(basic extraction)*")
                    
                    # Display year safely with fallback logic
                    year = doc.get('publication_year') or doc.get('year')
                    if year:
                        try:
                            year_int = int(year) if isinstance(year, (str, float)) else year
                            # Handle edge case where old regex captured only part of year
                            if year_int < 100:  # Likely partial year like "20"
                                # Try to use publication_year if it's more reasonable
                                pub_year = doc.get('publication_year')
                                if pub_year and int(pub_year) > 1900:
                                    year_int = int(pub_year)
                                else:
                                    year_int = f"20{year_int}" if year_int >= 0 and year_int <= 25 else year_int
                            
                            st.write(f"**📅 Year:** {year_int}")
                        except (ValueError, TypeError):
                            st.write(f"**📅 Year:** {year} *(format issue)*")
                    
                    # Display venue safely
                    venue = doc.get('venue') or doc.get('journal')
                    if venue:
                        st.write(f"**📚 Journal/Venue:** {venue}")
                    
                    if doc.get('doi'):
                        st.write(f"**🔗 DOI:** {doc['doi']}")
                    
                    # Enhanced citation information
                    st.write("**📖 Citation Format:**")
                    citation = generate_enhanced_citation(doc)
                    st.code(citation, language="text")
                    
                    # Show research domain and topics if available
                    if doc.get('research_domain'):
                        st.write(f"**🔬 Research Domain:** {doc['research_domain']}")
                    
                    if doc.get('key_topics'):
                        topics_str = ', '.join(doc['key_topics'])
                        st.write(f"**🏷️ Key Topics:** {topics_str}")
                    
                    # Show processing quality
                    quality = doc.get('processing_quality', 'unknown')
                    quality_icon = {
                        'high': '✅',
                        'high_multimodal': '🎯',
                        'medium': '⚠️',
                        'basic_fallback': '⚠️',
                        'low': '❌'
                    }.get(quality, '❓')
                    st.write(f"**🔧 Processing Quality:** {quality_icon} {quality}")
                    
                with col2:
                    # Processing stats
                    st.write("**🔧 Processing Info:**")
                    st.metric("Chunks Created", doc['total_chunks'])
                    st.write(f"**Page Range:** {doc.get('page_range', 'Unknown')}")
                    
                    if doc.get('sections'):
                        st.write("**Sections Found:**")
                        for section in doc['sections']:
                            st.write(f"• {section}")
                    
                    # Action buttons
                    st.write("**⚡ Actions:**")
                    col_a, col_b = st.columns(2)
                    
                    with col_a:
                        if st.button(f"🔍 Search", key=f"search_{i}"):
                            st.session_state.doc_filter = doc['document_id']
                            st.info(f"Document filter set to: {doc['title']}")
                    
                    with col_b:
                        if st.button(f"📊 Chunks", key=f"chunks_{i}"):
                            st.session_state.show_chunks = doc['document_id']
                    
                    # Delete button with confirmation
                    if st.button(f"🗑️ Remove", key=f"delete_{i}", type="secondary"):
                        st.session_state[f"confirm_delete_{i}"] = True
                    
                    # Confirmation dialog
                    if st.session_state.get(f"confirm_delete_{i}", False):
                        st.warning(f"⚠️ **Confirm deletion of:** {doc['title']}")
                        
                        # Show file path if available
                        file_path = doc.get('file_path') or doc.get('document_id', 'Unknown')
                        if file_path != 'Unknown':
                            st.code(f"File: {file_path}", language="text")
                        
                        st.write("This will:")
                        st.write("• Remove the document and all its chunks from the knowledge base")
                        st.write("• **Delete the physical file from disk permanently**")
                        st.write("• This action cannot be undone")
                        
                        col_yes, col_no = st.columns(2)
                        with col_yes:
                            if st.button(f"✅ Yes, Delete", key=f"confirm_yes_{i}"):
                                delete_document_from_db(doc['document_id'], doc['title'])
                                # Clear confirmation state
                                st.session_state[f"confirm_delete_{i}"] = False
                                st.rerun()
                        
                        with col_no:
                            if st.button(f"❌ Cancel", key=f"confirm_no_{i}"):
                                st.session_state[f"confirm_delete_{i}"] = False
                                st.rerun()
        
        # Show chunks if requested
        if hasattr(st.session_state, 'show_chunks') and st.session_state.show_chunks:
            show_document_chunks(st.session_state.show_chunks, documents)
    
    except Exception as e:
        st.error(f"Error loading document library: {str(e)}")

def generate_citation(doc: Dict[str, Any]) -> str:
    """Generate a formatted citation for a document"""
    authors = doc.get('authors', [])
    title = doc.get('title', 'Unknown Title')
    year = doc.get('year', 'n.d.')
    journal = doc.get('journal', '')
    doi = doc.get('doi', '')
    
    # Format authors
    if authors:
        if len(authors) == 1:
            author_str = authors[0]
        elif len(authors) == 2:
            author_str = f"{authors[0]} & {authors[1]}"
        else:
            author_str = f"{authors[0]} et al."
    else:
        author_str = "Unknown Author"
    
    # Basic citation format
    citation = f"{author_str} ({year}). {title}."
    
    if journal:
        citation += f" {journal}."
    
    if doi:
        citation += f" https://doi.org/{doi}"
    
    return citation

def show_document_chunks(doc_id: str, documents: List[Dict[str, Any]]):
    """Show individual chunks for a document"""
    doc = next((d for d in documents if d['document_id'] == doc_id), None)
    if not doc:
        return
    
    st.subheader(f"📄 Chunks in: {doc['title']}")
    
    try:
        # Get chunks for this document from the search engine
        search_engine = st.session_state.agent.search_engine
        collection_data = search_engine.collection.get(
            where={"document_id": doc_id},
            include=['documents', 'metadatas']
        )
        
        if not collection_data['ids']:
            st.warning("No chunks found for this document.")
            return
        
        for i, (chunk_id, content, metadata) in enumerate(zip(
            collection_data['ids'], 
            collection_data['documents'], 
            collection_data['metadatas']
        )):
            with st.expander(f"Chunk {i+1}: {metadata.get('section', 'Unknown')} (Page {metadata.get('page_number', '?')})", expanded=False):
                st.write(f"**Chunk ID:** `{chunk_id}`")
                st.write(f"**Section:** {metadata.get('section', 'Unknown')}")
                st.write(f"**Page:** {metadata.get('page_number', 'Unknown')}")
                st.write(f"**Tokens:** {metadata.get('token_count', 'Unknown')}")
                st.divider()
                st.write("**Content:**")
                st.write(content)
    
    except Exception as e:
        st.error(f"Error loading chunks: {str(e)}")
    
    # Clear the show_chunks state
    if st.button("← Back to Documents"):
        if hasattr(st.session_state, 'show_chunks'):
            del st.session_state.show_chunks

def chat_interface():
    """Main chat interface"""
    st.header("💬 Ask Questions About Your Papers")
    
    if st.session_state.agent is None:
        st.info("👆 Please upload and process documents first")
        return
    
    # Display document stats
    if st.session_state.documents_processed > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📚 Documents", f"{st.session_state.documents_processed}")
        with col2:
            st.metric("🤖 AI Model", st.session_state.config.llm_model)
        with col3:
            st.metric("🎯 Status", "Ready 🟢")
    
    # Chat messages container
    chat_container = st.container()
    
    # Display chat history
    with chat_container:
        for message in st.session_state.messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div class="chat-message user-message">
                    <strong>🧑 You:</strong> {message["content"]}
                </div>
                """, unsafe_allow_html=True)
            else:
                # Build assistant message with adaptive info
                confidence = message.get("confidence", 0.0)
                reasoning = message.get("reasoning", "")
                
                # Confidence indicator
                if confidence > 0.8:
                    confidence_icon = "🎯"
                    confidence_color = "#28a745"
                elif confidence > 0.6:
                    confidence_icon = "✅"
                    confidence_color = "#17a2b8"
                else:
                    confidence_icon = "⚠️"
                    confidence_color = "#ffc107"
                
                st.markdown(f"""
                <div class="chat-message assistant-message">
                    <strong>🤖 AI:</strong> {message["content"]}
                    {f'<br><small style="color: {confidence_color};">{confidence_icon} Confidence: {confidence:.0%}</small>' if confidence > 0 else ''}
                </div>
                """, unsafe_allow_html=True)
                
                # Display reasoning if available
                if reasoning:
                    with st.expander("🧠 AI Reasoning", expanded=False):
                        st.info(reasoning)
                
                # Display sources if available
                if "sources" in message:
                    with st.expander("📚 Sources", expanded=False):
                        for i, source in enumerate(message["sources"]):
                            st.markdown(f"""
                            <div class="source-card">
                                <strong>📄 {source.get('title', 'Unknown Document')}</strong><br>
                                <em>📍 Section: {source.get('section', 'N/A')} | Page: {source.get('page', 'N/A')} | 
                                Relevance: {source.get('relevance_score', 0):.2f}</em>
                                <div class="source-content">
                                    {source.get('content', '')[:300]}{'...' if len(source.get('content', '')) > 300 else ''}
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
    
    # Query input
    query = st.chat_input("Ask me anything about your research papers...")
    
    if query:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": query})
        
        # Get AI response
        with st.spinner("🔍 Analyzing your question..."):
            try:
                response = st.session_state.agent.query(query)
                
                # Add AI response with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response["answer"],
                    "sources": response.get("sources", []),
                    "confidence": response.get("confidence", 0.0),
                    "reasoning": response.get("reasoning", "")
                })
                
                # Show search feedback
                num_sources = len(response.get("sources", []))
                if num_sources > 0:
                    if response.get("confidence", 0.0) > 0.8:
                        st.success(f"🎯 Found {num_sources} highly relevant sources!")
                    elif response.get("confidence", 0.0) > 0.6:
                        st.info(f"✅ Found {num_sources} relevant sources.")
                    else:
                        st.info(f"📚 Found {num_sources} sources - consider refining your query.")
                else:
                    st.warning("🔍 No matching content found. Try different keywords.")
                
                # Rerun to update the chat
                st.rerun()
                
            except Exception as e:
                # Enhanced error handling with user-friendly messages
                error_msg = str(e)
                if "max_chunks" in error_msg:
                    st.error("🔧 Configuration issue detected. Please try again.")
                elif "No documents" in error_msg:
                    st.error("📄 No documents found. Please upload PDF files first.")
                elif "OpenAI" in error_msg or "API" in error_msg or "Ollama" in error_msg:
                    st.error("🔑 AI service temporarily unavailable. Please try again.")
                else:
                    st.error(f"❌ Unexpected error: {error_msg}")
                    st.info("💡 Try rephrasing your question.")

def main():
    """Main application"""
    # Initialize session state
    initialize_session_state()
    
    # Title and description
    st.title("🔬 Scientific Paper AI Analyzer")
    st.markdown("**Intelligent Research Assistant** - Just upload papers and ask questions.")
    
    # Sidebar configuration
    config = sidebar_configuration()
    
    # Main content in tabs
    tab1, tab2, tab3 = st.tabs(["📤 Upload", "💬 Chat", "� Documents"])
    
    with tab1:
        document_upload_section()
    
    with tab2:
        chat_interface()
    
    with tab3:
        documents_overview_section()

def delete_document_from_db(document_id: str, title: str):
    """Delete a specific document from the ChromaDB database AND remove file from disk"""
    try:
        if st.session_state.agent and st.session_state.agent.search_engine:
            collection = st.session_state.agent.search_engine.collection
            
            # Get all chunks for this document to find the file path
            results = collection.get(where={"document_id": document_id})
            
            file_path = None
            if results and results['metadatas']:
                # Look for file_path in metadata
                for metadata in results['metadatas']:
                    if metadata and 'file_path' in metadata:
                        file_path = metadata['file_path']
                        break
            
            # Delete from database
            if results and results['ids']:
                collection.delete(ids=results['ids'])
                chunk_count = len(results['ids'])
                
                # Delete physical file from disk
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        st.success(f"✅ Successfully removed '{title}' from knowledge base ({chunk_count} chunks) and deleted file from disk")
                    except OSError as e:
                        st.warning(f"⚠️ Removed '{title}' from knowledge base ({chunk_count} chunks) but couldn't delete file: {str(e)}")
                else:
                    st.success(f"✅ Successfully removed '{title}' from knowledge base ({chunk_count} chunks). File not found on disk.")
                
                # Update session state
                if hasattr(st.session_state, 'documents_processed'):
                    st.session_state.documents_processed -= chunk_count
            else:
                st.warning(f"⚠️ No chunks found for document: {title}")
                
    except Exception as e:
        st.error(f"❌ Error deleting document '{title}': {str(e)}")

def clear_all_documents():
    """Clear all documents from the ChromaDB database AND delete all files from disk"""
    try:
        if st.session_state.agent and st.session_state.agent.search_engine:
            collection = st.session_state.agent.search_engine.collection
            
            # Get all documents and their file paths before deletion
            all_results = collection.get()
            total_count = len(all_results.get('ids', []))
            
            if total_count == 0:
                st.info("No documents found to delete.")
                return
            
            # Collect unique file paths
            file_paths = set()
            if all_results.get('metadatas'):
                for metadata in all_results['metadatas']:
                    if metadata and 'file_path' in metadata:
                        file_paths.add(metadata['file_path'])
            
            # Use the new clear_all_documents method from search engine
            success = st.session_state.agent.search_engine.clear_all_documents()
            
            if not success:
                st.error("❌ Failed to clear documents from database")
                return
            
            # Delete physical files from disk
            deleted_files = 0
            failed_files = []
            
            for file_path in file_paths:
                if file_path and os.path.exists(file_path):
                    try:
                        os.remove(file_path)
                        deleted_files += 1
                    except OSError as e:
                        failed_files.append(f"{os.path.basename(file_path)}: {str(e)}")
            
            # Reset the agent to force reinitialization
            st.session_state.agent = None
            
            # Reset session state
            st.session_state.documents_processed = 0
            if hasattr(st.session_state, 'doc_filter'):
                del st.session_state.doc_filter
            if hasattr(st.session_state, 'show_chunks'):
                del st.session_state.show_chunks
            
            # Show results
            if failed_files:
                st.success(f"✅ Cleared all documents ({total_count} chunks) from knowledge base")
                st.success(f"✅ Deleted {deleted_files} files from disk")
                st.warning(f"⚠️ Could not delete {len(failed_files)} files:\n" + "\n".join(failed_files))
            else:
                st.success(f"✅ Successfully cleared all documents ({total_count} chunks) from knowledge base and deleted {deleted_files} files from disk")
            
            # Verify deletion by checking if any documents remain
            try:
                # Reinitialize search engine to check
                from src.intelligent_search import IntelligentSearch
                test_engine = IntelligentSearch(st.session_state.config)
                remaining_stats = test_engine.get_document_stats()
                
                if remaining_stats and remaining_stats.get('total_chunks', 0) > 0:
                    st.warning(f"⚠️ Warning: {remaining_stats['total_chunks']} chunks may still remain in storage")
                else:
                    st.success("✅ Verified: All documents successfully removed from persistent storage")
            except:
                st.success("✅ Documents cleared - collection reset successfully")
                
    except Exception as e:
        st.error(f"❌ Error clearing all documents: {str(e)}")
        # Reset agent even on error to force reinit
        st.session_state.agent = None

if __name__ == "__main__":
    main()
