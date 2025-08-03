"""
Intelligent Search Engine for Scientific Papers
Implements hierarchical search, cross-document synthesis, and advanced ranking
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import re
from collections import defaultdict, Counter
from dataclasses import dataclass
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from .document_processor import DocumentChunk

@dataclass
class SearchResult:
    """Enhanced search result with relevance scoring"""
    chunk: DocumentChunk
    similarity_score: float
    relevance_score: float
    rank: int
    highlights: List[str] = None

class IntelligentSearch:
    """Advanced search engine with hierarchical search and intelligent ranking"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize vector database with persistent storage
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection_name = "scientific_papers"
        
        try:
            self.collection = self.chroma_client.get_collection(self.collection_name)
            self.logger.info(f"Loaded existing collection with {self.collection.count()} documents")
        except:
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("Created new persistent collection")
        
        # Initialize embedding model
        self.embedding_model = None
        self._initialize_embedding_model()
        
        # Initialize TF-IDF for keyword matching
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        self.tfidf_matrix = None
        self.documents = []  # Initialize empty documents list
        
        # Initialize from persistent storage AFTER all components are set up
        if self.collection.count() > 0:
            self._initialize_from_persistent_storage()
        
        # Author detection patterns
        self.author_query_patterns = [
            r'\b(?:papers?|work|research|studies?)\s+(?:by|from|of)\s+([^?]+)',
            r'\b(?:what|which)\s+(?:papers?|work)\s+(?:has|did|does)\s+([^?]+)',
            r'\b([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s+(?:wrote|published|authored)',
        ]

    def _initialize_embedding_model(self):
        """Initialize the embedding model with robust fallback"""
        try:
            if self.config.embedding_model == "mxbai-embed-large":
                # Try Ollama embedding first
                from langchain_ollama import OllamaEmbeddings
                try:
                    self.embedding_model = OllamaEmbeddings(model="mxbai-embed-large")
                    # Test the model with a simple embedding
                    test_embedding = self.embedding_model.embed_query("test")
                    self.logger.info(f"Successfully initialized Ollama embedding model: {self.config.embedding_model}")
                    return
                except Exception as ollama_error:
                    self.logger.warning(f"Ollama embedding model '{self.config.embedding_model}' not available: {ollama_error}")
                    self.logger.info("Falling back to SentenceTransformer...")
            
            # Fallback to sentence transformers
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.logger.info(f"Initialized fallback embedding model: all-MiniLM-L6-v2")
            
        except Exception as e:
            self.logger.error(f"Error initializing embedding model: {e}")
            # Last resort fallback
            try:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                self.logger.info("Using last resort embedding model: all-MiniLM-L6-v2")
            except Exception as final_error:
                self.logger.error(f"Failed to initialize any embedding model: {final_error}")
                raise Exception("Could not initialize any embedding model")

    def _initialize_from_persistent_storage(self):
        """Initialize TF-IDF and document list from persistent storage"""
        try:
            if self.collection.count() > 0:
                self.logger.info("Initializing search system from persistent storage...")
                
                # Load documents from persistent storage to rebuild in-memory structures
                collection_data = self.collection.get(include=['documents', 'metadatas'])
                
                if collection_data and collection_data['ids']:
                    self.logger.info(f"Loading {len(collection_data['ids'])} chunks from persistent storage...")
                    
                    # Rebuild the documents list from stored data
                    from .document_processor import DocumentChunk
                    self.documents = []
                    
                    for chunk_id, content, metadata in zip(
                        collection_data['ids'],
                        collection_data['documents'], 
                        collection_data['metadatas']
                    ):
                        # Reconstruct DocumentChunk objects
                        chunk = DocumentChunk(
                            chunk_id=chunk_id,
                            document_id=metadata.get('document_id', 'unknown'),
                            content=content,
                            chunk_index=int(metadata.get('chunk_index', 0)),
                            token_count=int(metadata.get('token_count', 0)),
                            section=metadata.get('section'),
                            page_number=int(metadata.get('page_number', 0)) if metadata.get('page_number', '0').isdigit() else None,
                            metadata=metadata
                        )
                        self.documents.append(chunk)
                    
                    self.logger.info(f"Rebuilt {len(self.documents)} document chunks")
                    
                    # Now build TF-IDF index from the reconstructed documents
                    self._update_tfidf_index()
                    
        except Exception as e:
            self.logger.error(f"Error initializing from persistent storage: {e}")
            # Initialize empty structures
            self.documents = []

    def add_documents(self, chunks: List[DocumentChunk]):
        """Add document chunks to the search index"""
        if not chunks:
            return
        
        # Check for duplicates to avoid re-processing
        existing_ids = set()
        try:
            # Get existing document IDs
            existing_data = self.collection.get()
            existing_ids = set(existing_data['ids'])
        except:
            pass
        
        # Filter out already processed chunks
        new_chunks = [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
        
        if not new_chunks:
            self.logger.info("All documents already processed, skipping...")
            # Still need to update local documents list for TF-IDF
            self.documents.extend(chunks)
            self._update_tfidf_index()
            return
        
        self.logger.info(f"Adding {len(new_chunks)} new chunks (out of {len(chunks)} total)")
        self.documents.extend(new_chunks)
        
        # Prepare data for ChromaDB
        texts = [chunk.content for chunk in new_chunks]
        ids = [chunk.chunk_id for chunk in new_chunks]
        metadatas = []
        
        for chunk in new_chunks:
            metadata = chunk.metadata.copy()
            metadata.update({
                'document_id': chunk.document_id,
                'section': chunk.section or 'unknown',
                'page_number': chunk.page_number or 0,
                'chunk_index': chunk.chunk_index,
                'token_count': chunk.token_count
            })
            # Convert all values to strings for ChromaDB
            metadata = {k: str(v) for k, v in metadata.items() if v is not None}
            metadatas.append(metadata)
        
        # Generate embeddings
        try:
            if hasattr(self.embedding_model, 'embed_documents'):
                # Ollama embeddings
                embeddings = self.embedding_model.embed_documents(texts)
            else:
                # Sentence transformers
                embeddings = self.embedding_model.encode(texts).tolist()
            
            # Add to ChromaDB
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            
            self.logger.info(f"Added {len(chunks)} chunks to search index")
            
        except Exception as e:
            self.logger.error(f"Error adding documents to index: {e}")
        
        # Update TF-IDF matrix
        self._update_tfidf_index()

    def clear_all_documents(self):
        """Completely clear all documents from the collection and reset the search engine"""
        try:
            # Delete the entire collection
            self.chroma_client.delete_collection(self.collection_name)
            self.logger.info("Deleted ChromaDB collection")
            
            # Recreate the collection
            self.collection = self.chroma_client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            self.logger.info("Recreated empty ChromaDB collection")
            
            # Clear local data structures
            self.documents = []
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            
            self.logger.info("Cleared all documents and reset search engine")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
            # Fallback: try to delete all items from existing collection
            try:
                all_data = self.collection.get()
                if all_data.get('ids'):
                    self.collection.delete(ids=all_data['ids'])
                    self.documents = []
                    self.logger.info("Fallback: Cleared all items from collection")
                    return True
            except Exception as fallback_error:
                self.logger.error(f"Fallback deletion also failed: {fallback_error}")
                return False

    def _update_tfidf_index(self):
        """Update TF-IDF index for keyword-based search"""
        try:
            # Ensure we have documents to work with
            if not hasattr(self, 'documents'):
                self.documents = []
            
            # If we have local documents, use them
            if self.documents:
                texts = [doc.content for doc in self.documents]
                self.logger.info(f"Building TF-IDF index from {len(texts)} local documents")
            else:
                # Try to rebuild from persistent database
                self.logger.info("Rebuilding TF-IDF index from persistent database...")
                collection_data = self.collection.get(include=['documents'])
                if collection_data and collection_data['documents']:
                    texts = collection_data['documents']
                    self.logger.info(f"Building TF-IDF index from {len(texts)} stored documents")
                else:
                    self.logger.warning("No documents available for TF-IDF indexing")
                    return
            
            if texts:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
                self.logger.info(f"Successfully updated TF-IDF index with {len(texts)} documents")
            else:
                self.logger.warning("No text content available for TF-IDF indexing")
                
        except Exception as e:
            self.logger.error(f"Error updating TF-IDF index: {e}")

    def search(self, query: str, max_results: int = None, relaxed: bool = False) -> List[SearchResult]:
        """Intelligent hierarchical search with adaptive parameters"""
        
        # Intelligent defaults based on collection size and query
        if max_results is None:
            collection_size = len(self.documents)
            if collection_size < 10:
                max_results = min(50, collection_size * 5)  # Much more comprehensive for small collections
            elif collection_size < 100:
                max_results = 75  # Significantly increased
            else:
                max_results = 100  # No artificial limits for large collections
        
        # Detect query type
        query_type = self._detect_query_type(query)
        
        if query_type == "author":
            return self._search_by_author(query, max_results)
        else:
            return self._search_by_content(query, max_results, relaxed)

    def _detect_query_type(self, query: str) -> str:
        """Detect the type of query (author, topic, etc.)"""
        query_lower = query.lower()
        
        # Check for author queries
        author_keywords = ['author', 'wrote', 'published', 'by', 'papers by', 'work by']
        if any(keyword in query_lower for keyword in author_keywords):
            return "author"
        
        # Check for patterns
        for pattern in self.author_query_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                return "author"
        
        return "content"

    def _search_by_author(self, query: str, max_results: int) -> List[SearchResult]:
        """Enhanced search for papers by specific authors"""
        # Extract author name from query
        author_name = self._extract_author_from_query(query)
        
        if not author_name:
            # Fallback to content search if no author detected
            return self._search_by_content(query, max_results)
        
        self.logger.info(f"Searching for author: '{author_name}'")
        
        # Search in multiple ways
        author_results = []
        
        # 1. Search in document metadata
        for i, doc in enumerate(self.documents):
            authors = doc.metadata.get('authors', [])
            if self._author_matches(author_name, authors):
                relevance = self._calculate_author_relevance(author_name, authors)
                
                result = SearchResult(
                    chunk=doc,
                    similarity_score=1.0,  # Perfect match for author
                    relevance_score=relevance,
                    rank=i
                )
                author_results.append(result)
        
        # 2. Search in filename (common in academic papers)
        for i, doc in enumerate(self.documents):
            filename = doc.metadata.get('filename', '')
            if self._author_in_filename(author_name, filename):
                relevance = 0.9  # High relevance for filename match
                
                result = SearchResult(
                    chunk=doc,
                    similarity_score=0.95,
                    relevance_score=relevance,
                    rank=i
                )
                author_results.append(result)
        
        # 3. Search in document title
        for i, doc in enumerate(self.documents):
            title = doc.metadata.get('title', '')
            if self._author_in_text(author_name, title):
                relevance = 0.8  # Good relevance for title match
                
                result = SearchResult(
                    chunk=doc,
                    similarity_score=0.9,
                    relevance_score=relevance,
                    rank=i
                )
                author_results.append(result)
        
        # 4. Search in content for author citations/mentions
        if not author_results:
            # If no metadata matches, search content
            content_results = self._search_author_in_content(author_name, max_results)
            author_results.extend(content_results)
        
        # Remove duplicates and sort by relevance
        seen_chunks = set()
        unique_results = []
        for result in author_results:
            chunk_id = id(result.chunk)
            if chunk_id not in seen_chunks:
                seen_chunks.add(chunk_id)
                unique_results.append(result)
        
        unique_results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        self.logger.info(f"Found {len(unique_results)} results for author '{author_name}'")
        return unique_results[:max_results]

    def _author_in_filename(self, author_name: str, filename: str) -> bool:
        """Check if author name appears in filename"""
        if not filename or not author_name:
            return False
        
        filename_lower = filename.lower()
        author_parts = author_name.lower().split()
        
        # Check if all author parts appear in filename
        return all(part in filename_lower for part in author_parts if len(part) > 2)

    def _author_in_text(self, author_name: str, text: str) -> bool:
        """Check if author name appears in text"""
        if not text or not author_name:
            return False
        
        text_lower = text.lower()
        author_parts = author_name.lower().split()
        
        # Check for various author mention patterns
        full_name = ' '.join(author_parts)
        if full_name in text_lower:
            return True
        
        # Check for last name only
        if len(author_parts) > 1 and author_parts[-1] in text_lower:
            return True
        
        return False

    def _search_author_in_content(self, author_name: str, max_results: int) -> List[SearchResult]:
        """Search for author mentions in document content"""
        results = []
        author_parts = author_name.lower().split()
        
        for i, doc in enumerate(self.documents):
            content_lower = doc.content.lower()
            
            # Look for author name patterns in content
            score = 0
            if ' '.join(author_parts) in content_lower:
                score = 0.7  # Full name match
            elif len(author_parts) > 1 and author_parts[-1] in content_lower:
                score = 0.5  # Last name match
            
            if score > 0:
                result = SearchResult(
                    chunk=doc,
                    similarity_score=score,
                    relevance_score=score,
                    rank=i
                )
                results.append(result)
        
        return results

    def _extract_author_from_query(self, query: str) -> Optional[str]:
        """Extract author name from query with enhanced pattern matching"""
        query_lower = query.lower()
        
        # Enhanced patterns to handle various phrasings and typos
        enhanced_patterns = [
            r'papers?\s+(?:where|were|written|authored|published)?\s*by\s+([a-zA-Z\s]+?)(?:\s|$|,|\?|\.)',
            r'(?:work|research|papers?)\s+(?:of|from|by)\s+([a-zA-Z\s]+?)(?:\s|$|,|\?|\.)',
            r'author[s]?\s+([a-zA-Z\s]+?)(?:\s|$|,|\?|\.)',
            r'([a-zA-Z\s]+?)\s+(?:papers?|work|research|publications?)',
            r'written\s+by\s+([a-zA-Z\s]+?)(?:\s|$|,|\?|\.)',
            r'published\s+by\s+([a-zA-Z\s]+?)(?:\s|$|,|\?|\.)'
        ]
        
        for pattern in enhanced_patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                author_name = match.group(1).strip()
                # Clean up common words that might be captured
                stopwords = ['and', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'with', 'what', 'how', 'where', 'were']
                author_words = [word for word in author_name.split() if word.lower() not in stopwords]
                if author_words and len(author_words) <= 4:  # Reasonable author name length
                    return ' '.join(author_words)
        
        # Fallback: look for capitalized words after author triggers
        words = query.split()
        author_triggers = ['by', 'from', 'authored', 'wrote', 'published', 'author']
        
        for i, word in enumerate(words):
            if word.lower() in author_triggers and i + 1 < len(words):
                # Take next 1-3 words as author name
                author_words = []
                for j in range(i + 1, min(i + 4, len(words))):
                    if j < len(words) and words[j] and words[j][0].isupper():
                        author_words.append(words[j])
                    else:
                        break
                if author_words:
                    return ' '.join(author_words)
        
        return None

    def _author_matches(self, query_author: str, doc_authors: List[str]) -> bool:
        """Check if query author matches document authors with enhanced matching"""
        query_author = query_author.lower().strip()
        
        for author in doc_authors:
            author_lower = author.lower().strip()
            
            # Exact match
            if query_author == author_lower:
                return True
            
            # Split names for flexible matching
            query_parts = [part.strip() for part in query_author.split() if part.strip()]
            author_parts = [part.strip() for part in author_lower.split() if part.strip()]
            
            if not query_parts or not author_parts:
                continue
            
            # Check various matching strategies
            
            # 1. Last name matching
            if len(query_parts) > 0 and len(author_parts) > 0:
                if query_parts[-1] == author_parts[-1]:
                    return True
            
            # 2. First + Last name matching
            if len(query_parts) >= 2 and len(author_parts) >= 2:
                if (query_parts[0] == author_parts[0] and 
                    query_parts[-1] == author_parts[-1]):
                    return True
            
            # 3. Any significant part matching (for names with middle initials, etc.)
            significant_matches = 0
            for qp in query_parts:
                if len(qp) > 2:  # Only consider parts longer than 2 chars
                    for ap in author_parts:
                        if qp in ap or ap in qp:
                            significant_matches += 1
                            break
            
            # If most query parts match, consider it a match
            if len(query_parts) > 1 and significant_matches >= len(query_parts) - 1:
                return True
            elif len(query_parts) == 1 and significant_matches > 0:
                return True
            
            # 4. Fuzzy matching for common typos (simple edit distance)
            if self._fuzzy_name_match(query_author, author_lower):
                return True
        
        return False

    def _fuzzy_name_match(self, query_name: str, author_name: str) -> bool:
        """Simple fuzzy matching for name typos"""
        # Only apply to reasonably similar length names
        if abs(len(query_name) - len(author_name)) > 3:
            return False
        
        # Simple character-based similarity
        query_chars = set(query_name.replace(' ', ''))
        author_chars = set(author_name.replace(' ', ''))
        
        # Calculate overlap
        overlap = len(query_chars & author_chars)
        total = len(query_chars | author_chars)
        
        if total == 0:
            return False
        
        similarity = overlap / total
        return similarity >= 0.8  # 80% character overlap

    def _calculate_author_relevance(self, query_author: str, doc_authors: List[str]) -> float:
        """Calculate relevance score for author match"""
        if not doc_authors:
            return 0.0
        
        query_author = query_author.lower()
        max_score = 0.0
        
        for author in doc_authors:
            author_lower = author.lower()
            
            # Exact match
            if query_author == author_lower:
                return 1.0
            
            # Partial match scoring
            query_parts = set(query_author.split())
            author_parts = set(author_lower.split())
            
            if query_parts and author_parts:
                overlap = len(query_parts.intersection(author_parts))
                total = len(query_parts.union(author_parts))
                score = overlap / total if total > 0 else 0.0
                max_score = max(max_score, score)
        
        return max_score

    def _search_by_content(self, query: str, max_results: int, relaxed: bool = False) -> List[SearchResult]:
        """Content-based search with adaptive parameters"""
        
        # Adjust search intensity based on relaxed flag
        search_multiplier = 3 if relaxed else 2
        
        # Strategy 1: Vector similarity search
        vector_results = self._vector_search(query, max_results * search_multiplier)
        
        # Strategy 2: Keyword/TF-IDF search (always run this as backup)
        keyword_results = self._keyword_search(query, max_results * search_multiplier)
        
        # If vector search failed but we have documents, rely more heavily on keyword search
        if not vector_results and len(self.documents) > 0:
            self.logger.warning("Vector search returned no results, prioritizing keyword search")
            keyword_results = self._keyword_search(query, max_results * search_multiplier * 2)
            
            # Also try a broader keyword search
            if not keyword_results:
                # Try searching individual words from the query
                query_words = query.lower().split()
                for word in query_words:
                    if len(word) > 3:  # Only search meaningful words
                        word_results = self._keyword_search(word, max_results)
                        keyword_results.extend(word_results)
        
        # Strategy 3: Section-aware search
        section_boosted_results = self._apply_section_boost(vector_results, query)
        
        # Use adaptive similarity thresholds from config
        query_type = self._detect_query_type(query)
        base_threshold = self.config.get_adaptive_similarity_threshold(query_type, len(vector_results))
        
        if relaxed or not vector_results:
            # Lower thresholds for relaxed search or when vector search failed
            vector_threshold = max(0.1, base_threshold - 0.3)
            keyword_threshold = max(0.05, base_threshold - 0.4)
        else:
            # Use adaptive thresholds
            vector_threshold = base_threshold
            keyword_threshold = base_threshold - 0.1
        
        # Apply adaptive thresholds
        vector_results = [r for r in vector_results if r.similarity_score > vector_threshold]
        keyword_results = [r for r in keyword_results if r.similarity_score > keyword_threshold]
        
        # If still no results, do a final fallback search
        if not vector_results and not keyword_results and len(self.documents) > 0:
            self.logger.info("No results found, doing fallback content search...")
            fallback_results = self._fallback_content_search(query, max_results)
            return fallback_results
        
        # Combine and re-rank results
        combined_results = self._combine_and_rerank(
            vector_results, keyword_results, section_boosted_results, query
        )
        
        return combined_results[:max_results]

    def _vector_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Vector similarity search using embeddings"""
        try:
            # Check if we have any documents in the collection
            if self.collection.count() == 0:
                self.logger.warning("Vector search: No documents in collection")
                return []
            
            # Generate query embedding
            if hasattr(self.embedding_model, 'embed_query'):
                query_embedding = self.embedding_model.embed_query(query)
            else:
                query_embedding = self.embedding_model.encode([query])[0].tolist()
            
            # Search in ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(max_results, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['ids'][0]:
                self.logger.warning("Vector search returned no results")
                return []
            
            search_results = []
            
            for i, (doc_id, distance, metadata) in enumerate(zip(
                results['ids'][0], 
                results['distances'][0], 
                results['metadatas'][0]
            )):
                # Find the original chunk
                chunk = next((d for d in self.documents if d.chunk_id == doc_id), None)
                if chunk:
                    similarity = max(0, 1 - distance)  # Convert distance to similarity, ensure non-negative
                    
                    result = SearchResult(
                        chunk=chunk,
                        similarity_score=similarity,
                        relevance_score=similarity,
                        rank=i,
                        highlights=self._extract_highlights(chunk.content, query)
                    )
                    search_results.append(result)
                else:
                    self.logger.warning(f"Could not find chunk with ID: {doc_id}")
            
            self.logger.info(f"Vector search found {len(search_results)} results")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Error in vector search: {e}")
            # If vector search fails, return empty list so keyword search can take over
            return []

    def _keyword_search(self, query: str, max_results: int) -> List[SearchResult]:
        """TF-IDF based keyword search"""
        if self.tfidf_matrix is None or len(self.documents) == 0:
            self.logger.warning("Keyword search: No TF-IDF matrix or documents available")
            return []
        
        try:
            # Transform query
            query_vector = self.tfidf_vectorizer.transform([query])
            
            # Calculate similarities
            similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
            
            # Get top results
            top_indices = np.argsort(similarities)[::-1][:max_results]
            
            results = []
            for i, idx in enumerate(top_indices):
                if idx < len(self.documents) and similarities[idx] > 0:  # Only include non-zero similarities
                    chunk = self.documents[idx]
                    result = SearchResult(
                        chunk=chunk,
                        similarity_score=similarities[idx],
                        relevance_score=similarities[idx],
                        rank=i,
                        highlights=self._extract_highlights(chunk.content, query)
                    )
                    results.append(result)
            
            self.logger.info(f"Keyword search found {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            return []

    def _apply_section_boost(self, results: List[SearchResult], query: str) -> List[SearchResult]:
        """Boost results based on section relevance"""
        # Define section importance for different query types
        section_weights = {
            'abstract': 1.2,
            'introduction': 1.0,
            'methods': 1.1,
            'results': 1.3,
            'discussion': 1.2,
            'conclusion': 1.1,
        }
        
        # Detect query intent to adjust weights
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['method', 'approach', 'technique']):
            section_weights['methods'] = 1.5
        elif any(word in query_lower for word in ['result', 'finding', 'outcome']):
            section_weights['results'] = 1.5
        elif any(word in query_lower for word in ['conclusion', 'summary']):
            section_weights['conclusion'] = 1.5
        
        # Apply boosts
        boosted_results = []
        for result in results:
            section = result.chunk.section or 'unknown'
            boost = section_weights.get(section, 1.0)
            
            boosted_result = SearchResult(
                chunk=result.chunk,
                similarity_score=result.similarity_score,
                relevance_score=result.relevance_score * boost,
                rank=result.rank,
                highlights=result.highlights
            )
            boosted_results.append(boosted_result)
        
        return boosted_results

    def _combine_and_rerank(
        self, 
        vector_results: List[SearchResult], 
        keyword_results: List[SearchResult],
        section_results: List[SearchResult],
        query: str
    ) -> List[SearchResult]:
        """Combine different search strategies and re-rank"""
        
        # Create a dictionary to combine results by chunk ID
        combined = {}
        
        # Add vector results
        for result in vector_results:
            chunk_id = result.chunk.chunk_id
            combined[chunk_id] = {
                'result': result,
                'vector_score': result.similarity_score,
                'keyword_score': 0.0,
                'section_score': result.relevance_score
            }
        
        # Add keyword results
        for result in keyword_results:
            chunk_id = result.chunk.chunk_id
            if chunk_id in combined:
                combined[chunk_id]['keyword_score'] = result.similarity_score
            else:
                combined[chunk_id] = {
                    'result': result,
                    'vector_score': 0.0,
                    'keyword_score': result.similarity_score,
                    'section_score': result.relevance_score
                }
        
        # Calculate final scores
        final_results = []
        for chunk_id, data in combined.items():
            # Weighted combination of scores
            final_score = (
                0.5 * data['vector_score'] +
                0.3 * data['keyword_score'] +
                0.2 * data['section_score']
            )
            
            result = SearchResult(
                chunk=data['result'].chunk,
                similarity_score=data['vector_score'],
                relevance_score=final_score,
                rank=0,  # Will be updated after sorting
                highlights=data['result'].highlights
            )
            final_results.append(result)
        
        # Sort by final score and update ranks
        final_results.sort(key=lambda x: x.relevance_score, reverse=True)
        for i, result in enumerate(final_results):
            result.rank = i
        
        # Remove duplicates and apply diversity
        return self._apply_diversity_filter(final_results)

    def _apply_diversity_filter(self, results: List[SearchResult]) -> List[SearchResult]:
        """Apply diversity filtering to avoid too many results from same document"""
        if not results:
            return results
        
        filtered_results = []
        doc_count = defaultdict(int)
        max_per_doc = 8  # Increased maximum chunks per document for better coverage
        
        for result in results:
            doc_id = result.chunk.document_id
            
            if doc_count[doc_id] < max_per_doc:
                filtered_results.append(result)
                doc_count[doc_id] += 1
        
        return filtered_results

    def _extract_highlights(self, text: str, query: str, max_highlights: int = 3) -> List[str]:
        """Extract relevant highlights from text"""
        query_words = query.lower().split()
        sentences = re.split(r'[.!?]+', text)
        
        highlights = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Skip very short sentences
                continue
            
            sentence_lower = sentence.lower()
            
            # Count query word matches
            matches = sum(1 for word in query_words if word in sentence_lower)
            
            if matches > 0:
                highlights.append((sentence, matches))
        
        # Sort by match count and take top highlights
        highlights.sort(key=lambda x: x[1], reverse=True)
        
        return [h[0] for h in highlights[:max_highlights]]

    def _fallback_content_search(self, query: str, max_results: int) -> List[SearchResult]:
        """Fallback search when both vector and keyword search fail"""
        self.logger.info("Running fallback content search...")
        results = []
        query_lower = query.lower()
        query_words = [word.strip('.,!?') for word in query_lower.split() if len(word.strip('.,!?')) > 2]
        
        for i, doc in enumerate(self.documents):
            content_lower = doc.content.lower()
            score = 0
            matches = 0
            
            # Count word matches
            for word in query_words:
                if word in content_lower:
                    matches += 1
                    # Give higher score for exact phrase matches
                    score += content_lower.count(word)
            
            # Calculate relevance score
            if matches > 0:
                relevance_score = (matches / len(query_words)) * 0.7 + (score / len(content_lower.split())) * 0.3
                
                result = SearchResult(
                    chunk=doc,
                    similarity_score=relevance_score,
                    relevance_score=relevance_score,
                    rank=i,
                    highlights=self._extract_highlights(doc.content, query)
                )
                results.append(result)
        
        # Sort by relevance and return top results
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        self.logger.info(f"Fallback search found {len(results)} results")
        return results[:max_results]

    def get_document_details(self) -> List[Dict[str, Any]]:
        """Get detailed information about all documents in the collection"""
        try:
            # Get all data from persistent database
            collection_data = self.collection.get(include=['documents', 'metadatas'])
            
            if not collection_data['ids']:
                return []
            
            documents = []
            doc_info = {}  # Group chunks by document_id
            
            # Process all chunks and group by document
            for chunk_id, metadata in zip(collection_data['ids'], collection_data['metadatas']):
                if not metadata:
                    continue
                
                doc_id = metadata.get('document_id', 'unknown')
                
                if doc_id not in doc_info:
                    doc_info[doc_id] = {
                        'document_id': doc_id,
                        'title': metadata.get('title', 'Unknown Title'),
                        'paper_title': metadata.get('paper_title', metadata.get('title', 'Unknown Title')),
                        'title_confidence': metadata.get('title_confidence', 0.0),
                        'filename': metadata.get('filename', 'Unknown File'),
                        'file_path': metadata.get('file_path', ''),
                        'authors': [],
                        'enhanced_authors': [],
                        'year': metadata.get('year'),
                        'publication_year': metadata.get('publication_year'),
                        'journal': metadata.get('journal'),
                        'venue': metadata.get('venue'),
                        'doi': metadata.get('doi'),
                        'research_domain': metadata.get('research_domain'),
                        'key_topics': metadata.get('key_topics', []),
                        'document_type': metadata.get('document_type', 'unknown'),
                        'processing_quality': metadata.get('processing_quality', 'unknown'),
                        'total_chunks': 0,
                        'sections': set(),
                        'pages': set()
                    }
                
                # Update chunk count
                doc_info[doc_id]['total_chunks'] += 1
                
                # Add section if available
                if metadata.get('section'):
                    doc_info[doc_id]['sections'].add(metadata['section'])
                
                # Add page if available
                if metadata.get('page_number'):
                    try:
                        page_num = int(metadata['page_number'])
                        doc_info[doc_id]['pages'].add(page_num)
                    except:
                        pass
                
                # Handle authors (basic)
                if metadata.get('authors') and not doc_info[doc_id]['authors']:
                    author_str = metadata['authors']
                    if author_str.startswith('[') and author_str.endswith(']'):
                        try:
                            import ast
                            doc_info[doc_id]['authors'] = ast.literal_eval(author_str)
                        except:
                            doc_info[doc_id]['authors'] = [author_str]
                    else:
                        doc_info[doc_id]['authors'] = [author_str]
                
                # Handle enhanced authors
                if metadata.get('enhanced_authors') and not doc_info[doc_id]['enhanced_authors']:
                    enhanced_author_str = metadata['enhanced_authors']
                    if enhanced_author_str.startswith('[') and enhanced_author_str.endswith(']'):
                        try:
                            import ast
                            doc_info[doc_id]['enhanced_authors'] = ast.literal_eval(enhanced_author_str)
                        except:
                            doc_info[doc_id]['enhanced_authors'] = [enhanced_author_str]
                    else:
                        doc_info[doc_id]['enhanced_authors'] = [enhanced_author_str]
            
            # Convert to list and clean up
            for doc_data in doc_info.values():
                doc_data['sections'] = sorted(list(doc_data['sections']))
                doc_data['page_range'] = f"{min(doc_data['pages'])}-{max(doc_data['pages'])}" if doc_data['pages'] else "Unknown"
                del doc_data['pages']  # Remove the set, keep page_range
                
                # Convert year fields to integers safely
                for year_field in ['year', 'publication_year']:
                    if doc_data.get(year_field):
                        try:
                            year_val = doc_data[year_field]
                            if isinstance(year_val, str) and year_val.isdigit():
                                doc_data[year_field] = int(year_val)
                            elif isinstance(year_val, (int, float)):
                                doc_data[year_field] = int(year_val)
                        except (ValueError, TypeError):
                            # Keep original value if conversion fails
                            pass
                
                # Ensure confidence is a float
                if 'title_confidence' in doc_data:
                    try:
                        doc_data['title_confidence'] = float(doc_data['title_confidence'])
                    except (ValueError, TypeError):
                        doc_data['title_confidence'] = 0.0
                
                documents.append(doc_data)
            
            # Sort by title
            documents.sort(key=lambda x: x['title'])
            
            return documents
            
        except Exception as e:
            self.logger.error(f"Error getting document details: {e}")
            return []

    def get_document_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            # Get data from persistent database
            collection_data = self.collection.get()
            total_chunks = self.collection.count()
            
            if total_chunks == 0:
                return {}
            
            # Extract document IDs and metadata
            metadatas = collection_data.get('metadatas', [])
            doc_ids = set()
            sections = []
            authors = []
            
            for metadata in metadatas:
                if metadata:
                    doc_ids.add(metadata.get('document_id', 'unknown'))
                    if metadata.get('section'):
                        sections.append(metadata['section'])
                    
                    # Handle authors (stored as strings in metadata)
                    if metadata.get('authors'):
                        author_str = metadata['authors']
                        # Parse author string back to list
                        if author_str.startswith('[') and author_str.endswith(']'):
                            try:
                                import ast
                                authors.extend(ast.literal_eval(author_str))
                            except:
                                authors.append(author_str)
                        else:
                            authors.append(author_str)
            
            return {
                'total_documents': len(doc_ids),
                'total_chunks': total_chunks,
                'sections_found': dict(Counter(sections)),
                'unique_authors': len(set(authors)),
                'most_common_authors': Counter(authors).most_common(10)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting document stats: {e}")
            # Fallback to in-memory data
            if not self.documents:
                return {}
            
            doc_ids = set(doc.document_id for doc in self.documents)
            sections = [doc.section for doc in self.documents if doc.section]
            authors = []
            
            for doc in self.documents:
                if doc.metadata.get('authors'):
                    authors.extend(doc.metadata['authors'])
            
            return {
                'total_documents': len(doc_ids),
                'total_chunks': len(self.documents),
                'sections_found': dict(Counter(sections)),
                'unique_authors': len(set(authors)),
                'most_common_authors': Counter(authors).most_common(10)
            }
