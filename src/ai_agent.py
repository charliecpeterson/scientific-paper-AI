"""
AI Agent for Scientific Paper Analysis
Implements sophisticated reasoning and cross-document synthesis
"""

import re
from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

try:
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    logging.warning("Ollama not available, using fallback")

from .intelligent_search import IntelligentSearch, SearchResult

@dataclass
class AgentResponse:
    """Response from the AI agent"""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    reasoning: str = ""

class ScientificPaperAgent:
    """Intelligent AI agent for scientific paper analysis"""
    
    def __init__(self, search_engine: IntelligentSearch, config):
        self.search_engine = search_engine
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize LLM
        self.llm = self._initialize_llm()
        
        # Analysis templates
        self.templates = {
            'general_analysis': self._get_general_analysis_template(),
            'author_query': self._get_author_query_template(),
            'comparison': self._get_comparison_template(),
            'synthesis': self._get_synthesis_template(),
            'fact_verification': self._get_fact_verification_template()
        }

    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            if OLLAMA_AVAILABLE:
                return ChatOllama(
                    model=self.config.llm_model,
                    temperature=0.1,
                    base_url="http://localhost:11434"
                )
            else:
                # Fallback - you could implement other LLM providers here
                self.logger.warning("Using fallback LLM - implement your preferred provider")
                return None
        except Exception as e:
            self.logger.error(f"Error initializing LLM: {e}")
            return None

    def query(self, user_query: str) -> Dict[str, Any]:
        """Process user query with intelligent adaptive search"""
        try:
            # Step 1: Classify query to understand complexity and requirements
            query_type = self._classify_query(user_query)
            query_complexity = self._assess_query_complexity(user_query, query_type)
            
            # Step 2: Intelligent adaptive search with multiple passes if needed
            search_results = self._adaptive_search(user_query, query_type, query_complexity)
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information in your documents for this query. Try rephrasing your question or using different keywords.",
                    "sources": [],
                    "confidence": 0.0,
                    "reasoning": "No relevant content found despite comprehensive search across all documents"
                }
            
            # Step 3: Select appropriate template and generate response
            template = self.templates.get(query_type, self.templates['general_analysis'])
            response = self._generate_response(user_query, search_results, template)
            
            # Step 4: Format sources with intelligent ranking
            formatted_sources = self._format_sources(search_results)
            
            return {
                "answer": response.answer,
                "sources": formatted_sources,
                "confidence": response.confidence,
                "reasoning": response.reasoning
            }
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return {
                "answer": f"An error occurred while processing your query: {str(e)}",
                "sources": [],
                "confidence": 0.0
            }

    def _assess_query_complexity(self, query: str, query_type: str) -> str:
        """Assess query complexity to determine search strategy"""
        query_lower = query.lower()
        
        # Simple queries - direct factual questions
        simple_indicators = ['what is', 'define', 'who is', 'when did', 'where is']
        if any(indicator in query_lower for indicator in simple_indicators):
            return 'simple'
        
        # Complex synthesis queries
        synthesis_indicators = ['across all papers', 'overall', 'in general', 'consensus', 'what do papers say']
        if any(indicator in query_lower for indicator in synthesis_indicators):
            return 'synthesis'
        
        # Comparison queries
        comparison_indicators = ['compare', 'contrast', 'difference', 'versus', 'vs', 'similar']
        if any(indicator in query_lower for indicator in comparison_indicators):
            return 'comparison'
        
        # Verification queries
        verification_indicators = ['according to', 'verify', 'confirm', 'is it true', 'correct']
        if any(indicator in query_lower for indicator in verification_indicators):
            return 'verification'
        
        # Author queries
        if query_type == 'author_query':
            return 'author'
        
        return 'moderate'

    def _adaptive_search(self, query: str, query_type: str, complexity: str) -> List[SearchResult]:
        """Smart RAG search that adapts to find relevant content across ALL documents"""
        # Get collection stats for adaptive behavior
        collection_stats = self.search_engine.get_document_stats()
        total_docs = collection_stats.get('total_documents', 1)
        
        # Determine optimal search parameters - cast wider net for comprehensive coverage
        max_chunks = self.config.get_adaptive_max_chunks(complexity, total_docs)
        
        # Primary search across ALL documents
        initial_results = self.search_engine.search(query, max_results=max_chunks)
        
        # If no results, try progressively more permissive searches
        if not initial_results:
            self.logger.info("No initial results, trying relaxed search...")
            initial_results = self.search_engine.search(query, max_results=max_chunks, relaxed=True)
        
        # If still few results, expand search scope significantly
        if len(initial_results) < 5:
            self.logger.info(f"Found only {len(initial_results)} results, expanding search scope...")
            
            # Try with expanded query terms
            expanded_query = self._expand_query_terms(query)
            expanded_results = self.search_engine.search(expanded_query, max_results=max_chunks * 2, relaxed=True)
            
            # Combine and deduplicate results
            all_results = initial_results + expanded_results
            unique_results = self._deduplicate_results(all_results)
            
            if len(unique_results) > len(initial_results):
                initial_results = unique_results
        
        # For comprehensive coverage, ensure we have enough results
        if len(initial_results) < 10 and complexity in ['synthesis', 'author', 'comparison']:
            self.logger.info("Ensuring comprehensive coverage for complex query...")
            # Cast an even wider net
            broad_results = self.search_engine.search(query, max_results=max_chunks * 3, relaxed=True)
            if len(broad_results) > len(initial_results):
                initial_results = broad_results
        
        return initial_results[:max_chunks]  # Return up to max_chunks for processing

    def _expand_query_terms(self, query: str) -> str:
        """Expand query with related terms for better document coverage"""
        query_lower = query.lower()
        
        # Extract key terms and add synonyms/related terms
        expanded_terms = []
        
        # Add common scientific synonyms and related terms
        term_expansions = {
            'method': ['approach', 'technique', 'procedure', 'methodology'],
            'result': ['finding', 'outcome', 'conclusion', 'data'],
            'analysis': ['study', 'investigation', 'examination', 'evaluation'],
            'model': ['framework', 'system', 'approach', 'theory'],
            'effect': ['impact', 'influence', 'consequence', 'result'],
            'property': ['characteristic', 'feature', 'attribute', 'behavior'],
            'structure': ['organization', 'arrangement', 'composition', 'architecture'],
            'process': ['procedure', 'method', 'mechanism', 'pathway']
        }
        
        # Add relevant expansions
        words = query.split()
        for word in words:
            word_lower = word.lower().strip('.,?!')
            if word_lower in term_expansions:
                # Add one related term to avoid query explosion
                expanded_terms.append(term_expansions[word_lower][0])
        
        # Construct expanded query (keep it reasonable)
        if expanded_terms:
            return f"{query} {' '.join(expanded_terms[:2])}"  # Add max 2 expansion terms
        
        return query

    def _deduplicate_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Remove duplicate results while preserving best scores"""
        seen_content = {}
        unique_results = []
        
        for result in results:
            # Use content hash as key for deduplication
            content_key = hash(result.chunk.content[:200])  # Use first 200 chars as signature
            
            if content_key not in seen_content:
                seen_content[content_key] = result
                unique_results.append(result)
            else:
                # Keep the result with higher relevance score
                existing = seen_content[content_key]
                if result.relevance_score > existing.relevance_score:
                    # Replace in the list
                    idx = unique_results.index(existing)
                    unique_results[idx] = result
                    seen_content[content_key] = result
        
        return unique_results

    def _optimize_results(self, results: List[SearchResult], query: str, complexity: str) -> List[SearchResult]:
        """Optimize search results based on quality assessment"""
        if not results:
            return results
        
        # Check result quality metrics
        avg_relevance = sum(r.relevance_score for r in results) / len(results)
        
        # If results are low quality, try to expand search
        if avg_relevance < 0.4:  # Lower threshold for more inclusive search
            self.logger.info(f"Low relevance ({avg_relevance:.2f}), expanding search...")
            # Try with more permissive search
            expanded_results = self.search_engine.search(query, max_results=len(results) * 2, relaxed=True)
            if expanded_results and len(expanded_results) > len(results):
                return expanded_results[:len(results) * 2]  # Return more results
        
        return results

    def _analyze_query_complexity(self, query: str) -> str:
        """Analyze query complexity to determine optimal search parameters"""
        query_lower = query.lower()
        
        # Simple queries - direct factual questions
        simple_patterns = [
            'what is', 'define', 'definition of', 'meaning of', 'explain',
            'how many', 'when was', 'where is', 'who is', 'which'
        ]
        if any(pattern in query_lower for pattern in simple_patterns):
            return 'simple'
        
        # Comparison queries - need multiple sources
        comparison_patterns = [
            'compare', 'versus', 'vs', 'difference between', 'similarities',
            'contrast', 'both', 'either', 'better than', 'worse than'
        ]
        if any(pattern in query_lower for pattern in comparison_patterns):
            return 'comparison'
        
        # Synthesis queries - need broad coverage
        synthesis_patterns = [
            'what do papers say', 'according to literature', 'research shows',
            'studies suggest', 'consensus', 'overall', 'in general',
            'across studies', 'multiple papers', 'various authors'
        ]
        if any(pattern in query_lower for pattern in synthesis_patterns):
            return 'synthesis'
        
        # Verification queries - need thorough evidence
        verification_patterns = [
            'is it true', 'verify', 'confirm', 'accurate', 'correct',
            'evidence for', 'support for', 'proof that', 'validate'
        ]
        if any(pattern in query_lower for pattern in verification_patterns):
            return 'verification'
        
        # Author queries - need author-specific search
        author_patterns = [
            'author', 'wrote', 'published by', 'papers by', 'work by',
            'research by', 'findings by', 'studies by'
        ]
        if any(pattern in query_lower for pattern in author_patterns):
            return 'author'
        
        # Complex analysis by default for detailed questions
        complex_indicators = ['analyze', 'discuss', 'elaborate', 'comprehensive', 'detailed']
        if any(indicator in query_lower for indicator in complex_indicators) or len(query.split()) > 10:
            return 'synthesis'  # Treat complex questions as synthesis queries
        
        return 'general_analysis'  # Default for moderate complexity

    def _classify_query(self, query: str) -> str:
        """Classify query type for appropriate template selection"""
        query_lower = query.lower()
        
        # Author queries - looking for specific researchers
        if any(pattern in query_lower for pattern in ['author', 'wrote', 'published', 'by', 'papers by', 'work by']):
            return 'author_query'
        
        # Comparison queries - comparing concepts/methods
        if any(word in query_lower for word in ['compare', 'versus', 'vs', 'difference', 'similar', 'contrast']):
            return 'comparison'
        
        # Synthesis queries - asking about overall findings
        if any(word in query_lower for word in ['what do papers say', 'overall', 'in general', 'consensus', 'literature']):
            return 'synthesis'
        
        # Default to general analysis for most questions
        return 'general_analysis'

    def _generate_response(self, query: str, search_results: List[SearchResult], template: str) -> AgentResponse:
        """Generate intelligent response using LLM"""
        
        # Determine query complexity for adaptive context preparation
        complexity = self._analyze_query_complexity(query)
        total_docs = len(search_results)
        
        # Pre-process query to identify key concepts for better matching
        query_concepts = self._extract_query_concepts(query)
        
        # Prepare context from search results
        context = self._prepare_context(search_results, complexity, total_docs)
        
        # Add concept matching hints if relevant
        concept_guidance = self._generate_concept_guidance(query, query_concepts, search_results)
        
        # Create prompt with enhanced guidance
        enhanced_template = template
        if concept_guidance:
            enhanced_template = f"{template}\n\nCONCEPT MATCHING GUIDANCE:\n{concept_guidance}"
        
        prompt = enhanced_template.format(
            query=query,
            context=context,
            num_sources=len(search_results)
        )
        
        if self.llm is None:
            # Fallback response
            return self._generate_fallback_response(query, search_results)
        
        try:
            # Generate response using LLM
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response and extract confidence
            parsed_response = self._parse_llm_response(content)
            
            return AgentResponse(
                answer=parsed_response['answer'],
                sources=[],  # Will be added separately
                confidence=parsed_response['confidence'],
                reasoning=parsed_response.get('reasoning', '')
            )
            
        except Exception as e:
            self.logger.error(f"Error generating LLM response: {e}")
            return self._generate_fallback_response(query, search_results)

    def _extract_query_concepts(self, query: str) -> List[str]:
        """Extract key concepts from the user query"""
        concepts = []
        query_lower = query.lower()
        
        # Map common query terms to broader concepts
        concept_mapping = {
            'composite method': ['composite', 'ccca', 'correlation consistent composite approach'],
            'machine learning': ['ml', 'neural network', 'ai', 'artificial intelligence'],
            'optimization': ['optimize', 'minimization', 'maximization'],
            'computational': ['computation', 'calculate', 'simulate'],
            'experimental': ['experiment', 'measure', 'empirical']
        }
        
        for concept, synonyms in concept_mapping.items():
            if any(syn in query_lower for syn in synonyms) or concept in query_lower:
                concepts.append(concept)
        
        return concepts

    def _generate_concept_guidance(self, query: str, query_concepts: List[str], search_results: List[SearchResult]) -> str:
        """Generate guidance to help LLM make concept connections"""
        if not query_concepts:
            return ""
        
        guidance_parts = []
        
        # Check if we have potential semantic matches
        for concept in query_concepts:
            if concept == 'composite method':
                # Look for composite-related content in results
                composite_matches = []
                for result in search_results:
                    content_lower = result.chunk.content.lower()
                    if any(term in content_lower for term in ['composite', 'ccca', 'correlation consistent']):
                        composite_matches.append(result)
                
                if composite_matches:
                    guidance_parts.append(
                        f"NOTE: When looking for '{concept}', consider that 'ccCA' (correlation consistent Composite Approach), "
                        f"'composite approaches', and similar terminology are directly relevant."
                    )
        
        return '\n'.join(guidance_parts)

    def _prepare_context(self, search_results: List[SearchResult], complexity: str, total_docs: int) -> str:
        """Prepare context from search results for LLM using adaptive chunking"""
        context_parts = []
        
        # Use adaptive max_chunks instead of fixed value
        max_chunks = self.config.get_adaptive_max_chunks(complexity, total_docs)
        
        for i, result in enumerate(search_results[:max_chunks]):
            chunk = result.chunk
            
            # Analyze content for key terms and concepts to help LLM make connections
            content = chunk.content
            key_concepts = self._extract_key_concepts(content)
            
            # Create a structured context entry with clearer formatting and concept hints
            context_entry = f"""
=== SOURCE {i+1} ===
DOCUMENT: {chunk.metadata.get('title', 'Unknown Title')}
AUTHORS: {', '.join(chunk.metadata.get('authors', ['Unknown']))}
SECTION: {chunk.section or 'Unknown'}
PAGE: {chunk.page_number or 'Unknown'}
RELEVANCE: {result.relevance_score:.3f}
KEY_CONCEPTS: {', '.join(key_concepts) if key_concepts else 'None detected'}

CONTENT:
{content}

"""
            context_parts.append(context_entry)
        
        return '\n'.join(context_parts)

    def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key scientific concepts from content to help LLM make connections"""
        concepts = []
        content_lower = content.lower()
        
        # Define concept patterns to look for
        concept_patterns = {
            'composite methods': ['composite', 'ccca', 'correlation consistent composite approach'],
            'machine learning': ['machine learning', 'ml', 'neural network', 'deep learning', 'ai'],
            'optimization': ['optimization', 'minimization', 'maximization', 'optimize'],
            'modeling': ['model', 'modeling', 'simulation', 'computational'],
            'experimental': ['experiment', 'experimental', 'measurement', 'data'],
            'theoretical': ['theory', 'theoretical', 'calculation', 'computed'],
            'analysis': ['analysis', 'analyze', 'study', 'investigation'],
            'comparison': ['compare', 'comparison', 'versus', 'relative to']
        }
        
        for concept_name, patterns in concept_patterns.items():
            if any(pattern in content_lower for pattern in patterns):
                concepts.append(concept_name)
        
        # Also look for specific abbreviations and methods
        abbreviations = {
            'ccCA': 'correlation consistent Composite Approach',
            'DFT': 'Density Functional Theory',
            'MD': 'Molecular Dynamics',
            'QM': 'Quantum Mechanics',
            'AI': 'Artificial Intelligence',
            'ML': 'Machine Learning'
        }
        
        for abbrev, full_name in abbreviations.items():
            if abbrev.lower() in content_lower:
                concepts.append(f"{abbrev} ({full_name})")
        
        return concepts

    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract answer, confidence, and reasoning"""
        # Try to extract confidence score
        confidence_match = re.search(r'Confidence:\s*(\d+(?:\.\d+)?)', response, re.IGNORECASE)
        confidence = float(confidence_match.group(1)) / 100 if confidence_match else 0.8
        
        # Try to extract reasoning
        reasoning_match = re.search(r'Reasoning:\s*(.*?)(?=\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        
        # Clean up the main answer (remove confidence and reasoning sections)
        answer = re.sub(r'Confidence:\s*\d+(?:\.\d+)?%?', '', response, flags=re.IGNORECASE)
        answer = re.sub(r'Reasoning:\s*.*?(?=\n\n|\Z)', '', answer, flags=re.IGNORECASE | re.DOTALL)
        answer = answer.strip()
        
        return {
            'answer': answer,
            'confidence': confidence,
            'reasoning': reasoning
        }

    def _generate_fallback_response(self, query: str, search_results: List[SearchResult]) -> AgentResponse:
        """Generate a fallback response when LLM is not available"""
        if not search_results:
            return AgentResponse(
                answer="No relevant information found.",
                sources=[],
                confidence=0.0
            )
        
        # Simple extractive approach
        top_results = search_results[:3]
        answer_parts = []
        
        for result in top_results:
            chunk = result.chunk
            title = chunk.metadata.get('title', 'Unknown Document')
            section = chunk.section or 'unknown section'
            
            answer_parts.append(f"From '{title}' ({section}): {chunk.content[:200]}...")
        
        answer = "\n\n".join(answer_parts)
        
        return AgentResponse(
            answer=answer,
            sources=[],
            confidence=0.6,
            reasoning="Fallback extractive response due to LLM unavailability"
        )

    def _format_sources(self, search_results: List[SearchResult]) -> List[Dict[str, Any]]:
        """Format search results as source citations"""
        sources = []
        
        for result in search_results:
            chunk = result.chunk
            
            source = {
                'title': chunk.metadata.get('title', 'Unknown Document'),
                'authors': chunk.metadata.get('authors', ['Unknown']),
                'section': chunk.section or 'Unknown Section',
                'page': chunk.page_number or 'Unknown',
                'relevance_score': result.relevance_score,
                'content': chunk.content[:300] + "..." if len(chunk.content) > 300 else chunk.content,
                'highlights': result.highlights or [],
                'year': chunk.metadata.get('year'),
                'journal': chunk.metadata.get('journal')
            }
            
            sources.append(source)
        
        return sources

    def _get_general_analysis_template(self) -> str:
        """Template for general analysis queries"""
        return """You are a scientific research assistant. Your job is to read the provided sources and answer the user's question based ONLY on what you find in those sources.

IMPORTANT INSTRUCTIONS:
1. READ EACH SOURCE CAREFULLY - Look at the document title, authors, section, and content
2. ONLY use information that is actually present in the sources
3. CITE SPECIFIC PAPERS - Use the actual document titles and authors provided
4. DO NOT make up or hallucinate information
5. If you cannot find relevant information, say so clearly

User Question: {query}

Here are {num_sources} sources I found in your document collection:

{context}

TASK: Answer the user's question using ONLY the information from these sources. For each relevant piece of information, cite the specific document title and section.

FORMAT YOUR ANSWER AS:
**Papers that used [relevant topic]:**

1. **[ACTUAL PAPER TITLE]** by [AUTHORS]
   - Section: [SECTION NAME] 
   - What they did: [SPECIFIC DESCRIPTION FROM THE CONTENT]
   - Quote: "[RELEVANT QUOTE FROM THE SOURCE]"

2. **[NEXT PAPER TITLE]** by [AUTHORS]
   - [Same format]

If no papers discuss the topic, say: "I could not find papers in your collection that specifically discuss [topic]."

Remember: Use ONLY the actual paper titles, authors, and content provided in the sources above. Do not invent or assume information."""

    def _get_author_query_template(self) -> str:
        """Template for author-specific queries"""
        return """You are analyzing scientific papers to answer questions about specific authors and their work.

User Question: {query}

Found Papers ({num_sources} documents):
{context}

Instructions:
1. Identify all papers by the requested author(s)
2. Provide bibliographic information for each paper
3. Summarize the author's research contributions
4. Note any recurring themes or methodologies
5. If asking about specific findings, cite the relevant papers and sections

Format your response as:
[List of papers and analysis of the author's work]

Confidence: [0-100]%"""

    def _get_comparison_template(self) -> str:
        """Template for comparison queries"""
        return """You are comparing information across multiple scientific papers.

User Question: {query}

Sources for Comparison ({num_sources} found):
{context}

Instructions:
1. Identify the specific aspects being compared
2. Present findings from different papers clearly
3. Highlight similarities and differences
4. Note any conflicting results or methodologies
5. Provide a balanced synthesis

Format your response as:
[Structured comparison with clear citations]

Confidence: [0-100]%
Reasoning: [Quality of comparison data and potential limitations]"""

    def _get_synthesis_template(self) -> str:
        """Template for synthesis across multiple papers"""
        return """You are synthesizing information across multiple scientific papers to provide a comprehensive overview.

User Question: {query}

Multiple Sources ({num_sources} papers):
{context}

Instructions:
1. Identify common themes and findings across papers
2. Note areas of consensus and disagreement
3. Synthesize information into a coherent overview
4. Highlight any research gaps or future directions mentioned
5. Provide a balanced perspective representing all sources

Format your response as:
[Comprehensive synthesis with citations from multiple sources]

Confidence: [0-100]%
Reasoning: [Assessment of synthesis quality and source coverage]"""

    def _get_fact_verification_template(self) -> str:
        """Template for fact verification queries"""
        return """You are verifying specific claims or statements against scientific literature.

User Question: {query}

Verification Sources ({num_sources} found):
{context}

Instructions:
1. Identify the specific claim being verified
2. Search for supporting or contradicting evidence in the sources
3. Provide direct quotes and citations for evidence
4. State clearly whether the claim is supported, contradicted, or unclear
5. Note any limitations in the available evidence

Format your response as:
VERIFICATION RESULT: [SUPPORTED/CONTRADICTED/UNCLEAR/PARTIALLY SUPPORTED]

[Detailed analysis with specific citations and quotes]

Confidence: [0-100]%
Reasoning: [Quality and directness of evidence found]"""
