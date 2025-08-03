"""
Enhanced AI Agent with Multimodal Capabilities
Extends the existing AI agent to handle visual content from papers
"""

from typing import List, Dict, Any, Optional
import logging
from dataclasses import dataclass

from .ai_agent import AIAgent, AgentResponse
from .multimodal_processor import MultimodalChunk

class MultimodalAIAgent(AIAgent):
    """Enhanced AI Agent that can process visual content from scientific papers"""
    
    def __init__(self, config):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        
    def generate_response(self, query: str, search_results: List[Any]) -> AgentResponse:
        """Generate response using both text and visual context"""
        
        # Check if we have multimodal chunks
        has_visual_content = any(
            hasattr(result, 'visual_context') and result.visual_context 
            for result in search_results
        )
        
        if has_visual_content:
            return self._generate_multimodal_response(query, search_results)
        else:
            # Fallback to standard response
            return super().generate_response(query, search_results)
    
    def _generate_multimodal_response(self, query: str, search_results: List[Any]) -> AgentResponse:
        """Generate response using multimodal content"""
        
        # Determine query type
        query_type = self._analyze_query_type(query)
        
        # Get appropriate template
        if query_type == 'visual':
            template = self._get_visual_analysis_template()
        elif query_type == 'data':
            template = self._get_data_analysis_template()
        else:
            template = self._get_enhanced_general_template()
        
        # Prepare enhanced context
        context = self._prepare_multimodal_context(search_results)
        
        # Format prompt
        prompt = template.format(
            query=query,
            context=context,
            num_sources=len(search_results)
        )
        
        if self.llm is None:
            return self._generate_enhanced_fallback_response(query, search_results)
        
        try:
            # Generate response using LLM
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse response
            parsed_response = self._parse_llm_response(content)
            
            return AgentResponse(
                answer=parsed_response['answer'],
                sources=[],  # Will be added separately
                confidence=parsed_response['confidence'],
                reasoning=parsed_response.get('reasoning', '')
            )
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal LLM response: {e}")
            return self._generate_enhanced_fallback_response(query, search_results)
    
    def _analyze_query_type(self, query: str) -> str:
        """Analyze what type of query this is to use appropriate template"""
        query_lower = query.lower()
        
        visual_keywords = [
            'figure', 'graph', 'chart', 'plot', 'image', 'diagram', 'visualization',
            'show', 'display', 'illustrate', 'visual', 'picture'
        ]
        
        data_keywords = [
            'table', 'data', 'results', 'values', 'numbers', 'statistics',
            'measurements', 'experimental', 'compare', 'comparison'
        ]
        
        if any(keyword in query_lower for keyword in visual_keywords):
            return 'visual'
        elif any(keyword in query_lower for keyword in data_keywords):
            return 'data'
        else:
            return 'general'
    
    def _prepare_multimodal_context(self, search_results: List[Any]) -> str:
        """Prepare context that includes both text and visual information"""
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            context_part = f"**Source {i}:**\n"
            
            # Add document information
            if hasattr(result, 'metadata') and result.metadata:
                doc_info = []
                if 'filename' in result.metadata:
                    doc_info.append(f"Document: {result.metadata['filename']}")
                if 'page_number' in result.metadata:
                    doc_info.append(f"Page: {result.metadata['page_number']}")
                if 'section' in result.metadata and result.metadata['section']:
                    doc_info.append(f"Section: {result.metadata['section']}")
                
                if doc_info:
                    context_part += f"({', '.join(doc_info)})\n"
            
            # Add main content
            content = getattr(result, 'content', str(result))
            context_part += f"Text: {content}\n"
            
            # Add visual context if available
            if hasattr(result, 'visual_context') and result.visual_context:
                context_part += f"Visual Elements: {result.visual_context}\n"
            
            # Add specific visual elements
            if hasattr(result, 'images') and result.images:
                image_descriptions = [img.get('description', 'Image') for img in result.images[:2]]
                context_part += f"Images: {'; '.join(image_descriptions)}\n"
            
            if hasattr(result, 'tables') and result.tables:
                table_descriptions = [tbl.get('description', 'Table') for tbl in result.tables[:2]]
                context_part += f"Tables: {'; '.join(table_descriptions)}\n"
            
            if hasattr(result, 'figures') and result.figures:
                figure_info = [f"Figure {fig.get('figure_number', '')}: {fig.get('caption', '')[:100]}" 
                              for fig in result.figures[:2]]
                context_part += f"Figures: {'; '.join(figure_info)}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _get_visual_analysis_template(self) -> str:
        """Template for queries about visual content"""
        return """You are a scientific research assistant specialized in analyzing visual content from research papers.

The user is asking about visual elements (figures, charts, graphs, images) in scientific papers. Your job is to analyze the visual descriptions and provide detailed insights.

User Question: {query}

Sources with Visual Content ({num_sources} found):
{context}

IMPORTANT INSTRUCTIONS:
1. Focus on the visual elements described in the sources
2. Explain what the figures, charts, or images show
3. Interpret the data or trends visible in the visual content
4. Connect visual information to the research findings
5. Cite specific figures and their descriptions

FORMAT YOUR ANSWER AS:
**Visual Analysis Results:**

**Relevant Figures/Charts:**
- **Figure X** (from [Document]): [Description of what it shows]
  - Key findings: [What the visual data reveals]
  - Significance: [Why this visual is important]

**Data Interpretation:**
[Analysis of trends, patterns, or relationships shown in the visuals]

**Research Implications:**
[How the visual evidence supports or relates to the research question]

Remember: Base your analysis on the actual visual descriptions provided in the sources."""

    def _get_data_analysis_template(self) -> str:
        """Template for queries about data and tables"""
        return """You are a scientific research assistant specialized in analyzing data tables and experimental results from research papers.

The user is asking about data, tables, or experimental results. Focus on the quantitative information and data structures described in the sources.

User Question: {query}

Sources with Data Content ({num_sources} found):
{context}

IMPORTANT INSTRUCTIONS:
1. Focus on tables, data sets, and numerical results
2. Explain the structure and content of data tables
3. Identify key measurements, comparisons, or trends
4. Interpret statistical significance or experimental outcomes
5. Cite specific tables and data sources

FORMAT YOUR ANSWER AS:
**Data Analysis Results:**

**Relevant Tables/Data:**
- **Table X** (from [Document]): [Description of data structure]
  - Key measurements: [What was measured]
  - Results: [Main findings from the data]
  - Sample size/conditions: [Experimental parameters]

**Statistical Analysis:**
[Summary of significant results, trends, or comparisons]

**Experimental Context:**
[How the data relates to the research objectives]

Remember: Base your analysis on the actual table descriptions and data provided in the sources."""

    def _get_enhanced_general_template(self) -> str:
        """Enhanced general template that includes visual context"""
        return """You are a scientific research assistant with access to both textual content and visual elements from research papers.

You have enhanced context including text, images, tables, and figures from scientific papers. Use ALL available information to provide comprehensive answers.

User Question: {query}

Enhanced Sources ({num_sources} found):
{context}

IMPORTANT INSTRUCTIONS:
1. Use both textual content AND visual elements in your analysis
2. When available, reference specific figures, tables, or images
3. Provide a comprehensive answer drawing from all content types
4. Cite specific documents, sections, and visual elements
5. Explain how visual content supports or illustrates the textual findings

FORMAT YOUR ANSWER AS:
**Comprehensive Analysis:**

**Main Findings:**
[Primary answer based on textual content]

**Supporting Visual Evidence:**
[How images, figures, or tables support the findings]

**Detailed Sources:**
1. **[Document Title]** - [Section]
   - Text evidence: [Key textual information]
   - Visual evidence: [Relevant figures/tables if present]

**Synthesis:**
[How all the evidence (text + visual) addresses the question]

Remember: Provide the most complete answer possible using all available information types."""

    def _generate_enhanced_fallback_response(self, query: str, search_results: List[Any]) -> AgentResponse:
        """Enhanced fallback response that includes visual context"""
        
        # Check for visual content
        visual_elements = []
        text_summaries = []
        
        for result in search_results[:3]:  # Limit to top 3 results
            # Extract text summary
            content = getattr(result, 'content', str(result))
            text_summaries.append(content[:200] + "..." if len(content) > 200 else content)
            
            # Extract visual elements
            if hasattr(result, 'visual_context') and result.visual_context:
                visual_elements.append(result.visual_context)
        
        # Build enhanced fallback response
        answer = f"Based on the search results, I found {len(search_results)} relevant sources"
        
        if visual_elements:
            answer += f" including {len(visual_elements)} sources with visual content (figures, tables, or images)"
        
        answer += ":\n\n"
        
        # Add text summaries
        for i, summary in enumerate(text_summaries, 1):
            answer += f"**Source {i}:** {summary}\n\n"
        
        # Add visual context
        if visual_elements:
            answer += "**Visual Elements Found:**\n"
            for i, visual in enumerate(visual_elements, 1):
                answer += f"- Source {i}: {visual}\n"
            answer += "\n"
        
        answer += "*Note: Enhanced AI processing is temporarily unavailable. The above represents the key information found in your document collection.*"
        
        return AgentResponse(
            answer=answer,
            sources=[],
            confidence=60,  # Lower confidence for fallback
            reasoning="Fallback response with enhanced visual context extraction"
        )
