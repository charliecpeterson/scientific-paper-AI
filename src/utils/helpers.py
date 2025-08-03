"""
Helper utilities for the Scientific Paper AI Analyzer
"""

from typing import List, Dict, Any
import streamlit as st

def format_sources(sources: List[Dict[str, Any]]) -> str:
    """Format sources for display"""
    if not sources:
        return "No sources found."
    
    formatted = []
    for i, source in enumerate(sources, 1):
        authors = ", ".join(source.get('authors', ['Unknown']))
        title = source.get('title', 'Unknown Title')
        section = source.get('section', 'Unknown Section')
        page = source.get('page', 'Unknown')
        year = source.get('year', 'Unknown')
        
        formatted.append(f"""
**Source {i}:** {title}
- **Authors:** {authors}
- **Year:** {year}
- **Section:** {section}
- **Page:** {page}
- **Relevance:** {source.get('relevance_score', 0):.3f}
""")
    
    return "\n".join(formatted)

def display_document_stats(stats: Dict[str, Any]):
    """Display document collection statistics"""
    if not stats:
        st.info("No documents processed yet")
        return
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Documents", stats.get('total_documents', 0))
        st.metric("Total Chunks", stats.get('total_chunks', 0))
    
    with col2:
        st.metric("Unique Authors", stats.get('unique_authors', 0))
        
        if stats.get('sections_found'):
            st.write("**Sections Found:**")
            for section, count in stats['sections_found'].items():
                st.write(f"- {section.title()}: {count}")
    
    with col3:
        if stats.get('most_common_authors'):
            st.write("**Top Authors:**")
            for author, count in stats['most_common_authors'][:5]:
                st.write(f"- {author}: {count} papers")

def create_citation(source: Dict[str, Any]) -> str:
    """Create a formatted citation for a source"""
    authors = source.get('authors', ['Unknown'])
    if len(authors) > 3:
        author_str = f"{authors[0]} et al."
    else:
        author_str = ", ".join(authors)
    
    title = source.get('title', 'Unknown Title')
    year = source.get('year', 'Unknown')
    journal = source.get('journal', '')
    
    if journal:
        return f"{author_str} ({year}). {title}. {journal}."
    else:
        return f"{author_str} ({year}). {title}."

def highlight_text(text: str, query: str) -> str:
    """Highlight query terms in text"""
    import re
    
    query_words = query.lower().split()
    highlighted_text = text
    
    for word in query_words:
        if len(word) > 2:  # Only highlight words longer than 2 characters
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted_text = pattern.sub(f"**{word.upper()}**", highlighted_text)
    
    return highlighted_text
