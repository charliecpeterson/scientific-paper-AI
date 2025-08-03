# 🔬 Scientific Paper AI Analyzer

**Your Personal Research Assistant** - An intelligent AI agent that reads, understands, and analyzes your scientific paper collection, providing precise answers with complete source attribution.


## ✨ What Makes This Special

Unlike traditional document search tools, this AI agent **automatically optimizes itself** to give you the most accurate results possible:

## 🛠️ Core Features

### 📤 Intelligent Document Processing
- **Multi-format support**: PDF, Word (.docx), Plain Text
- **Smart section detection**: Automatically identifies Abstract, Introduction, Methods, Results, Discussion, Conclusion
- **Metadata extraction**: During the embedding of the docs, this app well try to extract, Authors, titles, citations, publication info, year, journal, page numbers for the text in order to give a esitmated citation (That can be used to label the papers).
- **Intelligent chunking**: Preserves context while breaking into semantic segments

### 🧠 Advanced Search Engine
- **Hierarchical search**: Document-level → Section-level → Chunk-level analysis
- **Author-aware queries**: Detects author questions and searches accordingly
- **Adaptive resource allocation**: Uses more content from highly relevant papers
- **Cross-document synthesis**: Finds patterns and relationships across papers
- **Intelligent re-ranking**: Removes duplicates and prioritizes diverse perspectives

### 🎛️ Flexible Configuration
**LLM Models** (via Ollama):
- `llama3.2` - Recommended for analysis
- `llama3.1` - High-quality alternative  
- `mistral` - Faster option
- Any Ollama-compatible model

**Embedding Models**:
- `mxbai-embed-large` (Ollama) - **Recommended**
- `nomic-embed-text` (Ollama) - Fast alternative


## Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running


**Run the Application**
```bash
streamlit run app.py
```

**Upload Papers & Start Analyzing!**
   - Drag & drop your PDF/Word documents
   - Ask intelligent questions about your research
   - Get precise answers with full source attribution

