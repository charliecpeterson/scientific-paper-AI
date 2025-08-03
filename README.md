# 🔬 Scientific Paper AI Analyzer

**Your Personal Research Assistant** - An intelligent AI agent that reads, understands, and analyzes your scientific paper collection, providing precise answers with complete source attribution.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Ollama](https://img.shields.io/badge/AI-Ollama-orange.svg)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)

## ✨ What Makes This Special

Unlike traditional document search tools, this AI agent **automatically optimizes itself** to give you the most accurate results possible:

- 🧠 **Zero Configuration Required** - Just upload papers and ask questions
- 🎯 **Adaptive Intelligence** - Automatically adjusts search depth, precision, and strategy based on your query
- 🔍 **Smart Query Understanding** - Detects if you want comparisons, verifications, author searches, or synthesis
- 📚 **Perfect Source Attribution** - Shows exact paper, section, and page for every claim  
- 🤖 **Intelligent Resource Management** - Uses more sources for complex queries, fewer for simple ones
- 🔒 **100% Private** - All processing happens locally with Ollama

**No sliders, no settings, no complexity** - the AI figures out the optimal approach for each question automatically.


## 🎯 Real-World Usage Examples

### Author-Specific Queries
```
You: "What papers are by deborah penchoff?"
AI: "Found 3 papers authored by Deborah A. Penchoff:
     1. Penchoff, D.A. et al. (2021). Rare Earth Elements..."
     2. Penchoff, D.A.; Harrison, R.J. (2018). Evaluation of computational..."
```

### Content Analysis
```
You: "According to the papers, is the statement 'DFT results show complexation of Cu metals' correct? Please provide what papers and where in the papers that discusses this."

AI: "VERIFICATION RESULT: PARTIALLY SUPPORTED

Based on my analysis of your papers, I found the following evidence:

**Supporting Evidence:**
1. **Martinez et al. (2023)** - Methods section, Page 15:
   'DFT calculations revealed stable Cu-ligand complexes with binding energies of -145 kJ/mol'

2. **Chen & Williams (2022)** - Results section, Page 8:
   'Quantum mechanical analysis shows significant Cu(II) complexation in aqueous solution'

**Limitations:**
- Only 2 of 15 papers directly address Cu complexation
- Most studies focus on other transition metals
- DFT methodology varies between studies

**Confidence: 75%**
**Sources: 2 papers, 3 specific sections**"
```

### Methodology Comparison
```
You: "Compare the computational methods used"
AI: "The papers employ three distinct computational approaches:
     DFT calculations (4 papers) vs Monte Carlo simulations (2 papers)..."
```

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

### 🔍 Smart Query Understanding
The system automatically detects and handles different query types:

- **Author queries**: "papers by [author]", "work done by [author]"
- **Topic analysis**: "findings about [topic]", "approaches to [subject]"
- **Methodology comparison**: "how do papers compare [methods]"
- **Cross-reference analysis**: "what do papers agree on"

## � Quick Start

### Prerequisites
- Python 3.8+
- [Ollama](https://ollama.ai/) installed and running
- At least 8GB RAM (16GB recommended for large document collections)

### Installation

1. **Clone and Setup**
```bash
git clone <your-repo>
cd paperai
python setup.py  # Automated setup script
```

2. **Install Ollama Models**
```bash
ollama pull llama3.2
ollama pull mxbai-embed-large
```

3. **Run the Application**
```bash
streamlit run app.py
```

4. **Upload Papers & Start Analyzing!**
   - Drag & drop your PDF/Word documents
   - Ask intelligent questions about your research
   - Get precise answers with full source attribution

### 🎯 Try the Demo
```bash
python demo.py  # See the system in action with sample papers
```

## �💻 Usage Modes

### 🌐 Web Interface (Recommended)
```bash
streamlit run app.py
```

**Features:**
- Drag-and-drop document upload
- Real-time model switching
- Interactive chat interface  
- Rich source visualization
- Document management dashboard

### 🔧 Advanced Configuration
Edit `.env` file to customize:
- Model selection
- Search parameters
- Processing settings
- Resource limits

## 🧠 Why This Agent is Superior

### 🎯 **Fully Adaptive Intelligence**
```
Simple Query → Few focused results → Fast, precise answers
Complex Synthesis → Broad search → Comprehensive analysis  
Fact Verification → Evidence-focused → High-precision sources
Author Search → Name-aware matching → Complete paper lists
```

### 🔍 **Zero-Configuration Search**
- **Automatically optimizes** chunk count based on query complexity
- **Dynamically adjusts** similarity thresholds for best results  
- **Intelligently expands** search when initial results are insufficient
- **Adapts strategy** based on your document collection size

### 🤖 **Smart Query Understanding**
- **Detects intent**: Comparison vs. verification vs. synthesis vs. author queries
- **Adjusts depth**: Simple questions get focused answers, complex ones get comprehensive analysis
- **Optimizes sources**: Uses 5 sources for definitions, 25+ for cross-paper comparisons
- **Self-corrects**: Automatically relaxes search if no good results found

### 📊 **Intelligent Resource Management**
- **Collection-aware**: Small collections get broader search, large ones get more selective
- **Quality-focused**: Prefers fewer high-quality results over many mediocre ones
- **Context-preserving**: Never breaks important semantic boundaries
- **Performance-optimized**: Automatically balances thoroughness with speed

