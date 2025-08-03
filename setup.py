#!/usr/bin/env python3
"""
Setup script for Scientific Paper AI Analyzer
"""

import os
import sys
import subprocess
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for the setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        raise RuntimeError("Python 3.8 or higher is required")
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def install_requirements():
    """Install required packages"""
    print("📦 Installing requirements...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_ollama():
    """Check if Ollama is available"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/version", timeout=5)
        if response.status_code == 200:
            print("✅ Ollama is running and accessible")
            return True
        else:
            print("⚠️  Ollama is not responding properly")
            return False
    except Exception:
        print("⚠️  Ollama is not running. Please start Ollama service.")
        print("   Install: https://ollama.ai/")
        print("   Then run: ollama pull llama3.2")
        print("           ollama pull mxbai-embed-large")
        return False

def create_test_directories():
    """Create necessary directories"""
    directories = ["data", "temp", "logs"]
    for dir_name in directories:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✅ Created directory: {dir_name}")

def test_imports():
    """Test if all imports work correctly"""
    print("🔍 Testing imports...")
    try:
        # Test core imports
        from src.document_processor import DocumentProcessor
        from src.intelligent_search import IntelligentSearch
        from src.ai_agent import ScientificPaperAgent
        from src.utils.config import Config
        print("✅ All core modules imported successfully")
        
        # Test external dependencies
        import streamlit
        import chromadb
        import PyPDF2
        print("✅ All external dependencies available")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def create_sample_config():
    """Create a sample configuration file"""
    config_content = """# Scientific Paper AI Analyzer Configuration

# Ollama Models (make sure these are pulled)
LLM_MODEL=llama3.2
EMBEDDING_MODEL=mxbai-embed-large

# Search Settings - no limits for comprehensive results
MAX_CHUNKS=200
SIMILARITY_THRESHOLD=0.7

# File Settings - no file size limits  
MAX_FILE_SIZE_MB=None

# Logging
LOG_LEVEL=INFO
"""
    
    with open(".env", "w") as f:
        f.write(config_content)
    print("✅ Created sample .env configuration file")

def run_basic_test():
    """Run a basic functionality test"""
    print("🧪 Running basic functionality test...")
    try:
        from src.utils.config import Config
        from src.document_processor import DocumentProcessor
        
        config = Config()
        processor = DocumentProcessor(config)
        
        # Create a test document
        test_content = """
# Test Scientific Paper

## Abstract
This is a test document for the Scientific Paper AI Analyzer.

## Introduction
The purpose of this test is to verify that document processing works correctly.

## Methods
We use intelligent chunking and section detection.

## Results
The system successfully processes documents.

## Conclusion
Everything works as expected.
"""
        
        # Save test document
        test_file = "test_document.txt"
        with open(test_file, "w") as f:
            f.write(test_content)
        
        # Process test document
        chunks = processor.process_document(test_file)
        
        # Clean up
        os.remove(test_file)
        
        if chunks:
            print(f"✅ Basic test passed - processed {len(chunks)} chunks")
            print(f"   Sections detected: {[c.section for c in chunks if c.section]}")
            return True
        else:
            print("❌ Basic test failed - no chunks processed")
            return False
            
    except Exception as e:
        print(f"❌ Basic test failed: {e}")
        return False

def main():
    """Main setup function"""
    logger = setup_logging()
    
    print("🔬 Scientific Paper AI Analyzer - Setup")
    print("=" * 50)
    
    # Check Python version
    try:
        check_python_version()
    except RuntimeError as e:
        print(f"❌ {e}")
        return False
    
    # Install requirements
    if not install_requirements():
        return False
    
    # Create directories
    create_test_directories()
    
    # Create config
    create_sample_config()
    
    # Test imports
    if not test_imports():
        print("❌ Setup failed - check your Python environment")
        return False
    
    # Check Ollama
    ollama_available = check_ollama()
    
    # Run basic test
    if not run_basic_test():
        return False
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Start the application: streamlit run app.py")
    
    if not ollama_available:
        print("2. Install and start Ollama for full functionality:")
        print("   - Visit: https://ollama.ai/")
        print("   - Run: ollama pull llama3.2")
        print("   - Run: ollama pull mxbai-embed-large")
    
    print("3. Upload your scientific papers and start analyzing!")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
