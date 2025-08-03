"""
Configuration and utility modules
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class Config:
    """Intelligent, adaptive configuration for the Scientific Paper AI Analyzer"""
    
    # Model settings (auto-detected best available)
    llm_model: str = "llama3.2"
    embedding_model: str = "mxbai-embed-large"
    
    # Adaptive search settings (automatically optimized)
    max_tokens_per_chunk: int = 1000  # Increased for better context
    
    # Processing settings - no file size limits
    max_file_size_mb: int = None  # No file size limit
    supported_formats: list = None
    
    # Intelligent features (always enabled for best results)
    enable_section_detection: bool = True
    enable_author_extraction: bool = True
    enable_cross_doc_synthesis: bool = True
    enable_adaptive_search: bool = True  # New: Smart search optimization
    
    def __post_init__(self):
        if self.supported_formats is None:
            self.supported_formats = ['.pdf', '.docx', '.txt', '.md']
    
    def get_adaptive_max_chunks(self, query_complexity: str, total_documents: int) -> int:
        """Intelligently determine optimal number of chunks based on query and collection size"""
        # Dynamic base calculation - scales with collection but caps for performance
        if total_documents <= 5:
            base_chunks = min(25, total_documents * 5)  # Small collections get comprehensive coverage
        elif total_documents <= 20:
            base_chunks = min(20, total_documents * 2)  # Medium collections get good coverage
        else:
            base_chunks = min(15, max(8, total_documents // 4))  # Large collections get selective coverage
        
        # Adjust based on query complexity
        complexity_multipliers = {
            'simple': 0.7,       # "What is X?" - fewer chunks needed
            'comparison': 1.8,   # "Compare A and B" - more chunks for comprehensive comparison
            'synthesis': 2.5,    # "What do papers say about X?" - need broad coverage
            'verification': 1.5, # "Is this true?" - need thorough evidence
            'author': 1.2,       # "Papers by X" - moderate expansion
            'general_analysis': 1.0  # Default complexity
        }
        
        multiplier = complexity_multipliers.get(query_complexity, 1.0)
        optimal_chunks = int(base_chunks * multiplier)
        
        # Dynamic caps based on collection size (significantly increased)
        if total_documents <= 10:
            max_cap = 100  # Much higher for small collections
        elif total_documents <= 50:
            max_cap = 150  # Higher for medium collections
        else:
            max_cap = 200  # Remove conservative limits for large collections
        
        return min(max_cap, max(5, optimal_chunks))  # Ensure minimum of 5 chunks
    
    def get_adaptive_similarity_threshold(self, query_type: str, num_results_found: int) -> float:
        """Dynamically adjust similarity threshold for optimal results"""
        
        # Base threshold adjustments based on result quantity
        if num_results_found < 2:
            base_adjustment = -0.3  # Much lower threshold if very few results
        elif num_results_found < 5:
            base_adjustment = -0.2  # Lower threshold if few results found
        elif num_results_found > 100:
            base_adjustment = 0.3   # Higher threshold if too many results
        elif num_results_found > 50:
            base_adjustment = 0.15  # Moderately higher threshold
        else:
            base_adjustment = 0.0   # Standard threshold
        
        # Adjust by query type
        type_thresholds = {
            'author': 0.55,      # More permissive for author matching
            'comparison': 0.75,  # Higher precision for comparisons
            'synthesis': 0.60,   # Broader scope for synthesis
            'verification': 0.70, # High precision for fact checking
            'simple': 0.65,      # Moderate precision for simple queries
            'general_analysis': 0.65  # Default precision
        }
        
        base_threshold = type_thresholds.get(query_type, 0.65)
        final_threshold = base_threshold + base_adjustment
        
        # Ensure reasonable bounds
        return max(0.3, min(0.9, final_threshold))
