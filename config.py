"""Configuration settings for the Company Researcher"""

from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ResearchConfig:
    """Configuration for research parameters"""
    
    # API Configuration
    openai_model: str = "gpt-4"
    openai_temperature: float = 0.1
    
    # Search Limits
    max_search_queries: int = 8
    max_search_results: int = 50
    max_reflections: int = 3
    
    # Concurrent Search Settings
    max_concurrent_searches: int = 3
    
    # Search Settings
    tavily_search_depth: str = "basic"  # "basic" or "advanced"
    tavily_max_results_per_query: int = 5
    
    # Content Processing
    max_content_length: int = 500  # Characters per search result
    recent_results_window: int = 10  # Number of recent results to process
    
    # Validation
    min_required_fields: int = 2  # Minimum fields required to consider research complete
    

class DefaultConfigs:
    """Pre-configured settings for different use cases"""
    
    @staticmethod
    def quick_research() -> ResearchConfig:
        """Fast research with minimal queries"""
        return ResearchConfig(
            max_search_queries=4,
            max_search_results=20,
            max_reflections=1,
            tavily_max_results_per_query=3
        )
    
    @staticmethod
    def thorough_research() -> ResearchConfig:
        """Comprehensive research with more queries"""
        return ResearchConfig(
            max_search_queries=12,
            max_search_results=80,
            max_reflections=5,
            tavily_search_depth="advanced",
            tavily_max_results_per_query=7
        )
    
    @staticmethod
    def balanced_research() -> ResearchConfig:
        """Default balanced approach"""
        return ResearchConfig()


# Environment variable requirements
REQUIRED_ENV_VARS = [
    "OPENAI_API_KEY",
    "TAVILY_API_KEY"
]


def validate_environment() -> Dict[str, Any]:
    """Validate that required environment variables are set"""
    import os
    
    missing_vars = []
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            missing_vars.append(var)
    
    return {
        "valid": len(missing_vars) == 0,
        "missing_variables": missing_vars,
        "message": f"Missing environment variables: {missing_vars}" if missing_vars else "All required environment variables are set"
    }