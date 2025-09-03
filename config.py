"""
Configuration settings for the Company Researcher.

This file contains configurable parameters for the LangGraph-based
company research system.
"""

from dataclasses import dataclass
from typing import Optional
import os

@dataclass
class ResearchConfig:
    """Configuration for the company research system."""
    
    # API Keys
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    # Search limits
    max_search_queries: int = 5
    max_search_results_per_query: int = 3
    max_reflection_steps: int = 2
    
    # LLM settings
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.1
    
    # Timeout settings (in seconds)
    search_timeout: int = 30
    llm_timeout: int = 60
    
    # Output settings
    save_intermediate_results: bool = False
    verbose_logging: bool = False
    
    def __post_init__(self):
        """Load API keys from environment if not provided."""
        if self.openai_api_key is None:
            self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if self.tavily_api_key is None:
            self.tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    def validate(self) -> bool:
        """Validate that required settings are present."""
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is required")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required")
        if self.max_search_queries < 1:
            raise ValueError("Max search queries must be at least 1")
        if self.max_search_results_per_query < 1:
            raise ValueError("Max search results per query must be at least 1")
        return True

# Default configuration
DEFAULT_CONFIG = ResearchConfig()

# Quick configuration presets
FAST_CONFIG = ResearchConfig(
    max_search_queries=3,
    max_search_results_per_query=2,
    max_reflection_steps=1
)

THOROUGH_CONFIG = ResearchConfig(
    max_search_queries=8,
    max_search_results_per_query=5,
    max_reflection_steps=3
)

DEMO_CONFIG = ResearchConfig(
    max_search_queries=2,
    max_search_results_per_query=1,
    max_reflection_steps=1,
    verbose_logging=True
)