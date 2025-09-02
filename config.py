from dataclasses import dataclass
from typing import Optional

@dataclass
class ResearchConfig:
    """Configuration settings for company research"""
    max_queries: int = 5
    max_search_results_per_query: int = 3
    max_reflection_steps: int = 3
    search_depth: str = "advanced"  # tavily search depth
    llm_temperature: float = 0.1
    max_content_length: int = 500  # max characters per search result for LLM processing
    
    def validate(self):
        """Validate configuration values"""
        if self.max_queries <= 0:
            raise ValueError("max_queries must be positive")
        if self.max_search_results_per_query <= 0:
            raise ValueError("max_search_results_per_query must be positive")
        if self.max_reflection_steps <= 0:
            raise ValueError("max_reflection_steps must be positive")
        if self.llm_temperature < 0 or self.llm_temperature > 2:
            raise ValueError("llm_temperature must be between 0 and 2")
        if self.max_content_length <= 0:
            raise ValueError("max_content_length must be positive")

# Default configuration
DEFAULT_CONFIG = ResearchConfig()

# Preset configurations for different use cases
QUICK_RESEARCH_CONFIG = ResearchConfig(
    max_queries=3,
    max_search_results_per_query=2,
    max_reflection_steps=2
)

THOROUGH_RESEARCH_CONFIG = ResearchConfig(
    max_queries=8,
    max_search_results_per_query=5,
    max_reflection_steps=5
)