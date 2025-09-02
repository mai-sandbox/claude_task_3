import os
from typing import Optional
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ResearchConfig(BaseModel):
    """Configuration for the company research system."""
    
    # API Keys
    anthropic_api_key: str = Field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY"))
    tavily_api_key: str = Field(default_factory=lambda: os.getenv("TAVILY_API_KEY"))
    
    # Search Configuration
    max_queries: int = Field(default=6, description="Maximum number of search queries per company")
    max_results_per_query: int = Field(default=3, description="Maximum search results per query")
    max_reflections: int = Field(default=2, description="Maximum number of reflection iterations")
    
    # LLM Configuration
    model_name: str = Field(default="claude-3-5-sonnet-20241022", description="Anthropic model to use")
    temperature: float = Field(default=0.1, description="LLM temperature for consistency")
    
    # Search Filtering
    excluded_domains: list = Field(
        default_factory=lambda: ["youtube.com", "twitter.com", "facebook.com"],
        description="Domains to exclude from search results"
    )
    
    # Logging and Tracing
    enable_tracing: bool = Field(default=True, description="Enable LangSmith tracing")
    langsmith_project: Optional[str] = Field(
        default_factory=lambda: os.getenv("LANGCHAIN_PROJECT"),
        description="LangSmith project name"
    )
    
    def validate_config(self) -> None:
        """Validate that required configuration is present."""
        errors = []
        
        if not self.anthropic_api_key:
            errors.append("ANTHROPIC_API_KEY is required")
        
        if not self.tavily_api_key:
            errors.append("TAVILY_API_KEY is required")
        
        if self.max_queries < 1:
            errors.append("max_queries must be at least 1")
        
        if self.max_results_per_query < 1:
            errors.append("max_results_per_query must be at least 1")
        
        if self.max_reflections < 0:
            errors.append("max_reflections must be non-negative")
        
        if errors:
            raise ValueError("Configuration errors: " + ", ".join(errors))
    
    @classmethod
    def from_env(cls) -> "ResearchConfig":
        """Create configuration from environment variables."""
        config = cls()
        config.validate_config()
        return config
    
    def to_dict(self) -> dict:
        """Convert configuration to dictionary, excluding sensitive data."""
        config_dict = self.dict()
        # Hide API keys in output
        if config_dict.get("anthropic_api_key"):
            config_dict["anthropic_api_key"] = f"{config_dict['anthropic_api_key'][:8]}..."
        if config_dict.get("tavily_api_key"):
            config_dict["tavily_api_key"] = f"{config_dict['tavily_api_key'][:8]}..."
        return config_dict


def get_default_config() -> ResearchConfig:
    """Get the default configuration from environment variables."""
    return ResearchConfig.from_env()


# Pre-defined configurations for different use cases
QUICK_RESEARCH_CONFIG = ResearchConfig(
    max_queries=3,
    max_results_per_query=2,
    max_reflections=1
)

THOROUGH_RESEARCH_CONFIG = ResearchConfig(
    max_queries=10,
    max_results_per_query=5,
    max_reflections=3
)

BALANCED_RESEARCH_CONFIG = ResearchConfig(
    max_queries=6,
    max_results_per_query=3,
    max_reflections=2
)