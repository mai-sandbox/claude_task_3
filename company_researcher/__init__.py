"""
Company Researcher - A LangGraph-based multi-node system for comprehensive company research.

This package provides tools to research companies using web search and LLM-based information extraction.
"""

from .graph import CompanyResearchGraph, create_company_researcher
from .models import CompanyInfo, GraphState
from .config import ResearchConfig, get_default_config
from .config import QUICK_RESEARCH_CONFIG, THOROUGH_RESEARCH_CONFIG, BALANCED_RESEARCH_CONFIG

__version__ = "1.0.0"
__all__ = [
    "CompanyResearchGraph",
    "create_company_researcher", 
    "CompanyInfo",
    "GraphState",
    "ResearchConfig",
    "get_default_config",
    "QUICK_RESEARCH_CONFIG",
    "THOROUGH_RESEARCH_CONFIG", 
    "BALANCED_RESEARCH_CONFIG"
]