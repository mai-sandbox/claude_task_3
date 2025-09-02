from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(default=None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(default=None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(default=None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(default=None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """Represents a search query to be executed"""
    query: str
    purpose: str  # What information this query is meant to find


class ResearchState(BaseModel):
    """State maintained throughout the research process"""
    company_name: str
    user_notes: Optional[str] = None
    messages: List[Dict[str, Any]] = Field(default_factory=list)
    search_queries_generated: List[SearchQuery] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    company_info: CompanyInfo = Field(default_factory=lambda: CompanyInfo(company_name=""))
    queries_executed: int = 0
    reflections_done: int = 0
    max_search_queries: int = 8
    max_search_results: int = 50
    max_reflections: int = 3
    research_complete: bool = False


class SearchResult(BaseModel):
    """Represents a search result from Tavily API"""
    title: str
    url: str
    content: str
    score: float
    raw_content: Optional[str] = None