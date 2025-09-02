from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")


class ResearchState(BaseModel):
    """State object for the company research workflow"""
    company_name: str
    user_notes: Optional[str] = None
    max_search_queries: int = Field(default=5, description="Maximum number of search queries per company")
    max_search_results: int = Field(default=3, description="Maximum search results per query")
    max_reflection_steps: int = Field(default=2, description="Maximum number of reflection steps")
    
    # Workflow state
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    company_info: CompanyInfo = Field(default_factory=lambda: CompanyInfo(company_name=""))
    messages: List[Dict[str, str]] = Field(default_factory=list)
    reflection_count: int = Field(default=0)
    is_complete: bool = Field(default=False)
    needs_more_info: bool = Field(default=True)


class SearchQuery(BaseModel):
    """Individual search query with context"""
    query: str
    purpose: str = Field(description="What information this query is trying to find")


class SearchResult(BaseModel):
    """Search result from Tavily API"""
    title: str
    url: str
    content: str
    score: float