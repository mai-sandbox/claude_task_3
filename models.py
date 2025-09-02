from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: Optional[List[str]] = Field(None, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """Search query with purpose"""
    query: str = Field(description="The search query text")
    purpose: str = Field(description="What information this query aims to find")


class ResearchState(BaseModel):
    """State object for the company research workflow"""
    company_name: str = Field(description="Name of the company to research")
    user_notes: Optional[str] = Field(None, description="Optional user notes about the company")
    
    # Configuration
    max_search_queries: int = Field(default=5, description="Maximum number of search queries per research session")
    max_search_results: int = Field(default=10, description="Maximum number of search results per query")
    max_reflection_steps: int = Field(default=3, description="Maximum number of reflection iterations")
    
    # State tracking
    search_queries_used: int = Field(default=0, description="Number of search queries used so far")
    reflection_steps_used: int = Field(default=0, description="Number of reflection steps used so far")
    
    # Data collection
    company_info: CompanyInfo = Field(default_factory=lambda: CompanyInfo(company_name=""))
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Raw search results")
    messages: List[Dict[str, str]] = Field(default_factory=list, description="Conversation history")
    
    # Control flow
    needs_more_research: bool = Field(default=True, description="Whether more research is needed")
    completed: bool = Field(default=False, description="Whether research is completed")