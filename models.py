from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field

class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")

class ResearchState(BaseModel):
    """State for the company research workflow"""
    company_name: str
    user_notes: Optional[str] = None
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    company_info: Optional[CompanyInfo] = None
    messages: List[str] = Field(default_factory=list)
    query_count: int = 0
    reflection_count: int = 0
    max_queries: int = 5
    max_search_results: int = 3
    max_reflections: int = 3
    needs_more_info: bool = True