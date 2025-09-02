from typing import List, Optional, Dict, Any, TypedDict
from pydantic import BaseModel, Field


class CompanyInfo(BaseModel):
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: Optional[List[str]] = Field(None, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    query: str = Field(description="Search query to execute")
    purpose: str = Field(description="What information this query aims to find")


class SearchResult(BaseModel):
    query: str
    content: str
    url: str
    score: float


class GraphState(TypedDict):
    company_name: str
    user_notes: Optional[str]
    messages: List[Dict[str, Any]]
    
    generated_queries: List[Dict[str, str]]
    search_results: List[Dict[str, Any]]
    extracted_info: Optional[Dict[str, Any]]
    
    queries_executed: int
    reflection_count: int
    is_complete: bool
    
    max_queries: int
    max_results_per_query: int
    max_reflections: int


class ReflectionResult(BaseModel):
    is_sufficient: bool = Field(description="Whether the information is sufficient")
    missing_info: List[str] = Field(description="List of missing information types")
    confidence_score: float = Field(description="Confidence in the completeness (0-1)")
    suggested_queries: List[str] = Field(description="Additional queries to improve information")