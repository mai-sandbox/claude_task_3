from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum

class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(default=None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(default=None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(default=None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(default=None, description="Known customers that use company's product/service")

class SearchQuery(BaseModel):
    """Search query with purpose"""
    query: str = Field(description="The search query text")
    purpose: str = Field(description="What information this query is trying to find")

class SearchResult(BaseModel):
    """Individual search result"""
    title: str
    content: str
    url: str

class ResearchState(BaseModel):
    """State for the company research workflow"""
    company_name: str = Field(description="Name of the company to research")
    user_notes: Optional[str] = Field(default=None, description="Optional user notes about the company")
    
    # Configuration
    max_search_queries: int = Field(default=8, description="Maximum number of search queries to execute")
    max_search_results: int = Field(default=5, description="Maximum number of search results per query")
    max_reflection_steps: int = Field(default=3, description="Maximum number of reflection iterations")
    
    # Working data
    search_queries: List[SearchQuery] = Field(default_factory=list, description="Generated search queries")
    search_results: List[SearchResult] = Field(default_factory=list, description="All search results")
    company_info: Optional[CompanyInfo] = Field(default=None, description="Extracted company information")
    
    # Progress tracking
    queries_executed: int = Field(default=0, description="Number of queries executed so far")
    reflection_count: int = Field(default=0, description="Number of reflection steps performed")
    messages: List[Dict[str, Any]] = Field(default_factory=list, description="Conversation history")
    is_complete: bool = Field(default=False, description="Whether research is complete")
    
    def add_message(self, role: str, content: str):
        """Add a message to the conversation history"""
        self.messages.append({"role": role, "content": content})
    
    def get_missing_fields(self) -> List[str]:
        """Get list of missing or incomplete fields in company_info"""
        if not self.company_info:
            return ["company_name", "founding_year", "founder_names", "product_description", "funding_summary", "notable_customers"]
        
        missing = []
        if not self.company_info.founding_year:
            missing.append("founding_year")
        if not self.company_info.founder_names:
            missing.append("founder_names")
        if not self.company_info.product_description:
            missing.append("product_description")
        if not self.company_info.funding_summary:
            missing.append("funding_summary")
        if not self.company_info.notable_customers:
            missing.append("notable_customers")
        
        return missing