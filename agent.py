"""
Company Research Agent using LangGraph
Multi-node graph for comprehensive company research using Tavily API
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import MessagesState
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# Import TavilySearch from the new package
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    try:
        from langchain_community.tools.tavily_search import TavilySearchResults
    except ImportError:
        print("Warning: Could not import TavilySearchResults")

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration constants
MAX_SEARCH_QUERIES = 5
MAX_SEARCH_RESULTS = 3
MAX_REFLECTION_STEPS = 2


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """A search query for company research"""
    query: str = Field(description="The search query to execute")
    purpose: str = Field(description="What information this query aims to find")


class SearchQueryList(BaseModel):
    """List of search queries"""
    queries: List[SearchQuery] = Field(description="List of search queries to execute")


class CompanyResearchState(TypedDict):
    """State for the company research workflow"""
    messages: List[Any]
    company_name: str
    notes: Optional[str]
    search_queries: List[SearchQuery]
    search_results: List[Dict[str, Any]]
    company_info: CompanyInfo
    reflection_count: int
    is_complete: bool
    queries_executed: int


# Initialize the LLM and tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize Tavily search tool with explicit API key
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is required")

search_tool = TavilySearchResults(
    max_results=MAX_SEARCH_RESULTS,
    tavily_api_key=tavily_api_key
)


def generate_search_queries(state: CompanyResearchState) -> Dict[str, Any]:
    """Generate targeted search queries for company research"""
    
    company_name = state["company_name"]
    notes = state.get("notes", "")
    
    prompt = f"""You are a research assistant tasked with generating search queries to gather comprehensive information about "{company_name}".
    
    Additional context: {notes if notes else "None provided"}
    
    Generate {MAX_SEARCH_QUERIES} specific, targeted search queries that will help gather information for:
    1. Company founding year and founders
    2. Product or service description
    3. Funding history and investors
    4. Notable customers or clients
    5. Recent developments or news
    
    Each query should be specific and likely to return relevant results."""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Generate search queries for company: {company_name}")
    ]
    
    response = model.with_structured_output(SearchQueryList).invoke(messages)
    queries = response.queries if hasattr(response, 'queries') else response
    
    return {
        "search_queries": queries,
        "messages": state["messages"] + [AIMessage(content=f"Generated {len(queries)} search queries for {company_name}")]
    }


def execute_parallel_searches(state: CompanyResearchState) -> Dict[str, Any]:
    """Execute search queries in parallel using Tavily API"""
    
    search_queries = state["search_queries"]
    all_results = []
    
    for query in search_queries[:MAX_SEARCH_QUERIES]:
        try:
            results = search_tool.invoke({"query": query.query})
            all_results.extend(results)
        except Exception as e:
            print(f"Search failed for query '{query.query}': {str(e)}")
            continue
    
    return {
        "search_results": all_results,
        "queries_executed": len(search_queries[:MAX_SEARCH_QUERIES]),
        "messages": state["messages"] + [AIMessage(content=f"Executed {len(search_queries[:MAX_SEARCH_QUERIES])} search queries, found {len(all_results)} results")]
    }


def extract_company_information(state: CompanyResearchState) -> Dict[str, Any]:
    """Extract structured company information from search results"""
    
    company_name = state["company_name"]
    search_results = state["search_results"]
    
    # Combine all search result content
    combined_content = "\n\n".join([
        f"Source: {result.get('url', 'Unknown')}\nContent: {result.get('content', '')}"
        for result in search_results
    ])
    
    prompt = f"""You are an expert research analyst. Extract comprehensive information about "{company_name}" from the provided search results.
    
    Search Results:
    {combined_content}
    
    Extract and structure the information accurately. If specific information is not found, leave those fields empty rather than guessing.
    Focus on factual, verifiable information from the search results."""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Extract company information for: {company_name}")
    ]
    
    company_info = model.with_structured_output(CompanyInfo).invoke(messages)
    
    return {
        "company_info": company_info,
        "messages": state["messages"] + [AIMessage(content=f"Extracted structured information for {company_name}")]
    }


def reflect_on_completeness(state: CompanyResearchState) -> Dict[str, Any]:
    """Reflect on whether we have sufficient company information"""
    
    company_info = state["company_info"]
    reflection_count = state.get("reflection_count", 0)
    
    # Check completeness based on filled fields
    required_fields = ["company_name"]
    optional_fields = ["founding_year", "founder_names", "product_description", "funding_summary", "notable_customers"]
    
    filled_optional = sum(1 for field in optional_fields if getattr(company_info, field, None))
    completeness_score = filled_optional / len(optional_fields)
    
    # Consider complete if we have at least 60% of optional information or hit max reflections
    is_complete = completeness_score >= 0.6 or reflection_count >= MAX_REFLECTION_STEPS
    
    reflection_message = f"Completeness assessment: {completeness_score:.1%} of optional fields filled. "
    
    if is_complete:
        reflection_message += "Research considered complete."
    else:
        reflection_message += f"Need more information. Reflection step {reflection_count + 1}/{MAX_REFLECTION_STEPS}"
    
    return {
        "is_complete": is_complete,
        "reflection_count": reflection_count + 1,
        "messages": state["messages"] + [AIMessage(content=reflection_message)]
    }


def generate_additional_queries(state: CompanyResearchState) -> Dict[str, Any]:
    """Generate additional targeted queries based on missing information"""
    
    company_info = state["company_info"]
    company_name = state["company_name"]
    
    # Identify missing information
    missing_info = []
    if not company_info.founding_year:
        missing_info.append("founding year")
    if not company_info.founder_names:
        missing_info.append("founders")
    if not company_info.product_description:
        missing_info.append("products/services")
    if not company_info.funding_summary:
        missing_info.append("funding history")
    if not company_info.notable_customers:
        missing_info.append("customers/clients")
    
    prompt = f"""Generate {min(3, len(missing_info))} highly specific search queries to find missing information about "{company_name}".

    Missing information: {', '.join(missing_info)}
    
    Create targeted queries that are likely to find this specific missing information."""
    
    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=f"Generate additional queries for missing info about: {company_name}")
    ]
    
    response = model.with_structured_output(SearchQueryList).invoke(messages)
    additional_queries = response.queries if hasattr(response, 'queries') else response
    
    return {
        "search_queries": additional_queries,
        "messages": state["messages"] + [AIMessage(content=f"Generated {len(additional_queries)} additional search queries")]
    }


def should_continue_research(state: CompanyResearchState) -> str:
    """Determine if research should continue or end"""
    return "end" if state["is_complete"] else "continue"


# Build the workflow graph
def create_company_researcher():
    """Create the company research workflow"""
    
    workflow = StateGraph(CompanyResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search", execute_parallel_searches)
    workflow.add_node("extract", extract_company_information)
    workflow.add_node("reflect", reflect_on_completeness)
    workflow.add_node("additional_queries", generate_additional_queries)
    workflow.add_node("additional_search", execute_parallel_searches)
    
    # Add edges
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "extract")
    workflow.add_edge("extract", "reflect")
    
    # Conditional routing after reflection
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "continue": "additional_queries",
            "end": END
        }
    )
    
    # Reflection loop
    workflow.add_edge("additional_queries", "additional_search")
    workflow.add_edge("additional_search", "extract")
    
    return workflow.compile()


# Export the compiled graph as 'app' for deployment
app = create_company_researcher()


def research_company(company_name: str, notes: Optional[str] = None) -> CompanyInfo:
    """
    Research a company and return structured information
    
    Args:
        company_name: Name of the company to research
        notes: Optional additional context or specific requirements
    
    Returns:
        CompanyInfo object with researched information
    """
    
    initial_state = CompanyResearchState(
        messages=[HumanMessage(content=f"Research company: {company_name}")],
        company_name=company_name,
        notes=notes,
        search_queries=[],
        search_results=[],
        company_info=CompanyInfo(company_name=company_name),
        reflection_count=0,
        is_complete=False,
        queries_executed=0
    )
    
    final_state = app.invoke(initial_state)
    return final_state["company_info"]


if __name__ == "__main__":
    # Example usage
    result = research_company(
        company_name="Anthropic",
        notes="Focus on AI safety and constitutional AI research"
    )
    
    print("=== Company Research Results ===")
    print(f"Company: {result.company_name}")
    print(f"Founded: {result.founding_year or 'Not found'}")
    print(f"Founders: {', '.join(result.founder_names) if result.founder_names else 'Not found'}")
    print(f"Product: {result.product_description or 'Not found'}")
    print(f"Funding: {result.funding_summary or 'Not found'}")
    print(f"Customers: {result.notable_customers or 'Not found'}")