from typing import List, Dict, Any, Annotated, Literal, Union
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import asyncio
import os

# Pydantic Models for structured data
class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: int = Field(default=None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: str = Field(default="", description="Brief description of the company's main product or service")
    funding_summary: str = Field(default="", description="Summary of the company's funding history")
    notable_customers: str = Field(default="", description="Known customers that use company's product/service")

class SearchQuery(BaseModel):
    """Represents a search query"""
    query: str = Field(description="The search query text")
    purpose: str = Field(description="What information this query is meant to find")

class GeneratedQueries(BaseModel):
    """Collection of search queries"""
    queries: List[SearchQuery] = Field(description="List of search queries to execute")

class ReflectionResult(BaseModel):
    """Result of reflection on gathered information"""
    is_sufficient: bool = Field(description="Whether the information is sufficient")
    missing_information: List[str] = Field(default_factory=list, description="List of missing information types")
    confidence_score: float = Field(description="Confidence in the completeness of information (0-1)")

# State definition
class ResearchState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    company_name: str
    user_notes: str
    company_info: CompanyInfo
    search_queries_used: List[str]
    search_results: List[Dict[str, Any]]
    reflection_count: int
    max_search_queries: int
    max_search_results: int
    max_reflection_steps: int
    is_complete: bool

# Configuration constants
DEFAULT_MAX_SEARCH_QUERIES = 5
DEFAULT_MAX_SEARCH_RESULTS = 3
DEFAULT_MAX_REFLECTION_STEPS = 2

# Initialize LLM and tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize search tool with error handling
def get_search_tool():
    """Get search tool with proper error handling"""
    try:
        return TavilySearchResults(max_results=3)
    except Exception as e:
        print(f"Warning: Could not initialize Tavily search tool: {e}")
        return None

search_tool = get_search_tool()

def generate_search_queries(state: ResearchState) -> Dict[str, Any]:
    """Generate search queries based on company name and existing information"""
    
    # Check what information we already have
    existing_info = state["company_info"]
    company_name = state["company_name"]
    user_notes = state.get("user_notes", "")
    
    # Determine what information is still needed
    missing_fields = []
    if not existing_info.founding_year:
        missing_fields.append("founding year")
    if not existing_info.founder_names:
        missing_fields.append("founder names")
    if not existing_info.product_description:
        missing_fields.append("product description")
    if not existing_info.funding_summary:
        missing_fields.append("funding history")
    if not existing_info.notable_customers:
        missing_fields.append("notable customers")
    
    # Create system prompt for query generation
    system_prompt = f"""You are a research assistant that generates strategic search queries to gather information about companies.
    
Company to research: {company_name}
User notes: {user_notes}

Information still needed: {', '.join(missing_fields) if missing_fields else 'General validation and additional details'}

Generate {min(state['max_search_queries'] - len(state['search_queries_used']), 3)} diverse and specific search queries that will help gather the missing information about this company. 

Make queries specific and targeted - avoid generic queries. Focus on factual information that can be found through web search.

Examples of good queries:
- "{company_name} founders founding team"
- "{company_name} funding rounds investment history"
- "{company_name} customers clients testimonials"
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate targeted search queries to research {company_name}")
    ]
    
    result = model.with_structured_output(GeneratedQueries).invoke(messages)
    
    # Limit queries based on remaining quota
    remaining_queries = state["max_search_queries"] - len(state["search_queries_used"])
    limited_queries = result.queries[:remaining_queries]
    
    return {
        "messages": [AIMessage(content=f"Generated {len(limited_queries)} search queries for {company_name}")],
        "search_queries_used": state["search_queries_used"] + [q.query for q in limited_queries]
    }

async def execute_web_searches(state: ResearchState) -> Dict[str, Any]:
    """Execute web searches in parallel for efficiency"""
    
    # Get new queries to execute
    all_queries = state["search_queries_used"]
    existing_results_count = len(state["search_results"])
    new_queries = all_queries[existing_results_count:]
    
    if not new_queries:
        return {"messages": [AIMessage(content="No new queries to execute")]}
    
    # Execute searches in parallel for speed
    async def search_single_query(query: str):
        try:
            if search_tool is None:
                return {"query": query, "results": [], "error": "Search tool not available"}
            results = search_tool.invoke({"query": query})
            return {"query": query, "results": results}
        except Exception as e:
            return {"query": query, "results": [], "error": str(e)}
    
    # Run searches in parallel
    search_tasks = [search_single_query(query) for query in new_queries]
    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
    
    # Process results
    valid_results = []
    for result in search_results:
        if isinstance(result, dict) and "results" in result:
            valid_results.append(result)
    
    return {
        "messages": [AIMessage(content=f"Completed {len(valid_results)} web searches")],
        "search_results": state["search_results"] + valid_results
    }

def extract_company_information(state: ResearchState) -> Dict[str, Any]:
    """Extract structured company information from search results"""
    
    company_name = state["company_name"]
    search_results = state["search_results"]
    user_notes = state.get("user_notes", "")
    
    # Prepare search results text
    results_text = ""
    for result in search_results:
        results_text += f"\nQuery: {result['query']}\n"
        if 'results' in result:
            for item in result['results']:
                if isinstance(item, dict):
                    title = item.get('title', '')
                    content = item.get('content', '')
                    url = item.get('url', '')
                    results_text += f"Title: {title}\nContent: {content}\nURL: {url}\n---\n"
    
    system_prompt = f"""You are an expert research analyst. Extract structured information about the company from the provided search results.

Company Name: {company_name}
User Notes: {user_notes}

Based on the search results below, extract accurate information for each field. Only include information that you can verify from the search results. If information is not available or unclear, leave the field empty or use appropriate defaults.

Be precise with:
- founding_year: Must be a specific year (integer)
- founder_names: Only include confirmed founder names
- product_description: Keep concise but informative
- funding_summary: Include key funding rounds and amounts if available
- notable_customers: Only include customers that are explicitly mentioned

Search Results:
{results_text}
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract company information for {company_name} from the search results")
    ]
    
    extracted_info = model.with_structured_output(CompanyInfo).invoke(messages)
    
    return {
        "messages": [AIMessage(content=f"Extracted information for {company_name}")],
        "company_info": extracted_info
    }

def reflect_on_information(state: ResearchState) -> Dict[str, Any]:
    """Reflect on the gathered information to determine if it's sufficient"""
    
    company_info = state["company_info"]
    company_name = state["company_name"]
    reflection_count = state["reflection_count"]
    
    # Analyze completeness
    info_dict = company_info.dict()
    
    system_prompt = f"""You are a quality analyst reviewing research completeness for company information.

Company: {company_name}
Current reflection step: {reflection_count + 1} of {state["max_reflection_steps"]}

Current information gathered:
- Company Name: {info_dict.get('company_name', 'Missing')}
- Founding Year: {info_dict.get('founding_year', 'Missing')}
- Founders: {', '.join(info_dict.get('founder_names', [])) if info_dict.get('founder_names') else 'Missing'}
- Product Description: {info_dict.get('product_description', 'Missing')}
- Funding Summary: {info_dict.get('funding_summary', 'Missing')}
- Notable Customers: {info_dict.get('notable_customers', 'Missing')}

Evaluate whether this information is sufficient for a basic company profile. Consider:
1. Are the core fields populated with meaningful information?
2. Is the information accurate and detailed enough?
3. What critical information is still missing?

Provide a confidence score (0.0 to 1.0) and determine if we need more research.
"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Evaluate the completeness and quality of the gathered company information")
    ]
    
    reflection = model.with_structured_output(ReflectionResult).invoke(messages)
    
    # Determine if research is complete
    is_sufficient = (
        reflection.is_sufficient or 
        reflection.confidence_score >= 0.7 or 
        state["reflection_count"] >= state["max_reflection_steps"] - 1 or
        len(state["search_queries_used"]) >= state["max_search_queries"]
    )
    
    return {
        "messages": [AIMessage(content=f"Reflection {reflection_count + 1}: Confidence {reflection.confidence_score:.2f}, Sufficient: {is_sufficient}")],
        "reflection_count": state["reflection_count"] + 1,
        "is_complete": is_sufficient
    }

# Routing function
def should_continue_research(state: ResearchState) -> Literal["generate_queries", "extract_info", "end"]:
    """Determine the next step in the research process"""
    
    # Check if research is complete
    if state.get("is_complete", False):
        return "end"
    
    # Check if we've hit search query limits
    if len(state["search_queries_used"]) >= state["max_search_queries"]:
        if not state.get("search_results"):
            return "extract_info"
        return "end"
    
    # Check if we have search results to process
    if len(state["search_results"]) < len(state["search_queries_used"]):
        return "extract_info"
    
    # Continue generating queries if we haven't reached limits
    return "generate_queries"

def should_execute_search(state: ResearchState) -> Literal["search", "extract_info"]:
    """Determine if we should execute searches or extract information"""
    
    # If we have new queries that haven't been searched, execute searches
    if len(state["search_queries_used"]) > len(state["search_results"]):
        return "search"
    
    return "extract_info"

# Build the graph
def create_research_graph():
    """Create the multi-node research graph"""
    
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search", execute_web_searches)
    workflow.add_node("extract_info", extract_company_information)
    workflow.add_node("reflect", reflect_on_information)
    
    # Add edges
    workflow.add_edge(START, "generate_queries")
    workflow.add_conditional_edges(
        "generate_queries",
        should_execute_search,
        {
            "search": "search",
            "extract_info": "extract_info"
        }
    )
    workflow.add_edge("search", "extract_info")
    workflow.add_edge("extract_info", "reflect")
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "generate_queries": "generate_queries",
            "extract_info": "extract_info",
            "end": END
        }
    )
    
    return workflow.compile()

# Export the compiled graph as 'app'
app = create_research_graph()

# Utility function to run research
def research_company(
    company_name: str,
    user_notes: str = "",
    max_search_queries: int = DEFAULT_MAX_SEARCH_QUERIES,
    max_search_results: int = DEFAULT_MAX_SEARCH_RESULTS,
    max_reflection_steps: int = DEFAULT_MAX_REFLECTION_STEPS
) -> CompanyInfo:
    """
    Research a company using the LangGraph workflow
    
    Args:
        company_name: Name of the company to research
        user_notes: Optional user notes/context
        max_search_queries: Maximum number of search queries to execute
        max_search_results: Maximum search results per query
        max_reflection_steps: Maximum number of reflection iterations
    
    Returns:
        CompanyInfo object with extracted information
    """
    
    initial_state = {
        "messages": [HumanMessage(content=f"Research the company: {company_name}")],
        "company_name": company_name,
        "user_notes": user_notes,
        "company_info": CompanyInfo(company_name=company_name),
        "search_queries_used": [],
        "search_results": [],
        "reflection_count": 0,
        "max_search_queries": max_search_queries,
        "max_search_results": max_search_results,
        "max_reflection_steps": max_reflection_steps,
        "is_complete": False
    }
    
    # Run the workflow
    final_state = app.invoke(initial_state)
    
    return final_state["company_info"]

if __name__ == "__main__":
    # Example usage
    company_info = research_company(
        company_name="Anthropic",
        user_notes="AI safety company",
        max_search_queries=3,
        max_reflection_steps=1
    )
    
    print("Company Research Results:")
    print(f"Name: {company_info.company_name}")
    print(f"Founded: {company_info.founding_year}")
    print(f"Founders: {', '.join(company_info.founder_names)}")
    print(f"Product: {company_info.product_description}")
    print(f"Funding: {company_info.funding_summary}")
    print(f"Customers: {company_info.notable_customers}")