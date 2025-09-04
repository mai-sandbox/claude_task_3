from typing import TypedDict, List, Optional, Dict, Any
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import asyncio
import os
from tavily import TavilyClient

class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(description="Year the company was founded", default=None)
    founder_names: Optional[List[str]] = Field(description="Names of the founding team members", default_factory=list)
    product_description: Optional[str] = Field(description="Brief description of the company's main product or service", default=None)
    funding_summary: Optional[str] = Field(description="Summary of the company's funding history", default=None)
    notable_customers: Optional[str] = Field(description="Known customers that use company's product/service", default=None)

class SearchQueries(BaseModel):
    """Generated search queries for company research"""
    queries: List[str] = Field(description="List of search queries to research the company")

class ReflectionResult(BaseModel):
    """Result of reflection on gathered information"""
    is_sufficient: bool = Field(description="Whether the gathered information is sufficient")
    missing_info: List[str] = Field(description="List of missing information categories")
    additional_queries: List[str] = Field(description="Additional search queries if more information is needed")

class CompanyResearchState(TypedDict):
    """State for the company research workflow"""
    messages: List[BaseMessage]
    company_name: str
    user_notes: Optional[str]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    company_info: Optional[CompanyInfo]
    reflection_count: int
    max_queries_per_iteration: int
    max_results_per_query: int
    max_reflections: int

# Configuration
MAX_QUERIES_PER_ITERATION = 3
MAX_RESULTS_PER_QUERY = 5
MAX_REFLECTIONS = 2

# Initialize model and tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
tavily_client = TavilyClient()

def generate_search_queries(state: CompanyResearchState) -> Dict[str, Any]:
    """Generate search queries to research the company"""
    company_name = state["company_name"]
    user_notes = state.get("user_notes", "")
    existing_info = state.get("company_info")
    
    system_prompt = f"""You are a research assistant tasked with generating search queries to research a company.
    
Company: {company_name}
User notes: {user_notes}

Generate {MAX_QUERIES_PER_ITERATION} specific, targeted search queries that will help gather comprehensive information about this company.
Focus on finding:
- Founding information (year, founders)
- Products and services
- Funding and investment history
- Notable customers and partnerships

Make queries specific and likely to return high-quality results."""

    if existing_info:
        system_prompt += f"\n\nExisting information gathered:\n{existing_info.model_dump_json(indent=2)}\n\nGenerate queries to fill in missing information."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate search queries for researching {company_name}")
    ]
    
    response = model.with_structured_output(SearchQueries).invoke(messages)
    queries = response.queries[:MAX_QUERIES_PER_ITERATION]
    
    return {
        "search_queries": queries,
        "messages": state["messages"] + [AIMessage(content=f"Generated {len(queries)} search queries: {', '.join(queries)}")]
    }

async def search_web_parallel(state: CompanyResearchState) -> Dict[str, Any]:
    """Perform parallel web searches using Tavily API"""
    queries = state["search_queries"]
    
    async def search_single_query(query: str) -> List[Dict[str, Any]]:
        try:
            response = tavily_client.search(
                query=query,
                max_results=MAX_RESULTS_PER_QUERY,
                search_depth="advanced"
            )
            return response.get("results", [])
        except Exception as e:
            print(f"Error searching for '{query}': {e}")
            return []
    
    # Perform parallel searches
    search_tasks = [search_single_query(query) for query in queries]
    results = await asyncio.gather(*search_tasks)
    
    # Flatten results
    all_results = []
    for query_results in results:
        all_results.extend(query_results)
    
    return {
        "search_results": all_results,
        "messages": state["messages"] + [AIMessage(content=f"Completed web search with {len(all_results)} results across {len(queries)} queries")]
    }

def extract_company_info(state: CompanyResearchState) -> Dict[str, Any]:
    """Extract structured company information from search results"""
    search_results = state["search_results"]
    company_name = state["company_name"]
    existing_info = state.get("company_info")
    
    # Prepare search results context
    results_text = "\n\n".join([
        f"Title: {result.get('title', 'N/A')}\nURL: {result.get('url', 'N/A')}\nContent: {result.get('content', 'N/A')}"
        for result in search_results[:20]  # Limit to first 20 results to avoid token limits
    ])
    
    system_prompt = f"""You are an expert researcher analyzing web search results to extract structured company information.

Company: {company_name}

Based on the search results provided, extract comprehensive information about the company. Be accurate and only include information that is explicitly stated in the search results. If information is not found, leave the field empty.

Search Results:
{results_text}"""

    if existing_info:
        system_prompt += f"\n\nExisting information:\n{existing_info.model_dump_json(indent=2)}\n\nUpdate and enhance the existing information with new findings."
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract structured information about {company_name} from the search results")
    ]
    
    response = model.with_structured_output(CompanyInfo).invoke(messages)
    
    return {
        "company_info": response,
        "messages": state["messages"] + [AIMessage(content=f"Extracted company information for {company_name}")]
    }

def reflect_on_completeness(state: CompanyResearchState) -> Dict[str, Any]:
    """Reflect on whether we have sufficient information about the company"""
    company_info = state["company_info"]
    reflection_count = state["reflection_count"]
    max_reflections = state["max_reflections"]
    
    if not company_info:
        return {
            "messages": state["messages"] + [AIMessage(content="No company information available for reflection")]
        }
    
    system_prompt = f"""You are an expert researcher evaluating the completeness of company information.

Current information about the company:
{company_info.model_dump_json(indent=2)}

Evaluate whether this information is comprehensive and sufficient for a good company profile. Consider:
- Is the basic company information complete?
- Are there significant gaps in key areas?
- Would additional research be valuable?

Current reflection iteration: {reflection_count + 1}/{max_reflections}

If information is insufficient and we haven't reached the maximum reflections, suggest additional specific search queries."""
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Evaluate the completeness of the company information and determine if more research is needed")
    ]
    
    response = model.with_structured_output(ReflectionResult).invoke(messages)
    
    new_reflection_count = reflection_count + 1
    
    result = {
        "reflection_count": new_reflection_count,
        "messages": state["messages"] + [AIMessage(content=f"Reflection {new_reflection_count}: Information {'sufficient' if response.is_sufficient else 'insufficient'}")]
    }
    
    # If information is insufficient and we haven't reached max reflections, set up for another iteration
    if not response.is_sufficient and new_reflection_count < max_reflections:
        result["search_queries"] = response.additional_queries[:MAX_QUERIES_PER_ITERATION]
        result["messages"].append(AIMessage(content=f"Additional queries needed: {', '.join(response.additional_queries)}"))
    
    return result

def should_continue_research(state: CompanyResearchState) -> str:
    """Determine whether to continue research or finish"""
    reflection_count = state["reflection_count"]
    max_reflections = state["max_reflections"]
    company_info = state.get("company_info")
    search_queries = state.get("search_queries", [])
    
    # If we haven't done reflection yet, go to reflection
    if reflection_count == 0:
        return "reflect"
    
    # If we've reached max reflections, finish
    if reflection_count >= max_reflections:
        return "finish"
    
    # If we have new search queries from reflection, continue research
    if search_queries:
        return "search"
    
    # Otherwise, finish
    return "finish"

# Build the graph
def create_company_researcher() -> StateGraph:
    workflow = StateGraph(CompanyResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search_web", search_web_parallel)
    workflow.add_node("extract_info", extract_company_info)
    workflow.add_node("reflect", reflect_on_completeness)
    
    # Add edges
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "search_web")
    workflow.add_edge("search_web", "extract_info")
    workflow.add_edge("extract_info", "reflect")
    
    # Add conditional edge from reflect
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "search": "generate_queries",
            "finish": END
        }
    )
    
    return workflow

# Create and compile the graph
graph = create_company_researcher().compile()
app = graph

if __name__ == "__main__":
    # Test the workflow
    initial_state = {
        "messages": [HumanMessage(content="Research OpenAI company")],
        "company_name": "OpenAI",
        "user_notes": "Focus on their AI models and recent developments",
        "search_queries": [],
        "search_results": [],
        "company_info": None,
        "reflection_count": 0,
        "max_queries_per_iteration": MAX_QUERIES_PER_ITERATION,
        "max_results_per_query": MAX_RESULTS_PER_QUERY,
        "max_reflections": MAX_REFLECTIONS
    }
    
    print("Starting company research for OpenAI...")
    result = app.invoke(initial_state)
    
    print("\n=== Final Company Information ===")
    if result["company_info"]:
        print(result["company_info"].model_dump_json(indent=2))
    else:
        print("No company information extracted")
    
    print(f"\nCompleted with {result['reflection_count']} reflection steps")