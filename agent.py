"""Company Researcher - A LangGraph agent for comprehensive company research using web search."""

from typing import Dict, List, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_anthropic import ChatAnthropic
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
import asyncio
import json


# Pydantic Models for Structured Data
class CompanyInfo(BaseModel):
    """Basic information about a company."""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(default=None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(default=None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(default=None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(default=None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """A search query for company research."""
    query: str = Field(description="The search query to execute")
    purpose: str = Field(description="What information this query is meant to find")


class SearchQueries(BaseModel):
    """Collection of search queries for company research."""
    queries: List[SearchQuery] = Field(description="List of search queries to execute")


class QualityAssessment(BaseModel):
    """Assessment of research quality and completeness."""
    is_sufficient: bool = Field(description="Whether the current information is sufficient")
    missing_fields: List[str] = Field(description="List of missing or incomplete fields")
    reasoning: str = Field(description="Explanation of the quality assessment")
    suggested_queries: List[str] = Field(default_factory=list, description="Additional queries to improve information")


# State Schema
class ResearchState(TypedDict):
    messages: List[BaseMessage]
    company_name: str
    user_notes: Optional[str]
    company_info: CompanyInfo
    search_results: List[Dict[str, Any]]
    queries_executed: int
    reflection_count: int
    max_queries: int
    max_search_results: int
    max_reflections: int
    is_complete: bool


# Configuration
MAX_QUERIES_DEFAULT = 8
MAX_SEARCH_RESULTS_DEFAULT = 5
MAX_REFLECTIONS_DEFAULT = 3

# Initialize LLM and Tools
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize search tool with error handling
try:
    search_tool = TavilySearchResults(max_results=MAX_SEARCH_RESULTS_DEFAULT)
except Exception as e:
    print(f"Warning: Could not initialize Tavily search tool: {e}")
    print("Make sure TAVILY_API_KEY is set in your environment")
    search_tool = None


def generate_queries_node(state: ResearchState) -> Dict[str, Any]:
    """Generate search queries based on company name and current information gaps."""
    company_name = state["company_name"]
    user_notes = state.get("user_notes", "")
    current_info = state["company_info"]
    queries_executed = state.get("queries_executed", 0)
    max_queries = state.get("max_queries", MAX_QUERIES_DEFAULT)
    
    remaining_queries = max_queries - queries_executed
    if remaining_queries <= 0:
        return {"messages": state["messages"]}
    
    # Determine what information is missing
    missing_info = []
    if not current_info.founding_year:
        missing_info.append("founding year")
    if not current_info.founder_names:
        missing_info.append("founder names")
    if not current_info.product_description:
        missing_info.append("product/service description")
    if not current_info.funding_summary:
        missing_info.append("funding history")
    if not current_info.notable_customers:
        missing_info.append("notable customers")
    
    prompt = f"""Generate {min(remaining_queries, 4)} specific search queries to research {company_name}.

Current information gaps: {', '.join(missing_info) if missing_info else 'Basic information verification'}

User notes: {user_notes if user_notes else 'None'}

Focus on finding:
- Company founding details (year, founders)
- Main products or services
- Funding history and investors
- Notable customers or case studies
- Recent news or developments

Generate diverse queries that will help fill information gaps."""

    messages = [SystemMessage(content=prompt)]
    
    structured_llm = model.with_structured_output(SearchQueries)
    response = structured_llm.invoke(messages)
    
    # Store generated queries in messages for tracking
    query_message = AIMessage(content=f"Generated queries: {[q.query for q in response.queries]}")
    
    return {
        "messages": state["messages"] + [query_message],
        "pending_queries": response.queries
    }


async def execute_search_queries(queries: List[SearchQuery], max_results: int) -> List[Dict[str, Any]]:
    """Execute search queries in parallel."""
    async def search_single_query(query: SearchQuery):
        try:
            if search_tool is None:
                return {
                    "query": query.query,
                    "purpose": query.purpose,
                    "results": [],
                    "error": "Search tool not initialized - missing TAVILY_API_KEY"
                }
            
            results = search_tool.invoke({"query": query.query})
            return {
                "query": query.query,
                "purpose": query.purpose,
                "results": results if isinstance(results, list) else [results]
            }
        except Exception as e:
            return {
                "query": query.query,
                "purpose": query.purpose,
                "results": [],
                "error": str(e)
            }
    
    # Execute searches in parallel
    tasks = [search_single_query(query) for query in queries]
    search_results = await asyncio.gather(*tasks)
    
    return search_results


def web_search_node(state: ResearchState) -> Dict[str, Any]:
    """Execute web searches using Tavily API."""
    pending_queries = state.get("pending_queries", [])
    if not pending_queries:
        return {"messages": state["messages"]}
    
    max_results = state.get("max_search_results", MAX_SEARCH_RESULTS_DEFAULT)
    
    # Execute searches
    search_results = asyncio.run(execute_search_queries(pending_queries, max_results))
    
    # Update state
    all_search_results = state.get("search_results", []) + search_results
    queries_executed = state.get("queries_executed", 0) + len(pending_queries)
    
    search_message = AIMessage(content=f"Executed {len(pending_queries)} search queries, found {sum(len(r.get('results', [])) for r in search_results)} total results")
    
    return {
        "messages": state["messages"] + [search_message],
        "search_results": all_search_results,
        "queries_executed": queries_executed,
        "pending_queries": []
    }


def extract_information_node(state: ResearchState) -> Dict[str, Any]:
    """Extract and structure company information from search results."""
    search_results = state.get("search_results", [])
    current_info = state["company_info"]
    company_name = state["company_name"]
    
    if not search_results:
        return {"messages": state["messages"]}
    
    # Compile all search result content
    search_content = ""
    for result_batch in search_results:
        search_content += f"\nQuery: {result_batch.get('query', 'Unknown')}\n"
        for result in result_batch.get("results", []):
            if isinstance(result, dict):
                search_content += f"Title: {result.get('title', '')}\n"
                search_content += f"Content: {result.get('content', '')}\n"
                search_content += f"URL: {result.get('url', '')}\n\n"
    
    prompt = f"""Extract and update company information for {company_name} based on the search results below.

Current information:
{current_info.model_dump_json(indent=2)}

Search Results:
{search_content}

Please update the CompanyInfo with any new information found. Preserve existing information unless you find more accurate data. Be specific and factual."""

    messages = [SystemMessage(content=prompt)]
    
    structured_llm = model.with_structured_output(CompanyInfo)
    updated_info = structured_llm.invoke(messages)
    
    # Preserve company name if not found in search
    if not updated_info.company_name:
        updated_info.company_name = company_name
    
    extraction_message = AIMessage(content=f"Updated company information based on {len(search_results)} search result batches")
    
    return {
        "messages": state["messages"] + [extraction_message],
        "company_info": updated_info
    }


def reflection_node(state: ResearchState) -> Dict[str, Any]:
    """Assess the quality and completeness of gathered information."""
    company_info = state["company_info"]
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", MAX_REFLECTIONS_DEFAULT)
    queries_executed = state.get("queries_executed", 0)
    max_queries = state.get("max_queries", MAX_QUERIES_DEFAULT)
    
    prompt = f"""Assess the quality and completeness of the following company information:

{company_info.model_dump_json(indent=2)}

Consider:
1. Are the required fields filled (company_name is required)?
2. How complete is the information across all fields?
3. Is the information specific and detailed enough?
4. Are there critical gaps that need more research?

Current status:
- Queries executed: {queries_executed}/{max_queries}
- Reflection round: {reflection_count + 1}/{max_reflections}

Provide assessment and suggest up to 3 additional queries if needed."""

    messages = [SystemMessage(content=prompt)]
    
    structured_llm = model.with_structured_output(QualityAssessment)
    assessment = structured_llm.invoke(messages)
    
    # Determine if we should continue searching
    should_continue = (
        not assessment.is_sufficient and 
        reflection_count < max_reflections and 
        queries_executed < max_queries and
        assessment.suggested_queries
    )
    
    reflection_message = AIMessage(
        content=f"Quality assessment: {'Sufficient' if assessment.is_sufficient else 'Needs more research'}. "
                f"Missing: {', '.join(assessment.missing_fields) if assessment.missing_fields else 'None'}"
    )
    
    return {
        "messages": state["messages"] + [reflection_message],
        "reflection_count": reflection_count + 1,
        "is_complete": assessment.is_sufficient or not should_continue,
        "quality_assessment": assessment
    }


def should_continue_research(state: ResearchState) -> str:
    """Determine if we should continue research or end."""
    is_complete = state.get("is_complete", False)
    queries_executed = state.get("queries_executed", 0)
    max_queries = state.get("max_queries", MAX_QUERIES_DEFAULT)
    reflection_count = state.get("reflection_count", 0)
    max_reflections = state.get("max_reflections", MAX_REFLECTIONS_DEFAULT)
    
    # End if complete or reached limits
    if is_complete or queries_executed >= max_queries or reflection_count >= max_reflections:
        return "end"
    
    # Check if we have suggested queries from last reflection
    quality_assessment = state.get("quality_assessment")
    if quality_assessment and quality_assessment.suggested_queries:
        return "continue"
    
    return "end"


def format_final_response(state: ResearchState) -> Dict[str, Any]:
    """Format the final research results."""
    company_info = state["company_info"]
    queries_executed = state.get("queries_executed", 0)
    reflection_count = state.get("reflection_count", 0)
    
    final_message = AIMessage(
        content=f"""Company Research Complete for {company_info.company_name}

Results:
{company_info.model_dump_json(indent=2)}

Research Summary:
- Search queries executed: {queries_executed}
- Reflection rounds: {reflection_count}
- Data completeness: {'High' if reflection_count == 0 else 'Moderate' if reflection_count < 2 else 'Basic'}
"""
    )
    
    return {"messages": state["messages"] + [final_message]}


# Create the graph
def create_company_research_graph():
    """Create the company research workflow graph."""
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_queries_node)
    workflow.add_node("web_search", web_search_node)
    workflow.add_node("extract_info", extract_information_node)
    workflow.add_node("reflect", reflection_node)
    workflow.add_node("format_response", format_final_response)
    
    # Add edges
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "web_search")
    workflow.add_edge("web_search", "extract_info")
    workflow.add_edge("extract_info", "reflect")
    
    # Conditional edge for continuing research
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "continue": "generate_queries",
            "end": "format_response"
        }
    )
    
    workflow.add_edge("format_response", END)
    
    return workflow.compile()


# Export the compiled graph as 'app' for deployment
app = create_company_research_graph()


# Utility function for easy invocation
def research_company(
    company_name: str, 
    user_notes: Optional[str] = None,
    max_queries: int = MAX_QUERIES_DEFAULT,
    max_search_results: int = MAX_SEARCH_RESULTS_DEFAULT,
    max_reflections: int = MAX_REFLECTIONS_DEFAULT
) -> CompanyInfo:
    """Research a company and return structured information."""
    initial_state = {
        "messages": [HumanMessage(content=f"Research company: {company_name}")],
        "company_name": company_name,
        "user_notes": user_notes,
        "company_info": CompanyInfo(company_name=company_name),
        "search_results": [],
        "queries_executed": 0,
        "reflection_count": 0,
        "max_queries": max_queries,
        "max_search_results": max_search_results,
        "max_reflections": max_reflections,
        "is_complete": False
    }
    
    result = app.invoke(initial_state)
    return result["company_info"]


if __name__ == "__main__":
    # Example usage
    company_info = research_company(
        company_name="OpenAI",
        user_notes="Focus on recent developments and funding",
        max_queries=6
    )
    print(json.dumps(company_info.model_dump(), indent=2))