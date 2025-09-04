from typing import List, Dict, Any, Optional, TypedDict
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
import asyncio
import os
from tavily import TavilyClient

# Pydantic models for structured output
class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: Optional[List[str]] = Field(None, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")

class SearchQuery(BaseModel):
    """A search query to find company information"""
    query: str = Field(description="The search query")
    purpose: str = Field(description="What information this query is trying to find")

class SearchQueries(BaseModel):
    """List of search queries for company research"""
    queries: List[SearchQuery] = Field(description="List of search queries")

class ReflectionDecision(BaseModel):
    """Decision on whether we have sufficient information"""
    has_sufficient_info: bool = Field(description="Whether we have sufficient information about the company")
    missing_info: List[str] = Field(description="List of missing information types")
    confidence_score: float = Field(description="Confidence in the current information (0-1)")

# State schema
class ResearchState(TypedDict):
    messages: List[BaseMessage]
    company_name: str
    user_notes: Optional[str]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    company_info: CompanyInfo
    reflection_count: int
    max_search_queries: int
    max_search_results: int
    max_reflection_steps: int

# Configuration constants
MAX_SEARCH_QUERIES = 5
MAX_SEARCH_RESULTS = 3
MAX_REFLECTION_STEPS = 2

# Initialize LLM
model = ChatAnthropic(model="claude-3-5-sonnet-20241022")

# Initialize Tavily client
tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def generate_search_queries_node(state: ResearchState) -> Dict[str, Any]:
    """Generate search queries to gather company information"""
    
    system_prompt = f"""You are a research assistant tasked with generating search queries to gather comprehensive information about a company.

Company: {state['company_name']}
Additional notes: {state.get('user_notes', 'None')}

Generate {state['max_search_queries']} specific search queries that will help gather information for these fields:
- Company name (official name)
- Founding year
- Founder names
- Product/service description
- Funding history
- Notable customers

Make queries specific and varied to get comprehensive information. Avoid duplicate or overly similar queries."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate search queries for researching {state['company_name']}")
    ]
    
    structured_llm = model.with_structured_output(SearchQueries)
    result = structured_llm.invoke(messages)
    
    # Extract just the query strings
    query_strings = [q.query for q in result.queries[:state['max_search_queries']]]
    
    return {
        "search_queries": query_strings,
        "messages": state["messages"] + [
            HumanMessage(content=f"Generate search queries for {state['company_name']}"),
            AIMessage(content=f"Generated {len(query_strings)} search queries: {', '.join(query_strings)}")
        ]
    }

async def perform_web_search_node(state: ResearchState) -> Dict[str, Any]:
    """Perform web searches using Tavily API"""
    
    search_results = []
    
    # Perform searches in parallel
    async def search_single_query(query: str):
        try:
            # Use Tavily's search method
            response = tavily_client.search(
                query=query,
                search_depth="advanced",
                max_results=state['max_search_results']
            )
            return {
                "query": query,
                "results": response.get("results", [])
            }
        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            return {"query": query, "results": []}
    
    # Run searches in parallel
    search_tasks = [search_single_query(query) for query in state['search_queries']]
    search_results = await asyncio.gather(*search_tasks)
    
    # Flatten all results
    all_results = []
    for search_result in search_results:
        for result in search_result["results"]:
            all_results.append({
                "query": search_result["query"],
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "url": result.get("url", "")
            })
    
    return {
        "search_results": all_results,
        "messages": state["messages"] + [
            AIMessage(content=f"Completed web searches. Found {len(all_results)} total results across {len(state['search_queries'])} queries.")
        ]
    }

def extract_company_info_node(state: ResearchState) -> Dict[str, Any]:
    """Extract structured company information from search results"""
    
    # Prepare search results text
    search_context = ""
    for i, result in enumerate(state['search_results'][:15], 1):  # Limit to prevent context overflow
        search_context += f"\n--- Result {i} (Query: {result['query']}) ---\n"
        search_context += f"Title: {result['title']}\n"
        search_context += f"Content: {result['content'][:500]}...\n"  # Truncate long content
        search_context += f"URL: {result['url']}\n"
    
    system_prompt = f"""You are a research analyst tasked with extracting structured company information from web search results.

Company being researched: {state['company_name']}
User notes: {state.get('user_notes', 'None')}

Based on the search results below, extract and structure the company information. If certain information is not found or unclear, leave those fields as null/None.

Be precise and factual. Only include information that you can verify from the search results.

Search Results:
{search_context}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract structured company information for {state['company_name']} from the search results.")
    ]
    
    structured_llm = model.with_structured_output(CompanyInfo)
    company_info = structured_llm.invoke(messages)
    
    return {
        "company_info": company_info,
        "messages": state["messages"] + [
            HumanMessage(content="Extract company information from search results"),
            AIMessage(content=f"Extracted company information: {company_info.model_dump_json(indent=2)}")
        ]
    }

def reflection_node(state: ResearchState) -> Dict[str, Any]:
    """Reflect on the gathered information and decide if more research is needed"""
    
    current_info = state['company_info']
    
    system_prompt = f"""You are a research quality assessor. Review the gathered company information and determine if it's sufficient and accurate.

Company: {state['company_name']}
Current information gathered:
{current_info.model_dump_json(indent=2)}

Evaluate:
1. How complete is the information?
2. Are there any obvious gaps?
3. Is the information consistent and credible?
4. Do we have enough detail for each field?

Consider that we want comprehensive information covering:
- Company name (required)
- Founding year
- Founder names
- Product/service description
- Funding history
- Notable customers

Current reflection step: {state['reflection_count'] + 1}/{state['max_reflection_steps']}"""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content="Assess the quality and completeness of the gathered company information.")
    ]
    
    structured_llm = model.with_structured_output(ReflectionDecision)
    decision = structured_llm.invoke(messages)
    
    reflection_count = state['reflection_count'] + 1
    
    return {
        "reflection_count": reflection_count,
        "messages": state["messages"] + [
            HumanMessage(content="Assess information completeness"),
            AIMessage(content=f"Reflection decision: sufficient_info={decision.has_sufficient_info}, confidence={decision.confidence_score}, missing={decision.missing_info}")
        ]
    }

def should_continue_research(state: ResearchState) -> str:
    """Determine if we should continue research or end"""
    
    # Check if we've hit reflection limit
    if state['reflection_count'] >= state['max_reflection_steps']:
        return "end"
    
    # Get the last reflection decision
    current_info = state['company_info']
    
    # Simple heuristic: if we have basic info, we're done
    if (current_info.company_name and 
        current_info.product_description and 
        (current_info.founding_year or current_info.founder_names)):
        return "end"
    
    # Otherwise, continue research
    return "continue_research"

# Build the graph
def create_company_research_graph():
    """Create the company research graph"""
    
    graph_builder = StateGraph(ResearchState)
    
    # Add nodes
    graph_builder.add_node("generate_queries", generate_search_queries_node)
    graph_builder.add_node("web_search", perform_web_search_node)
    graph_builder.add_node("extract_info", extract_company_info_node)
    graph_builder.add_node("reflect", reflection_node)
    
    # Add edges
    graph_builder.add_edge(START, "generate_queries")
    graph_builder.add_edge("generate_queries", "web_search")
    graph_builder.add_edge("web_search", "extract_info")
    graph_builder.add_edge("extract_info", "reflect")
    
    # Conditional edge from reflect
    graph_builder.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "continue_research": "generate_queries",
            "end": END
        }
    )
    
    return graph_builder.compile()

# Create and export the compiled graph
graph = create_company_research_graph()
app = graph

def run_company_research(company_name: str, user_notes: str = None) -> CompanyInfo:
    """Helper function to run company research"""
    
    initial_state: ResearchState = {
        "messages": [HumanMessage(content=f"Research company: {company_name}")],
        "company_name": company_name,
        "user_notes": user_notes,
        "search_queries": [],
        "search_results": [],
        "company_info": CompanyInfo(company_name=company_name),
        "reflection_count": 0,
        "max_search_queries": MAX_SEARCH_QUERIES,
        "max_search_results": MAX_SEARCH_RESULTS,
        "max_reflection_steps": MAX_REFLECTION_STEPS
    }
    
    result = app.invoke(initial_state)
    return result["company_info"]

if __name__ == "__main__":
    # Example usage
    company_name = input("Enter company name: ")
    user_notes = input("Enter optional notes (press enter to skip): ") or None
    
    try:
        company_info = run_company_research(company_name, user_notes)
        print("\n=== Company Research Results ===")
        print(company_info.model_dump_json(indent=2))
    except Exception as e:
        print(f"Error during research: {e}")