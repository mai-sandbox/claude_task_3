from typing import List, Dict, Any, Optional
from typing_extensions import TypedDict
import asyncio
from concurrent.futures import ThreadPoolExecutor
import json

from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from tavily import TavilyClient


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """Represents a search query to be executed"""
    query: str = Field(description="The search query text")
    priority: int = Field(description="Priority level (1-5, 5 being highest)")


class SearchQueries(BaseModel):
    """Collection of search queries for company research"""
    queries: List[SearchQuery] = Field(description="List of search queries to execute")


class ReflectionResult(BaseModel):
    """Result of reflection analysis"""
    is_sufficient: bool = Field(description="Whether the current information is sufficient")
    missing_fields: List[str] = Field(default_factory=list, description="List of missing or incomplete fields")
    confidence_score: float = Field(description="Confidence score from 0-1 for the current information quality")
    reasoning: str = Field(description="Explanation of the reflection decision")


class ResearchState(TypedDict):
    """State for the company research workflow"""
    messages: List[BaseMessage]
    company_name: str
    user_notes: Optional[str]
    search_queries: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    company_info: Dict[str, Any]
    reflection_count: int
    max_reflections: int
    max_search_queries: int
    max_search_results: int
    is_complete: bool


# Initialize models and clients
model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)

# Initialize Tavily client with error handling
try:
    tavily_client = TavilyClient()
except Exception as e:
    print(f"Warning: Tavily client initialization failed: {e}")
    tavily_client = None

# Configuration constants
MAX_SEARCH_QUERIES = 8
MAX_SEARCH_RESULTS = 5
MAX_REFLECTIONS = 3


def generate_search_queries(state: ResearchState) -> Dict[str, Any]:
    """Generate targeted search queries for company research"""
    company_name = state["company_name"]
    user_notes = state.get("user_notes", "")
    reflection_count = state.get("reflection_count", 0)
    
    system_prompt = f"""You are a research expert. Generate {MAX_SEARCH_QUERIES} specific search queries to gather comprehensive information about the company "{company_name}".
    
    User notes: {user_notes if user_notes else "None provided"}
    
    Focus on finding:
    - Company founding information (year, founders)
    - Product/service descriptions
    - Funding history and financial information
    - Notable customers and partnerships
    - Recent news and developments
    
    Reflection round: {reflection_count}
    
    Create diverse queries that will capture different aspects of the company. Prioritize queries based on importance (1-5, 5 being most critical).
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Generate search queries for researching {company_name}")
    ]
    
    structured_model = model.with_structured_output(SearchQueries)
    result = structured_model.invoke(messages)
    
    search_queries = [{"query": q.query, "priority": q.priority} for q in result.queries]
    
    ai_message = AIMessage(content=f"Generated {len(search_queries)} search queries for {company_name}")
    
    return {
        "messages": state["messages"] + [ai_message],
        "search_queries": search_queries
    }


def execute_web_searches(state: ResearchState) -> Dict[str, Any]:
    """Execute web searches using Tavily API with parallel processing"""
    search_queries = state["search_queries"]
    max_results = state.get("max_search_results", MAX_SEARCH_RESULTS)
    
    def search_single_query(query_data):
        try:
            if tavily_client is None:
                return {
                    "query": query_data["query"],
                    "priority": query_data["priority"],
                    "results": [],
                    "answer": "Error: Tavily client not available (missing API key)"
                }
            
            query = query_data["query"]
            response = tavily_client.search(
                query=query,
                max_results=max_results,
                include_answer=True,
                include_raw_content=False
            )
            return {
                "query": query,
                "priority": query_data["priority"],
                "results": response.get("results", []),
                "answer": response.get("answer", "")
            }
        except Exception as e:
            return {
                "query": query_data["query"],
                "priority": query_data["priority"],
                "results": [],
                "answer": f"Error: {str(e)}"
            }
    
    # Execute searches in parallel
    with ThreadPoolExecutor(max_workers=4) as executor:
        search_results = list(executor.map(search_single_query, search_queries))
    
    ai_message = AIMessage(content=f"Completed {len(search_results)} web searches")
    
    return {
        "messages": state["messages"] + [ai_message],
        "search_results": search_results
    }


def extract_company_information(state: ResearchState) -> Dict[str, Any]:
    """Extract structured company information from search results"""
    company_name = state["company_name"]
    search_results = state["search_results"]
    
    # Prepare search context
    search_context = ""
    for result in search_results:
        search_context += f"\n--- Query: {result['query']} (Priority: {result['priority']}) ---\n"
        if result.get("answer"):
            search_context += f"Answer: {result['answer']}\n"
        
        for item in result.get("results", [])[:3]:  # Limit to top 3 results per query
            search_context += f"Title: {item.get('title', 'N/A')}\n"
            search_context += f"Content: {item.get('content', 'N/A')[:500]}...\n"
            search_context += f"URL: {item.get('url', 'N/A')}\n\n"
    
    system_prompt = f"""You are an expert research analyst. Extract comprehensive information about "{company_name}" from the provided search results.

    Fill out the CompanyInfo structure with accurate information. Only include information you can verify from the search results.
    
    For fields where you cannot find definitive information, leave them empty or null rather than guessing.
    
    Search Results:
    {search_context}
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Extract company information for {company_name} from the search results")
    ]
    
    structured_model = model.with_structured_output(CompanyInfo)
    company_info = structured_model.invoke(messages)
    
    # Convert to dict for state storage
    company_info_dict = company_info.dict()
    
    ai_message = AIMessage(content=f"Extracted company information for {company_name}")
    
    return {
        "messages": state["messages"] + [ai_message],
        "company_info": company_info_dict
    }


def reflect_on_information(state: ResearchState) -> Dict[str, Any]:
    """Analyze if the gathered information is sufficient and complete"""
    company_info = state["company_info"]
    company_name = state["company_name"]
    reflection_count = state["reflection_count"]
    max_reflections = state["max_reflections"]
    
    # Create summary of current information
    info_summary = f"""
    Company Name: {company_info.get('company_name', 'Missing')}
    Founding Year: {company_info.get('founding_year', 'Missing')}
    Founders: {company_info.get('founder_names', [])}
    Product Description: {company_info.get('product_description', 'Missing')}
    Funding Summary: {company_info.get('funding_summary', 'Missing')}
    Notable Customers: {company_info.get('notable_customers', 'Missing')}
    """
    
    system_prompt = f"""You are a quality assurance expert for company research. Analyze the current information gathered about "{company_name}" and determine if it's sufficient for a comprehensive company profile.
    
    Current reflection count: {reflection_count}/{max_reflections}
    
    Current Information:
    {info_summary}
    
    Evaluate:
    1. Completeness: Are key fields populated with meaningful information?
    2. Quality: Is the information specific and valuable?
    3. Accuracy: Does the information seem consistent and reliable?
    
    Consider the information sufficient if:
    - Company name is confirmed
    - At least 4 out of 6 total fields have meaningful content
    - The information provides a good overview of the company
    
    Be more lenient if this is the final reflection round.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Evaluate the completeness of information for {company_name}")
    ]
    
    structured_model = model.with_structured_output(ReflectionResult)
    reflection = structured_model.invoke(messages)
    
    # Determine if we should continue or finish
    is_complete = (
        reflection.is_sufficient or 
        reflection_count >= max_reflections or
        reflection.confidence_score >= 0.8
    )
    
    ai_message = AIMessage(
        content=f"Reflection {reflection_count + 1}: {'Sufficient' if reflection.is_sufficient else 'More research needed'} - {reflection.reasoning}"
    )
    
    return {
        "messages": state["messages"] + [ai_message],
        "reflection_count": reflection_count + 1,
        "is_complete": is_complete
    }


def should_continue_research(state: ResearchState) -> str:
    """Determine if more research is needed"""
    return "end" if state["is_complete"] else "generate_queries"


# Build the graph
def create_company_research_graph():
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("execute_searches", execute_web_searches)
    workflow.add_node("extract_info", extract_company_information)
    workflow.add_node("reflect", reflect_on_information)
    
    # Add edges
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "execute_searches")
    workflow.add_edge("execute_searches", "extract_info")
    workflow.add_edge("extract_info", "reflect")
    
    # Conditional edge based on reflection
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "generate_queries": "generate_queries",
            "end": END
        }
    )
    
    return workflow.compile()


# Export the compiled graph as 'app'
graph = create_company_research_graph()
app = graph


# Helper function to run research
def research_company(company_name: str, user_notes: str = "", max_reflections: int = MAX_REFLECTIONS):
    """Run company research with the specified parameters"""
    initial_state = ResearchState(
        messages=[HumanMessage(content=f"Research company: {company_name}")],
        company_name=company_name,
        user_notes=user_notes,
        search_queries=[],
        search_results=[],
        company_info={},
        reflection_count=0,
        max_reflections=max_reflections,
        max_search_queries=MAX_SEARCH_QUERIES,
        max_search_results=MAX_SEARCH_RESULTS,
        is_complete=False
    )
    
    result = app.invoke(initial_state)
    return result["company_info"]


if __name__ == "__main__":
    # Example usage
    company = "OpenAI"
    notes = "Focus on recent developments and AI safety initiatives"
    
    print(f"Researching {company}...")
    result = research_company(company, notes)
    print(json.dumps(result, indent=2))