from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.graph.message import add_messages
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from tavily import TavilyClient
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: Optional[List[str]] = Field(None, description="Names of the founding team members")
    product_description: Optional[str] = Field(None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """A search query for finding company information"""
    query: str = Field(description="The search query to execute")
    purpose: str = Field(description="What information this query is meant to find")


class SearchQueries(BaseModel):
    """Collection of search queries"""
    queries: List[SearchQuery] = Field(description="List of search queries to execute")


class ReflectionResult(BaseModel):
    """Result of reflection on gathered information"""
    is_sufficient: bool = Field(description="Whether the gathered information is sufficient")
    missing_info: List[str] = Field(description="List of information that is still missing or incomplete")
    additional_queries: Optional[List[SearchQuery]] = Field(None, description="Additional queries to execute if more info is needed")


class CompanyResearchState(MessagesState):
    """State for company research workflow"""
    company_name: str
    notes: Optional[str] = None
    company_info: Optional[CompanyInfo] = None
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    queries_executed: int = 0
    max_queries: int = 5
    max_results_per_query: int = 3
    reflection_count: int = 0
    max_reflections: int = 2


# Constants
MAX_QUERIES_PER_COMPANY = 5
MAX_RESULTS_PER_QUERY = 3
MAX_REFLECTION_STEPS = 2

# Initialize LLM
model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)

# Initialize Tavily client
try:
    tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
except Exception as e:
    print(f"Warning: Tavily client not initialized - {e}")
    tavily_client = None


def generate_search_queries(state: CompanyResearchState) -> Dict[str, Any]:
    """Generate search queries for gathering company information"""
    
    system_prompt = f"""You are a research assistant tasked with generating effective search queries to gather comprehensive information about a company.

Company: {state['company_name']}
Additional notes: {state.get('notes', 'None')}

You need to generate search queries that will help find:
1. Company name (official name)
2. Founding year 
3. Founder names
4. Product/service description
5. Funding history and summary
6. Notable customers

Generate {min(MAX_QUERIES_PER_COMPANY - state['queries_executed'], 3)} specific, targeted search queries that will help find this information efficiently. Each query should have a clear purpose.

Focus on finding factual, recent information from reliable sources."""

    messages = [SystemMessage(content=system_prompt)]
    if state.get('messages'):
        messages.extend(state['messages'][-3:])  # Include recent context
    
    structured_llm = model.with_structured_output(SearchQueries)
    result = structured_llm.invoke(messages)
    
    # Limit queries based on remaining quota
    remaining_queries = MAX_QUERIES_PER_COMPANY - state['queries_executed']
    if len(result.queries) > remaining_queries:
        result.queries = result.queries[:remaining_queries]
    
    return {
        "messages": add_messages(state["messages"], [AIMessage(content=f"Generated {len(result.queries)} search queries for {state['company_name']}")]),
        "generated_queries": result.queries
    }


def execute_web_search(state: CompanyResearchState) -> Dict[str, Any]:
    """Execute web searches using Tavily API with parallel processing"""
    
    if "generated_queries" not in state or not state["generated_queries"]:
        return {"messages": add_messages(state["messages"], [AIMessage(content="No queries to execute")])}
    
    def search_single_query(query: SearchQuery) -> List[Dict[str, Any]]:
        """Execute a single search query"""
        try:
            if not tavily_client:
                return [{
                    'query_purpose': query.purpose,
                    'query': query.query,
                    'error': 'Tavily client not available',
                    'title': 'Mock Search Result',
                    'content': f'Mock search result for: {query.query}',
                    'url': 'https://example.com',
                    'score': 0.5
                }]
            
            results = tavily_client.search(
                query=query.query,
                max_results=MAX_RESULTS_PER_QUERY,
                search_depth="advanced"
            )
            
            # Extract relevant information from results
            processed_results = []
            for result in results.get('results', []):
                processed_results.append({
                    'query_purpose': query.purpose,
                    'query': query.query,
                    'title': result.get('title', ''),
                    'content': result.get('content', ''),
                    'url': result.get('url', ''),
                    'score': result.get('score', 0)
                })
            
            return processed_results
            
        except Exception as e:
            return [{
                'query_purpose': query.purpose,
                'query': query.query,
                'error': str(e),
                'title': 'Search Error',
                'content': f'Failed to search: {str(e)}',
                'url': '',
                'score': 0
            }]
    
    # Execute searches in parallel
    with ThreadPoolExecutor(max_workers=3) as executor:
        search_futures = {
            executor.submit(search_single_query, query): query 
            for query in state["generated_queries"]
        }
        
        all_results = []
        for future in search_futures:
            results = future.result()
            all_results.extend(results)
    
    # Update state
    updated_search_results = state.get("search_results", []) + all_results
    new_queries_executed = state["queries_executed"] + len(state["generated_queries"])
    
    return {
        "messages": add_messages(state["messages"], [AIMessage(content=f"Executed {len(state['generated_queries'])} search queries, found {len(all_results)} results")]),
        "search_results": updated_search_results,
        "queries_executed": new_queries_executed,
        "generated_queries": []  # Clear executed queries
    }


def extract_company_info(state: CompanyResearchState) -> Dict[str, Any]:
    """Extract structured company information from search results"""
    
    if not state.get("search_results"):
        return {"messages": add_messages(state["messages"], [AIMessage(content="No search results to process")])}
    
    # Prepare context from search results
    search_context = ""
    for result in state["search_results"]:
        if not result.get('error'):
            search_context += f"Query Purpose: {result['query_purpose']}\n"
            search_context += f"Title: {result['title']}\n"
            search_context += f"Content: {result['content'][:500]}...\n"
            search_context += f"URL: {result['url']}\n\n"
    
    system_prompt = f"""You are a research analyst extracting structured information about a company from search results.

Company Name: {state['company_name']}
Additional Notes: {state.get('notes', 'None')}

Based on the following search results, extract the requested information about the company. If some information is not available or unclear from the search results, leave those fields as null.

Be precise and factual. Only include information that is clearly supported by the search results.

Search Results:
{search_context}

Extract the information into the structured format. For arrays like founder_names, provide actual names if found, otherwise null.
"""

    messages = [SystemMessage(content=system_prompt)]
    structured_llm = model.with_structured_output(CompanyInfo)
    
    try:
        extracted_info = structured_llm.invoke(messages)
        
        # Ensure company name is set
        if not extracted_info.company_name:
            extracted_info.company_name = state['company_name']
        
        return {
            "messages": add_messages(state["messages"], [AIMessage(content=f"Extracted company information for {extracted_info.company_name}")]),
            "company_info": extracted_info
        }
        
    except Exception as e:
        # Fallback: create minimal company info
        fallback_info = CompanyInfo(company_name=state['company_name'])
        return {
            "messages": add_messages(state["messages"], [AIMessage(content=f"Error extracting info, using fallback: {str(e)}")]),
            "company_info": fallback_info
        }


def reflect_on_information(state: CompanyResearchState) -> Dict[str, Any]:
    """Reflect on the gathered information and determine if more research is needed"""
    
    if not state.get("company_info"):
        return {
            "messages": add_messages(state["messages"], [AIMessage(content="No company info to reflect on")]),
            "reflection_result": ReflectionResult(is_sufficient=False, missing_info=["All company information"])
        }
    
    company_info = state["company_info"]
    info_summary = f"""
Company Name: {company_info.company_name}
Founding Year: {company_info.founding_year or 'Not found'}
Founders: {company_info.founder_names or 'Not found'}  
Product Description: {company_info.product_description or 'Not found'}
Funding Summary: {company_info.funding_summary or 'Not found'}
Notable Customers: {company_info.notable_customers or 'Not found'}
"""

    system_prompt = f"""You are evaluating the completeness and quality of gathered company information.

Current Information Gathered:
{info_summary}

Search Context:
- Total queries executed: {state['queries_executed']}/{MAX_QUERIES_PER_COMPANY}
- Reflection count: {state['reflection_count']}/{MAX_REFLECTION_STEPS}
- Total search results: {len(state.get('search_results', []))}

Evaluate whether the information is sufficient for a comprehensive company profile. Consider:
1. Are the key facts present and accurate?
2. Is the information detailed enough to be useful?
3. Are there critical gaps that would benefit from additional research?

If more research is needed AND we haven't exceeded limits, suggest specific targeted queries to fill the gaps.
Be selective - only request additional searches for truly important missing information.
"""

    messages = [SystemMessage(content=system_prompt)]
    structured_llm = model.with_structured_output(ReflectionResult)
    
    reflection_result = structured_llm.invoke(messages)
    
    # Check limits - force sufficient if we've hit limits
    if (state['queries_executed'] >= MAX_QUERIES_PER_COMPANY or 
        state['reflection_count'] >= MAX_REFLECTION_STEPS):
        reflection_result.is_sufficient = True
        reflection_result.additional_queries = None
    
    return {
        "messages": add_messages(state["messages"], [AIMessage(content=f"Reflection complete. Sufficient: {reflection_result.is_sufficient}")]),
        "reflection_result": reflection_result,
        "reflection_count": state["reflection_count"] + 1
    }


def should_continue_research(state: CompanyResearchState) -> str:
    """Determine if we should continue research or end"""
    
    # Check if we have reflection result
    if not state.get("reflection_result"):
        return "reflect"
    
    reflection = state["reflection_result"]
    
    # If sufficient or hit limits, end
    if (reflection.is_sufficient or 
        state['queries_executed'] >= MAX_QUERIES_PER_COMPANY or
        state['reflection_count'] >= MAX_REFLECTION_STEPS):
        return "end"
    
    # If we have additional queries to execute, continue research
    if reflection.additional_queries:
        return "continue_research"
    
    return "end"


def continue_research(state: CompanyResearchState) -> Dict[str, Any]:
    """Continue research with additional queries from reflection"""
    
    if not state.get("reflection_result") or not state["reflection_result"].additional_queries:
        return {"messages": add_messages(state["messages"], [AIMessage(content="No additional queries to execute")])}
    
    additional_queries = state["reflection_result"].additional_queries
    
    return {
        "messages": add_messages(state["messages"], [AIMessage(content=f"Continuing research with {len(additional_queries)} additional queries")]),
        "generated_queries": additional_queries,
        "reflection_result": None  # Clear reflection to allow new cycle
    }


# Build the graph
def create_company_researcher():
    """Create the company research workflow graph"""
    
    workflow = StateGraph(CompanyResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", generate_search_queries)
    workflow.add_node("search", execute_web_search)
    workflow.add_node("extract_info", extract_company_info)
    workflow.add_node("reflect", reflect_on_information)
    workflow.add_node("continue_research", continue_research)
    
    # Add edges
    workflow.add_edge(START, "generate_queries")
    workflow.add_edge("generate_queries", "search")
    workflow.add_edge("search", "extract_info")
    workflow.add_edge("extract_info", "reflect")
    workflow.add_edge("continue_research", "search")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "continue_research": "continue_research",
            "end": END,
            "reflect": "reflect"
        }
    )
    
    return workflow.compile()


# Export the app
app = create_company_researcher()