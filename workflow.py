from langgraph.graph import StateGraph, END
from typing import Dict, Any
from models import ResearchState
from nodes import CompanyResearchNodes


def should_continue_research(state: ResearchState) -> str:
    """Conditional edge function to determine if research should continue"""
    if state.is_complete:
        return "end"
    elif state.needs_more_info and state.reflection_count < state.max_reflection_steps:
        return "search_more"
    else:
        return "end"


def create_company_research_workflow(openai_api_key: str, tavily_api_key: str) -> StateGraph:
    """Create the LangGraph workflow for company research"""
    
    # Initialize the nodes
    nodes = CompanyResearchNodes(openai_api_key, tavily_api_key)
    
    # Create the state graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("generate_queries", nodes.generate_search_queries)
    workflow.add_node("search_web", nodes.search_web_parallel)
    workflow.add_node("extract_info", nodes.extract_company_info)
    workflow.add_node("reflect", nodes.reflect_on_completeness)
    
    # Set entry point
    workflow.set_entry_point("generate_queries")
    
    # Add edges
    workflow.add_edge("generate_queries", "search_web")
    workflow.add_edge("search_web", "extract_info")
    workflow.add_edge("extract_info", "reflect")
    
    # Add conditional edges from reflect node
    workflow.add_conditional_edges(
        "reflect",
        should_continue_research,
        {
            "search_more": "generate_queries",  # Loop back for more searches
            "end": END
        }
    )
    
    # Compile the workflow
    return workflow.compile()


def run_company_research(
    company_name: str, 
    user_notes: str = None,
    openai_api_key: str = None,
    tavily_api_key: str = None,
    max_search_queries: int = 5,
    max_search_results: int = 3,
    max_reflection_steps: int = 2
) -> Dict[str, Any]:
    """Run the complete company research workflow"""
    
    # Create initial state
    initial_state = ResearchState(
        company_name=company_name,
        user_notes=user_notes,
        max_search_queries=max_search_queries,
        max_search_results=max_search_results,
        max_reflection_steps=max_reflection_steps,
        company_info={"company_name": company_name}  # Initialize with company name
    )
    
    # Create and run workflow
    workflow = create_company_research_workflow(openai_api_key, tavily_api_key)
    
    # Execute the workflow
    result = workflow.invoke(initial_state)
    
    # Return structured result
    return {
        "company_info": result.company_info.dict(),
        "search_queries_used": result.search_queries,
        "total_search_results": len(result.search_results),
        "reflection_steps": result.reflection_count,
        "workflow_messages": result.messages,
        "is_complete": result.is_complete
    }