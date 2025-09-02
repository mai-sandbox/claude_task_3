import asyncio
from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models import ResearchState, CompanyInfo
from nodes import CompanyResearchNodes


class CompanyResearcher:
    """Main company research workflow using LangGraph"""
    
    def __init__(self, openai_api_key: str = None, tavily_api_key: str = None):
        self.nodes = CompanyResearchNodes(
            openai_api_key=openai_api_key,
            tavily_api_key=tavily_api_key
        )
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the state graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self.nodes.generate_search_queries)
        workflow.add_node("search_web", self._search_web_wrapper)  # Async wrapper needed
        workflow.add_node("extract_info", self.nodes.extract_information)
        workflow.add_node("reflect", self.nodes.reflect_and_decide)
        
        # Define the workflow edges
        workflow.set_entry_point("generate_queries")
        
        # Flow: generate_queries -> search_web -> extract_info -> reflect
        workflow.add_edge("generate_queries", "search_web")
        workflow.add_edge("search_web", "extract_info")
        workflow.add_edge("extract_info", "reflect")
        
        # Conditional edge from reflect
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "continue": "generate_queries",  # Loop back for more research
                "end": END  # Complete the research
            }
        )
        
        # Add memory saver for state persistence
        memory = MemorySaver()
        app = workflow.compile(checkpointer=memory)
        
        return app
    
    def _search_web_wrapper(self, state: ResearchState) -> ResearchState:
        """Synchronous wrapper for the async search_web method"""
        try:
            # Run the async search in the current event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If we're already in an async context, create a new task
                task = asyncio.create_task(self.nodes.search_web(state))
                # This is a workaround for nested async calls
                return asyncio.run(self._run_in_new_loop(state))
            else:
                return asyncio.run(self.nodes.search_web(state))
        except Exception as e:
            state.messages.append({
                "type": "error",
                "content": f"Error in search wrapper: {str(e)}"
            })
            return state
    
    async def _run_in_new_loop(self, state: ResearchState) -> ResearchState:
        """Run search in a new event loop"""
        return await self.nodes.search_web(state)
    
    def _should_continue(self, state: ResearchState) -> str:
        """Decide whether to continue research or end"""
        if state.completed or not state.needs_more_research:
            return "end"
        return "continue"
    
    def research_company(
        self, 
        company_name: str, 
        user_notes: str = None,
        max_search_queries: int = 5,
        max_search_results: int = 10,
        max_reflection_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Research a company and return structured information
        
        Args:
            company_name: Name of the company to research
            user_notes: Optional user notes about what to focus on
            max_search_queries: Maximum number of search queries to perform
            max_search_results: Maximum number of results per search query
            max_reflection_steps: Maximum number of reflection iterations
            
        Returns:
            Dictionary with research results and metadata
        """
        
        # Initialize the research state
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            max_search_queries=max_search_queries,
            max_search_results=max_search_results,
            max_reflection_steps=max_reflection_steps,
            company_info=CompanyInfo(company_name=company_name)
        )
        
        # Configure the graph execution
        config = {"configurable": {"thread_id": f"research_{company_name}"}}
        
        try:
            # Execute the workflow
            result = self.graph.invoke(initial_state, config=config)
            
            # Return structured results
            return {
                "success": True,
                "company_info": result.company_info.model_dump(),
                "research_metadata": {
                    "search_queries_used": result.search_queries_used,
                    "reflection_steps_used": result.reflection_steps_used,
                    "total_search_results": len(result.search_results),
                    "completed": result.completed
                },
                "messages": result.messages,
                "raw_search_results": result.search_results
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "company_info": initial_state.company_info.model_dump(),
                "research_metadata": {
                    "search_queries_used": 0,
                    "reflection_steps_used": 0,
                    "total_search_results": 0,
                    "completed": False
                },
                "messages": [{"type": "error", "content": f"Graph execution failed: {str(e)}"}],
                "raw_search_results": []
            }
    
    async def research_company_async(
        self, 
        company_name: str, 
        user_notes: str = None,
        max_search_queries: int = 5,
        max_search_results: int = 10,
        max_reflection_steps: int = 3
    ) -> Dict[str, Any]:
        """
        Async version of research_company
        """
        # For now, we'll use the synchronous version
        # In a production environment, you'd want to implement proper async graph execution
        return self.research_company(
            company_name, user_notes, max_search_queries, 
            max_search_results, max_reflection_steps
        )


# Convenience function for easy usage
def research_company(
    company_name: str,
    user_notes: str = None,
    openai_api_key: str = None,
    tavily_api_key: str = None,
    max_search_queries: int = 5,
    max_search_results: int = 10,
    max_reflection_steps: int = 3
) -> Dict[str, Any]:
    """
    Convenience function to research a company
    
    Args:
        company_name: Name of the company to research
        user_notes: Optional user notes about what to focus on
        openai_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
        tavily_api_key: Tavily API key (or set TAVILY_API_KEY env var)
        max_search_queries: Maximum number of search queries to perform
        max_search_results: Maximum number of results per search query
        max_reflection_steps: Maximum number of reflection iterations
        
    Returns:
        Dictionary with research results and metadata
    """
    researcher = CompanyResearcher(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key
    )
    
    return researcher.research_company(
        company_name=company_name,
        user_notes=user_notes,
        max_search_queries=max_search_queries,
        max_search_results=max_search_results,
        max_reflection_steps=max_reflection_steps
    )