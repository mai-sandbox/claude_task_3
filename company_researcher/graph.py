from typing import Dict, Any, TypedDict
from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic
from tavily import TavilyClient
import os

from .models import GraphState
from .nodes import QueryGenerationNode, SearchNode, InformationExtractionNode, ReflectionNode


class CompanyResearchGraph:
    def __init__(
        self, 
        anthropic_api_key: str = None,
        tavily_api_key: str = None,
        max_queries: int = 6,
        max_results_per_query: int = 3,
        max_reflections: int = 2
    ):
        # Initialize API keys
        self.anthropic_api_key = anthropic_api_key or os.getenv("ANTHROPIC_API_KEY")
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        
        if not self.anthropic_api_key:
            raise ValueError("Anthropic API key is required")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key is required")
        
        # Initialize clients and nodes
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=self.anthropic_api_key,
            temperature=0.1
        )
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        
        # Initialize nodes
        self.query_node = QueryGenerationNode(self.llm)
        self.search_node = SearchNode(self.tavily_client)
        self.extraction_node = InformationExtractionNode(self.llm)
        self.reflection_node = ReflectionNode(self.llm)
        
        # Configuration
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query
        self.max_reflections = max_reflections
        
        # Build graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        # Create state graph
        workflow = StateGraph(GraphState)
        
        # Add nodes
        workflow.add_node("query_generation", self._query_generation_wrapper)
        workflow.add_node("search", self._search_wrapper)
        workflow.add_node("extraction", self._extraction_wrapper)
        workflow.add_node("reflection", self._reflection_wrapper)
        
        # Define the flow
        workflow.set_entry_point("query_generation")
        
        workflow.add_edge("query_generation", "search")
        workflow.add_edge("search", "extraction")
        workflow.add_edge("extraction", "reflection")
        
        # Conditional edge from reflection
        workflow.add_conditional_edges(
            "reflection",
            self._should_continue,
            {
                "continue": "query_generation",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _query_generation_wrapper(self, state: GraphState) -> GraphState:
        state["max_queries"] = self.max_queries
        state["max_results_per_query"] = self.max_results_per_query
        state["max_reflections"] = self.max_reflections
        return self.query_node.execute(state)
    
    def _search_wrapper(self, state: GraphState) -> GraphState:
        return self.search_node.execute(state)
    
    def _extraction_wrapper(self, state: GraphState) -> GraphState:
        return self.extraction_node.execute(state)
    
    def _reflection_wrapper(self, state: GraphState) -> GraphState:
        return self.reflection_node.execute(state)
    
    def _should_continue(self, state: GraphState) -> str:
        # Continue if not complete and haven't exceeded limits
        if (not state["is_complete"] and 
            state["queries_executed"] < self.max_queries and 
            state["reflection_count"] < self.max_reflections):
            return "continue"
        return "end"
    
    def research_company(
        self, 
        company_name: str, 
        user_notes: str = None
    ) -> Dict[str, Any]:
        """
        Research a company and return structured information.
        
        Args:
            company_name: Name of the company to research
            user_notes: Optional additional context or specific areas of interest
            
        Returns:
            Dictionary containing the research results and metadata
        """
        # Initialize state
        initial_state: GraphState = {
            "company_name": company_name,
            "user_notes": user_notes,
            "messages": [],
            "generated_queries": [],
            "search_results": [],
            "extracted_info": None,
            "queries_executed": 0,
            "reflection_count": 0,
            "is_complete": False,
            "max_queries": self.max_queries,
            "max_results_per_query": self.max_results_per_query,
            "max_reflections": self.max_reflections
        }
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Return structured results
        return {
            "company_info": final_state["extracted_info"],
            "research_metadata": {
                "queries_executed": final_state["queries_executed"],
                "results_found": len(final_state["search_results"]),
                "reflections_performed": final_state["reflection_count"],
                "is_complete": final_state["is_complete"]
            },
            "messages": final_state["messages"],
            "search_results": final_state["search_results"]
        }
    
    def research_company_sync(
        self, 
        company_name: str, 
        user_notes: str = None
    ) -> Dict[str, Any]:
        """
        Synchronous version of research_company for easier use.
        """
        return self.research_company(company_name, user_notes)


def create_company_researcher(
    anthropic_api_key: str = None,
    tavily_api_key: str = None,
    max_queries: int = 6,
    max_results_per_query: int = 3,
    max_reflections: int = 2
) -> CompanyResearchGraph:
    """
    Factory function to create a CompanyResearchGraph instance.
    """
    return CompanyResearchGraph(
        anthropic_api_key=anthropic_api_key,
        tavily_api_key=tavily_api_key,
        max_queries=max_queries,
        max_results_per_query=max_results_per_query,
        max_reflections=max_reflections
    )