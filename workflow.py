from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from models import ResearchState
from nodes import CompanyResearchNodes

class CompanyResearchWorkflow:
    def __init__(self):
        self.nodes = CompanyResearchNodes()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        # Create the state graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self._generate_queries_node)
        workflow.add_node("execute_searches", self._execute_searches_node)
        workflow.add_node("extract_info", self._extract_info_node)
        workflow.add_node("reflect", self._reflect_node)
        
        # Set entry point
        workflow.set_entry_point("generate_queries")
        
        # Add edges
        workflow.add_edge("generate_queries", "execute_searches")
        workflow.add_edge("execute_searches", "extract_info")
        workflow.add_edge("extract_info", "reflect")
        
        # Add conditional edge from reflection
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue_research,
            {
                "continue": "generate_queries",
                "finish": END
            }
        )
        
        return workflow
    
    async def _generate_queries_node(self, state: ResearchState) -> Dict[str, Any]:
        """Wrapper for generate_search_queries node"""
        result = await self.nodes.generate_search_queries(state)
        return result
    
    async def _execute_searches_node(self, state: ResearchState) -> Dict[str, Any]:
        """Wrapper for execute_searches node"""
        result = await self.nodes.execute_searches(state)
        return result
    
    async def _extract_info_node(self, state: ResearchState) -> Dict[str, Any]:
        """Wrapper for extract_company_info node"""
        result = await self.nodes.extract_company_info(state)
        return result
    
    async def _reflect_node(self, state: ResearchState) -> Dict[str, Any]:
        """Wrapper for reflect_on_completeness node"""
        result = await self.nodes.reflect_on_completeness(state)
        return result
    
    def _should_continue_research(self, state: ResearchState) -> str:
        """Determine if we should continue research or finish"""
        # Check if we've reached limits or have sufficient info
        if (state.reflection_count >= state.max_reflections or 
            state.query_count >= state.max_queries or 
            not state.needs_more_info):
            return "finish"
        return "continue"
    
    async def research_company(
        self, 
        company_name: str, 
        user_notes: str = None,
        max_queries: int = 5,
        max_search_results: int = 3,
        max_reflections: int = 3
    ) -> ResearchState:
        """
        Research a company and return comprehensive information
        
        Args:
            company_name: Name of the company to research
            user_notes: Optional additional context from user
            max_queries: Maximum number of search queries to execute
            max_search_results: Maximum results per search query
            max_reflections: Maximum number of reflection iterations
        
        Returns:
            ResearchState with company information and research history
        """
        # Initialize state
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            max_queries=max_queries,
            max_search_results=max_search_results,
            max_reflections=max_reflections
        )
        
        # Compile and run workflow
        app = self.workflow.compile(checkpointer=MemorySaver())
        
        # Execute the workflow
        config = {"configurable": {"thread_id": f"research_{company_name}_{hash(company_name)}"}}
        
        final_state = None
        async for state in app.astream(initial_state, config):
            final_state = state
        
        # Extract the final state from the last step
        if final_state:
            # The state is nested in the step result, extract it
            for step_name, step_result in final_state.items():
                if isinstance(step_result, ResearchState):
                    return step_result
                # If it's a dict with the state fields, create ResearchState
                elif isinstance(step_result, dict) and 'company_name' in step_result:
                    return ResearchState(**step_result)
        
        return initial_state