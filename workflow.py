from typing import Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.message import MessagesState
from schemas import ResearchState
from nodes import CompanyResearchNodes

class CompanyResearchWorkflow:
    def __init__(self):
        self.nodes = CompanyResearchNodes()
        self.workflow = self._build_workflow()
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self.nodes.generate_search_queries)
        workflow.add_node("search", self.nodes.execute_search_queries)
        workflow.add_node("extract", self.nodes.extract_company_information)
        workflow.add_node("reflect", self.nodes.reflect_and_decide)
        
        # Define the workflow edges
        workflow.set_entry_point("generate_queries")
        
        # generate_queries -> search
        workflow.add_edge("generate_queries", "search")
        
        # search -> extract
        workflow.add_edge("search", "extract")
        
        # extract -> reflect
        workflow.add_edge("extract", "reflect")
        
        # reflect -> conditional routing
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "continue": "generate_queries",  # Go back for more research
                "finish": END                    # Complete the research
            }
        )
        
        return workflow.compile()
    
    def _should_continue(self, state: Dict[str, Any]) -> str:
        """Determine if the workflow should continue or finish"""
        # Convert dict back to ResearchState for checking
        if isinstance(state, dict):
            research_state = ResearchState(**state)
        else:
            research_state = state
            
        if research_state.is_complete:
            return "finish"
        else:
            return "continue"
    
    async def research_company(
        self, 
        company_name: str, 
        user_notes: str = None,
        max_search_queries: int = 8,
        max_search_results: int = 5,
        max_reflection_steps: int = 3
    ) -> ResearchState:
        """
        Main method to research a company
        
        Args:
            company_name: Name of the company to research
            user_notes: Optional notes about the company
            max_search_queries: Maximum number of search queries to execute
            max_search_results: Maximum results per search query
            max_reflection_steps: Maximum reflection iterations
            
        Returns:
            ResearchState: Final state with company information
        """
        
        # Initialize the research state
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            max_search_queries=max_search_queries,
            max_search_results=max_search_results,
            max_reflection_steps=max_reflection_steps
        )
        
        initial_state.add_message("user", f"Starting research for {company_name}")
        if user_notes:
            initial_state.add_message("user", f"User notes: {user_notes}")
        
        try:
            # Execute the workflow (convert to dict for LangGraph)
            initial_dict = initial_state.model_dump()
            final_dict = await self.workflow.ainvoke(initial_dict)
            
            # Convert back to ResearchState
            final_state = ResearchState(**final_dict)
            final_state.add_message("system", "Research workflow completed")
            return final_state
            
        except Exception as e:
            initial_state.add_message("error", f"Workflow execution failed: {str(e)}")
            return initial_state
    
    def print_research_summary(self, state: ResearchState) -> None:
        """Print a formatted summary of the research results"""
        print(f"\n{'='*60}")
        print(f"COMPANY RESEARCH SUMMARY")
        print(f"{'='*60}")
        
        if state.company_info:
            info = state.company_info
            print(f"Company Name: {info.company_name}")
            print(f"Founding Year: {info.founding_year or 'Not found'}")
            print(f"Founders: {', '.join(info.founder_names) if info.founder_names else 'Not found'}")
            print(f"Product/Service: {info.product_description or 'Not found'}")
            print(f"Funding Summary: {info.funding_summary or 'Not found'}")
            print(f"Notable Customers: {info.notable_customers or 'Not found'}")
        else:
            print("No company information extracted")
        
        print(f"\n{'='*60}")
        print(f"RESEARCH STATISTICS")
        print(f"{'='*60}")
        print(f"Search queries executed: {state.queries_executed}")
        print(f"Search results collected: {len(state.search_results)}")
        print(f"Reflection steps: {state.reflection_count}")
        print(f"Messages: {len(state.messages)}")
        
        missing_fields = state.get_missing_fields()
        if missing_fields:
            print(f"Missing information: {', '.join(missing_fields)}")
        else:
            print("All information fields populated")
        
        print(f"\n{'='*60}")
        print(f"CONVERSATION HISTORY")
        print(f"{'='*60}")
        for i, message in enumerate(state.messages[-10:], 1):  # Show last 10 messages
            role = message.get('role', 'unknown').upper()
            content = message.get('content', '')[:100] + ('...' if len(message.get('content', '')) > 100 else '')
            print(f"{i}. [{role}] {content}")
        
        print(f"\n{'='*60}")

# Example usage function
async def main():
    """Example usage of the company research workflow"""
    
    # Initialize the workflow
    researcher = CompanyResearchWorkflow()
    
    # Research a company
    company_name = "Anthropic"
    user_notes = "AI safety company that created Claude"
    
    print(f"Starting research for: {company_name}")
    if user_notes:
        print(f"User notes: {user_notes}")
    
    # Execute the research
    result = await researcher.research_company(
        company_name=company_name,
        user_notes=user_notes,
        max_search_queries=6,
        max_search_results=4,
        max_reflection_steps=2
    )
    
    # Print the results
    researcher.print_research_summary(result)
    
    return result

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())