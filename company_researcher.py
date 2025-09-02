import asyncio
import json
import os
from typing import Dict, Any
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

from models import ResearchState, CompanyInfo
from nodes import QueryGenerationNode, WebSearchNode, InformationExtractionNode, ReflectionNode


class CompanyResearcher:
    """Main company researcher using LangGraph"""
    
    def __init__(
        self, 
        openai_api_key: str = None,
        tavily_api_key: str = None,
        model: str = "gpt-4",
        max_search_queries: int = 8,
        max_search_results: int = 50,
        max_reflections: int = 3
    ):
        # Load environment variables
        load_dotenv()
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=openai_api_key or os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0.1
        )
        
        # Initialize Tavily client
        self.tavily_client = TavilyClient(
            api_key=tavily_api_key or os.getenv("TAVILY_API_KEY")
        )
        
        # Configuration
        self.max_search_queries = max_search_queries
        self.max_search_results = max_search_results
        self.max_reflections = max_reflections
        
        # Initialize nodes
        self.query_node = QueryGenerationNode(self.llm)
        self.search_node = WebSearchNode(self.tavily_client)
        self.extraction_node = InformationExtractionNode(self.llm)
        self.reflection_node = ReflectionNode(self.llm)
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        
        # Create the state graph
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self.query_node)
        workflow.add_node("web_search", self.search_node) 
        workflow.add_node("extract_info", self.extraction_node)
        workflow.add_node("reflect", self.reflection_node)
        
        # Define the workflow edges
        workflow.set_entry_point("generate_queries")
        
        # Generate queries -> Web search
        workflow.add_edge("generate_queries", "web_search")
        
        # Web search -> Extract information
        workflow.add_edge("web_search", "extract_info")
        
        # Extract information -> Reflect
        workflow.add_edge("extract_info", "reflect")
        
        # Conditional edge from reflect
        def should_continue(state: ResearchState) -> str:
            """Determine if research should continue"""
            if state.research_complete:
                return END
            else:
                return "generate_queries"  # Generate more queries
        
        workflow.add_conditional_edges(
            "reflect",
            should_continue
        )
        
        return workflow.compile()
    
    async def research_company(
        self, 
        company_name: str, 
        user_notes: str = None
    ) -> Dict[str, Any]:
        """Research a company and return structured information"""
        
        # Initialize state
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            company_info=CompanyInfo(company_name=company_name),
            max_search_queries=self.max_search_queries,
            max_search_results=self.max_search_results,
            max_reflections=self.max_reflections
        )
        
        print(f"🔍 Starting research for: {company_name}")
        if user_notes:
            print(f"📝 User notes: {user_notes}")
        
        # Run the graph
        final_state = await self.graph.ainvoke(initial_state)
        
        # Prepare results
        results = {
            "company_info": final_state.company_info.model_dump(),
            "research_summary": {
                "queries_executed": final_state.queries_executed,
                "max_queries": final_state.max_search_queries,
                "results_collected": len(final_state.search_results),
                "reflections_done": final_state.reflections_done,
                "research_complete": final_state.research_complete
            },
            "messages": final_state.messages,
            "search_queries": [q.model_dump() for q in final_state.search_queries_generated]
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]) -> None:
        """Print formatted research results"""
        
        company_info = results["company_info"]
        summary = results["research_summary"]
        
        print("\n" + "="*50)
        print(f"📊 COMPANY RESEARCH RESULTS")
        print("="*50)
        
        print(f"\n🏢 Company: {company_info['company_name']}")
        
        if company_info.get('founding_year'):
            print(f"📅 Founded: {company_info['founding_year']}")
        
        if company_info.get('founder_names'):
            founders = ", ".join(company_info['founder_names'])
            print(f"👥 Founders: {founders}")
        
        if company_info.get('product_description'):
            print(f"🚀 Product: {company_info['product_description']}")
        
        if company_info.get('funding_summary'):
            print(f"💰 Funding: {company_info['funding_summary']}")
        
        if company_info.get('notable_customers'):
            print(f"🤝 Customers: {company_info['notable_customers']}")
        
        print(f"\n📈 Research Summary:")
        print(f"   • Search queries: {summary['queries_executed']}/{summary['max_queries']}")
        print(f"   • Results collected: {summary['results_collected']}")
        print(f"   • Reflections: {summary['reflections_done']}")
        print(f"   • Complete: {'✅' if summary['research_complete'] else '❌'}")
        
        print("\n" + "="*50)


# Convenience function for quick research
async def research_company(
    company_name: str,
    user_notes: str = None,
    **kwargs
) -> Dict[str, Any]:
    """Quick research function"""
    researcher = CompanyResearcher(**kwargs)
    return await researcher.research_company(company_name, user_notes)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Example research
        results = await research_company(
            company_name="OpenAI",
            user_notes="Focus on recent developments and partnerships"
        )
        
        # Print results
        researcher = CompanyResearcher()
        researcher.print_results(results)
        
        # Save to file
        with open("research_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: research_results.json")
    
    # Run the example
    asyncio.run(main())