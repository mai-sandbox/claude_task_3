import os
import json
import asyncio
from typing import List, Dict, Any, Optional, TypedDict, Annotated
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient


class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: str = Field("", description="Brief description of the company's main product or service")
    funding_summary: str = Field("", description="Summary of the company's funding history")
    notable_customers: str = Field("", description="Known customers that use company's product/service")


class ResearchState(TypedDict):
    """State for the company research workflow"""
    company_name: str
    user_notes: str
    messages: Annotated[list[BaseMessage], add_messages]
    company_info: CompanyInfo
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    queries_executed: int
    reflection_count: int
    max_queries: int
    max_reflections: int
    max_results_per_query: int
    is_complete: bool


class CompanyResearcher:
    def __init__(
        self,
        anthropic_api_key: str,
        tavily_api_key: str,
        max_queries: int = 6,
        max_reflections: int = 3,
        max_results_per_query: int = 5
    ):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            api_key=anthropic_api_key,
            temperature=0.1
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.max_queries = max_queries
        self.max_reflections = max_reflections
        self.max_results_per_query = max_results_per_query
        
        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self.generate_queries_node)
        workflow.add_node("web_search", self.web_search_node)
        workflow.add_node("extract_info", self.extract_info_node)
        workflow.add_node("reflect", self.reflect_node)
        workflow.add_node("final_response", self.final_response_node)
        
        # Add edges
        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", "web_search")
        workflow.add_edge("web_search", "extract_info")
        workflow.add_edge("extract_info", "reflect")
        
        # Conditional edge from reflect
        workflow.add_conditional_edges(
            "reflect",
            self.should_continue_research,
            {
                "continue": "generate_queries",
                "finish": "final_response"
            }
        )
        workflow.add_edge("final_response", END)
        
        # Compile with memory
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    def generate_queries_node(self, state: ResearchState) -> ResearchState:
        """Generate search queries based on current company info"""
        company_info = state["company_info"]
        user_notes = state.get("user_notes", "")
        queries_executed = state.get("queries_executed", 0)
        max_queries = state["max_queries"]
        
        # Determine what information is missing
        missing_fields = []
        if not company_info.founding_year:
            missing_fields.append("founding year")
        if not company_info.founder_names:
            missing_fields.append("founder names")
        if not company_info.product_description:
            missing_fields.append("product/service description")
        if not company_info.funding_summary:
            missing_fields.append("funding history")
        if not company_info.notable_customers:
            missing_fields.append("notable customers")

        remaining_queries = max_queries - queries_executed
        
        prompt = f"""Generate {min(remaining_queries, 3)} targeted search queries to find information about {company_info.company_name}.

Current information we have:
- Company name: {company_info.company_name}
- Founding year: {company_info.founding_year or "Unknown"}
- Founders: {", ".join(company_info.founder_names) or "Unknown"}
- Product description: {company_info.product_description or "Unknown"}
- Funding: {company_info.funding_summary or "Unknown"}
- Customers: {company_info.notable_customers or "Unknown"}

Missing information we need: {", ".join(missing_fields)}

User notes: {user_notes}

Generate specific search queries that will help fill in the missing information. Focus on the most important missing fields first.
Return only the search queries, one per line, without numbering or bullet points."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        return {
            **state,
            "search_queries": queries[:remaining_queries],
            "messages": [HumanMessage(content=f"Generated {len(queries)} search queries")]
        }

    def web_search_node(self, state: ResearchState) -> ResearchState:
        """Execute web searches in parallel using Tavily"""
        queries = state["search_queries"]
        max_results = state["max_results_per_query"]
        
        def search_query(query: str) -> List[Dict[str, Any]]:
            try:
                response = self.tavily_client.search(
                    query=query,
                    max_results=max_results,
                    search_depth="advanced"
                )
                return response.get("results", [])
            except Exception as e:
                print(f"Search error for query '{query}': {e}")
                return []
        
        # Execute searches in parallel
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            search_results = list(executor.map(search_query, queries))
        
        # Flatten results
        all_results = []
        for results in search_results:
            all_results.extend(results)
        
        return {
            **state,
            "search_results": all_results,
            "queries_executed": state.get("queries_executed", 0) + len(queries),
            "messages": state["messages"] + [HumanMessage(content=f"Executed {len(queries)} searches, found {len(all_results)} results")]
        }

    def extract_info_node(self, state: ResearchState) -> ResearchState:
        """Extract company information from search results"""
        search_results = state["search_results"]
        current_info = state["company_info"]
        
        # Combine search result content
        search_content = ""
        for result in search_results:
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            search_content += f"Title: {title}\nContent: {content}\nURL: {url}\n\n"
        
        if not search_content.strip():
            return {
                **state,
                "messages": state["messages"] + [HumanMessage(content="No search content to extract from")]
            }

        prompt = f"""Extract and update company information for {current_info.company_name} based on the following search results.

Current information:
{current_info.model_dump_json(indent=2)}

Search results:
{search_content[:8000]}  # Limit content length

Please extract any new or updated information and provide the complete company information in the following JSON format:
{{
  "company_name": "Official company name",
  "founding_year": year_as_integer_or_null,
  "founder_names": ["founder1", "founder2"],
  "product_description": "Brief description of main product/service",
  "funding_summary": "Summary of funding history including rounds and amounts",
  "notable_customers": "Known customers or client types"
}}

Keep existing information if no new information is found. Only update with factual information from the search results."""

        response = self.llm.invoke([HumanMessage(content=prompt)])
        
        try:
            # Extract JSON from response
            json_start = response.content.find('{')
            json_end = response.content.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response.content[json_start:json_end]
                extracted_data = json.loads(json_str)
                updated_info = CompanyInfo(**extracted_data)
            else:
                updated_info = current_info
        except Exception as e:
            print(f"Error parsing extracted info: {e}")
            updated_info = current_info

        return {
            **state,
            "company_info": updated_info,
            "messages": state["messages"] + [HumanMessage(content=f"Extracted information and updated company profile")]
        }

    def reflect_node(self, state: ResearchState) -> ResearchState:
        """Reflect on the completeness of information"""
        company_info = state["company_info"]
        queries_executed = state["queries_executed"]
        max_queries = state["max_queries"]
        reflection_count = state.get("reflection_count", 0)
        max_reflections = state["max_reflections"]
        
        # Check completeness
        completeness_score = 0
        total_fields = 6  # Total number of fields in CompanyInfo
        
        if company_info.company_name:
            completeness_score += 1
        if company_info.founding_year:
            completeness_score += 1
        if company_info.founder_names:
            completeness_score += 1
        if company_info.product_description:
            completeness_score += 1
        if company_info.funding_summary:
            completeness_score += 1
        if company_info.notable_customers:
            completeness_score += 1
        
        completeness_percentage = (completeness_score / total_fields) * 100
        
        # Decision logic
        is_complete = (
            completeness_percentage >= 70 or  # 70% or more fields filled
            queries_executed >= max_queries or  # Max queries reached
            reflection_count >= max_reflections  # Max reflections reached
        )
        
        reflection_message = f"Reflection {reflection_count + 1}: Completeness {completeness_percentage:.0f}% ({completeness_score}/{total_fields} fields), Queries used: {queries_executed}/{max_queries}"
        
        return {
            **state,
            "reflection_count": reflection_count + 1,
            "is_complete": is_complete,
            "messages": state["messages"] + [HumanMessage(content=reflection_message)]
        }

    def should_continue_research(self, state: ResearchState) -> str:
        """Determine if we should continue research or finish"""
        return "finish" if state["is_complete"] else "continue"

    def final_response_node(self, state: ResearchState) -> ResearchState:
        """Generate final response"""
        company_info = state["company_info"]
        
        return {
            **state,
            "messages": state["messages"] + [AIMessage(content=f"Research complete for {company_info.company_name}")]
        }

    def research_company(
        self, 
        company_name: str, 
        user_notes: str = "",
        thread_id: str = "default"
    ) -> CompanyInfo:
        """Main method to research a company"""
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            messages=[],
            company_info=CompanyInfo(company_name=company_name),
            search_queries=[],
            search_results=[],
            queries_executed=0,
            reflection_count=0,
            max_queries=self.max_queries,
            max_reflections=self.max_reflections,
            max_results_per_query=self.max_results_per_query,
            is_complete=False
        )
        
        config = {"configurable": {"thread_id": thread_id}}
        
        # Run the graph
        final_state = self.graph.invoke(initial_state, config)
        
        return final_state["company_info"]


def main():
    """Example usage"""
    # Get API keys from environment
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not anthropic_key or not tavily_key:
        print("Please set ANTHROPIC_API_KEY and TAVILY_API_KEY environment variables")
        return
    
    # Create researcher
    researcher = CompanyResearcher(
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        max_queries=6,
        max_reflections=3,
        max_results_per_query=5
    )
    
    # Research a company
    company_name = "OpenAI"
    user_notes = "Focus on recent developments and funding"
    
    print(f"Researching {company_name}...")
    result = researcher.research_company(company_name, user_notes)
    
    print("\nResearch Results:")
    print(json.dumps(result.model_dump(), indent=2))


if __name__ == "__main__":
    main()