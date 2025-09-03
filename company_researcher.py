"""
Company Researcher using LangGraph - Multi-node graph for comprehensive company research
"""
import asyncio
import json
import os
from typing import List, Optional, Dict, Any, Annotated
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import operator

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from tavily import TavilyClient

load_dotenv()

@dataclass
class Config:
    """Configuration settings for the company researcher"""
    max_search_queries: int = 5
    max_search_results_per_query: int = 3
    max_reflection_steps: int = 2
    model_name: str = "gpt-4o-mini"

class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(default=None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(default=None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(default=None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(default=None, description="Known customers that use company's product/service")

class ResearchState(BaseModel):
    """State for the research workflow"""
    messages: Annotated[List, add_messages] = Field(default_factory=list)
    company_name: str = ""
    user_notes: str = ""
    company_info: CompanyInfo = Field(default_factory=CompanyInfo)
    search_queries: List[str] = Field(default_factory=list)
    search_results: List[Dict[str, Any]] = Field(default_factory=list)
    queries_executed: int = 0
    reflection_steps: int = 0
    needs_more_research: bool = True
    config: Config = Field(default_factory=Config)

class CompanyResearcher:
    """Main company researcher class using LangGraph"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.llm = ChatOpenAI(model=self.config.model_name, temperature=0.1)
        self.tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the multi-node research graph"""
        workflow = StateGraph(ResearchState)
        
        workflow.add_node("generate_queries", self._generate_queries_node)
        workflow.add_node("parallel_search", self._parallel_search_node)
        workflow.add_node("extract_info", self._extract_info_node)
        workflow.add_node("reflect", self._reflect_node)
        
        workflow.add_edge(START, "generate_queries")
        workflow.add_edge("generate_queries", "parallel_search")
        workflow.add_edge("parallel_search", "extract_info")
        workflow.add_edge("extract_info", "reflect")
        
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue_research,
            {
                "continue": "generate_queries",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _generate_queries_node(self, state: ResearchState) -> Dict[str, Any]:
        """Generate search queries based on missing information"""
        current_info = state.company_info.model_dump()
        missing_fields = [k for k, v in current_info.items() 
                         if k != "company_name" and (v is None or (isinstance(v, list) and len(v) == 0))]
        
        system_message = SystemMessage(content=f"""
        You are a research query generator. Generate specific search queries to find missing information about {state.company_name}.
        
        Missing information fields: {missing_fields}
        User notes: {state.user_notes}
        Queries already executed: {state.queries_executed}
        Max queries allowed: {state.config.max_search_queries}
        
        Generate {min(3, state.config.max_search_queries - state.queries_executed)} specific, targeted search queries.
        Each query should focus on finding specific missing information.
        
        Return only a JSON list of query strings, no other text.
        """)
        
        human_message = HumanMessage(content=f"Generate search queries for {state.company_name}")
        
        response = self.llm.invoke([system_message, human_message])
        
        try:
            queries = json.loads(response.content.strip())
            if not isinstance(queries, list):
                queries = [str(queries)]
        except:
            queries = [f"{state.company_name} company information"]
        
        queries = queries[:state.config.max_search_queries - state.queries_executed]
        
        return {
            "search_queries": queries,
            "messages": [AIMessage(content=f"Generated {len(queries)} search queries: {queries}")]
        }
    
    def _parallel_search_node(self, state: ResearchState) -> Dict[str, Any]:
        """Execute searches in parallel using Tavily API"""
        def search_single_query(query: str) -> List[Dict[str, Any]]:
            try:
                response = self.tavily_client.search(
                    query=query,
                    max_results=state.config.max_search_results_per_query,
                    search_depth="advanced"
                )
                return response.get("results", [])
            except Exception as e:
                print(f"Search error for query '{query}': {e}")
                return []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            search_results = list(executor.map(search_single_query, state.search_queries))
        
        flattened_results = [result for sublist in search_results for result in sublist]
        
        return {
            "search_results": flattened_results,
            "queries_executed": state.queries_executed + len(state.search_queries),
            "messages": [AIMessage(content=f"Executed {len(state.search_queries)} searches, found {len(flattened_results)} results")]
        }
    
    def _extract_info_node(self, state: ResearchState) -> Dict[str, Any]:
        """Extract and aggregate information from search results"""
        if not state.search_results:
            return {"messages": [AIMessage(content="No search results to process")]}
        
        search_content = "\n\n".join([
            f"Title: {result.get('title', 'N/A')}\nContent: {result.get('content', 'N/A')}\nURL: {result.get('url', 'N/A')}"
            for result in state.search_results[:15]  # Limit to prevent token overflow
        ])
        
        current_info_json = state.company_info.model_dump_json(indent=2)
        
        system_message = SystemMessage(content=f"""
        You are an expert information extractor. Extract and update company information from search results.
        
        Current company information:
        {current_info_json}
        
        Instructions:
        1. Extract information to fill missing fields
        2. Do not overwrite existing accurate information unless you find more accurate/complete data
        3. For lists (like founder_names), combine existing with new unique entries
        4. Be conservative - only include information you're confident about
        5. Return the complete CompanyInfo JSON object with all fields
        """)
        
        human_message = HumanMessage(content=f"Search results:\n{search_content}")
        
        response = self.llm.invoke([system_message, human_message])
        
        try:
            updated_info_dict = json.loads(response.content.strip())
            updated_info = CompanyInfo.model_validate(updated_info_dict)
        except Exception as e:
            print(f"Error parsing extracted info: {e}")
            updated_info = state.company_info
        
        return {
            "company_info": updated_info,
            "messages": [AIMessage(content=f"Extracted and updated company information")]
        }
    
    def _reflect_node(self, state: ResearchState) -> Dict[str, Any]:
        """Reflect on completeness of information and decide next steps"""
        current_info = state.company_info.model_dump()
        missing_fields = [k for k, v in current_info.items() 
                         if k != "company_name" and (v is None or (isinstance(v, list) and len(v) == 0))]
        
        completeness_score = (6 - len(missing_fields)) / 6  # 6 total fields including company_name
        
        system_message = SystemMessage(content=f"""
        You are a research quality assessor. Evaluate the completeness and quality of company research.
        
        Company: {state.company_name}
        Current information completeness: {completeness_score:.1%}
        Missing fields: {missing_fields}
        Queries executed: {state.queries_executed}/{state.config.max_search_queries}
        Reflection steps: {state.reflection_steps}/{state.config.max_reflection_steps}
        
        Assess if the research is sufficient or needs more work.
        Consider:
        1. Information completeness (aim for at least 70%)
        2. Quality and accuracy of existing information
        3. Remaining query budget
        4. Whether additional searches would likely yield better results
        
        Return JSON: {{"needs_more_research": true/false, "reasoning": "explanation"}}
        """)
        
        current_info_display = json.dumps(current_info, indent=2)
        human_message = HumanMessage(content=f"Current information:\n{current_info_display}")
        
        response = self.llm.invoke([system_message, human_message])
        
        try:
            assessment = json.loads(response.content.strip())
            needs_more = assessment.get("needs_more_research", False)
            reasoning = assessment.get("reasoning", "Assessment completed")
        except:
            needs_more = completeness_score < 0.7 and state.queries_executed < state.config.max_search_queries
            reasoning = "Default assessment based on completeness score"
        
        return {
            "needs_more_research": needs_more and state.reflection_steps < state.config.max_reflection_steps,
            "reflection_steps": state.reflection_steps + 1,
            "messages": [AIMessage(content=f"Reflection: {reasoning}. Completeness: {completeness_score:.1%}")]
        }
    
    def _should_continue_research(self, state: ResearchState) -> str:
        """Determine if research should continue"""
        if (state.needs_more_research and 
            state.queries_executed < state.config.max_search_queries and 
            state.reflection_steps <= state.config.max_reflection_steps):
            return "continue"
        return "end"
    
    def research_company(self, company_name: str, user_notes: str = "") -> CompanyInfo:
        """Main method to research a company"""
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            company_info=CompanyInfo(company_name=company_name),
            config=self.config
        )
        
        final_state = self.graph.invoke(initial_state)
        return final_state["company_info"]

def main():
    """Demo the company researcher"""
    researcher = CompanyResearcher()
    
    company = input("Enter company name to research: ").strip()
    notes = input("Enter any additional notes (optional): ").strip()
    
    if not company:
        print("Please provide a company name.")
        return
    
    print(f"\nResearching {company}...")
    print("-" * 50)
    
    try:
        result = researcher.research_company(company, notes)
        print("\n🔍 Research Complete!")
        print("=" * 50)
        print(json.dumps(result.model_dump(), indent=2))
    except Exception as e:
        print(f"Error during research: {e}")

if __name__ == "__main__":
    main()