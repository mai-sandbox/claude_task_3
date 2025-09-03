#!/usr/bin/env python3
import json
import asyncio
from typing import Dict, List, Optional, TypedDict, Annotated
from pydantic import BaseModel, Field
from tavily import TavilyClient
import os
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser


class CompanyInfo(BaseModel):
    """Schema for company information"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(default=None, description="Year the company was founded")
    founder_names: Optional[List[str]] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(default=None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(default=None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(default=None, description="Known customers that use company's product/service")


class SearchQuery(BaseModel):
    """Schema for search queries"""
    query: str = Field(description="Search query to execute")
    purpose: str = Field(description="What information this query aims to find")


class ResearchState(TypedDict):
    """State for the research workflow"""
    messages: Annotated[List[BaseMessage], add_messages]
    company_name: str
    user_notes: Optional[str]
    company_info: CompanyInfo
    search_queries: List[SearchQuery]
    search_results: List[Dict]
    query_count: int
    max_queries: int
    max_results_per_query: int
    reflection_count: int
    max_reflections: int
    completed: bool
    error: Optional[str]


class CompanyResearcher:
    """Multi-node LangGraph company researcher"""
    
    def __init__(
        self,
        tavily_api_key: str,
        openai_api_key: str,
        max_queries: int = 5,
        max_results_per_query: int = 3,
        max_reflections: int = 2
    ):
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4", temperature=0)
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query
        self.max_reflections = max_reflections
        
        # Build the graph
        self.graph = self._build_graph()
    
    def _build_graph(self):
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("initialize", self.initialize_research)
        workflow.add_node("generate_queries", self.generate_search_queries)
        workflow.add_node("execute_searches", self.execute_searches)
        workflow.add_node("extract_information", self.extract_information)
        workflow.add_node("reflect_and_evaluate", self.reflect_and_evaluate)
        workflow.add_node("finalize", self.finalize_research)
        
        # Add edges
        workflow.add_edge(START, "initialize")
        workflow.add_edge("initialize", "generate_queries")
        workflow.add_edge("generate_queries", "execute_searches")
        workflow.add_edge("execute_searches", "extract_information")
        workflow.add_edge("extract_information", "reflect_and_evaluate")
        
        # Conditional edge for reflection
        workflow.add_conditional_edges(
            "reflect_and_evaluate",
            self.should_continue_research,
            {
                "continue": "generate_queries",
                "finish": "finalize"
            }
        )
        
        workflow.add_edge("finalize", END)
        
        return workflow.compile()
    
    async def initialize_research(self, state: ResearchState) -> Dict:
        """Initialize the research state"""
        return {
            "messages": [SystemMessage(content=f"Starting research for company: {state['company_name']}")],
            "company_info": CompanyInfo(company_name=state["company_name"]),
            "search_queries": [],
            "search_results": [],
            "query_count": 0,
            "max_queries": self.max_queries,
            "max_results_per_query": self.max_results_per_query,
            "reflection_count": 0,
            "max_reflections": self.max_reflections,
            "completed": False,
            "error": None
        }
    
    async def generate_search_queries(self, state: ResearchState) -> Dict:
        """Generate search queries based on current company info gaps"""
        current_info = state["company_info"]
        user_notes = state.get("user_notes", "")
        
        # Create prompt to generate queries
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a research assistant. Generate search queries to find missing information about a company.
            
            Current information about {company_name}:
            - Founding year: {founding_year}
            - Founders: {founders}
            - Product description: {product_description}
            - Funding summary: {funding_summary}
            - Notable customers: {notable_customers}
            
            User notes: {user_notes}
            
            Generate {remaining_queries} specific search queries to fill in missing information.
            Focus on the most important gaps first.
            
            Return as JSON array of objects with 'query' and 'purpose' fields."""),
            ("user", "Generate search queries for missing company information.")
        ])
        
        remaining_queries = min(
            self.max_queries - state["query_count"],
            3  # Generate max 3 queries per iteration
        )
        
        if remaining_queries <= 0:
            return {"search_queries": []}
        
        response = await self.llm.ainvoke(
            prompt.format_messages(
                company_name=current_info.company_name,
                founding_year=current_info.founding_year or "Unknown",
                founders=", ".join(current_info.founder_names) if current_info.founder_names else "Unknown",
                product_description=current_info.product_description or "Unknown",
                funding_summary=current_info.funding_summary or "Unknown",
                notable_customers=current_info.notable_customers or "Unknown",
                user_notes=user_notes,
                remaining_queries=remaining_queries
            )
        )
        
        try:
            # Parse JSON response
            queries_data = json.loads(response.content)
            queries = [SearchQuery(**q) for q in queries_data]
            
            return {
                "search_queries": queries,
                "messages": [AIMessage(content=f"Generated {len(queries)} search queries")]
            }
        except Exception as e:
            return {
                "search_queries": [],
                "messages": [AIMessage(content=f"Error generating queries: {str(e)}")]
            }
    
    async def execute_searches(self, state: ResearchState) -> Dict:
        """Execute search queries in parallel using Tavily API"""
        queries = state["search_queries"]
        if not queries:
            return {"search_results": []}
        
        search_tasks = []
        for query in queries:
            search_tasks.append(self._execute_single_search(query.query))
        
        # Execute searches in parallel
        try:
            results = await asyncio.gather(*search_tasks, return_exceptions=True)
            
            search_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    print(f"Search failed for query '{queries[i].query}': {result}")
                    continue
                
                search_results.extend(result)
            
            return {
                "search_results": search_results,
                "query_count": state["query_count"] + len(queries),
                "messages": [AIMessage(content=f"Executed {len(queries)} searches, found {len(search_results)} results")]
            }
        except Exception as e:
            return {
                "search_results": [],
                "messages": [AIMessage(content=f"Error executing searches: {str(e)}")]
            }
    
    async def _execute_single_search(self, query: str) -> List[Dict]:
        """Execute a single search query"""
        try:
            response = self.tavily_client.search(
                query=query,
                max_results=self.max_results_per_query,
                search_depth="advanced"
            )
            return response.get("results", [])
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []
    
    async def extract_information(self, state: ResearchState) -> Dict:
        """Extract relevant information from search results"""
        search_results = state["search_results"]
        current_info = state["company_info"]
        
        if not search_results:
            return {"company_info": current_info}
        
        # Combine all search result content
        content = "\n\n".join([
            f"Title: {result.get('title', '')}\nContent: {result.get('content', '')}"
            for result in search_results
        ])
        
        # Create prompt to extract information
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert information extractor. Given search results about a company, extract and update the company information.
            
            Current company information:
            {current_info}
            
            Search results:
            {content}
            
            Extract relevant information and return updated company info as JSON matching this schema:
            {{
                "company_name": "string",
                "founding_year": integer or null,
                "founder_names": ["string"] or [],
                "product_description": "string" or null,
                "funding_summary": "string" or null,
                "notable_customers": "string" or null
            }}
            
            Only include information you are confident about from the search results.
            Keep existing information if no better information is found."""),
            ("user", "Extract and update company information from the search results.")
        ])
        
        try:
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    current_info=current_info.model_dump_json(indent=2),
                    content=content[:8000]  # Limit content length
                )
            )
            
            # Parse JSON response
            updated_info_data = json.loads(response.content)
            updated_info = CompanyInfo(**updated_info_data)
            
            return {
                "company_info": updated_info,
                "messages": [AIMessage(content="Extracted information from search results")]
            }
        except Exception as e:
            return {
                "company_info": current_info,
                "messages": [AIMessage(content=f"Error extracting information: {str(e)}")]
            }
    
    async def reflect_and_evaluate(self, state: ResearchState) -> Dict:
        """Evaluate if we have sufficient information about the company"""
        current_info = state["company_info"]
        
        # Create prompt to evaluate completeness
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an evaluation expert. Assess if the company information is sufficient and complete.
            
            Company information:
            {company_info}
            
            Evaluate the completeness and quality of information. Consider:
            1. Are the core fields filled with meaningful information?
            2. Is the information detailed enough to understand the company?
            3. Are there critical gaps that need more research?
            
            Return JSON with:
            {{
                "is_sufficient": boolean,
                "completeness_score": float (0-1),
                "missing_areas": ["string"],
                "reasoning": "string"
            }}"""),
            ("user", "Evaluate if the company information is sufficient.")
        ])
        
        try:
            response = await self.llm.ainvoke(
                prompt.format_messages(
                    company_info=current_info.model_dump_json(indent=2)
                )
            )
            
            evaluation = json.loads(response.content)
            
            return {
                "reflection_count": state["reflection_count"] + 1,
                "messages": [
                    AIMessage(content=f"Reflection {state['reflection_count'] + 1}: {evaluation['reasoning']}")
                ]
            }
        except Exception as e:
            return {
                "reflection_count": state["reflection_count"] + 1,
                "messages": [AIMessage(content=f"Error in reflection: {str(e)}")]
            }
    
    def should_continue_research(self, state: ResearchState) -> str:
        """Determine if research should continue or finish"""
        # Stop if max queries reached
        if state["query_count"] >= self.max_queries:
            return "finish"
        
        # Stop if max reflections reached
        if state["reflection_count"] >= self.max_reflections:
            return "finish"
        
        # Get the last evaluation from messages
        try:
            last_message = state["messages"][-1]
            if "Reflection" in last_message.content:
                # Simple heuristic: continue if reflection mentions "missing" or "gaps"
                if any(word in last_message.content.lower() for word in ["missing", "gaps", "insufficient", "unknown"]):
                    return "continue"
        except:
            pass
        
        return "finish"
    
    async def finalize_research(self, state: ResearchState) -> Dict:
        """Finalize the research and prepare final output"""
        return {
            "completed": True,
            "messages": [AIMessage(content="Research completed successfully")]
        }
    
    async def research_company(self, company_name: str, user_notes: Optional[str] = None) -> CompanyInfo:
        """Main entry point to research a company"""
        initial_state = {
            "messages": [],
            "company_name": company_name,
            "user_notes": user_notes,
            "company_info": CompanyInfo(company_name=company_name),
            "search_queries": [],
            "search_results": [],
            "query_count": 0,
            "max_queries": self.max_queries,
            "max_results_per_query": self.max_results_per_query,
            "reflection_count": 0,
            "max_reflections": self.max_reflections,
            "completed": False,
            "error": None
        }
        
        try:
            final_state = await self.graph.ainvoke(initial_state)
            return final_state["company_info"]
        except Exception as e:
            print(f"Research error: {e}")
            return CompanyInfo(company_name=company_name)


async def main():
    """Example usage"""
    # Initialize the researcher
    researcher = CompanyResearcher(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_queries=5,
        max_results_per_query=3,
        max_reflections=2
    )
    
    # Research a company
    company_info = await researcher.research_company(
        company_name="Anthropic",
        user_notes="Focus on AI safety and Claude chatbot"
    )
    
    print("Research Results:")
    print(company_info.model_dump_json(indent=2))


if __name__ == "__main__":
    asyncio.run(main())