#!/usr/bin/env python3
"""
Company Research Agent using LangGraph
A multi-node graph system for researching companies using the Tavily API
"""

import os
import asyncio
from typing import TypedDict, List, Optional, Dict, Any
from enum import Enum
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class CompanyInfo(BaseModel):
    """Basic information about a company"""
    company_name: str = Field(description="Official name of the company")
    founding_year: Optional[int] = Field(default=None, description="Year the company was founded")
    founder_names: List[str] = Field(default_factory=list, description="Names of the founding team members")
    product_description: Optional[str] = Field(default=None, description="Brief description of the company's main product or service")
    funding_summary: Optional[str] = Field(default=None, description="Summary of the company's funding history")
    notable_customers: Optional[str] = Field(default=None, description="Known customers that use company's product/service")

class ResearchState(TypedDict):
    """State for the research process"""
    company_name: str
    user_notes: Optional[str]
    generated_queries: List[str]
    search_results: List[Dict[str, Any]]
    extracted_info: CompanyInfo
    reflection_count: int
    max_reflections: int
    max_queries: int
    max_results_per_query: int
    messages: List[Dict[str, str]]
    is_complete: bool

@dataclass
class Config:
    """Configuration for the company researcher"""
    max_queries: int = 5
    max_results_per_query: int = 3
    max_reflections: int = 3
    openai_api_key: str = ""
    tavily_api_key: str = ""

class CompanyResearcher:
    """Main company research agent using LangGraph"""
    
    def __init__(self, config: Config):
        self.config = config
        self.llm = ChatOpenAI(
            api_key=config.openai_api_key,
            model="gpt-4",
            temperature=0.1
        )
        self.tavily_client = TavilyClient(api_key=config.tavily_api_key)
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self._generate_queries_node)
        workflow.add_node("search_web", self._search_web_node)
        workflow.add_node("extract_info", self._extract_info_node)
        workflow.add_node("reflect", self._reflect_node)
        
        # Add edges
        workflow.add_edge("generate_queries", "search_web")
        workflow.add_edge("search_web", "extract_info")
        workflow.add_edge("extract_info", "reflect")
        
        # Conditional edge from reflect
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue_research,
            {
                "continue": "generate_queries",
                "end": END
            }
        )
        
        # Set entry point
        workflow.set_entry_point("generate_queries")
        
        return workflow.compile()
    
    def _generate_queries_node(self, state: ResearchState) -> Dict[str, Any]:
        """Generate search queries for the company research"""
        company_name = state["company_name"]
        user_notes = state.get("user_notes", "")
        current_info = state.get("extracted_info")
        
        # Determine what information we still need
        missing_fields = []
        if current_info:
            if not current_info.founding_year:
                missing_fields.append("founding year")
            if not current_info.founder_names:
                missing_fields.append("founders")
            if not current_info.product_description:
                missing_fields.append("products/services")
            if not current_info.funding_summary:
                missing_fields.append("funding history")
            if not current_info.notable_customers:
                missing_fields.append("notable customers")
        else:
            missing_fields = ["founding year", "founders", "products/services", "funding history", "notable customers"]
        
        system_prompt = f"""You are a research query generator. Generate specific, targeted search queries to find information about a company.

Company: {company_name}
User Notes: {user_notes}
Missing Information: {', '.join(missing_fields) if missing_fields else 'General company information'}

Generate {state['max_queries']} specific search queries that will help find the missing information about this company.
Focus on factual, verifiable information. Make queries specific and varied to get comprehensive results.

Return only the queries as a JSON list of strings."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate search queries for researching {company_name}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            # Extract JSON from response
            content = response.content
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            queries = json.loads(json_str)
            
            # Limit to max_queries
            queries = queries[:state['max_queries']]
            
        except Exception as e:
            # Fallback queries
            queries = [
                f"{company_name} company founding history founders",
                f"{company_name} products services description",
                f"{company_name} funding investment history",
                f"{company_name} notable customers clients",
                f"{company_name} company information overview"
            ][:state['max_queries']]
        
        # Log the queries
        new_message = {"role": "system", "content": f"Generated queries: {queries}"}
        updated_messages = state["messages"] + [new_message]
        
        return {
            "generated_queries": queries,
            "messages": updated_messages
        }
    
    def _search_web_node(self, state: ResearchState) -> Dict[str, Any]:
        """Search the web using Tavily API in parallel"""
        queries = state["generated_queries"]
        max_results = state["max_results_per_query"]
        
        all_results = []
        
        def search_single_query(query: str) -> List[Dict[str, Any]]:
            """Search a single query"""
            try:
                response = self.tavily_client.search(
                    query=query,
                    max_results=max_results,
                    include_answer=True,
                    include_domains=[]
                )
                return response.get("results", [])
            except Exception as e:
                print(f"Error searching query '{query}': {e}")
                return []
        
        # Parallel search execution
        with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as executor:
            future_to_query = {executor.submit(search_single_query, query): query for query in queries}
            
            for future in as_completed(future_to_query):
                query = future_to_query[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                except Exception as e:
                    print(f"Error processing results for query '{query}': {e}")
        
        # Log search results
        new_message = {"role": "system", "content": f"Found {len(all_results)} search results"}
        updated_messages = state["messages"] + [new_message]
        
        return {
            "search_results": all_results,
            "messages": updated_messages
        }
    
    def _extract_info_node(self, state: ResearchState) -> Dict[str, Any]:
        """Extract structured information from search results"""
        company_name = state["company_name"]
        search_results = state["search_results"]
        current_info = state.get("extracted_info")
        
        # Prepare search results text
        results_text = ""
        for i, result in enumerate(search_results[:15]):  # Limit to avoid token limits
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            results_text += f"\n--- Result {i+1} ---\nTitle: {title}\nURL: {url}\nContent: {content}\n"
        
        system_prompt = f"""You are an information extraction expert. Extract structured information about a company from search results.

Company: {company_name}

Current Information: {current_info.model_dump() if current_info else 'None'}

Extract and update the following information based on the search results:
- company_name: Official name of the company
- founding_year: Year the company was founded (integer)
- founder_names: Names of the founding team members (list of strings)
- product_description: Brief description of the company's main product or service
- funding_summary: Summary of the company's funding history
- notable_customers: Known customers that use company's product/service

Important guidelines:
1. Only include information that is explicitly mentioned in the search results
2. If current information exists, update/enhance it with new findings
3. If information is not found, leave the field as null/empty
4. Be factual and avoid speculation
5. For founding_year, only provide a 4-digit integer year
6. For founder_names, provide a list even if there's only one founder

Return the information as a JSON object matching the CompanyInfo schema."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract information about {company_name} from these search results:\n{results_text}")
        ]
        
        try:
            response = self.llm.invoke(messages)
            content = response.content
            
            # Extract JSON from response
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            else:
                json_str = content.strip()
            
            extracted_data = json.loads(json_str)
            extracted_info = CompanyInfo(**extracted_data)
            
        except Exception as e:
            print(f"Error extracting information: {e}")
            # Fallback to current info or empty
            extracted_info = current_info or CompanyInfo(company_name=company_name)
        
        # Log extraction
        new_message = {"role": "system", "content": f"Extracted information for {company_name}"}
        updated_messages = state["messages"] + [new_message]
        
        return {
            "extracted_info": extracted_info,
            "messages": updated_messages
        }
    
    def _reflect_node(self, state: ResearchState) -> Dict[str, Any]:
        """Reflect on the gathered information and decide if more research is needed"""
        extracted_info = state["extracted_info"]
        reflection_count = state["reflection_count"]
        max_reflections = state["max_reflections"]
        
        # Check completeness
        completeness_score = 0
        total_fields = 5  # excluding company_name which is required
        
        if extracted_info.founding_year:
            completeness_score += 1
        if extracted_info.founder_names:
            completeness_score += 1
        if extracted_info.product_description:
            completeness_score += 1
        if extracted_info.funding_summary:
            completeness_score += 1
        if extracted_info.notable_customers:
            completeness_score += 1
        
        completeness_percentage = (completeness_score / total_fields) * 100
        
        # Check if we should continue
        should_continue = (
            completeness_percentage < 80 and  # Less than 80% complete
            reflection_count < max_reflections  # Haven't exceeded max reflections
        )
        
        # Log reflection
        new_message = {
            "role": "system", 
            "content": f"Reflection {reflection_count + 1}: Completeness {completeness_percentage:.1f}%. {'Continuing research' if should_continue else 'Research complete'}"
        }
        updated_messages = state["messages"] + [new_message]
        
        return {
            "reflection_count": reflection_count + 1,
            "is_complete": not should_continue,
            "messages": updated_messages
        }
    
    def _should_continue_research(self, state: ResearchState) -> str:
        """Determine if research should continue"""
        return "end" if state["is_complete"] else "continue"
    
    def research_company(self, company_name: str, user_notes: Optional[str] = None) -> CompanyInfo:
        """Main method to research a company"""
        initial_state = ResearchState(
            company_name=company_name,
            user_notes=user_notes,
            generated_queries=[],
            search_results=[],
            extracted_info=CompanyInfo(company_name=company_name),
            reflection_count=0,
            max_reflections=self.config.max_reflections,
            max_queries=self.config.max_queries,
            max_results_per_query=self.config.max_results_per_query,
            messages=[{"role": "user", "content": f"Research company: {company_name}"}],
            is_complete=False
        )
        
        # Run the graph
        final_state = self.graph.invoke(initial_state)
        
        # Print conversation log
        print("\n=== Research Process Log ===")
        for msg in final_state["messages"]:
            role = msg["role"].upper()
            content = msg["content"]
            print(f"[{role}] {content}")
        
        return final_state["extracted_info"]

def main():
    """CLI interface for the company researcher"""
    print("🔍 Company Research Agent")
    print("=" * 50)
    
    # Check for required environment variables
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set it in a .env file or export it:")
        print("export OPENAI_API_KEY='your-api-key'")
        return
    
    if not tavily_key:
        print("❌ Error: TAVILY_API_KEY environment variable not set")
        print("Please set it in a .env file or export it:")
        print("export TAVILY_API_KEY='your-api-key'")
        return
    
    # Get user input
    company_name = input("\nEnter company name to research: ").strip()
    if not company_name:
        print("❌ Company name is required")
        return
    
    user_notes = input("Enter optional notes (press Enter to skip): ").strip()
    user_notes = user_notes if user_notes else None
    
    # Configuration
    config = Config(
        max_queries=5,
        max_results_per_query=3,
        max_reflections=3,
        openai_api_key=openai_key,
        tavily_api_key=tavily_key
    )
    
    # Initialize and run research
    researcher = CompanyResearcher(config)
    
    print(f"\n🚀 Starting research for: {company_name}")
    print("This may take a few moments...")
    
    try:
        result = researcher.research_company(company_name, user_notes)
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 RESEARCH RESULTS")
        print("=" * 50)
        
        print(f"🏢 Company Name: {result.company_name}")
        print(f"📅 Founded: {result.founding_year or 'Not found'}")
        print(f"👥 Founders: {', '.join(result.founder_names) if result.founder_names else 'Not found'}")
        print(f"🎯 Product/Service: {result.product_description or 'Not found'}")
        print(f"💰 Funding: {result.funding_summary or 'Not found'}")
        print(f"🤝 Notable Customers: {result.notable_customers or 'Not found'}")
        
        # Save results to JSON file
        filename = f"{company_name.replace(' ', '_').lower()}_research.json"
        with open(filename, 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ An error occurred during research: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()