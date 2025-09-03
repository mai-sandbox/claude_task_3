from typing import TypedDict, List, Dict, Any, Optional, Annotated
from dataclasses import dataclass
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
import asyncio
import json
import os
from datetime import datetime

@dataclass
class CompanyInfo:
    company_name: str
    founding_year: Optional[int] = None
    founder_names: List[str] = None
    product_description: Optional[str] = None
    funding_summary: Optional[str] = None
    notable_customers: Optional[str] = None
    
    def __post_init__(self):
        if self.founder_names is None:
            self.founder_names = []

class ResearchState(TypedDict):
    company_name: str
    notes: Optional[str]
    messages: List[BaseMessage]
    company_info: Dict[str, Any]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    queries_used: int
    results_processed: int
    reflection_count: int
    final_result: Optional[Dict[str, Any]]

class CompanyResearcher:
    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: str,
        max_search_queries: int = 5,
        max_search_results: int = 3,
        max_reflections: int = 2
    ):
        self.llm = ChatOpenAI(api_key=openai_api_key, model="gpt-4o-mini")
        self.search_tool = TavilySearch(
            api_key=tavily_api_key,
            max_results=max_search_results
        )
        self.max_search_queries = max_search_queries
        self.max_search_results = max_search_results
        self.max_reflections = max_reflections
        
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(ResearchState)
        
        # Add nodes
        workflow.add_node("generate_queries", self._generate_queries)
        workflow.add_node("web_search", self._web_search)
        workflow.add_node("extract_data", self._extract_data)
        workflow.add_node("reflect", self._reflect)
        
        # Add edges
        workflow.set_entry_point("generate_queries")
        workflow.add_edge("generate_queries", "web_search")
        workflow.add_edge("web_search", "extract_data")
        workflow.add_edge("extract_data", "reflect")
        
        # Conditional edge from reflect
        workflow.add_conditional_edges(
            "reflect",
            self._should_continue,
            {
                "continue": "generate_queries",
                "end": END
            }
        )
        
        return workflow.compile()

    async def _generate_queries(self, state: ResearchState) -> Dict[str, Any]:
        company_name = state["company_name"]
        notes = state.get("notes", "")
        current_info = state.get("company_info", {})
        queries_used = state.get("queries_used", 0)
        
        # Determine what information we still need
        needed_info = []
        if not current_info.get("founding_year"):
            needed_info.append("founding year")
        if not current_info.get("founder_names"):
            needed_info.append("founders")
        if not current_info.get("product_description"):
            needed_info.append("products/services")
        if not current_info.get("funding_summary"):
            needed_info.append("funding history")
        if not current_info.get("notable_customers"):
            needed_info.append("customers")
        
        remaining_queries = self.max_search_queries - queries_used
        
        prompt = f"""
        Generate {min(remaining_queries, 3)} specific search queries to find information about {company_name}.
        
        Company: {company_name}
        Additional notes: {notes if notes else "None"}
        
        We still need information about: {', '.join(needed_info) if needed_info else 'general verification'}
        
        Generate queries that are:
        1. Specific and targeted
        2. Likely to return official company information
        3. Focused on the missing information we need
        
        Return only the queries, one per line, without numbering or bullets.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        # Limit to remaining query budget
        queries = queries[:remaining_queries]
        
        return {
            "search_queries": queries,
            "messages": state["messages"] + [HumanMessage(content=prompt), response]
        }

    async def _web_search(self, state: ResearchState) -> Dict[str, Any]:
        queries = state["search_queries"]
        existing_results = state.get("search_results", [])
        
        # Execute searches in parallel
        search_tasks = []
        for query in queries:
            search_tasks.append(self._execute_search(query))
        
        new_results = await asyncio.gather(*search_tasks)
        
        # Flatten and combine results
        all_results = existing_results + [result for batch in new_results for result in batch]
        
        return {
            "search_results": all_results,
            "queries_used": state.get("queries_used", 0) + len(queries),
            "results_processed": state.get("results_processed", 0) + len([result for batch in new_results for result in batch])
        }

    async def _execute_search(self, query: str) -> List[Dict[str, Any]]:
        try:
            results = await self.search_tool.ainvoke({"query": query})
            return results if isinstance(results, list) else [results]
        except Exception as e:
            print(f"Search failed for query '{query}': {e}")
            return []

    async def _extract_data(self, state: ResearchState) -> Dict[str, Any]:
        company_name = state["company_name"]
        search_results = state["search_results"]
        current_info = state.get("company_info", {})
        
        # Prepare search results summary for the LLM
        results_text = "\n\n".join([
            f"Source: {result.get('url', 'Unknown')}\nTitle: {result.get('title', 'Unknown')}\nContent: {result.get('content', '')[:500]}..."
            for result in search_results[-10:]  # Use last 10 results to avoid token limits
        ])
        
        prompt = f"""
        Extract and update company information from the search results below.

        Company: {company_name}
        Current information: {json.dumps(current_info, indent=2)}

        Search Results:
        {results_text}

        Extract and return ONLY a JSON object with this exact structure:
        {{
            "company_name": "{company_name}",
            "founding_year": <year as integer or null>,
            "founder_names": [<list of founder names as strings>],
            "product_description": "<brief description or null>",
            "funding_summary": "<funding summary or null>",
            "notable_customers": "<notable customers or null>"
        }}

        Rules:
        1. Only include information you can verify from the search results
        2. For founding_year, use integer year only (e.g., 2015, not "2015")
        3. For founder_names, use a list of strings even if there's only one founder
        4. Keep descriptions concise (1-2 sentences)
        5. Use null for missing information
        6. Return valid JSON only, no other text
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        
        try:
            extracted_info = json.loads(response.content.strip())
            
            # Merge with existing info, prioritizing new non-null values
            updated_info = current_info.copy()
            for key, value in extracted_info.items():
                if value is not None:
                    if key == "founder_names" and isinstance(value, list):
                        # Merge founder lists, removing duplicates
                        existing_founders = updated_info.get("founder_names", [])
                        updated_info["founder_names"] = list(set(existing_founders + value))
                    else:
                        updated_info[key] = value
            
            return {
                "company_info": updated_info,
                "messages": state["messages"] + [HumanMessage(content=prompt), response]
            }
        except json.JSONDecodeError as e:
            print(f"Failed to parse LLM response as JSON: {e}")
            return {"company_info": current_info}

    async def _reflect(self, state: ResearchState) -> Dict[str, Any]:
        company_info = state["company_info"]
        reflection_count = state.get("reflection_count", 0)
        
        # Check completeness
        completeness_score = self._calculate_completeness(company_info)
        
        prompt = f"""
        Evaluate the completeness and quality of the company information gathered:

        Company Information:
        {json.dumps(company_info, indent=2)}

        Completeness Score: {completeness_score}/5

        Assess:
        1. Is the information sufficient for a comprehensive company overview?
        2. Are there any critical gaps that need more research?
        3. Is the information accurate and consistent?

        Respond with either:
        - "SUFFICIENT" if the information is complete enough
        - "NEEDS_MORE" if critical information is missing and we should search more

        Include a brief explanation of your decision.
        """
        
        response = await self.llm.ainvoke([HumanMessage(content=prompt)])
        decision = "SUFFICIENT" if "SUFFICIENT" in response.content else "NEEDS_MORE"
        
        return {
            "reflection_count": reflection_count + 1,
            "final_result": company_info if decision == "SUFFICIENT" else None,
            "messages": state["messages"] + [HumanMessage(content=prompt), response]
        }

    def _should_continue(self, state: ResearchState) -> str:
        # Check if we should continue or end
        reflection_count = state.get("reflection_count", 0)
        queries_used = state.get("queries_used", 0)
        final_result = state.get("final_result")
        
        # End conditions
        if final_result is not None:  # LLM determined info is sufficient
            return "end"
        if reflection_count >= self.max_reflections:  # Max reflections reached
            return "end"
        if queries_used >= self.max_search_queries:  # Max queries used
            return "end"
        
        return "continue"

    def _calculate_completeness(self, company_info: Dict[str, Any]) -> int:
        score = 0
        if company_info.get("company_name"):
            score += 1
        if company_info.get("founding_year"):
            score += 1
        if company_info.get("founder_names") and len(company_info["founder_names"]) > 0:
            score += 1
        if company_info.get("product_description"):
            score += 1
        if company_info.get("funding_summary"):
            score += 1
        if company_info.get("notable_customers"):
            score += 1
        return min(score, 5)  # Cap at 5

    async def research_company(self, company_name: str, notes: Optional[str] = None) -> Dict[str, Any]:
        initial_state = {
            "company_name": company_name,
            "notes": notes,
            "messages": [HumanMessage(content=f"Research company: {company_name}")],
            "company_info": {"company_name": company_name},
            "search_queries": [],
            "search_results": [],
            "queries_used": 0,
            "results_processed": 0,
            "reflection_count": 0,
            "final_result": None
        }
        
        final_state = await self.graph.ainvoke(initial_state)
        
        # Return the final company info
        result = final_state.get("final_result") or final_state.get("company_info", {})
        
        # Add metadata
        result["_metadata"] = {
            "queries_used": final_state.get("queries_used", 0),
            "results_processed": final_state.get("results_processed", 0),
            "reflection_count": final_state.get("reflection_count", 0),
            "completeness_score": self._calculate_completeness(result),
            "timestamp": datetime.now().isoformat()
        }
        
        return result

# Example usage
async def main():
    # Initialize with API keys
    researcher = CompanyResearcher(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        max_search_queries=5,
        max_search_results=3,
        max_reflections=2
    )
    
    # Research a company
    result = await researcher.research_company(
        company_name="Anthropic",
        notes="AI safety company, Claude chatbot"
    )
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    asyncio.run(main())