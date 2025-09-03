import os
import json
from typing import TypedDict, List, Optional, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import asyncio
from concurrent.futures import ThreadPoolExecutor

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from tavily import TavilyClient


@dataclass
class CompanyInfo:
    company_name: str
    founding_year: Optional[int] = None
    founder_names: Optional[List[str]] = None
    product_description: Optional[str] = None
    funding_summary: Optional[str] = None
    notable_customers: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def is_complete(self) -> bool:
        return all([
            self.company_name,
            self.founding_year is not None,
            self.founder_names,
            self.product_description,
            self.funding_summary,
            self.notable_customers
        ])

    def get_missing_fields(self) -> List[str]:
        missing = []
        if not self.founding_year:
            missing.append("founding_year")
        if not self.founder_names:
            missing.append("founder_names")
        if not self.product_description:
            missing.append("product_description")
        if not self.funding_summary:
            missing.append("funding_summary")
        if not self.notable_customers:
            missing.append("notable_customers")
        return missing


class ResearchState(TypedDict):
    company_name: str
    notes: str
    messages: List[Dict[str, str]]
    company_info: Dict[str, Any]
    search_queries: List[str]
    search_results: List[Dict[str, Any]]
    queries_executed: int
    reflection_count: int
    max_queries: int
    max_results_per_query: int
    max_reflections: int
    completed: bool


class CompanyResearcher:
    def __init__(
        self,
        openai_api_key: str,
        tavily_api_key: str,
        max_queries: int = 5,
        max_results_per_query: int = 3,
        max_reflections: int = 2
    ):
        self.llm = ChatOpenAI(
            api_key=openai_api_key,
            model="gpt-4o-mini",
            temperature=0
        )
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
        self.max_queries = max_queries
        self.max_results_per_query = max_results_per_query
        self.max_reflections = max_reflections
        
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(ResearchState)
        
        graph.add_node("initialize", self._initialize_research)
        graph.add_node("generate_queries", self._generate_search_queries)
        graph.add_node("execute_searches", self._execute_searches)
        graph.add_node("extract_information", self._extract_information)
        graph.add_node("reflect", self._reflect_and_evaluate)
        
        graph.add_edge(START, "initialize")
        graph.add_edge("initialize", "generate_queries")
        graph.add_edge("generate_queries", "execute_searches")
        graph.add_edge("execute_searches", "extract_information")
        graph.add_edge("extract_information", "reflect")
        
        graph.add_conditional_edges(
            "reflect",
            self._should_continue_research,
            {
                "continue": "generate_queries",
                "finish": END
            }
        )
        
        return graph.compile(checkpointer=self.checkpointer)

    def _initialize_research(self, state: ResearchState) -> ResearchState:
        company_info = CompanyInfo(company_name=state["company_name"])
        
        return {
            **state,
            "company_info": company_info.to_dict(),
            "search_queries": [],
            "search_results": [],
            "queries_executed": 0,
            "reflection_count": 0,
            "max_queries": self.max_queries,
            "max_results_per_query": self.max_results_per_query,
            "max_reflections": self.max_reflections,
            "completed": False,
            "messages": [{"role": "system", "content": f"Starting research for company: {state['company_name']}"}]
        }

    def _generate_search_queries(self, state: ResearchState) -> ResearchState:
        company_info = CompanyInfo(**state["company_info"])
        missing_fields = company_info.get_missing_fields()
        
        if state["reflection_count"] > 0:
            query_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research assistant generating targeted search queries for company information.
                Based on the current information and missing fields, generate specific search queries to find the missing information.
                
                Current company information:
                {current_info}
                
                Missing fields: {missing_fields}
                
                Generate 2-3 specific search queries that will help find the missing information.
                Make queries specific and targeted. Return only the queries, one per line."""),
                ("human", "Company: {company_name}\nAdditional notes: {notes}")
            ])
        else:
            query_prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a research assistant generating search queries for comprehensive company information.
                
                Generate 3-4 diverse search queries to gather information about:
                - Company founding year and founders
                - Products/services and business description
                - Funding history and financial information
                - Notable customers and partnerships
                
                Make queries specific and varied to get comprehensive coverage.
                Return only the queries, one per line."""),
                ("human", "Company: {company_name}\nAdditional notes: {notes}")
            ])
        
        response = self.llm.invoke(
            query_prompt.format(
                company_name=state["company_name"],
                notes=state["notes"],
                current_info=json.dumps(state["company_info"], indent=2),
                missing_fields=", ".join(missing_fields) if missing_fields else "None"
            )
        )
        
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        queries = queries[:min(len(queries), self.max_queries - state["queries_executed"])]
        
        new_messages = state["messages"] + [
            {"role": "assistant", "content": f"Generated {len(queries)} search queries: {queries}"}
        ]
        
        return {
            **state,
            "search_queries": queries,
            "messages": new_messages
        }

    def _execute_searches(self, state: ResearchState) -> ResearchState:
        queries = state["search_queries"]
        all_results = []
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_query = {
                executor.submit(self._search_tavily, query): query 
                for query in queries
            }
            
            for future in future_to_query:
                query = future_to_query[future]
                try:
                    results = future.result()
                    for result in results:
                        result['source_query'] = query
                    all_results.extend(results)
                except Exception as e:
                    print(f"Search failed for query '{query}': {e}")
        
        new_messages = state["messages"] + [
            {"role": "system", "content": f"Executed {len(queries)} searches, found {len(all_results)} results"}
        ]
        
        return {
            **state,
            "search_results": all_results,
            "queries_executed": state["queries_executed"] + len(queries),
            "messages": new_messages
        }

    def _search_tavily(self, query: str) -> List[Dict[str, Any]]:
        try:
            response = self.tavily_client.search(
                query=query,
                max_results=self.max_results_per_query
            )
            return response.get('results', [])
        except Exception as e:
            print(f"Tavily search error: {e}")
            return []

    def _extract_information(self, state: ResearchState) -> ResearchState:
        search_results_text = self._format_search_results(state["search_results"])
        current_info = CompanyInfo(**state["company_info"])
        
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert information extractor. Based on the search results, extract and update company information.

Current company information:
{current_info}

Search results:
{search_results}

Extract the following information and update the current data:
- company_name (official name)
- founding_year (integer year)
- founder_names (array of founder names)
- product_description (brief description of main product/service)
- funding_summary (summary of funding history)
- notable_customers (known customers that use the product/service)

Return the information as JSON with the exact field names above. Only include fields where you have reliable information.
If information is not found or unclear, exclude that field from the response."""),
            ("human", "Please extract and return the company information as JSON.")
        ])
        
        response = self.llm.invoke(
            extraction_prompt.format(
                current_info=json.dumps(current_info.to_dict(), indent=2),
                search_results=search_results_text
            )
        )
        
        try:
            extracted_data = json.loads(response.content)
            updated_info = current_info.to_dict()
            
            for key, value in extracted_data.items():
                if key in updated_info and value is not None:
                    updated_info[key] = value
                    
        except json.JSONDecodeError:
            updated_info = current_info.to_dict()
        
        new_messages = state["messages"] + [
            {"role": "assistant", "content": f"Extracted information: {json.dumps(extracted_data, indent=2)}"}
        ]
        
        return {
            **state,
            "company_info": updated_info,
            "messages": new_messages
        }

    def _reflect_and_evaluate(self, state: ResearchState) -> ResearchState:
        company_info = CompanyInfo(**state["company_info"])
        missing_fields = company_info.get_missing_fields()
        
        reflection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are evaluating the completeness of company research.

Current company information:
{company_info}

Missing fields: {missing_fields}
Queries executed: {queries_executed}/{max_queries}
Reflection count: {reflection_count}/{max_reflections}

Evaluate if we have sufficient information or need more research.
Consider:
1. Are the key fields populated with reliable information?
2. Is the information comprehensive enough?
3. Are there critical gaps that more searches could fill?

Respond with either "SUFFICIENT" or "NEED_MORE" and explain your reasoning."""),
            ("human", "Should we continue research or is the information sufficient?")
        ])
        
        response = self.llm.invoke(
            reflection_prompt.format(
                company_info=json.dumps(company_info.to_dict(), indent=2),
                missing_fields=", ".join(missing_fields) if missing_fields else "None",
                queries_executed=state["queries_executed"],
                max_queries=state["max_queries"],
                reflection_count=state["reflection_count"],
                max_reflections=state["max_reflections"]
            )
        )
        
        evaluation = response.content
        should_continue = (
            "NEED_MORE" in evaluation and
            state["queries_executed"] < state["max_queries"] and
            state["reflection_count"] < state["max_reflections"] and
            len(missing_fields) > 0
        )
        
        new_messages = state["messages"] + [
            {"role": "assistant", "content": f"Reflection {state['reflection_count'] + 1}: {evaluation}"}
        ]
        
        return {
            **state,
            "reflection_count": state["reflection_count"] + 1,
            "completed": not should_continue,
            "messages": new_messages
        }

    def _should_continue_research(self, state: ResearchState) -> str:
        return "finish" if state["completed"] else "continue"

    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        formatted = []
        for i, result in enumerate(results[:10], 1):
            formatted.append(f"""
Result {i} (Query: {result.get('source_query', 'Unknown')}):
Title: {result.get('title', 'No title')}
URL: {result.get('url', 'No URL')}
Content: {result.get('content', 'No content')[:500]}...
""")
        return "\n".join(formatted)

    async def research_company(
        self, 
        company_name: str, 
        notes: str = "",
        thread_id: str = "default"
    ) -> Dict[str, Any]:
        initial_state: ResearchState = {
            "company_name": company_name,
            "notes": notes,
            "messages": [],
            "company_info": {},
            "search_queries": [],
            "search_results": [],
            "queries_executed": 0,
            "reflection_count": 0,
            "max_queries": self.max_queries,
            "max_results_per_query": self.max_results_per_query,
            "max_reflections": self.max_reflections,
            "completed": False
        }
        
        config = {"configurable": {"thread_id": thread_id}}
        
        final_state = await self.graph.ainvoke(initial_state, config=config)
        
        return {
            "company_info": final_state["company_info"],
            "research_summary": {
                "queries_executed": final_state["queries_executed"],
                "reflection_count": final_state["reflection_count"],
                "total_results_found": len(final_state["search_results"]),
                "completed": final_state["completed"]
            },
            "messages": final_state["messages"]
        }


async def main():
    # Example usage
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_api_key or not tavily_api_key:
        print("Please set OPENAI_API_KEY and TAVILY_API_KEY environment variables")
        return
    
    researcher = CompanyResearcher(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        max_queries=5,
        max_results_per_query=3,
        max_reflections=2
    )
    
    # Research a company
    result = await researcher.research_company(
        company_name="Anthropic",
        notes="AI safety company, created Claude"
    )
    
    print("Company Research Results:")
    print("=" * 50)
    print(json.dumps(result["company_info"], indent=2))
    print("\nResearch Summary:")
    print(json.dumps(result["research_summary"], indent=2))


if __name__ == "__main__":
    asyncio.run(main())