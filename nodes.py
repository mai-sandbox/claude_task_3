import asyncio
from typing import Dict, Any, List
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage
from tavily import TavilyClient
from models import ResearchState, CompanyInfo
import os
from dotenv import load_dotenv

load_dotenv()

class CompanyResearchNodes:
    def __init__(self):
        self.llm = ChatAnthropic(
            model="claude-3-5-sonnet-20241022",
            temperature=0.1
        )
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    async def generate_search_queries(self, state: ResearchState) -> Dict[str, Any]:
        """Generate search queries to gather company information"""
        if state.query_count >= state.max_queries:
            state.messages.append(f"Reached maximum query limit ({state.max_queries})")
            return {"search_queries": [], "messages": state.messages, "query_count": state.query_count}
        
        existing_info = ""
        if state.company_info:
            existing_info = f"\nExisting information we have:\n{state.company_info.model_dump_json(indent=2)}"
        
        user_context = f"\nUser notes: {state.user_notes}" if state.user_notes else ""
        
        prompt = f"""
        Generate specific search queries to find comprehensive information about the company "{state.company_name}".
        
        We need to gather the following information:
        - Company name (official name)
        - Founding year
        - Founder names
        - Product/service description
        - Funding history and summary
        - Notable customers
        
        {user_context}
        {existing_info}
        
        Generate {state.max_queries - state.query_count} focused search queries that will help us find this information.
        Each query should target specific aspects of the company.
        
        Return only the search queries, one per line, no numbering or formatting.
        """
        
        messages = [
            SystemMessage(content="You are a research assistant that generates focused search queries for company information gathering."),
            HumanMessage(content=prompt)
        ]
        
        response = await self.llm.ainvoke(messages)
        queries = [q.strip() for q in response.content.split('\n') if q.strip()]
        
        state.search_queries.extend(queries[:state.max_queries - state.query_count])
        state.messages.append(f"Generated {len(queries)} search queries")
        
        return {
            "search_queries": state.search_queries,
            "messages": state.messages,
            "query_count": state.query_count
        }
    
    async def execute_searches(self, state: ResearchState) -> Dict[str, Any]:
        """Execute search queries in parallel using Tavily"""
        if not state.search_queries or state.query_count >= state.max_queries:
            return {"search_results": state.search_results, "messages": state.messages, "query_count": state.query_count}
        
        async def search_single_query(query: str) -> List[Dict[str, Any]]:
            try:
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=state.max_search_results
                )
                return response.get('results', [])
            except Exception as e:
                print(f"Error searching for '{query}': {e}")
                return []
        
        # Execute searches in parallel
        search_tasks = []
        queries_to_execute = state.search_queries[state.query_count:state.max_queries]
        
        for query in queries_to_execute:
            search_tasks.append(search_single_query(query))
        
        if search_tasks:
            search_results_batch = await asyncio.gather(*search_tasks)
            
            # Flatten and add results
            for results in search_results_batch:
                state.search_results.extend(results)
            
            state.query_count = min(len(state.search_queries), state.max_queries)
            state.messages.append(f"Executed {len(queries_to_execute)} search queries, got {sum(len(r) for r in search_results_batch)} results")
        
        return {
            "search_results": state.search_results,
            "messages": state.messages,
            "query_count": state.query_count
        }
    
    async def extract_company_info(self, state: ResearchState) -> Dict[str, Any]:
        """Extract structured company information from search results"""
        if not state.search_results:
            state.messages.append("No search results to extract information from")
            return {"company_info": state.company_info, "messages": state.messages}
        
        # Prepare search results for LLM
        search_content = ""
        for i, result in enumerate(state.search_results[:20]):  # Limit to avoid token limits
            search_content += f"\n--- Result {i+1} ---\n"
            search_content += f"Title: {result.get('title', 'N/A')}\n"
            search_content += f"URL: {result.get('url', 'N/A')}\n"
            search_content += f"Content: {result.get('content', 'N/A')[:500]}...\n"
        
        existing_info = ""
        if state.company_info:
            existing_info = f"\nExisting information:\n{state.company_info.model_dump_json(indent=2)}\n"
        
        prompt = f"""
        Extract and compile comprehensive information about "{state.company_name}" from the following search results.
        
        {existing_info}
        
        Search Results:
        {search_content}
        
        Please extract and return ONLY a JSON object with the following structure:
        {{
            "company_name": "official company name",
            "founding_year": year_as_integer_or_null,
            "founder_names": ["founder1", "founder2"],
            "product_description": "brief description of main product/service",
            "funding_summary": "summary of funding history including rounds and amounts",
            "notable_customers": "known customers or clients"
        }}
        
        Rules:
        - If information is not found, use null for optional fields or empty array for founder_names
        - Be accurate and only include verified information
        - Combine information from multiple sources when available
        - For funding_summary, include round types, amounts, and investors when available
        - For notable_customers, list well-known companies or organizations that use their services
        
        Return only the JSON object, no other text.
        """
        
        messages = [
            SystemMessage(content="You are a data extraction specialist. Extract structured company information from search results and return only valid JSON."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            # Parse JSON response
            import json
            company_data = json.loads(response.content.strip())
            state.company_info = CompanyInfo(**company_data)
            state.messages.append("Successfully extracted company information")
        except Exception as e:
            state.messages.append(f"Error extracting company information: {e}")
            if not state.company_info:
                state.company_info = CompanyInfo(company_name=state.company_name)
        
        return {"company_info": state.company_info, "messages": state.messages}
    
    async def reflect_on_completeness(self, state: ResearchState) -> Dict[str, Any]:
        """Evaluate if we have sufficient information about the company"""
        if state.reflection_count >= state.max_reflections:
            state.needs_more_info = False
            state.messages.append(f"Reached maximum reflection limit ({state.max_reflections})")
            return {
                "needs_more_info": state.needs_more_info,
                "reflection_count": state.reflection_count,
                "messages": state.messages
            }
        
        if not state.company_info:
            state.needs_more_info = True
            state.reflection_count += 1
            state.messages.append("No company information available, need more searches")
            return {
                "needs_more_info": state.needs_more_info,
                "reflection_count": state.reflection_count,
                "messages": state.messages
            }
        
        info_dict = state.company_info.model_dump()
        
        prompt = f"""
        Evaluate the completeness and quality of the following company information for "{state.company_name}":
        
        {info_dict}
        
        Consider:
        1. Are the core fields (company_name, founding_year, founder_names, product_description) well-populated?
        2. Is the funding_summary comprehensive enough?
        3. Are notable_customers identified?
        4. Is the information accurate and detailed enough for a comprehensive company profile?
        
        Respond with only "SUFFICIENT" if the information is comprehensive enough, or "INSUFFICIENT" if we need more research.
        If INSUFFICIENT, briefly explain what key information is missing.
        """
        
        messages = [
            SystemMessage(content="You are an information quality assessor. Evaluate if company information is sufficient for a comprehensive profile."),
            HumanMessage(content=prompt)
        ]
        
        try:
            response = await self.llm.ainvoke(messages)
            assessment = response.content.strip().upper()
            
            if "SUFFICIENT" in assessment:
                state.needs_more_info = False
                state.messages.append("Company information assessment: SUFFICIENT")
            else:
                state.needs_more_info = True
                state.messages.append(f"Company information assessment: INSUFFICIENT - {assessment}")
            
            state.reflection_count += 1
            
        except Exception as e:
            state.needs_more_info = False  # Default to stopping on error
            state.messages.append(f"Error during reflection: {e}")
        
        return {
            "needs_more_info": state.needs_more_info,
            "reflection_count": state.reflection_count,
            "messages": state.messages
        }