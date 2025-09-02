import asyncio
import json
from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from models import ResearchState, CompanyInfo, SearchQuery
from tavily_client import TavilySearchClient


class CompanyResearchNodes:
    """Collection of nodes for the company research workflow"""
    
    def __init__(self, openai_api_key: str = None, tavily_api_key: str = None):
        self.llm = ChatOpenAI(
            model="gpt-4",
            temperature=0.3,
            api_key=openai_api_key
        )
        self.tavily_client = TavilySearchClient(api_key=tavily_api_key)
    
    def generate_search_queries(self, state: ResearchState) -> ResearchState:
        """Generate targeted search queries for company research"""
        # Check if we've reached the max number of queries
        if state.search_queries_used >= state.max_search_queries:
            state.messages.append({
                "type": "system",
                "content": f"Reached maximum search queries limit ({state.max_search_queries})"
            })
            return state
        
        # Determine what information is missing
        missing_info = self._identify_missing_info(state.company_info)
        
        # Create prompt for query generation
        system_prompt = f"""You are a research assistant. Generate specific search queries to find information about the company "{state.company_name}".

Current information we have:
{json.dumps(state.company_info.model_dump(), indent=2)}

Missing information to find: {', '.join(missing_info)}

User notes: {state.user_notes or 'None'}

Generate 2-3 specific, targeted search queries that would help find the missing information. 
Return only a JSON array of queries, no other text.

Example format:
["OpenAI founding year", "Sam Altman Ilya Sutskever OpenAI founders", "OpenAI ChatGPT product description"]
"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate search queries for {state.company_name}")
            ]
            
            response = self.llm(messages)
            queries = json.loads(response.content)
            
            # Limit queries to avoid exceeding max
            remaining_queries = state.max_search_queries - state.search_queries_used
            queries = queries[:remaining_queries]
            
            state.messages.append({
                "type": "query_generation",
                "content": f"Generated {len(queries)} search queries: {queries}"
            })
            
            # Store queries for the next node
            state.__dict__['current_queries'] = queries
            
        except Exception as e:
            state.messages.append({
                "type": "error",
                "content": f"Error generating queries: {str(e)}"
            })
            state.__dict__['current_queries'] = []
        
        return state
    
    async def search_web(self, state: ResearchState) -> ResearchState:
        """Execute web searches in parallel"""
        queries = state.__dict__.get('current_queries', [])
        
        if not queries:
            state.messages.append({
                "type": "system",
                "content": "No queries to search"
            })
            return state
        
        try:
            # Perform parallel searches
            search_results = await self.tavily_client.search_parallel(
                queries, 
                max_results=state.max_search_results
            )
            
            # Flatten and store results
            all_results = []
            for i, results in enumerate(search_results):
                for result in results:
                    result['source_query'] = queries[i]
                    all_results.append(result)
            
            state.search_results.extend(all_results)
            state.search_queries_used += len(queries)
            
            state.messages.append({
                "type": "search",
                "content": f"Executed {len(queries)} searches, found {len(all_results)} total results"
            })
            
        except Exception as e:
            state.messages.append({
                "type": "error",
                "content": f"Error during web search: {str(e)}"
            })
        
        # Clean up temporary queries
        if 'current_queries' in state.__dict__:
            del state.__dict__['current_queries']
        
        return state
    
    def extract_information(self, state: ResearchState) -> ResearchState:
        """Extract and structure information from search results"""
        if not state.search_results:
            state.messages.append({
                "type": "system",
                "content": "No search results to process"
            })
            return state
        
        # Prepare search results summary for the LLM
        results_summary = self._format_search_results(state.search_results[-20:])  # Use last 20 results
        
        system_prompt = f"""You are a data extraction expert. Extract company information from the provided search results.

Company Name: {state.company_name}
User Notes: {state.user_notes or 'None'}

Current company information:
{json.dumps(state.company_info.model_dump(), indent=2)}

Search Results:
{results_summary}

Please update the company information with any new details found in the search results. 
Return ONLY a JSON object matching the CompanyInfo schema:

{{
    "company_name": "string",
    "founding_year": integer or null,
    "founder_names": ["string"] or null,
    "product_description": "string" or null,
    "funding_summary": "string" or null,
    "notable_customers": "string" or null
}}

Important:
- Only include information that is clearly stated in the search results
- Keep descriptions concise but informative
- If founding_year is found, it must be an integer
- If founder_names is found, it should be an array of individual names
"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please extract and update the company information.")
            ]
            
            response = self.llm(messages)
            updated_info = json.loads(response.content)
            
            # Update the company info
            state.company_info = CompanyInfo(**updated_info)
            
            state.messages.append({
                "type": "extraction",
                "content": f"Updated company information from {len(state.search_results)} search results"
            })
            
        except Exception as e:
            state.messages.append({
                "type": "error",
                "content": f"Error extracting information: {str(e)}"
            })
        
        return state
    
    def reflect_and_decide(self, state: ResearchState) -> ResearchState:
        """Reflect on the gathered information and decide if more research is needed"""
        state.reflection_steps_used += 1
        
        # Check limits
        if state.reflection_steps_used >= state.max_reflection_steps:
            state.needs_more_research = False
            state.completed = True
            state.messages.append({
                "type": "reflection",
                "content": f"Reached maximum reflection steps ({state.max_reflection_steps}). Completing research."
            })
            return state
        
        if state.search_queries_used >= state.max_search_queries:
            state.needs_more_research = False
            state.completed = True
            state.messages.append({
                "type": "reflection", 
                "content": f"Reached maximum search queries ({state.max_search_queries}). Completing research."
            })
            return state
        
        # Evaluate information completeness
        missing_info = self._identify_missing_info(state.company_info)
        completeness_score = self._calculate_completeness_score(state.company_info)
        
        system_prompt = f"""You are a research quality evaluator. Assess whether we have sufficient information about "{state.company_name}".

Current information:
{json.dumps(state.company_info.model_dump(), indent=2)}

Missing information: {', '.join(missing_info)}
Completeness score: {completeness_score}/100

Search queries used: {state.search_queries_used}/{state.max_search_queries}
Reflection steps used: {state.reflection_steps_used}/{state.max_reflection_steps}

User Notes: {state.user_notes or 'None'}

Respond with only "CONTINUE" if more research is needed, or "COMPLETE" if we have sufficient information.

Guidelines for decision:
- COMPLETE if completeness score >= 70 and we have basic company info
- COMPLETE if we've found the company name and at least 3 other pieces of information
- CONTINUE if important information is missing and we haven't reached limits
- CONTINUE if the user notes suggest specific information that's still missing
"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Should we continue research or complete?")
            ]
            
            response = self.llm(messages)
            decision = response.content.strip().upper()
            
            if decision == "COMPLETE":
                state.needs_more_research = False
                state.completed = True
            else:
                state.needs_more_research = True
            
            state.messages.append({
                "type": "reflection",
                "content": f"Decision: {decision}. Completeness: {completeness_score}%, Missing: {missing_info}"
            })
            
        except Exception as e:
            state.messages.append({
                "type": "error",
                "content": f"Error in reflection: {str(e)}"
            })
            # Default to complete on error
            state.needs_more_research = False
            state.completed = True
        
        return state
    
    def _identify_missing_info(self, company_info: CompanyInfo) -> List[str]:
        """Identify what information is missing from the company profile"""
        missing = []
        
        if not company_info.founding_year:
            missing.append("founding_year")
        if not company_info.founder_names:
            missing.append("founder_names")
        if not company_info.product_description:
            missing.append("product_description")
        if not company_info.funding_summary:
            missing.append("funding_summary")
        if not company_info.notable_customers:
            missing.append("notable_customers")
            
        return missing
    
    def _calculate_completeness_score(self, company_info: CompanyInfo) -> int:
        """Calculate a completeness score (0-100) for the company information"""
        total_fields = 6  # including company_name
        filled_fields = 1  # company_name is required
        
        if company_info.founding_year:
            filled_fields += 1
        if company_info.founder_names:
            filled_fields += 1
        if company_info.product_description:
            filled_fields += 1
        if company_info.funding_summary:
            filled_fields += 1
        if company_info.notable_customers:
            filled_fields += 1
        
        return int((filled_fields / total_fields) * 100)
    
    def _format_search_results(self, results: List[Dict[str, Any]]) -> str:
        """Format search results for LLM consumption"""
        formatted = []
        
        for i, result in enumerate(results):
            formatted.append(f"""
Result {i+1}:
Title: {result.get('title', 'N/A')}
URL: {result.get('url', 'N/A')}
Content: {result.get('content', 'N/A')[:500]}...
Source Query: {result.get('source_query', 'N/A')}
""")
        
        return "\n".join(formatted)