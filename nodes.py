import asyncio
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tavily import TavilyClient
from schemas import ResearchState, SearchQuery, SearchResult, CompanyInfo
import json
import os
from dotenv import load_dotenv

load_dotenv()

class CompanyResearchNodes:
    def __init__(self):
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self.tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    
    def _ensure_research_state(self, state) -> ResearchState:
        """Convert state dict to ResearchState object if needed"""
        if isinstance(state, dict):
            return ResearchState(**state)
        return state
    
    def _state_to_dict(self, state: ResearchState) -> Dict[str, Any]:
        """Convert ResearchState back to dict for LangGraph"""
        return state.model_dump()
    
    def _extract_info_manually(self, results_text: str, company_name: str) -> Dict[str, Any]:
        """Manual fallback extraction when JSON parsing fails"""
        # Initialize with minimal data
        company_data = {
            "company_name": company_name,
            "founding_year": None,
            "founder_names": [],
            "product_description": None,
            "funding_summary": None,
            "notable_customers": None
        }
        
        # Try to extract basic information using keyword matching
        text_lower = results_text.lower()
        
        # Look for founding year
        import re
        year_matches = re.findall(r'founded.*?(\d{4})|established.*?(\d{4})', text_lower)
        if year_matches:
            for match_group in year_matches:
                for year in match_group:
                    if year and 1800 <= int(year) <= 2030:
                        company_data["founding_year"] = int(year)
                        break
                if company_data["founding_year"]:
                    break
        
        # Look for founders
        founder_patterns = [
            r'founded by ([^.]+)',
            r'co-founded by ([^.]+)',
            r'founder[s]?:?\s*([^.]+)',
            r'created by ([^.]+)'
        ]
        for pattern in founder_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                founders_text = matches[0]
                # Extract names (simple approach)
                potential_founders = []
                for name_part in founders_text.split(' and '):
                    clean_name = re.sub(r'[^a-zA-Z\s]', '', name_part).strip().title()
                    if clean_name and len(clean_name.split()) >= 2:
                        potential_founders.append(clean_name)
                if potential_founders:
                    company_data["founder_names"] = potential_founders
                    break
        
        return company_data
    
    async def generate_search_queries(self, state) -> Dict[str, Any]:
        """Generate search queries to research the company"""
        state = self._ensure_research_state(state)
        state.add_message("system", f"Generating search queries for {state.company_name}")
        
        # Determine what information we still need
        missing_fields = state.get_missing_fields() if state.company_info else [
            "founding_year", "founder_names", "product_description", "funding_summary", "notable_customers"
        ]
        
        user_notes_context = f"\nUser notes: {state.user_notes}" if state.user_notes else ""
        
        system_prompt = f"""You are a research assistant tasked with generating specific search queries to find comprehensive information about a company.

Company: {state.company_name}{user_notes_context}

Missing information needed: {', '.join(missing_fields)}

Generate {min(state.max_search_queries - state.queries_executed, 6)} diverse and specific search queries that will help find the missing information. Each query should target specific aspects of the company.

Focus on:
1. Company founding information (founders, founding year)
2. Product/service descriptions and main offerings
3. Funding rounds, investments, and financial history
4. Notable customers, partnerships, and case studies
5. Company background and history

Return your response as a JSON array of objects, each with "query" and "purpose" fields.
Example format:
[
    {{"query": "'{state.company_name}' founders founding year history", "purpose": "Find founding information and founders"}},
    {{"query": "'{state.company_name}' product service description", "purpose": "Understand main products/services"}}
]"""

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Generate search queries for researching {state.company_name}")
            ]
            
            response = await self.llm.ainvoke(messages)
            queries_data = json.loads(response.content)
            
            new_queries = [SearchQuery(**query_data) for query_data in queries_data]
            state.search_queries.extend(new_queries)
            
            state.add_message("assistant", f"Generated {len(new_queries)} search queries")
            
        except Exception as e:
            state.add_message("error", f"Failed to generate queries: {str(e)}")
            # Fallback queries
            fallback_queries = [
                SearchQuery(query=f"{state.company_name} company information", purpose="General company information"),
                SearchQuery(query=f"{state.company_name} founders founding year", purpose="Founding information"),
                SearchQuery(query=f"{state.company_name} products services", purpose="Products and services"),
                SearchQuery(query=f"{state.company_name} funding investors", purpose="Funding information"),
                SearchQuery(query=f"{state.company_name} customers clients", purpose="Customer information")
            ]
            state.search_queries.extend(fallback_queries[:state.max_search_queries - state.queries_executed])
        
        return self._state_to_dict(state)
    
    async def execute_search_queries(self, state) -> Dict[str, Any]:
        """Execute search queries in parallel using Tavily API"""
        state = self._ensure_research_state(state)
        queries_to_execute = state.search_queries[state.queries_executed:]
        if not queries_to_execute:
            return self._state_to_dict(state)
        
        state.add_message("system", f"Executing {len(queries_to_execute)} search queries in parallel")
        
        async def search_single_query(query: SearchQuery) -> List[SearchResult]:
            try:
                response = self.tavily.search(
                    query=query.query,
                    max_results=state.max_search_results,
                    include_domains=None,
                    exclude_domains=["facebook.com", "twitter.com", "instagram.com"]
                )
                
                results = []
                for result in response.get("results", []):
                    results.append(SearchResult(
                        title=result.get("title", ""),
                        content=result.get("content", ""),
                        url=result.get("url", "")
                    ))
                
                return results
                
            except Exception as e:
                state.add_message("error", f"Search failed for '{query.query}': {str(e)}")
                return []
        
        # Execute searches in parallel
        search_tasks = [search_single_query(query) for query in queries_to_execute]
        search_results_lists = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Collect all results
        for results in search_results_lists:
            if isinstance(results, list):
                state.search_results.extend(results)
            else:
                state.add_message("error", f"Search task failed: {str(results)}")
        
        state.queries_executed = len(state.search_queries)
        state.add_message("assistant", f"Completed searches, collected {len(state.search_results)} results")
        
        return self._state_to_dict(state)
    
    async def extract_company_information(self, state) -> Dict[str, Any]:
        """Extract structured company information from search results"""
        state = self._ensure_research_state(state)
        if not state.search_results:
            state.add_message("error", "No search results to extract information from")
            return self._state_to_dict(state)
        
        state.add_message("system", "Extracting company information from search results")
        
        # Prepare search results content for analysis
        search_content = []
        for i, result in enumerate(state.search_results[:20]):  # Limit to prevent token overflow
            search_content.append(f"Result {i+1}:\nTitle: {result.title}\nContent: {result.content[:500]}...\nURL: {result.url}\n")
        
        results_text = "\n".join(search_content)
        
        system_prompt = f"""You are a research analyst tasked with extracting structured company information from web search results.

Company being researched: {state.company_name}

Extract the following information and return it as a JSON object:
- company_name: Official name of the company
- founding_year: Year the company was founded (integer)
- founder_names: Array of founder names
- product_description: Brief description of main product/service
- funding_summary: Summary of funding history
- notable_customers: Known customers or clients

If information is not found or unclear, use null for missing fields (except company_name which should always be provided).
Be factual and only include information that is clearly stated in the search results.

Return only the JSON object, no additional text."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Search Results:\n{results_text}\n\nExtract company information for {state.company_name}")
            ]
            
            response = await self.llm.ainvoke(messages)
            
            # Clean up the response content to extract JSON
            content = response.content.strip()
            
            # Try to find JSON in the response
            if content.startswith('```json'):
                # Remove markdown code blocks
                content = content[7:-3].strip()
            elif content.startswith('```'):
                # Remove any code blocks
                lines = content.split('\n')
                json_lines = []
                in_json = False
                for line in lines:
                    if line.strip().startswith('{') and not in_json:
                        in_json = True
                    if in_json:
                        json_lines.append(line)
                    if line.strip().endswith('}') and in_json:
                        break
                content = '\n'.join(json_lines)
            
            # Find JSON object in content
            start_idx = content.find('{')
            end_idx = content.rfind('}') + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
            else:
                json_str = content
            
            # Parse the JSON response
            try:
                company_data = json.loads(json_str)
            except json.JSONDecodeError as je:
                # If JSON parsing fails, try to extract info manually
                state.add_message("error", f"JSON parsing failed: {str(je)}")
                company_data = self._extract_info_manually(results_text, state.company_name)
            
            # Ensure company_name is set
            company_data["company_name"] = state.company_name
            
            state.company_info = CompanyInfo(**company_data)
            state.add_message("assistant", f"Extracted company information successfully")
            
        except Exception as e:
            state.add_message("error", f"Failed to extract company information: {str(e)}")
            # Create minimal company info
            state.company_info = CompanyInfo(company_name=state.company_name)
        
        return self._state_to_dict(state)
    
    async def reflect_and_decide(self, state) -> Dict[str, Any]:
        """Reflect on the gathered information and decide if more research is needed"""
        state = self._ensure_research_state(state)
        if not state.company_info:
            state.add_message("error", "No company information to reflect on")
            return self._state_to_dict(state)
        
        state.add_message("system", f"Reflecting on gathered information (step {state.reflection_count + 1}/{state.max_reflection_steps})")
        
        missing_fields = state.get_missing_fields()
        
        company_info_summary = f"""
Current Company Information:
- Name: {state.company_info.company_name}
- Founding Year: {state.company_info.founding_year or 'Unknown'}
- Founders: {', '.join(state.company_info.founder_names) if state.company_info.founder_names else 'Unknown'}
- Product Description: {state.company_info.product_description or 'Unknown'}
- Funding Summary: {state.company_info.funding_summary or 'Unknown'}
- Notable Customers: {state.company_info.notable_customers or 'Unknown'}

Missing Information: {', '.join(missing_fields) if missing_fields else 'None'}
"""
        
        system_prompt = f"""You are a research quality assessor. Evaluate if the current company research is sufficient or if more research is needed.

{company_info_summary}

Research Status:
- Queries executed: {state.queries_executed}/{state.max_search_queries}
- Reflection steps: {state.reflection_count}/{state.max_reflection_steps}
- Search results collected: {len(state.search_results)}

Determine if:
1. The information is sufficient and complete
2. More research is needed and possible (within limits)

Respond with a JSON object:
{{
    "is_sufficient": boolean,
    "reasoning": "explanation of the decision",
    "priority_missing": ["list", "of", "most", "important", "missing", "fields"]
}}"""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content="Assess the research quality and completeness")
            ]
            
            response = await self.llm.ainvoke(messages)
            assessment = json.loads(response.content)
            
            state.reflection_count += 1
            
            # Determine if research is complete
            is_complete = (
                assessment.get("is_sufficient", False) or
                state.reflection_count >= state.max_reflection_steps or
                state.queries_executed >= state.max_search_queries or
                len(missing_fields) <= 1  # Allow completion if only 1 field is missing
            )
            
            state.is_complete = is_complete
            
            if is_complete:
                state.add_message("assistant", f"Research completed. Reasoning: {assessment.get('reasoning', 'Limits reached')}")
            else:
                state.add_message("assistant", f"More research needed. Reasoning: {assessment.get('reasoning', 'Information incomplete')}")
                # Reset search queries for next iteration if we have room for more queries
                if state.queries_executed < state.max_search_queries:
                    state.search_queries = []  # Clear previous queries for new targeted ones
            
        except Exception as e:
            state.add_message("error", f"Reflection failed: {str(e)}")
            # Default to completion if reflection fails
            state.is_complete = True
        
        return self._state_to_dict(state)