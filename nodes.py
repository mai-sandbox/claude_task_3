import asyncio
import json
from typing import List, Dict, Any
from openai import OpenAI
from tavily import TavilyClient
from models import ResearchState, SearchQuery, SearchResult, CompanyInfo


class CompanyResearchNodes:
    def __init__(self, openai_api_key: str, tavily_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.tavily_client = TavilyClient(api_key=tavily_api_key)
    
    def generate_search_queries(self, state: ResearchState) -> ResearchState:
        """Generate search queries to find company information"""
        state.messages.append({
            "role": "system", 
            "content": f"Generating search queries for {state.company_name}"
        })
        
        # Identify missing information
        missing_fields = []
        if not state.company_info.founding_year:
            missing_fields.append("founding year")
        if not state.company_info.founder_names:
            missing_fields.append("founders")
        if not state.company_info.product_description:
            missing_fields.append("product/service description")
        if not state.company_info.funding_summary:
            missing_fields.append("funding history")
        if not state.company_info.notable_customers:
            missing_fields.append("notable customers")
        
        prompt = f"""
        Generate {min(state.max_search_queries, len(missing_fields) + 2)} specific search queries to research {state.company_name}.
        
        Company: {state.company_name}
        User notes: {state.user_notes or "None"}
        
        Missing information: {', '.join(missing_fields) if missing_fields else 'General company information'}
        
        Generate queries that will help find:
        1. Basic company information (founding year, founders)
        2. Product/service details
        3. Funding and business information
        4. Notable customers or partnerships
        
        Return as JSON array of objects with 'query' and 'purpose' fields.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )
            
            queries_data = json.loads(response.choices[0].message.content)
            state.search_queries = [item['query'] for item in queries_data[:state.max_search_queries]]
            
        except Exception as e:
            # Fallback queries if LLM fails
            state.search_queries = [
                f"{state.company_name} company information founding year founders",
                f"{state.company_name} product service description",
                f"{state.company_name} funding investment history",
                f"{state.company_name} notable customers clients partnerships"
            ][:state.max_search_queries]
        
        return state
    
    def search_web_parallel(self, state: ResearchState) -> ResearchState:
        """Perform web searches in parallel using Tavily API"""
        state.messages.append({
            "role": "system", 
            "content": f"Searching web for {len(state.search_queries)} queries"
        })
        
        def search_single_query(query: str) -> List[Dict[str, Any]]:
            try:
                response = self.tavily_client.search(
                    query=query,
                    search_depth="basic",
                    max_results=state.max_search_results
                )
                return response.get('results', [])
            except Exception as e:
                print(f"Search failed for query '{query}': {e}")
                return []
        
        # Execute searches in parallel
        all_results = []
        for query in state.search_queries:
            results = search_single_query(query)
            for result in results:
                result['query'] = query  # Track which query produced this result
            all_results.extend(results)
        
        state.search_results = all_results
        return state
    
    def extract_company_info(self, state: ResearchState) -> ResearchState:
        """Extract and structure company information from search results"""
        state.messages.append({
            "role": "system", 
            "content": "Extracting and structuring company information"
        })
        
        # Compile all search content
        search_content = "\n\n".join([
            f"Source: {result.get('title', 'Unknown')} ({result.get('url', 'No URL')})\n{result.get('content', '')}"
            for result in state.search_results
        ])
        
        prompt = f"""
        Extract company information from the search results below and return it as JSON matching this exact schema:
        
        {{
            "company_name": "string",
            "founding_year": integer or null,
            "founder_names": ["array", "of", "strings"],
            "product_description": "string or null",
            "funding_summary": "string or null",
            "notable_customers": "string or null"
        }}
        
        Company: {state.company_name}
        User notes: {state.user_notes or "None"}
        
        Search Results:
        {search_content[:8000]}  # Limit content to avoid token limits
        
        Important:
        - Use the exact company name provided: {state.company_name}
        - founding_year should be an integer year or null
        - founder_names should be an array of founder names
        - If information is not found, use null for optional fields
        - Be concise but informative
        
        Return only the JSON object, no other text.
        """
        
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            
            extracted_data = json.loads(response.choices[0].message.content)
            state.company_info = CompanyInfo(**extracted_data)
            
        except Exception as e:
            print(f"Failed to extract company info: {e}")
            # Keep existing company_info if extraction fails
        
        return state
    
    def reflect_on_completeness(self, state: ResearchState) -> ResearchState:
        """Reflect on whether we have sufficient information"""
        state.messages.append({
            "role": "system", 
            "content": f"Reflection step {state.reflection_count + 1}"
        })
        
        # Check completeness
        info = state.company_info
        missing_info = []
        
        if not info.founding_year:
            missing_info.append("founding year")
        if not info.founder_names:
            missing_info.append("founders")
        if not info.product_description:
            missing_info.append("product description")
        if not info.funding_summary:
            missing_info.append("funding information")
        if not info.notable_customers:
            missing_info.append("notable customers")
        
        # Decision logic
        if len(missing_info) <= 2:  # Allow some missing info
            state.is_complete = True
            state.needs_more_info = False
        elif state.reflection_count >= state.max_reflection_steps:
            state.is_complete = True
            state.needs_more_info = False
            state.messages.append({
                "role": "system", 
                "content": f"Maximum reflection steps reached. Proceeding with available information."
            })
        else:
            state.needs_more_info = True
            state.reflection_count += 1
            
            # Generate new queries for missing information
            new_queries = []
            for missing in missing_info[:2]:  # Focus on top 2 missing items
                if missing == "founding year":
                    new_queries.append(f"{state.company_name} founded when year established")
                elif missing == "founders":
                    new_queries.append(f"{state.company_name} founders co-founders CEO founder")
                elif missing == "product description":
                    new_queries.append(f"{state.company_name} what does company do product service")
                elif missing == "funding information":
                    new_queries.append(f"{state.company_name} funding investment series A B C valuation")
                elif missing == "notable customers":
                    new_queries.append(f"{state.company_name} customers clients case studies partnerships")
            
            state.search_queries = new_queries
        
        return state