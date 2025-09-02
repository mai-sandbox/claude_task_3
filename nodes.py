import asyncio
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from tavily import TavilyClient
import os
from models import ResearchState, SearchQuery, CompanyInfo, SearchResult


class QueryGenerationNode:
    """Node responsible for generating search queries"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def __call__(self, state: ResearchState) -> ResearchState:
        """Generate search queries for company research"""
        
        # Don't generate more queries if we've reached the limit
        if state.queries_executed >= state.max_search_queries:
            return state
        
        # Determine how many queries to generate
        remaining_queries = state.max_search_queries - state.queries_executed
        queries_to_generate = min(4, remaining_queries)  # Generate up to 4 at a time
        
        # Build context from previous research
        context = ""
        if state.search_results:
            context = f"\nPrevious search results summary: {len(state.search_results)} results found"
        
        user_notes_context = f"\nUser notes: {state.user_notes}" if state.user_notes else ""
        
        system_prompt = f"""You are a research assistant. Generate {queries_to_generate} specific search queries to find information about the company "{state.company_name}".

Focus on finding:
1. Company founding year and founders
2. Product/service descriptions
3. Funding history and investors
4. Notable customers and partnerships
5. Recent news and developments

Current research progress:
- Company: {state.company_name}
- Queries executed: {state.queries_executed}/{state.max_search_queries}
- Current info: {state.company_info.model_dump()}
{context}{user_notes_context}

Generate diverse, specific queries that will help fill missing information. Return exactly {queries_to_generate} queries, each on a new line in this format:
QUERY: [search query]
PURPOSE: [what information this query aims to find]

Example:
QUERY: {state.company_name} founders founding team history
PURPOSE: Find founding year and founder names
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Generate {queries_to_generate} search queries for researching {state.company_name}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        # Parse the response to extract queries
        queries = []
        lines = response.content.strip().split('\n')
        
        current_query = None
        current_purpose = None
        
        for line in lines:
            line = line.strip()
            if line.startswith('QUERY:'):
                current_query = line[6:].strip()
            elif line.startswith('PURPOSE:'):
                current_purpose = line[8:].strip()
                if current_query and current_purpose:
                    queries.append(SearchQuery(query=current_query, purpose=current_purpose))
                    current_query = None
                    current_purpose = None
        
        # Add to state
        state.search_queries_generated.extend(queries)
        state.messages.append({
            "type": "query_generation",
            "queries": [q.model_dump() for q in queries],
            "total_generated": len(queries)
        })
        
        return state


class WebSearchNode:
    """Node responsible for executing web searches"""
    
    def __init__(self, tavily_client: TavilyClient):
        self.tavily_client = tavily_client
    
    async def __call__(self, state: ResearchState) -> ResearchState:
        """Execute web searches for pending queries"""
        
        # Get unexecuted queries
        unexecuted_queries = state.search_queries_generated[state.queries_executed:]
        
        if not unexecuted_queries:
            return state
        
        # Execute searches in parallel (but limit concurrency)
        search_tasks = []
        for query in unexecuted_queries:
            if state.queries_executed >= state.max_search_queries:
                break
            search_tasks.append(self._search_query(query.query))
            state.queries_executed += 1
        
        if search_tasks:
            # Execute searches with limited concurrency
            semaphore = asyncio.Semaphore(3)  # Max 3 concurrent searches
            
            async def limited_search(task):
                async with semaphore:
                    return await task
            
            search_results = await asyncio.gather(*[limited_search(task) for task in search_tasks])
            
            # Process and store results
            for i, results in enumerate(search_results):
                if results:
                    for result in results:
                        if len(state.search_results) < state.max_search_results:
                            state.search_results.append({
                                "query": unexecuted_queries[i].query,
                                "purpose": unexecuted_queries[i].purpose,
                                "title": result.get("title", ""),
                                "url": result.get("url", ""),
                                "content": result.get("content", ""),
                                "score": result.get("score", 0.0)
                            })
            
            state.messages.append({
                "type": "web_search",
                "queries_executed": len(search_tasks),
                "results_found": len([r for r in search_results if r]),
                "total_results": len(state.search_results)
            })
        
        return state
    
    async def _search_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute a single search query"""
        try:
            # Use Tavily search
            response = self.tavily_client.search(
                query=query,
                search_depth="basic",
                include_images=False,
                include_answer=False,
                max_results=5
            )
            return response.get("results", [])
        except Exception as e:
            print(f"Search error for query '{query}': {e}")
            return []


class InformationExtractionNode:
    """Node responsible for extracting structured information from search results"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def __call__(self, state: ResearchState) -> ResearchState:
        """Extract company information from search results"""
        
        if not state.search_results:
            return state
        
        # Prepare search results for processing
        search_context = ""
        for i, result in enumerate(state.search_results[-10:]):  # Use last 10 results
            search_context += f"\n--- Result {i+1} ---\n"
            search_context += f"Query: {result['query']}\n"
            search_context += f"Title: {result['title']}\n"
            search_context += f"Content: {result['content'][:500]}...\n"
        
        system_prompt = f"""You are an expert information extraction assistant. Extract and update company information from the provided search results.

Current company information:
{state.company_info.model_dump_json(indent=2)}

Based on the search results below, update the company information. Follow these rules:
1. Only update fields with confident, factual information
2. For founder_names, provide a list of individual names
3. Keep descriptions concise but informative
4. If you find conflicting information, use the most recent/reliable source
5. Leave fields as null/empty if no reliable information is found

Return the updated information in this exact JSON format:
{{
    "company_name": "string",
    "founding_year": integer or null,
    "founder_names": ["name1", "name2"] or [],
    "product_description": "string" or null,
    "funding_summary": "string" or null,
    "notable_customers": "string" or null
}}

Search Results:
{search_context}
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract and update information for {state.company_name}")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        try:
            # Parse the JSON response
            import json
            # Find JSON in the response
            content = response.content.strip()
            if content.startswith('```json'):
                content = content[7:]
            if content.endswith('```'):
                content = content[:-3]
            
            extracted_data = json.loads(content)
            
            # Update the company info
            state.company_info = CompanyInfo(**extracted_data)
            
            state.messages.append({
                "type": "information_extraction",
                "extracted_fields": list(extracted_data.keys()),
                "updated_info": extracted_data
            })
            
        except Exception as e:
            print(f"Error parsing extracted information: {e}")
            state.messages.append({
                "type": "information_extraction_error",
                "error": str(e),
                "response": response.content
            })
        
        return state


class ReflectionNode:
    """Node responsible for determining if research is complete"""
    
    def __init__(self, llm: ChatOpenAI):
        self.llm = llm
    
    async def __call__(self, state: ResearchState) -> ResearchState:
        """Determine if we have sufficient information about the company"""
        
        # Check if we've exceeded reflection limit
        if state.reflections_done >= state.max_reflections:
            state.research_complete = True
            return state
        
        # Check if we've used all search queries
        if state.queries_executed >= state.max_search_queries:
            state.research_complete = True
            return state
        
        current_info = state.company_info.model_dump()
        
        system_prompt = f"""You are a research quality assessor. Evaluate if we have sufficient information about the company "{state.company_name}".

Current information gathered:
{json.dumps(current_info, indent=2)}

Research progress:
- Search queries executed: {state.queries_executed}/{state.max_search_queries}
- Search results collected: {len(state.search_results)}
- Reflections done: {state.reflections_done}/{state.max_reflections}

Evaluation criteria:
1. Do we have the company name? (Required)
2. Do we have founding year and/or founders?
3. Do we have a product/service description?
4. Do we have funding information?
5. Do we have notable customers?

Respond with either:
SUFFICIENT: [brief reason why research is complete]
OR
INSUFFICIENT: [specific gaps that need more research]

Focus on the most critical missing information that would make this research valuable.
"""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content="Is the current company information sufficient?")
        ]
        
        response = await self.llm.ainvoke(messages)
        
        state.reflections_done += 1
        
        if response.content.strip().startswith('SUFFICIENT'):
            state.research_complete = True
            state.messages.append({
                "type": "reflection",
                "decision": "sufficient",
                "reason": response.content.strip()[11:].strip(),
                "reflection_number": state.reflections_done
            })
        else:
            state.research_complete = False
            state.messages.append({
                "type": "reflection", 
                "decision": "insufficient",
                "gaps": response.content.strip()[13:].strip(),
                "reflection_number": state.reflections_done
            })
        
        return state