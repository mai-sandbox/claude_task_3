import asyncio
import os
from typing import List, Dict, Any
from tavily import TavilyClient


class TavilySearchClient:
    """Client for interacting with Tavily Search API"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY environment variable is required")
        self.client = TavilyClient(api_key=self.api_key)
    
    def search(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Perform a single search query"""
        try:
            response = self.client.search(
                query=query,
                max_results=max_results,
                include_domains=[],
                exclude_domains=[],
                include_answer=False,
                include_raw_content=False,
                include_images=False
            )
            
            return response.get("results", [])
        except Exception as e:
            print(f"Error searching for '{query}': {str(e)}")
            return []
    
    async def search_parallel(self, queries: List[str], max_results: int = 10) -> List[List[Dict[str, Any]]]:
        """Perform multiple searches in parallel"""
        loop = asyncio.get_event_loop()
        
        # Create tasks for each search query
        tasks = [
            loop.run_in_executor(None, self.search, query, max_results)
            for query in queries
        ]
        
        # Execute all searches in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions and return clean results
        clean_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Error in search for query '{queries[i]}': {str(result)}")
                clean_results.append([])
            else:
                clean_results.append(result)
        
        return clean_results