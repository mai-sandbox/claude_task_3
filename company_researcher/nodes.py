import asyncio
from typing import List, Dict, Any
from langchain_anthropic import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from tavily import TavilyClient
import os
import json

from .models import GraphState, SearchQuery, SearchResult, CompanyInfo, ReflectionResult


class QueryGenerationNode:
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are a research assistant tasked with generating search queries to gather comprehensive information about a company.

Company: {company_name}
User Notes: {user_notes}
Queries Already Executed: {queries_executed}
Maximum Queries Allowed: {max_queries}

Previous Messages: {messages}

Generate {remaining_queries} diverse search queries to find the following information:
1. Company founding information (year, founders)
2. Product/service description
3. Funding history and investment rounds
4. Notable customers or partnerships
5. Recent news and developments
6. Market position and competitors

Return ONLY a JSON list of objects with "query" and "purpose" fields. Each query should be specific and targeted.
Example:
[
    {{"query": "OpenAI founding year founders Sam Altman", "purpose": "Find founding information"}},
    {{"query": "OpenAI ChatGPT product description AI language model", "purpose": "Understand main product"}}
]
""")

    def execute(self, state: GraphState) -> GraphState:
        remaining_queries = min(
            state["max_queries"] - state["queries_executed"],
            state["max_queries"]
        )
        
        if remaining_queries <= 0:
            state["messages"].append({"type": "system", "content": "Maximum queries reached"})
            return state
        
        # Use sync invoke for compatibility
        response = self.llm.invoke(
            self.prompt.format_messages(
                company_name=state["company_name"],
                user_notes=state["user_notes"] or "None",
                queries_executed=state["queries_executed"],
                max_queries=state["max_queries"],
                remaining_queries=remaining_queries,
                messages=state["messages"][-3:] if state["messages"] else []
            )
        )
        
        try:
            queries_data = json.loads(response.content)
            state["generated_queries"] = queries_data[:remaining_queries]
            
            state["messages"].append({
                "type": "query_generation", 
                "content": f"Generated {len(state['generated_queries'])} queries"
            })
        except Exception as e:
            state["messages"].append({"type": "error", "content": f"Query generation failed: {str(e)}"})
        
        return state


class SearchNode:
    def __init__(self, tavily_client: TavilyClient):
        self.tavily_client = tavily_client

    def execute_single_search(self, query: Dict[str, str], max_results: int) -> List[Dict[str, Any]]:
        try:
            response = self.tavily_client.search(
                query["query"], 
                max_results=max_results,
                include_domains=None,
                exclude_domains=["youtube.com", "twitter.com", "facebook.com"]
            )
            
            results = []
            for result in response.get("results", []):
                results.append({
                    "query": query["query"],
                    "content": result.get("content", ""),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.0)
                })
            return results
        except Exception as e:
            print(f"Search failed for query '{query['query']}': {str(e)}")
            return []

    def execute(self, state: GraphState) -> GraphState:
        if not state["generated_queries"]:
            state["messages"].append({"type": "error", "content": "No queries to execute"})
            return state
        
        # Execute searches sequentially to avoid rate limits
        all_results = []
        for query in state["generated_queries"]:
            results = self.execute_single_search(query, state["max_results_per_query"])
            all_results.extend(results)
        
        state["search_results"].extend(all_results)
        state["queries_executed"] += len(state["generated_queries"])
        state["generated_queries"] = []  # Clear processed queries
        
        state["messages"].append({
            "type": "search_execution",
            "content": f"Executed searches, found {len(all_results)} results"
        })
        
        return state


class InformationExtractionNode:
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are an expert information extractor. Analyze the search results and extract structured information about the company.

Company: {company_name}
User Notes: {user_notes}

Search Results:
{search_results}

Current Extracted Information:
{current_info}

Extract and update the following information in valid JSON format. Return ONLY the JSON, no additional text:

{{
    "company_name": "string",
    "founding_year": null,
    "founder_names": null,
    "product_description": "string or null",
    "funding_summary": "string or null",
    "notable_customers": "string or null"
}}

IMPORTANT RULES:
1. Return ONLY valid JSON - no markdown, no explanations, no additional text
2. Use null (not "null" string) for missing information  
3. For founder_names, use an array like ["Name1", "Name2"] or null
4. For founding_year, use a number like 2023 or null
5. Only include information explicitly mentioned in the search results
6. Be concise but comprehensive in descriptions
""")

    def execute(self, state: GraphState) -> GraphState:
        if not state["search_results"]:
            state["messages"].append({"type": "error", "content": "No search results to process"})
            return state
        
        # Format search results for the prompt
        formatted_results = []
        for result in state["search_results"][-20:]:  # Use latest 20 results to avoid token limits
            formatted_results.append(f"Query: {result['query']}\nContent: {result['content'][:500]}...\nURL: {result['url']}\n")
        
        current_info = state["extracted_info"] if state["extracted_info"] else {}
        
        response = self.llm.invoke(
            self.prompt.format_messages(
                company_name=state["company_name"],
                user_notes=state["user_notes"] or "None",
                search_results="\n---\n".join(formatted_results),
                current_info=current_info
            )
        )
        
        try:
            # Try to extract JSON from the response
            content = response.content.strip()
            
            # Handle cases where LLM adds markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            extracted_data = json.loads(content)
            state["extracted_info"] = extracted_data
            
            state["messages"].append({
                "type": "information_extraction",
                "content": f"Updated company information for {extracted_data.get('company_name', 'Unknown')}"
            })
        except Exception as e:
            state["messages"].append({"type": "error", "content": f"Information extraction failed: {str(e)}. Response: {response.content[:200]}"})
        
        return state


class ReflectionNode:
    def __init__(self, llm: ChatAnthropic):
        self.llm = llm
        self.prompt = ChatPromptTemplate.from_template("""
You are a quality assessment expert. Evaluate the completeness and quality of the extracted company information.

Company: {company_name}
Extracted Information:
{extracted_info}

Messages History:
{messages}

Reflection Count: {reflection_count}
Max Reflections: {max_reflections}

Assess the information quality and completeness. Return ONLY valid JSON, no additional text:

{{
    "is_sufficient": true,
    "missing_info": [],
    "confidence_score": 0.8,
    "suggested_queries": []
}}

IMPORTANT RULES:
1. Return ONLY valid JSON - no markdown, no explanations
2. Use boolean values: true/false (not strings)
3. Use numbers for confidence_score (0.0-1.0)
4. missing_info should list missing fields like ["founding_year", "founder_names"]
5. suggested_queries should be specific search strings

Criteria for sufficiency:
- Company name: Required
- At least 3 of the other 5 fields should have meaningful information
- Product description should be comprehensive
- Information should be recent and accurate

If information is insufficient and we haven't reached max reflections, suggest specific queries to fill gaps.
""")

    def execute(self, state: GraphState) -> GraphState:
        if not state["extracted_info"]:
            state["messages"].append({"type": "error", "content": "No information to reflect on"})
            return state
        
        response = self.llm.invoke(
            self.prompt.format_messages(
                company_name=state["company_name"],
                extracted_info=state["extracted_info"],
                messages=state["messages"][-5:],
                reflection_count=state["reflection_count"],
                max_reflections=state["max_reflections"]
            )
        )
        
        try:
            # Try to extract JSON from the response
            content = response.content.strip()
            
            # Handle cases where LLM adds markdown formatting
            if content.startswith("```json"):
                content = content.replace("```json", "").replace("```", "").strip()
            elif content.startswith("```"):
                content = content.replace("```", "").strip()
            
            reflection_data = json.loads(content)
            
            state["reflection_count"] += 1
            
            if reflection_data["is_sufficient"] or state["reflection_count"] >= state["max_reflections"]:
                state["is_complete"] = True
                state["messages"].append({
                    "type": "reflection_complete",
                    "content": f"Research complete. Confidence: {reflection_data['confidence_score']:.2f}"
                })
            else:
                # Generate new queries based on reflection
                suggested_queries = [
                    {"query": query, "purpose": "Fill information gaps"}
                    for query in reflection_data["suggested_queries"][:2]  # Limit to 2 additional queries
                ]
                state["generated_queries"] = suggested_queries
                
                state["messages"].append({
                    "type": "reflection_continue",
                    "content": f"Need more information. Missing: {', '.join(reflection_data['missing_info'])}"
                })
        
        except Exception as e:
            state["messages"].append({"type": "error", "content": f"Reflection failed: {str(e)}. Response: {response.content[:200]}"})
            state["is_complete"] = True  # Fail safe
        
        return state