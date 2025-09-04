#!/usr/bin/env python3
"""
Example usage of the Company Researcher agent.
Demonstrates how to research a company and get structured information.
"""

from langchain_core.messages import HumanMessage
from agent import app

def research_company(company_name: str, user_notes: str = "") -> dict:
    """
    Research a company using the LangGraph-based researcher.
    
    Args:
        company_name: Name of the company to research
        user_notes: Optional additional context or focus areas
        
    Returns:
        Dictionary containing the research results
    """
    
    # Define the initial state for the research workflow
    initial_state = {
        "messages": [HumanMessage(content=f"Research {company_name}")],
        "company_name": company_name,
        "user_notes": user_notes,
        "search_queries": [],
        "search_results": [],
        "company_info": None,
        "reflection_count": 0,
        "max_queries_per_iteration": 3,  # Max queries per search iteration
        "max_results_per_query": 5,      # Max results per query
        "max_reflections": 2             # Max reflection/retry cycles
    }
    
    # Invoke the research workflow
    result = app.invoke(initial_state)
    
    return result

def main():
    # Example 1: Basic company research
    print("Example 1: Researching OpenAI")
    print("-" * 40)
    
    result = research_company("OpenAI", "Focus on their AI models and recent developments")
    
    if result["company_info"]:
        info = result["company_info"]
        print(f"Company: {info.company_name}")
        print(f"Founded: {info.founding_year}")
        print(f"Founders: {info.founder_names}")
        print(f"Product: {info.product_description}")
        print(f"Funding: {info.funding_summary}")
        print(f"Customers: {info.notable_customers}")
    
    print(f"\nResearch completed with {result['reflection_count']} reflection steps")
    print(f"Total search results: {len(result['search_results'])}")
    
    # Example 2: Research with specific focus
    print("\n\nExample 2: Researching Stripe with payment focus")
    print("-" * 50)
    
    result2 = research_company(
        "Stripe", 
        "Focus on payment processing capabilities and developer tools"
    )
    
    if result2["company_info"]:
        info2 = result2["company_info"]
        print(f"Company: {info2.company_name}")
        print(f"Product: {info2.product_description}")
        print(f"Notable Customers: {info2.notable_customers}")

if __name__ == "__main__":
    main()