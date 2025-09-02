#!/usr/bin/env python3
"""
Example usage of the Company Researcher built with LangGraph

This example demonstrates how to use the company research system to gather
structured information about companies using web search.
"""

import os
import json
from dotenv import load_dotenv
from company_researcher import research_company, CompanyResearcher

# Load environment variables
load_dotenv()

def example_basic_usage():
    """Basic example of researching a company"""
    print("=== Basic Company Research Example ===")
    
    # Research a well-known company
    result = research_company(
        company_name="OpenAI",
        user_notes="Focus on recent developments and funding information",
        max_search_queries=3,
        max_search_results=5
    )
    
    if result["success"]:
        print(f"Research completed successfully!")
        print(f"Company: {result['company_info']['company_name']}")
        print(f"Founding Year: {result['company_info']['founding_year']}")
        print(f"Founders: {result['company_info']['founder_names']}")
        print(f"Product: {result['company_info']['product_description']}")
        print(f"Funding: {result['company_info']['funding_summary']}")
        print(f"Notable Customers: {result['company_info']['notable_customers']}")
        
        metadata = result["research_metadata"]
        print(f"\nResearch Stats:")
        print(f"- Search queries used: {metadata['search_queries_used']}")
        print(f"- Reflection steps: {metadata['reflection_steps_used']}")
        print(f"- Total search results: {metadata['total_search_results']}")
    else:
        print(f"Research failed: {result['error']}")
    
    return result

def example_advanced_usage():
    """Advanced example with custom configuration"""
    print("\n=== Advanced Company Research Example ===")
    
    # Create a researcher instance for more control
    researcher = CompanyResearcher()
    
    # Research with custom parameters
    result = researcher.research_company(
        company_name="Anthropic", 
        user_notes="I want to understand their AI safety focus and constitutional AI approach",
        max_search_queries=4,
        max_search_results=8,
        max_reflection_steps=2
    )
    
    if result["success"]:
        print("Advanced research completed!")
        
        # Pretty print the full company info
        print(json.dumps(result["company_info"], indent=2))
        
        # Show the research process
        print("\nResearch Process:")
        for i, message in enumerate(result["messages"]):
            print(f"{i+1}. [{message['type']}] {message['content']}")
    
    return result

def example_multiple_companies():
    """Example of researching multiple companies"""
    print("\n=== Multiple Companies Research Example ===")
    
    companies = [
        ("SpaceX", "Focus on recent launches and achievements"),
        ("Tesla", "Focus on electric vehicle technology and market position"),
        ("Neuralink", "Focus on brain-computer interface technology")
    ]
    
    results = []
    
    for company_name, notes in companies:
        print(f"\nResearching {company_name}...")
        
        result = research_company(
            company_name=company_name,
            user_notes=notes,
            max_search_queries=2,  # Limit queries for faster processing
            max_search_results=5
        )
        
        if result["success"]:
            print(f"✓ {company_name} research completed")
            company_info = result["company_info"]
            print(f"  Founded: {company_info['founding_year']}")
            print(f"  Product: {company_info['product_description'][:100] if company_info['product_description'] else 'N/A'}...")
        else:
            print(f"✗ {company_name} research failed: {result['error']}")
        
        results.append(result)
    
    return results

def main():
    """Main function to run examples"""
    print("Company Researcher Examples")
    print("===========================")
    
    # Check if required API keys are set
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables")
        print("Please set your OpenAI API key in the .env file or as an environment variable")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Warning: TAVILY_API_KEY not found in environment variables")
        print("Please set your Tavily API key in the .env file or as an environment variable")
        return
    
    try:
        # Run basic example
        basic_result = example_basic_usage()
        
        # Run advanced example
        advanced_result = example_advanced_usage()
        
        # Run multiple companies example
        multiple_results = example_multiple_companies()
        
        print("\n=== All Examples Completed ===")
        print("Check the results above to see the structured company information!")
        
    except Exception as e:
        print(f"Error running examples: {str(e)}")
        print("Make sure you have:")
        print("1. Installed all dependencies: pip install -r requirements.txt")
        print("2. Set your API keys in the .env file")
        print("3. Have a stable internet connection")

if __name__ == "__main__":
    main()