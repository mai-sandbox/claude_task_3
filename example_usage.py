#!/usr/bin/env python3
"""
Example usage script for the LangGraph Company Researcher

This script demonstrates how to use the CompanyResearcher class to research companies.
Make sure to set your API keys in environment variables:
- TAVILY_API_KEY: Your Tavily API key for web searches
- OPENAI_API_KEY: Your OpenAI API key for LLM calls
"""

import asyncio
import os
from dotenv import load_dotenv
from company_researcher import CompanyResearcher

# Load environment variables from .env file
load_dotenv()


async def research_single_company():
    """Example: Research a single company"""
    print("=== Single Company Research Example ===")
    
    # Initialize the researcher with custom settings
    researcher = CompanyResearcher(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_queries=5,              # Max 5 search queries per company
        max_results_per_query=3,    # Max 3 results per search query
        max_reflections=2           # Max 2 reflection rounds
    )
    
    # Research a company with optional user notes
    company_info = await researcher.research_company(
        company_name="OpenAI",
        user_notes="Focus on ChatGPT, GPT models, and recent developments"
    )
    
    print(f"\nResearch Results for {company_info.company_name}:")
    print("=" * 50)
    print(f"Founding Year: {company_info.founding_year}")
    print(f"Founders: {', '.join(company_info.founder_names) if company_info.founder_names else 'Not found'}")
    print(f"Product Description: {company_info.product_description}")
    print(f"Funding Summary: {company_info.funding_summary}")
    print(f"Notable Customers: {company_info.notable_customers}")


async def research_multiple_companies():
    """Example: Research multiple companies in batch"""
    print("\n\n=== Multiple Companies Research Example ===")
    
    companies_to_research = [
        {"name": "Anthropic", "notes": "Focus on Claude AI and AI safety"},
        {"name": "Stripe", "notes": "Payment processing and fintech"},
        {"name": "Notion", "notes": "Productivity and note-taking software"}
    ]
    
    # Initialize researcher with lower limits for faster batch processing
    researcher = CompanyResearcher(
        tavily_api_key=os.getenv("TAVILY_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        max_queries=3,              # Reduced for batch processing
        max_results_per_query=2,    # Reduced for batch processing
        max_reflections=1           # Reduced for batch processing
    )
    
    # Research companies in parallel
    research_tasks = []
    for company in companies_to_research:
        task = researcher.research_company(
            company_name=company["name"],
            user_notes=company["notes"]
        )
        research_tasks.append(task)
    
    # Wait for all research to complete
    results = await asyncio.gather(*research_tasks)
    
    # Display results
    for i, company_info in enumerate(results):
        print(f"\n--- {company_info.company_name} ---")
        print(f"Founded: {company_info.founding_year or 'Unknown'}")
        print(f"Product: {company_info.product_description or 'Not found'}")
        print(f"Funding: {company_info.funding_summary or 'Not found'}")


async def main():
    """Main function to run examples"""
    print("Company Researcher - Example Usage")
    print("=" * 40)
    
    # Check if API keys are set
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY environment variable not set!")
        print("Please set your Tavily API key in the .env file or environment variables")
        return
    
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY environment variable not set!")
        print("Please set your OpenAI API key in the .env file or environment variables")
        return
    
    try:
        # Run single company research
        await research_single_company()
        
        # Run multiple companies research
        await research_multiple_companies()
        
        print("\n\n✅ All research completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error during research: {e}")
        print("Please check your API keys and internet connection")


if __name__ == "__main__":
    asyncio.run(main())