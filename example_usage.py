"""
Example usage of the Company Researcher built with LangGraph.

This example shows how to use the CompanyResearcher to gather information
about a company using the Tavily API for web searches.
"""

import asyncio
import json
import os
from company_researcher import CompanyResearcher

async def research_example_companies():
    """Example function to research multiple companies."""
    
    # Check for required API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or not tavily_key:
        print("Error: Please set OPENAI_API_KEY and TAVILY_API_KEY environment variables")
        return
    
    # Initialize the researcher
    researcher = CompanyResearcher(
        openai_api_key=openai_key,
        tavily_api_key=tavily_key,
        max_search_queries=4,  # Limit queries for demo
        max_search_results=2,  # Limit results per query
        max_reflections=1      # Limit reflection steps
    )
    
    # Example companies to research
    companies = [
        {
            "name": "OpenAI",
            "notes": "AI research company, created ChatGPT and GPT models"
        },
        {
            "name": "Stripe", 
            "notes": "Payment processing platform for online businesses"
        },
        {
            "name": "Figma",
            "notes": "Collaborative design tool, web-based interface design"
        }
    ]
    
    print("🔍 Starting Company Research with LangGraph\n")
    print("=" * 60)
    
    for i, company in enumerate(companies, 1):
        print(f"\n📊 Researching Company {i}/{len(companies)}: {company['name']}")
        print("-" * 40)
        
        try:
            # Research the company
            result = await researcher.research_company(
                company_name=company["name"],
                notes=company["notes"]
            )
            
            # Display results
            print_company_info(result)
            
        except Exception as e:
            print(f"❌ Error researching {company['name']}: {e}")
    
    print("\n" + "=" * 60)
    print("✅ Research complete!")

def print_company_info(company_info):
    """Pretty print company information."""
    
    print(f"\n🏢 Company: {company_info.get('company_name', 'Unknown')}")
    
    if company_info.get('founding_year'):
        print(f"📅 Founded: {company_info['founding_year']}")
    
    if company_info.get('founder_names'):
        founders = ", ".join(company_info['founder_names'])
        print(f"👥 Founders: {founders}")
    
    if company_info.get('product_description'):
        print(f"🛠️  Products: {company_info['product_description']}")
    
    if company_info.get('funding_summary'):
        print(f"💰 Funding: {company_info['funding_summary']}")
    
    if company_info.get('notable_customers'):
        print(f"🎯 Customers: {company_info['notable_customers']}")
    
    # Metadata
    if company_info.get('_metadata'):
        meta = company_info['_metadata']
        print(f"\n📈 Research Stats:")
        print(f"   • Queries used: {meta.get('queries_used', 0)}")
        print(f"   • Results processed: {meta.get('results_processed', 0)}")
        print(f"   • Reflections: {meta.get('reflection_count', 0)}")
        print(f"   • Completeness: {meta.get('completeness_score', 0)}/5")

async def research_single_company():
    """Research a single company with detailed output."""
    
    # Check for API keys
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key or not tavily_key:
        print("Error: Please set OPENAI_API_KEY and TAVILY_API_KEY environment variables")
        return
    
    # Initialize researcher
    researcher = CompanyResearcher(
        openai_api_key=openai_key,
        tavily_api_key=tavily_key,
        max_search_queries=5,
        max_search_results=3,
        max_reflections=2
    )
    
    # Get company name from user input
    company_name = input("\nEnter company name to research: ").strip()
    notes = input("Enter any additional notes (optional): ").strip() or None
    
    print(f"\n🔍 Researching {company_name}...")
    print("-" * 50)
    
    try:
        # Research the company
        result = await researcher.research_company(
            company_name=company_name,
            notes=notes
        )
        
        # Save results to file
        timestamp = result.get('_metadata', {}).get('timestamp', 'unknown')
        filename = f"research_{company_name.replace(' ', '_').lower()}_{timestamp[:10]}.json"
        
        with open(filename, 'w') as f:
            json.dump(result, f, indent=2)
        
        print_company_info(result)
        print(f"\n💾 Results saved to: {filename}")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("Company Researcher - LangGraph Demo")
    print("=" * 40)
    print("1. Research example companies")
    print("2. Research single company")
    
    choice = input("\nSelect option (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(research_example_companies())
    elif choice == "2":
        asyncio.run(research_single_company())
    else:
        print("Invalid choice. Please run again and select 1 or 2.")