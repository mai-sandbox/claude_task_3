"""
Example usage of the Company Researcher system.

This script demonstrates how to use the company researcher to gather
information about different companies.
"""

import asyncio
import json
from company_researcher import create_company_researcher, BALANCED_RESEARCH_CONFIG


def research_company_example():
    """Example of researching a single company."""
    
    # Create the research system
    researcher = create_company_researcher(
        max_queries=4,  # Limit for demo
        max_results_per_query=2,
        max_reflections=1
    )
    
    # Research a company
    print("🔍 Starting research on OpenAI...")
    
    results = researcher.research_company_sync(
        company_name="OpenAI",
        user_notes="Focus on recent developments and funding"
    )
    
    # Display results
    print("\n📊 Research Results:")
    print("=" * 50)
    
    if results["company_info"]:
        company_info = results["company_info"]
        print(f"Company: {company_info.get('company_name', 'N/A')}")
        print(f"Founded: {company_info.get('founding_year', 'N/A')}")
        print(f"Founders: {', '.join(company_info.get('founder_names', []) or ['N/A'])}")
        print(f"Product: {company_info.get('product_description', 'N/A')[:200]}...")
        print(f"Funding: {company_info.get('funding_summary', 'N/A')[:200]}...")
        print(f"Notable Customers: {company_info.get('notable_customers', 'N/A')[:200]}...")
    
    print(f"\n📈 Metadata:")
    metadata = results["research_metadata"]
    print(f"- Queries executed: {metadata['queries_executed']}")
    print(f"- Results found: {metadata['results_found']}")
    print(f"- Reflections: {metadata['reflections_performed']}")
    print(f"- Complete: {metadata['is_complete']}")
    
    print(f"\n📝 Process Messages:")
    for msg in results["messages"][-5:]:  # Show last 5 messages
        print(f"- [{msg['type']}] {msg['content']}")
    
    return results


def research_company_sync_example():
    """Example of using the synchronous interface."""
    
    print("\n🔄 Testing synchronous interface...")
    
    researcher = create_company_researcher(
        max_queries=3,
        max_results_per_query=2,
        max_reflections=1
    )
    
    # Use synchronous method
    results = researcher.research_company_sync(
        company_name="Anthropic",
        user_notes="Focus on AI safety and recent models"
    )
    
    print(f"✅ Synchronous research completed!")
    print(f"Company researched: {results['company_info']['company_name'] if results['company_info'] else 'N/A'}")
    print(f"Queries executed: {results['research_metadata']['queries_executed']}")
    
    return results


def batch_research_example():
    """Example of researching multiple companies."""
    
    companies = [
        ("Google", "Focus on recent AI developments"),
        ("Microsoft", "Focus on cloud services and AI"),
        ("Tesla", "Focus on autonomous driving and energy")
    ]
    
    researcher = create_company_researcher(
        max_queries=3,
        max_results_per_query=2,
        max_reflections=1
    )
    
    print("\n🚀 Starting batch research...")
    
    # Research companies sequentially
    results = []
    for company, notes in companies:
        try:
            result = researcher.research_company_sync(company, notes)
            results.append(result)
        except Exception as e:
            results.append(e)
    
    print("\n📋 Batch Research Summary:")
    print("=" * 50)
    
    for i, (company_name, _) in enumerate(companies):
        result = results[i]
        if isinstance(result, Exception):
            print(f"❌ {company_name}: Failed - {str(result)}")
        else:
            metadata = result["research_metadata"]
            company_info = result["company_info"]
            name = company_info["company_name"] if company_info else "N/A"
            print(f"✅ {name}: {metadata['queries_executed']} queries, "
                  f"{metadata['results_found']} results")


def configuration_example():
    """Example of using different configurations."""
    
    print("\n⚙️ Configuration Examples:")
    print("=" * 30)
    
    # Quick research configuration
    quick_researcher = create_company_researcher(
        max_queries=2,
        max_results_per_query=2,
        max_reflections=1
    )
    
    # Thorough research configuration  
    thorough_researcher = create_company_researcher(
        max_queries=8,
        max_results_per_query=4,
        max_reflections=3
    )
    
    print("Quick Config: 2 queries, 2 results each, 1 reflection")
    print("Thorough Config: 8 queries, 4 results each, 3 reflections")
    
    return quick_researcher, thorough_researcher


def main():
    """Main example function."""
    print("🏢 Company Researcher Demo")
    print("=" * 40)
    
    try:
        # Single company research
        research_company_example()
        
        # Synchronous interface
        research_company_sync_example()
        
        # Configuration examples
        configuration_example()
        
        # Batch research (commented out to avoid rate limits)
        # batch_research_example()
        
    except Exception as e:
        print(f"❌ Error during demo: {str(e)}")
        print("Make sure your API keys are set in the .env file")


if __name__ == "__main__":
    # Run the demo
    main()