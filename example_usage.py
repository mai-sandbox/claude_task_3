#!/usr/bin/env python3
"""
Example usage of the Company Researcher

This script demonstrates different ways to use the company researcher:
1. Basic research
2. Research with user notes
3. Custom configuration
4. Batch research for multiple companies
"""

import asyncio
import json
from typing import List, Dict, Any

from company_researcher import CompanyResearcher, research_company
from config import DefaultConfigs, validate_environment


async def example_basic_research():
    """Example 1: Basic company research"""
    print("\n🔍 Example 1: Basic Research")
    print("-" * 40)
    
    results = await research_company(
        company_name="Stripe",
        user_notes="Focus on payment processing and recent expansion"
    )
    
    # Print results
    researcher = CompanyResearcher()
    researcher.print_results(results)
    
    return results


async def example_custom_config():
    """Example 2: Research with custom configuration"""
    print("\n🔍 Example 2: Custom Configuration (Quick Research)")
    print("-" * 40)
    
    # Use quick research configuration
    config = DefaultConfigs.quick_research()
    
    researcher = CompanyResearcher(
        max_search_queries=config.max_search_queries,
        max_search_results=config.max_search_results,
        max_reflections=config.max_reflections
    )
    
    results = await researcher.research_company(
        company_name="Anthropic",
        user_notes="AI safety company, focus on Claude"
    )
    
    researcher.print_results(results)
    return results


async def example_thorough_research():
    """Example 3: Thorough research with more queries"""
    print("\n🔍 Example 3: Thorough Research")
    print("-" * 40)
    
    config = DefaultConfigs.thorough_research()
    
    researcher = CompanyResearcher(
        max_search_queries=config.max_search_queries,
        max_search_results=config.max_search_results,
        max_reflections=config.max_reflections
    )
    
    results = await researcher.research_company(
        company_name="Databricks",
        user_notes="Data and AI platform, recent IPO rumors"
    )
    
    researcher.print_results(results)
    return results


async def example_batch_research():
    """Example 4: Research multiple companies"""
    print("\n🔍 Example 4: Batch Research")
    print("-" * 40)
    
    companies = [
        {"name": "Notion", "notes": "Productivity and note-taking platform"},
        {"name": "Figma", "notes": "Design collaboration tool, acquired by Adobe"},
        {"name": "Linear", "notes": "Project management for software teams"}
    ]
    
    results = {}
    
    # Use quick config for batch processing
    config = DefaultConfigs.quick_research()
    researcher = CompanyResearcher(
        max_search_queries=config.max_search_queries,
        max_search_results=config.max_search_results,
        max_reflections=config.max_reflections
    )
    
    for company in companies:
        print(f"\n🔍 Researching {company['name']}...")
        
        result = await researcher.research_company(
            company_name=company['name'],
            user_notes=company['notes']
        )
        
        results[company['name']] = result
        
        # Brief summary
        info = result['company_info']
        print(f"✅ {company['name']}: {info.get('product_description', 'N/A')[:100]}...")
    
    return results


async def interactive_research():
    """Interactive research - ask user for company name"""
    print("\n🔍 Interactive Research")
    print("-" * 40)
    
    try:
        company_name = input("Enter company name to research: ").strip()
        if not company_name:
            print("❌ No company name provided")
            return None
        
        user_notes = input("Enter any specific notes (optional): ").strip()
        if not user_notes:
            user_notes = None
        
        print(f"\n🚀 Starting research for: {company_name}")
        
        results = await research_company(
            company_name=company_name,
            user_notes=user_notes
        )
        
        researcher = CompanyResearcher()
        researcher.print_results(results)
        
        # Ask if user wants to save results
        save = input("\nSave results to file? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"{company_name.lower().replace(' ', '_')}_research.json"
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"💾 Results saved to: {filename}")
        
        return results
        
    except KeyboardInterrupt:
        print("\n❌ Research cancelled by user")
        return None


def save_results(results: Dict[str, Any], filename: str):
    """Save research results to JSON file"""
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"💾 Results saved to: {filename}")


async def main():
    """Main function to run examples"""
    
    print("🚀 Company Researcher Examples")
    print("=" * 50)
    
    # Validate environment
    env_check = validate_environment()
    if not env_check['valid']:
        print(f"❌ {env_check['message']}")
        print("Please set the required environment variables:")
        for var in env_check['missing_variables']:
            print(f"   - {var}")
        return
    
    print("✅ Environment validated")
    
    # Menu
    print("\nChoose an example to run:")
    print("1. Basic Research (Stripe)")
    print("2. Custom Config - Quick Research (Anthropic)")  
    print("3. Thorough Research (Databricks)")
    print("4. Batch Research (Multiple companies)")
    print("5. Interactive Research (Your choice)")
    print("6. Run all examples")
    
    try:
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == "1":
            results = await example_basic_research()
            save_results(results, "stripe_research.json")
            
        elif choice == "2":
            results = await example_custom_config()
            save_results(results, "anthropic_research.json")
            
        elif choice == "3":
            results = await example_thorough_research()
            save_results(results, "databricks_research.json")
            
        elif choice == "4":
            results = await example_batch_research()
            save_results(results, "batch_research.json")
            
        elif choice == "5":
            results = await interactive_research()
            
        elif choice == "6":
            print("\n🚀 Running all examples...")
            
            results1 = await example_basic_research()
            save_results(results1, "stripe_research.json")
            
            results2 = await example_custom_config()
            save_results(results2, "anthropic_research.json")
            
            results3 = await example_batch_research()
            save_results(results3, "batch_research.json")
            
            print("\n✅ All examples completed!")
            
        else:
            print("❌ Invalid choice")
            
    except KeyboardInterrupt:
        print("\n❌ Cancelled by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())