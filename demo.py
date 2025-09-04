#!/usr/bin/env python3
"""
Demo script for the Company Researcher built with LangGraph

This script demonstrates how to use the company researcher to gather
structured information about any company using web search and AI analysis.
"""

from agent import research_company, CompanyInfo
import json
import os

def demo_research():
    """Demonstrate the company researcher with example companies"""
    
    # Check if API keys are available
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY environment variable not set")
    if not os.getenv("TAVILY_API_KEY"):  
        print("⚠️  TAVILY_API_KEY environment variable not set")
    
    print("🔍 Company Researcher Demo")
    print("=" * 50)
    
    # Example companies to research
    test_companies = [
        {
            "name": "Stripe",
            "notes": "Payment processing company"
        },
        {
            "name": "OpenAI", 
            "notes": "AI research company behind ChatGPT"
        }
    ]
    
    for company in test_companies:
        print(f"\n📊 Researching: {company['name']}")
        print(f"📝 Notes: {company['notes']}")
        print("-" * 40)
        
        try:
            # Research the company with limited queries for demo
            company_info = research_company(
                company_name=company["name"],
                user_notes=company["notes"],
                max_search_queries=3,  # Limit for demo
                max_reflection_steps=1  # Single reflection
            )
            
            # Display results
            print(f"🏢 Company: {company_info.company_name}")
            print(f"📅 Founded: {company_info.founding_year or 'Unknown'}")
            print(f"👥 Founders: {', '.join(company_info.founder_names) if company_info.founder_names else 'Unknown'}")
            print(f"🚀 Product: {company_info.product_description or 'Unknown'}")
            print(f"💰 Funding: {company_info.funding_summary or 'Unknown'}")
            print(f"🎯 Customers: {company_info.notable_customers or 'Unknown'}")
            
        except Exception as e:
            print(f"❌ Error researching {company['name']}: {str(e)}")
            
    print("\n" + "=" * 50)
    print("Demo completed!")

def interactive_research():
    """Interactive mode for researching custom companies"""
    
    print("\n🔍 Interactive Company Research")
    print("=" * 50)
    
    while True:
        company_name = input("\nEnter company name (or 'quit' to exit): ").strip()
        if company_name.lower() in ['quit', 'exit', 'q']:
            break
            
        user_notes = input("Enter any notes about the company (optional): ").strip()
        
        try:
            print(f"\n🔄 Researching {company_name}...")
            
            company_info = research_company(
                company_name=company_name,
                user_notes=user_notes,
                max_search_queries=5,
                max_reflection_steps=2
            )
            
            print(f"\n📊 Research Results for {company_name}")
            print("-" * 40)
            print(f"🏢 Name: {company_info.company_name}")
            print(f"📅 Founded: {company_info.founding_year or 'Unknown'}")
            print(f"👥 Founders: {', '.join(company_info.founder_names) if company_info.founder_names else 'Unknown'}")  
            print(f"🚀 Product: {company_info.product_description or 'Unknown'}")
            print(f"💰 Funding: {company_info.funding_summary or 'Unknown'}")
            print(f"🎯 Customers: {company_info.notable_customers or 'Unknown'}")
            
            # Option to save results
            save = input("\nSave results to JSON file? (y/n): ").strip().lower()
            if save == 'y':
                filename = f"{company_name.replace(' ', '_').lower()}_research.json"
                with open(filename, 'w') as f:
                    json.dump(company_info.dict(), f, indent=2)
                print(f"✅ Results saved to {filename}")
                
        except Exception as e:
            print(f"❌ Error researching {company_name}: {str(e)}")

if __name__ == "__main__":
    print("🏢 LangGraph Company Researcher")
    print("Built with multi-node graph, parallel search, and reflection")
    print()
    
    mode = input("Choose mode:\n1. Demo with example companies\n2. Interactive research\nEnter choice (1/2): ").strip()
    
    if mode == "1":
        demo_research()
    elif mode == "2":
        interactive_research()
    else:
        print("Invalid choice. Running demo mode...")
        demo_research()