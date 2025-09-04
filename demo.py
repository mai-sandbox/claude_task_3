#!/usr/bin/env python3
"""
Company Research Agent Demo
Demonstrates the multi-node LangGraph workflow for company research
"""

from agent import research_company, CompanyInfo

def main():
    """Run interactive company research demo"""
    
    print("🔍 Company Research Agent - LangGraph Demo")
    print("=" * 60)
    print("This agent uses a multi-node LangGraph workflow to research companies:")
    print("• Query Generation → Parallel Search → Information Extraction → Reflection")
    print("• Supports up to 5 search queries and 2 reflection loops per company")
    print("• Uses Tavily API for web search and Claude for analysis")
    print()
    
    # Demo companies
    demo_companies = [
        {
            "name": "Anthropic",
            "notes": "Focus on AI safety and constitutional AI research"
        },
        {
            "name": "OpenAI",
            "notes": "Research ChatGPT and GPT models, funding history"
        },
        {
            "name": "Scale AI",
            "notes": "Data labeling and AI infrastructure company"
        }
    ]
    
    while True:
        print("\n" + "="*60)
        print("Select an option:")
        print("1. Research Anthropic (pre-configured)")
        print("2. Research OpenAI (pre-configured)")
        print("3. Research Scale AI (pre-configured)")
        print("4. Research custom company")
        print("5. Exit")
        print("="*60)
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "5":
            print("\n👋 Thanks for using the Company Research Agent!")
            break
        
        company_name = None
        notes = None
        
        if choice in ["1", "2", "3"]:
            idx = int(choice) - 1
            company_name = demo_companies[idx]["name"]
            notes = demo_companies[idx]["notes"]
        elif choice == "4":
            company_name = input("\nEnter company name: ").strip()
            notes = input("Enter optional research notes (or press Enter): ").strip()
            if not notes:
                notes = None
        else:
            print("❌ Invalid choice. Please try again.")
            continue
        
        if not company_name:
            print("❌ Company name is required.")
            continue
        
        print(f"\n🔍 Researching: {company_name}")
        if notes:
            print(f"📝 Research focus: {notes}")
        print("-" * 50)
        
        try:
            result = research_company(company_name=company_name, notes=notes)
            
            print("\n📊 RESEARCH RESULTS")
            print("=" * 50)
            print(f"🏢 Company: {result.company_name}")
            print(f"📅 Founded: {result.founding_year or 'Not found'}")
            
            if result.founder_names:
                print(f"👥 Founders: {', '.join(result.founder_names)}")
            else:
                print("👥 Founders: Not found")
            
            print(f"\n📦 Product/Service:")
            print(f"   {result.product_description or 'Not found'}")
            
            print(f"\n💰 Funding Summary:")
            print(f"   {result.funding_summary or 'Not found'}")
            
            print(f"\n🎯 Notable Customers:")
            print(f"   {result.notable_customers or 'Not found'}")
            
            print("\n✅ Research completed successfully!")
            
        except Exception as e:
            print(f"❌ Error during research: {str(e)}")
        
        input("\nPress Enter to continue...")

if __name__ == "__main__":
    main()