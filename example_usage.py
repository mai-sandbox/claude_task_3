import asyncio
import os
import json
from company_researcher import CompanyResearcher


async def demo_research():
    # Load API keys from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_api_key:
        print("❌ OPENAI_API_KEY environment variable not set")
        return
    
    if not tavily_api_key:
        print("❌ TAVILY_API_KEY environment variable not set")
        return
    
    # Initialize the researcher
    researcher = CompanyResearcher(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        max_queries=4,           # Maximum search queries per company
        max_results_per_query=3, # Maximum results per search query
        max_reflections=2        # Maximum reflection steps
    )
    
    # Example companies to research
    companies = [
        {
            "name": "Anthropic",
            "notes": "AI safety company that created Claude"
        },
        {
            "name": "OpenAI",
            "notes": "Created ChatGPT and GPT models"
        },
        {
            "name": "Stripe",
            "notes": "Payment processing company"
        }
    ]
    
    for company in companies:
        print(f"\n🔍 Researching {company['name']}...")
        print("=" * 60)
        
        try:
            result = await researcher.research_company(
                company_name=company["name"],
                notes=company["notes"],
                thread_id=f"research_{company['name'].lower()}"
            )
            
            print("📊 Company Information:")
            print(json.dumps(result["company_info"], indent=2))
            
            print("\n📈 Research Summary:")
            summary = result["research_summary"]
            print(f"  • Queries executed: {summary['queries_executed']}")
            print(f"  • Reflection steps: {summary['reflection_count']}")
            print(f"  • Results found: {summary['total_results_found']}")
            print(f"  • Research completed: {summary['completed']}")
            
            # Show research process (optional)
            if os.getenv("SHOW_RESEARCH_PROCESS"):
                print("\n📝 Research Process:")
                for i, message in enumerate(result["messages"], 1):
                    print(f"  {i}. [{message['role']}] {message['content']}")
                    
        except Exception as e:
            print(f"❌ Error researching {company['name']}: {e}")
        
        print("\n" + "-" * 60)


async def single_company_demo():
    """Demo researching a single company with detailed output"""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_api_key or not tavily_api_key:
        print("Please set OPENAI_API_KEY and TAVILY_API_KEY environment variables")
        return
    
    researcher = CompanyResearcher(
        openai_api_key=openai_api_key,
        tavily_api_key=tavily_api_key,
        max_queries=3,
        max_results_per_query=2,
        max_reflections=1
    )
    
    company_name = input("Enter company name to research: ").strip()
    notes = input("Enter any additional notes (optional): ").strip()
    
    if not company_name:
        print("Company name is required!")
        return
    
    print(f"\n🔍 Starting research for: {company_name}")
    print("Please wait while I gather information...")
    
    result = await researcher.research_company(
        company_name=company_name,
        notes=notes,
        thread_id=f"interactive_{company_name.lower().replace(' ', '_')}"
    )
    
    print("\n" + "=" * 80)
    print(f"📋 RESEARCH RESULTS FOR: {company_name.upper()}")
    print("=" * 80)
    
    info = result["company_info"]
    for field, value in info.items():
        if value:
            field_display = field.replace('_', ' ').title()
            if isinstance(value, list):
                value_display = ", ".join(str(v) for v in value)
            else:
                value_display = str(value)
            print(f"• {field_display}: {value_display}")
    
    print(f"\n📊 Research completed using {result['research_summary']['queries_executed']} queries")


if __name__ == "__main__":
    print("Company Researcher Demo")
    print("=" * 30)
    print("1. Demo multiple companies")
    print("2. Research single company interactively")
    
    choice = input("\nChoose option (1 or 2): ").strip()
    
    if choice == "1":
        asyncio.run(demo_research())
    elif choice == "2":
        asyncio.run(single_company_demo())
    else:
        print("Invalid choice. Running multiple companies demo...")
        asyncio.run(demo_research())