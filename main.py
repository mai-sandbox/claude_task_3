import asyncio
import json
from typing import Optional
from workflow import CompanyResearchWorkflow
from models import ResearchState

async def main():
    """Main function to run company research"""
    print("🔍 Company Research Assistant")
    print("=" * 50)
    
    # Get user input
    company_name = input("Enter company name: ").strip()
    if not company_name:
        print("Company name is required!")
        return
    
    user_notes = input("Enter optional notes (press Enter to skip): ").strip()
    if not user_notes:
        user_notes = None
    
    # Configuration
    max_queries = int(input("Max search queries (default 5): ") or "5")
    max_search_results = int(input("Max results per query (default 3): ") or "3")
    max_reflections = int(input("Max reflection steps (default 3): ") or "3")
    
    print(f"\n🚀 Starting research for: {company_name}")
    if user_notes:
        print(f"📝 User notes: {user_notes}")
    
    print(f"⚙️  Configuration:")
    print(f"   - Max queries: {max_queries}")
    print(f"   - Max results per query: {max_search_results}")
    print(f"   - Max reflections: {max_reflections}")
    print()
    
    # Initialize workflow
    workflow = CompanyResearchWorkflow()
    
    try:
        # Run research
        result = await workflow.research_company(
            company_name=company_name,
            user_notes=user_notes,
            max_queries=max_queries,
            max_search_results=max_search_results,
            max_reflections=max_reflections
        )
        
        # Display results
        print("✅ Research Complete!")
        print("=" * 50)
        
        if result.company_info:
            print("📊 Company Information:")
            print(json.dumps(result.company_info.model_dump(), indent=2, ensure_ascii=False))
        else:
            print("❌ No company information could be gathered.")
        
        print(f"\n📈 Research Statistics:")
        print(f"   - Search queries executed: {result.query_count}")
        print(f"   - Search results collected: {len(result.search_results)}")
        print(f"   - Reflection iterations: {result.reflection_count}")
        
        print(f"\n📋 Research Log:")
        for i, message in enumerate(result.messages, 1):
            print(f"   {i}. {message}")
            
    except Exception as e:
        print(f"❌ Error during research: {e}")
        import traceback
        traceback.print_exc()

def run_research_interactive():
    """Interactive function to research companies"""
    while True:
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
        
        # Ask if user wants to continue
        continue_research = input("\nWould you like to research another company? (y/n): ").strip().lower()
        if continue_research not in ['y', 'yes']:
            print("👋 Goodbye!")
            break
        print()

async def research_company_api(
    company_name: str,
    user_notes: Optional[str] = None,
    max_queries: int = 5,
    max_search_results: int = 3,
    max_reflections: int = 3
) -> ResearchState:
    """
    API function to research a company programmatically
    
    Args:
        company_name: Name of the company to research
        user_notes: Optional additional context
        max_queries: Maximum number of search queries
        max_search_results: Maximum results per query  
        max_reflections: Maximum reflection iterations
        
    Returns:
        ResearchState with complete research results
    """
    workflow = CompanyResearchWorkflow()
    return await workflow.research_company(
        company_name=company_name,
        user_notes=user_notes,
        max_queries=max_queries,
        max_search_results=max_search_results,
        max_reflections=max_reflections
    )

if __name__ == "__main__":
    run_research_interactive()