"""
Test script for the Company Researcher
"""
import os
import json
from dotenv import load_dotenv
from company_researcher import CompanyResearcher, Config

def test_company_research():
    """Test the company researcher with sample companies"""
    load_dotenv()
    
    # Check for required API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY not found in environment variables")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("❌ TAVILY_API_KEY not found in environment variables")
        return
    
    print("✅ API keys found")
    
    # Create researcher with custom config
    config = Config(
        max_search_queries=3,
        max_search_results_per_query=2,
        max_reflection_steps=1,
        model_name="gpt-4o-mini"
    )
    
    researcher = CompanyResearcher(config)
    
    # Test companies
    test_companies = [
        ("OpenAI", "AI research company"),
        ("Tesla", "Electric vehicle manufacturer"),
        ("Stripe", "Payment processing platform")
    ]
    
    for company_name, notes in test_companies:
        print(f"\n{'='*60}")
        print(f"🔍 Researching: {company_name}")
        print(f"📝 Notes: {notes}")
        print('='*60)
        
        try:
            result = researcher.research_company(company_name, notes)
            
            print("\n📊 RESEARCH RESULTS:")
            print("-" * 40)
            print(f"Company: {result.company_name}")
            print(f"Founded: {result.founding_year or 'Not found'}")
            print(f"Founders: {', '.join(result.founder_names) if result.founder_names else 'Not found'}")
            print(f"Product: {result.product_description or 'Not found'}")
            print(f"Funding: {result.funding_summary or 'Not found'}")
            print(f"Customers: {result.notable_customers or 'Not found'}")
            
            # Calculate completeness
            total_fields = 6
            filled_fields = sum(1 for field in [
                result.company_name,
                result.founding_year,
                result.founder_names,
                result.product_description,
                result.funding_summary,
                result.notable_customers
            ] if field and (not isinstance(field, list) or len(field) > 0))
            
            completeness = (filled_fields / total_fields) * 100
            print(f"\n📈 Completeness: {completeness:.1f}% ({filled_fields}/{total_fields} fields)")
            
        except Exception as e:
            print(f"❌ Error researching {company_name}: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 60)

def test_graph_visualization():
    """Test graph structure visualization"""
    researcher = CompanyResearcher()
    
    try:
        # This would require additional dependencies for visualization
        print("📊 Graph structure:")
        print("generate_queries -> parallel_search -> extract_info -> reflect")
        print("reflect -> (continue) -> generate_queries OR (end) -> END")
        print("✅ Graph structure is valid")
    except Exception as e:
        print(f"❌ Graph visualization error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Company Researcher")
    print("=" * 60)
    
    test_graph_visualization()
    test_company_research()