#!/usr/bin/env python3
"""
Test script for the CompanyResearcher
"""
import os
import json
from dotenv import load_dotenv
from company_researcher import CompanyResearcher

def test_company_research():
    """Test the company researcher with a real example"""
    load_dotenv()
    
    # Get API keys
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not anthropic_key or not tavily_key:
        print("❌ Missing required API keys!")
        print("Please ensure ANTHROPIC_API_KEY and TAVILY_API_KEY are set in .env")
        return
    
    # Create researcher with smaller limits for testing
    researcher = CompanyResearcher(
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        max_queries=4,  # Smaller for testing
        max_reflections=2,  # Smaller for testing
        max_results_per_query=3  # Smaller for testing
    )
    
    print("🔍 Testing Company Researcher with Anthropic...")
    print("-" * 50)
    
    # Test with a well-known company
    company_name = "Anthropic"
    user_notes = "Focus on Claude AI model and safety research"
    
    try:
        result = researcher.research_company(
            company_name=company_name,
            user_notes=user_notes,
            thread_id="test-anthropic"
        )
        
        print(f"✅ Research completed for {company_name}!")
        print("\n📊 Results:")
        print("=" * 50)
        print(json.dumps(result.model_dump(), indent=2))
        print("=" * 50)
        
        # Test completeness
        fields_filled = 0
        total_fields = 6
        
        if result.company_name: fields_filled += 1
        if result.founding_year: fields_filled += 1
        if result.founder_names: fields_filled += 1
        if result.product_description: fields_filled += 1
        if result.funding_summary: fields_filled += 1
        if result.notable_customers: fields_filled += 1
        
        completeness = (fields_filled / total_fields) * 100
        print(f"\n📈 Completeness: {completeness:.1f}% ({fields_filled}/{total_fields} fields filled)")
        
        if completeness >= 50:
            print("✅ Test PASSED - Good information coverage!")
        else:
            print("⚠️  Test PARTIAL - Some information missing, but system working")
        
    except Exception as e:
        print(f"❌ Test FAILED with error: {e}")
        import traceback
        traceback.print_exc()

def test_small_company():
    """Test with a smaller/newer company to see how it handles limited info"""
    load_dotenv()
    
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not anthropic_key or not tavily_key:
        return
    
    researcher = CompanyResearcher(
        anthropic_api_key=anthropic_key,
        tavily_api_key=tavily_key,
        max_queries=3,
        max_reflections=2,
        max_results_per_query=2
    )
    
    print("\n🔍 Testing with smaller company...")
    print("-" * 50)
    
    try:
        result = researcher.research_company(
            company_name="Replit",
            user_notes="Online coding platform",
            thread_id="test-replit"
        )
        
        print("✅ Small company research completed!")
        print("\n📊 Results:")
        print(json.dumps(result.model_dump(), indent=2))
        
    except Exception as e:
        print(f"❌ Small company test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting Company Researcher Tests")
    print("=" * 50)
    
    test_company_research()
    test_small_company()
    
    print("\n🎉 Tests completed!")