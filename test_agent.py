#!/usr/bin/env python3
"""
Test script for the company researcher agent
"""

import os
from agent import app, CompanyResearchState
from langchain_core.messages import HumanMessage

def test_company_researcher():
    """Test the company researcher with a sample company"""
    
    # Test input
    test_state = CompanyResearchState(
        company_name="OpenAI",
        notes="Focus on recent developments and business model",
        messages=[HumanMessage(content="Research OpenAI company information")]
    )
    
    print(f"🔍 Starting research for: {test_state['company_name']}")
    print(f"📝 Notes: {test_state.get('notes', 'None')}")
    print("-" * 50)
    
    # Run the research workflow
    try:
        result = app.invoke(test_state)
        
        print("✅ Research completed!")
        print(f"📊 Queries executed: {result['queries_executed']}")
        print(f"🔄 Reflections: {result['reflection_count']}")
        print(f"📋 Search results: {len(result['search_results'])}")
        
        if result.get('company_info'):
            company_info = result['company_info']
            print("\n🏢 Company Information:")
            print(f"  Name: {company_info.company_name}")
            print(f"  Founded: {company_info.founding_year}")
            print(f"  Founders: {company_info.founder_names}")
            print(f"  Product: {company_info.product_description}")
            print(f"  Funding: {company_info.funding_summary}")
            print(f"  Customers: {company_info.notable_customers}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        return False

if __name__ == "__main__":
    # Check for required API keys
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("⚠️  ANTHROPIC_API_KEY not found in environment")
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  TAVILY_API_KEY not found in environment")
        print("ℹ️  You can get a Tavily API key from: https://tavily.com/")
    
    print("🧪 Testing Company Researcher Agent")
    success = test_company_researcher()
    
    if success:
        print("\n✅ Test completed successfully!")
    else:
        print("\n❌ Test failed!")