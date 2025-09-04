#!/usr/bin/env python3
"""
Test script for the company researcher agent.
This script demonstrates how to use the company researcher with different companies.
"""

import os
import asyncio
from langchain_core.messages import HumanMessage
from agent import app, CompanyInfo

def test_company_researcher(company_name: str, user_notes: str = ""):
    """Test the company researcher with a given company"""
    print(f"\n{'='*60}")
    print(f"RESEARCHING: {company_name}")
    print(f"{'='*60}")
    
    initial_state = {
        "messages": [HumanMessage(content=f"Research {company_name}")],
        "company_name": company_name,
        "user_notes": user_notes,
        "search_queries": [],
        "search_results": [],
        "company_info": None,
        "reflection_count": 0,
        "max_queries_per_iteration": 3,
        "max_results_per_query": 5,
        "max_reflections": 2
    }
    
    try:
        print("Starting research...")
        result = app.invoke(initial_state)
        
        print("\n📊 FINAL RESULTS:")
        print("-" * 40)
        
        if result["company_info"]:
            info = result["company_info"]
            print(f"Company Name: {info.company_name}")
            print(f"Founded: {info.founding_year or 'Not found'}")
            print(f"Founders: {', '.join(info.founder_names) if info.founder_names else 'Not found'}")
            print(f"Product: {info.product_description or 'Not found'}")
            print(f"Funding: {info.funding_summary or 'Not found'}")
            print(f"Notable Customers: {info.notable_customers or 'Not found'}")
        else:
            print("❌ No company information extracted")
        
        print(f"\n🔄 Reflection steps completed: {result['reflection_count']}")
        print(f"📈 Total search results gathered: {len(result['search_results'])}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during research: {e}")
        return None

def main():
    """Main test function"""
    # Check for required environment variables
    required_vars = ["ANTHROPIC_API_KEY", "TAVILY_API_KEY"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Please set these in your .env file or environment")
        print("   See .env.example for the required format")
        return
    
    print("🚀 Company Researcher Test Suite")
    print("================================")
    
    # Test cases
    test_cases = [
        {
            "company_name": "Anthropic",
            "user_notes": "Focus on AI safety and Claude models"
        },
        {
            "company_name": "Stripe", 
            "user_notes": "Payment processing company"
        },
        {
            "company_name": "Figma",
            "user_notes": "Design and collaboration tools"
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test Case {i}/{len(test_cases)}")
        result = test_company_researcher(
            test_case["company_name"], 
            test_case["user_notes"]
        )
        results.append(result)
        
        # Add a brief pause between tests
        if i < len(test_cases):
            print("\nWaiting 2 seconds before next test...")
            asyncio.sleep(2)
    
    print(f"\n✅ Testing completed! Processed {len([r for r in results if r is not None])} companies successfully.")

if __name__ == "__main__":
    main()