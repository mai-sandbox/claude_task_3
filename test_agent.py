#!/usr/bin/env python3
"""
Test script for the Company Research Agent
"""

from agent import research_company, CompanyInfo

def test_company_research():
    """Test the company research functionality"""
    
    print("🔍 Testing Company Research Agent")
    print("=" * 50)
    
    # Test case 1: Well-known tech company
    print("\n📊 Researching: Anthropic")
    print("-" * 30)
    
    try:
        result = research_company(
            company_name="Anthropic",
            notes="Focus on AI safety and constitutional AI research"
        )
        
        print(f"Company: {result.company_name}")
        print(f"Founded: {result.founding_year or 'Not found'}")
        print(f"Founders: {', '.join(result.founder_names) if result.founder_names else 'Not found'}")
        print(f"Product: {result.product_description or 'Not found'}")
        print(f"Funding: {result.funding_summary or 'Not found'}")
        print(f"Customers: {result.notable_customers or 'Not found'}")
        
        print("\n✅ Research completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    test_company_research()