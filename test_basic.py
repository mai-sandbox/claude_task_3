#!/usr/bin/env python3
"""
Basic test script for the Company Researcher
"""

import asyncio
import sys
from company_researcher import research_company
from config import validate_environment


async def test_basic_research():
    """Test basic research functionality"""
    
    # Validate environment
    env_check = validate_environment()
    if not env_check['valid']:
        print(f"❌ Environment validation failed: {env_check['message']}")
        return False
    
    print("✅ Environment validation passed")
    
    try:
        print("🔍 Testing basic research with a quick search...")
        
        # Test with a well-known company
        results = await research_company(
            company_name="Google",
            user_notes="Quick test - focus on core information",
            max_search_queries=3,  # Limit for quick test
            max_reflections=1
        )
        
        # Check results
        company_info = results.get('company_info', {})
        research_summary = results.get('research_summary', {})
        
        print(f"✅ Research completed successfully!")
        print(f"   Company: {company_info.get('company_name', 'N/A')}")
        print(f"   Queries executed: {research_summary.get('queries_executed', 0)}")
        print(f"   Results collected: {research_summary.get('results_collected', 0)}")
        print(f"   Research complete: {research_summary.get('research_complete', False)}")
        
        # Basic validation
        if company_info.get('company_name'):
            print("✅ Company name extracted successfully")
        else:
            print("⚠️  No company name extracted")
        
        if research_summary.get('queries_executed', 0) > 0:
            print("✅ Search queries executed successfully")
        else:
            print("❌ No search queries executed")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function"""
    print("🧪 Company Researcher Basic Test")
    print("=" * 40)
    
    try:
        success = asyncio.run(test_basic_research())
        
        if success:
            print("\n✅ All tests passed! The company researcher is working correctly.")
            sys.exit(0)
        else:
            print("\n❌ Tests failed. Please check the configuration and API keys.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n❌ Test cancelled by user")
        sys.exit(1)


if __name__ == "__main__":
    main()