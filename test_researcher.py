"""
Test script for the Company Researcher.

This script runs basic tests to ensure the system is working correctly.
"""

import asyncio
import json
import os
import sys
from unittest.mock import Mock, patch

def test_imports():
    """Test that all modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        from company_researcher import CompanyResearcher, CompanyInfo, ResearchState
        from config import ResearchConfig, DEFAULT_CONFIG
        print("✅ All imports successful")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config():
    """Test configuration validation."""
    print("🧪 Testing configuration...")
    
    try:
        from config import ResearchConfig
        
        # Test default config
        config = ResearchConfig()
        print("✅ Default config created")
        
        # Test validation with missing keys
        config.openai_api_key = None
        config.tavily_api_key = None
        
        try:
            config.validate()
            print("❌ Validation should have failed")
            return False
        except ValueError:
            print("✅ Validation correctly fails for missing keys")
        
        return True
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_company_info():
    """Test CompanyInfo dataclass."""
    print("🧪 Testing CompanyInfo...")
    
    try:
        from company_researcher import CompanyInfo
        
        # Test creation with minimal data
        info = CompanyInfo(company_name="Test Company")
        assert info.company_name == "Test Company"
        assert info.founder_names == []
        print("✅ CompanyInfo creation successful")
        
        # Test with full data
        info = CompanyInfo(
            company_name="Test Corp",
            founding_year=2020,
            founder_names=["John Doe", "Jane Smith"],
            product_description="Test product",
            funding_summary="Series A",
            notable_customers="Big Corp"
        )
        assert len(info.founder_names) == 2
        print("✅ CompanyInfo with full data successful")
        
        return True
    except Exception as e:
        print(f"❌ CompanyInfo test failed: {e}")
        return False

def test_graph_structure():
    """Test that the graph can be built without API keys."""
    print("🧪 Testing graph structure...")
    
    try:
        from company_researcher import CompanyResearcher
        
        # Create with dummy keys to test structure
        researcher = CompanyResearcher(
            openai_api_key="dummy",
            tavily_api_key="dummy",
            max_search_queries=2,
            max_search_results=1,
            max_reflections=1
        )
        
        # Check that graph exists
        assert hasattr(researcher, 'graph')
        print("✅ Graph structure created successfully")
        
        return True
    except Exception as e:
        print(f"❌ Graph structure test failed: {e}")
        return False

async def test_mock_research():
    """Test research flow with mocked APIs."""
    print("🧪 Testing research flow with mocks...")
    
    try:
        from company_researcher import CompanyResearcher
        
        # Mock responses
        mock_llm_response = Mock()
        mock_llm_response.content = '''{"company_name": "Test Corp", "founding_year": 2020, "founder_names": ["John Doe"], "product_description": "Test product", "funding_summary": null, "notable_customers": null}'''
        
        mock_search_results = [
            {
                "url": "https://example.com",
                "title": "Test Corp Information",
                "content": "Test Corp was founded in 2020 by John Doe. They make test products."
            }
        ]
        
        with patch('company_researcher.ChatOpenAI') as mock_chat, \
             patch('company_researcher.TavilySearch') as mock_search:
            
            # Setup mocks
            async def mock_llm_ainvoke(*args, **kwargs):
                return mock_llm_response
            
            async def mock_search_ainvoke(*args, **kwargs):
                return mock_search_results
                
            mock_chat.return_value.ainvoke = mock_llm_ainvoke
            mock_search.return_value.ainvoke = mock_search_ainvoke
            
            researcher = CompanyResearcher(
                openai_api_key="test",
                tavily_api_key="test",
                max_search_queries=1,
                max_search_results=1,
                max_reflections=1
            )
            
            # Test research
            result = await researcher.research_company("Test Corp")
            
            assert result["company_name"] == "Test Corp"
            print("✅ Mock research flow successful")
            return True
            
    except Exception as e:
        print(f"❌ Mock research test failed: {e}")
        return False

def run_all_tests():
    """Run all tests."""
    print("🚀 Running Company Researcher Tests")
    print("=" * 40)
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("CompanyInfo", test_company_info),
        ("Graph Structure", test_graph_structure),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 20)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run async test
    print(f"\n📋 Mock Research Flow")
    print("-" * 20)
    try:
        result = asyncio.run(test_mock_research())
        results.append(("Mock Research", result))
    except Exception as e:
        print(f"❌ Mock Research failed with exception: {e}")
        results.append(("Mock Research", False))
    
    # Summary
    print("\n" + "=" * 40)
    print("📊 Test Results Summary")
    print("-" * 20)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system appears to be working correctly.")
        if not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
            print("⚠️  Note: Configure API keys to run full integration tests")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)