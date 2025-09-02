"""
Test script for the Company Researcher system.

Run this to verify that the system is working correctly.
"""

import asyncio
import sys
from company_researcher import create_company_researcher
from company_researcher.config import get_default_config


def test_basic_functionality():
    """Test basic functionality of the researcher."""
    
    print("🧪 Testing Basic Functionality")
    print("=" * 40)
    
    try:
        # Test configuration
        config = get_default_config()
        print(f"✅ Configuration loaded: {config.max_queries} queries, {config.max_reflections} reflections")
        
        # Create researcher
        researcher = create_company_researcher(
            max_queries=2,  # Small test
            max_results_per_query=1,
            max_reflections=1
        )
        print("✅ Researcher instance created")
        
        # Test with a well-known company
        print("\n🔍 Testing research with Apple Inc...")
        
        results = researcher.research_company_sync(
            company_name="Apple Inc",
            user_notes="Test run - basic information only"
        )
        
        # Verify results structure
        assert "company_info" in results, "Missing company_info in results"
        assert "research_metadata" in results, "Missing research_metadata in results"
        assert "messages" in results, "Missing messages in results"
        
        company_info = results["company_info"]
        metadata = results["research_metadata"]
        
        print(f"✅ Research completed:")
        print(f"   - Company name: {company_info['company_name'] if company_info else 'N/A'}")
        print(f"   - Queries executed: {metadata['queries_executed']}")
        print(f"   - Results found: {metadata['results_found']}")
        print(f"   - Complete: {metadata['is_complete']}")
        
        # Basic validation
        if company_info and company_info.get("company_name"):
            print("✅ Successfully extracted company information")
        else:
            print("⚠️ Warning: No company information extracted")
        
        if metadata["queries_executed"] > 0:
            print("✅ Queries were executed")
        else:
            print("❌ No queries were executed")
        
        if len(results["messages"]) > 0:
            print("✅ Process messages were recorded")
        else:
            print("⚠️ Warning: No process messages recorded")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False


def test_configuration():
    """Test configuration validation."""
    
    print("\n⚙️ Testing Configuration")
    print("=" * 30)
    
    try:
        config = get_default_config()
        
        # Test configuration validation
        config.validate_config()
        print("✅ Configuration validation passed")
        
        # Test configuration display (should hide API keys)
        config_dict = config.to_dict()
        
        if "anthropic_api_key" in config_dict:
            if config_dict["anthropic_api_key"].endswith("..."):
                print("✅ API keys are properly masked in output")
            else:
                print("⚠️ Warning: API keys might not be masked")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {str(e)}")
        return False


def test_sync_interface():
    """Test the synchronous interface."""
    
    print("\n🔄 Testing Synchronous Interface")
    print("=" * 35)
    
    try:
        researcher = create_company_researcher(
            max_queries=1,
            max_results_per_query=1, 
            max_reflections=1
        )
        
        # Test sync method
        results = researcher.research_company_sync(
            company_name="Microsoft",
            user_notes="Sync test"
        )
        
        print(f"✅ Synchronous research completed")
        print(f"   - Company: {results['company_info']['company_name'] if results['company_info'] else 'N/A'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sync interface test failed: {str(e)}")
        return False


def run_all_tests():
    """Run all tests."""
    
    print("🚀 Company Researcher Test Suite")
    print("=" * 50)
    
    test_results = []
    
    # Configuration test
    test_results.append(test_configuration())
    
    # Basic functionality test
    test_results.append(test_basic_functionality())
    
    # Sync interface test  
    test_results.append(test_sync_interface())
    
    # Summary
    print(f"\n📊 Test Summary")
    print("=" * 20)
    
    passed = sum(test_results)
    total = len(test_results)
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("🎉 All tests passed!")
        return True
    else:
        print(f"⚠️ {total - passed} test(s) failed")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n✅ System is ready to use!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed. Check your configuration.")
        sys.exit(1)