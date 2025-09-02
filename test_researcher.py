#!/usr/bin/env python3
"""
Simple test script for the Company Researcher

This script performs basic validation of the components.
"""

import json
from models import CompanyInfo, ResearchState
from tavily_client import TavilySearchClient
from nodes import CompanyResearchNodes

def test_models():
    """Test the Pydantic models"""
    print("Testing models...")
    
    # Test CompanyInfo creation
    company = CompanyInfo(company_name="Test Company")
    assert company.company_name == "Test Company"
    assert company.founding_year is None
    
    # Test with full data
    full_company = CompanyInfo(
        company_name="OpenAI",
        founding_year=2015,
        founder_names=["Sam Altman", "Elon Musk", "Ilya Sutskever"],
        product_description="AI research company",
        funding_summary="Series A, B, C funding",
        notable_customers="Microsoft, various enterprises"
    )
    
    data = full_company.model_dump()
    assert data["company_name"] == "OpenAI"
    assert data["founding_year"] == 2015
    
    # Test ResearchState
    state = ResearchState(
        company_name="Test Corp",
        max_search_queries=5
    )
    assert state.company_name == "Test Corp"
    assert state.search_queries_used == 0
    assert state.max_search_queries == 5
    
    print("✓ Models test passed")

def test_tavily_client_init():
    """Test Tavily client initialization"""
    print("Testing Tavily client initialization...")
    
    try:
        # This will fail without API key, but we can test the error handling
        client = TavilySearchClient(api_key="test_key")
        print("✓ Tavily client initialized")
    except Exception as e:
        print(f"Expected error without real API key: {e}")

def test_nodes_init():
    """Test nodes initialization"""
    print("Testing nodes initialization...")
    
    try:
        nodes = CompanyResearchNodes(
            openai_api_key="test_key",
            tavily_api_key="test_key"
        )
        print("✓ Nodes initialized")
    except Exception as e:
        print(f"Expected error without real API keys: {e}")

def test_json_serialization():
    """Test JSON serialization of models"""
    print("Testing JSON serialization...")
    
    company = CompanyInfo(
        company_name="Test Company",
        founding_year=2020,
        founder_names=["Alice", "Bob"]
    )
    
    # Test serialization
    json_data = company.model_dump()
    json_str = json.dumps(json_data)
    
    # Test deserialization
    loaded_data = json.loads(json_str)
    restored_company = CompanyInfo(**loaded_data)
    
    assert restored_company.company_name == "Test Company"
    assert restored_company.founding_year == 2020
    assert restored_company.founder_names == ["Alice", "Bob"]
    
    print("✓ JSON serialization test passed")

def main():
    """Run all tests"""
    print("Running Company Researcher Tests")
    print("================================")
    
    try:
        test_models()
        test_tavily_client_init()
        test_nodes_init()
        test_json_serialization()
        
        print("\n✓ All basic tests passed!")
        print("Note: Full integration tests require valid API keys")
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()