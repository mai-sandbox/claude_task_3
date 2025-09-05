#!/usr/bin/env python3
"""
Test script for the Company Research Agent
Tests the system without requiring API keys by using mock data
"""

import json
from unittest.mock import Mock, patch
from company_researcher import CompanyResearcher, Config, CompanyInfo

def test_with_mock_data():
    """Test the company researcher with mock data"""
    print("🧪 Testing Company Research Agent with Mock Data")
    print("=" * 60)
    
    # Create config with dummy API keys for testing
    config = Config(
        max_queries=3,
        max_results_per_query=2,
        max_reflections=2,
        openai_api_key="test-openai-key",
        tavily_api_key="test-tavily-key"
    )
    
    # Mock search results
    mock_search_results = [
        {
            "title": "OpenAI - Artificial Intelligence Research Company",
            "content": "OpenAI was founded in 2015 by Sam Altman, Elon Musk, Greg Brockman, and others. The company develops artificial intelligence systems and is known for ChatGPT and GPT models.",
            "url": "https://openai.com/about"
        },
        {
            "title": "OpenAI Funding History",
            "content": "OpenAI has raised billions in funding, including a recent investment from Microsoft. The company has received over $11 billion in total funding.",
            "url": "https://example.com/funding"
        },
        {
            "title": "OpenAI Customers and Partners",
            "content": "OpenAI serves major customers including Microsoft, which integrates GPT into Office products, and many other enterprises use OpenAI's API.",
            "url": "https://example.com/customers"
        }
    ]
    
    # Mock LLM responses
    mock_queries_response = Mock()
    mock_queries_response.content = '''```json
[
    "OpenAI founding history founders 2015",
    "OpenAI products ChatGPT GPT models AI",
    "OpenAI funding Microsoft investment history"
]
```'''
    
    mock_extraction_response = Mock()
    mock_extraction_response.content = '''```json
{
    "company_name": "OpenAI",
    "founding_year": 2015,
    "founder_names": ["Sam Altman", "Elon Musk", "Greg Brockman"],
    "product_description": "AI research company developing large language models like ChatGPT and GPT-4",
    "funding_summary": "Raised over $11 billion including major investment from Microsoft",
    "notable_customers": "Microsoft, various enterprises using OpenAI API"
}
```'''
    
    # Create researcher with mocked dependencies
    with patch('company_researcher.ChatOpenAI') as mock_openai, \
         patch('company_researcher.TavilyClient') as mock_tavily:
        
        # Setup mocks
        mock_llm = Mock()
        mock_llm.invoke.side_effect = [mock_queries_response, mock_extraction_response]
        mock_openai.return_value = mock_llm
        
        mock_tavily_client = Mock()
        mock_tavily_client.search.return_value = {"results": mock_search_results}
        mock_tavily.return_value = mock_tavily_client
        
        # Create and test researcher
        researcher = CompanyResearcher(config)
        
        print("🚀 Running mock research for: OpenAI")
        
        # Run research
        result = researcher.research_company("OpenAI", "AI company test")
        
        # Display results
        print("\n" + "=" * 50)
        print("📊 MOCK RESEARCH RESULTS")
        print("=" * 50)
        
        print(f"🏢 Company Name: {result.company_name}")
        print(f"📅 Founded: {result.founding_year}")
        print(f"👥 Founders: {', '.join(result.founder_names)}")
        print(f"🎯 Product/Service: {result.product_description}")
        print(f"💰 Funding: {result.funding_summary}")
        print(f"🤝 Notable Customers: {result.notable_customers}")
        
        # Validate results
        assert result.company_name == "OpenAI"
        assert result.founding_year == 2015
        assert "Sam Altman" in result.founder_names
        assert result.product_description is not None
        assert result.funding_summary is not None
        
        print("\n✅ All tests passed!")
        
        # Save test results
        test_filename = "test_results.json"
        with open(test_filename, 'w') as f:
            json.dump(result.model_dump(), f, indent=2)
        
        print(f"💾 Test results saved to: {test_filename}")

def test_state_structure():
    """Test that the state structure is correct"""
    print("\n🔍 Testing State Structure")
    print("-" * 30)
    
    # Test CompanyInfo model
    company_info = CompanyInfo(
        company_name="Test Company",
        founding_year=2020,
        founder_names=["John Doe", "Jane Smith"],
        product_description="Test product",
        funding_summary="Series A funding",
        notable_customers="Customer A, Customer B"
    )
    
    # Convert to dict and back
    data = company_info.model_dump()
    restored = CompanyInfo(**data)
    
    assert restored.company_name == "Test Company"
    assert restored.founding_year == 2020
    assert len(restored.founder_names) == 2
    
    print("✅ State structure tests passed!")

if __name__ == "__main__":
    try:
        test_state_structure()
        test_with_mock_data()
        
        print("\n🎉 All tests completed successfully!")
        print("\nTo test with real API keys:")
        print("1. Copy .env.example to .env")
        print("2. Add your OPENAI_API_KEY and TAVILY_API_KEY")
        print("3. Run: python company_researcher.py")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()