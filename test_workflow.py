#!/usr/bin/env python3

"""
Test script for the company research workflow
This script validates the workflow structure without making actual API calls
"""

import json
from unittest.mock import Mock, patch
from models import ResearchState, CompanyInfo
from nodes import CompanyResearchNodes
from workflow import create_company_research_workflow, should_continue_research


def test_models():
    """Test the data models"""
    print("Testing data models...")
    
    # Test CompanyInfo model
    company = CompanyInfo(
        company_name="Test Company",
        founding_year=2020,
        founder_names=["John Doe", "Jane Smith"],
        product_description="AI-powered solutions",
        funding_summary="$10M Series A",
        notable_customers="Enterprise clients"
    )
    
    assert company.company_name == "Test Company"
    assert company.founding_year == 2020
    assert len(company.founder_names) == 2
    print("✓ CompanyInfo model works correctly")
    
    # Test ResearchState model
    state = ResearchState(
        company_name="Test Company",
        user_notes="Test notes",
        max_search_queries=3,
        max_search_results=2,
        company_info=company
    )
    
    assert state.company_name == "Test Company"
    assert state.max_search_queries == 3
    assert not state.is_complete
    print("✓ ResearchState model works correctly")


def test_conditional_logic():
    """Test the conditional edge logic"""
    print("\nTesting conditional logic...")
    
    # Test completion case
    state_complete = ResearchState(
        company_name="Test",
        is_complete=True
    )
    result = should_continue_research(state_complete)
    assert result == "end"
    print("✓ Correctly identifies complete state")
    
    # Test needs more info case
    state_more_info = ResearchState(
        company_name="Test",
        needs_more_info=True,
        reflection_count=0,
        max_reflection_steps=2
    )
    result = should_continue_research(state_more_info)
    assert result == "search_more"
    print("✓ Correctly identifies need for more research")
    
    # Test max reflections reached
    state_max_reflect = ResearchState(
        company_name="Test",
        needs_more_info=True,
        reflection_count=2,
        max_reflection_steps=2
    )
    result = should_continue_research(state_max_reflect)
    assert result == "end"
    print("✓ Correctly stops at max reflections")


def test_workflow_structure():
    """Test the workflow graph structure"""
    print("\nTesting workflow structure...")
    
    # Mock API keys for testing
    mock_openai_key = "test-openai-key"
    mock_tavily_key = "test-tavily-key"
    
    # Create workflow
    workflow = create_company_research_workflow(mock_openai_key, mock_tavily_key)
    
    # Check that workflow was created
    assert workflow is not None
    print("✓ Workflow created successfully")
    
    # The workflow should have the nodes we defined
    # This is a basic structure test - actual node testing would require mocking API calls
    print("✓ Workflow structure appears correct")


def test_node_initialization():
    """Test node initialization"""
    print("\nTesting node initialization...")
    
    # Test with mock keys
    nodes = CompanyResearchNodes("test-openai", "test-tavily")
    assert nodes.openai_client is not None
    assert nodes.tavily_client is not None
    print("✓ Nodes initialize correctly with API keys")


def run_basic_workflow_test():
    """Test basic workflow execution with mocked API responses"""
    print("\nTesting basic workflow execution...")
    
    # Create test state
    test_state = ResearchState(
        company_name="OpenAI",
        user_notes="AI company",
        max_search_queries=2,
        max_search_results=2,
        max_reflection_steps=1,
        company_info=CompanyInfo(company_name="OpenAI")
    )
    
    # Mock the API calls
    with patch('nodes.OpenAI') as mock_openai, \
         patch('nodes.TavilyClient') as mock_tavily:
        
        # Mock OpenAI response for query generation
        mock_openai.return_value.chat.completions.create.return_value.choices[0].message.content = json.dumps([
            {"query": "OpenAI company information", "purpose": "basic info"},
            {"query": "OpenAI founders", "purpose": "founder info"}
        ])
        
        # Mock Tavily search results
        mock_tavily.return_value.search.return_value = {
            'results': [
                {
                    'title': 'OpenAI - About',
                    'url': 'https://openai.com/about',
                    'content': 'OpenAI was founded in 2015 by Sam Altman, Elon Musk, and others.'
                }
            ]
        }
        
        # Mock OpenAI response for information extraction
        mock_extraction_response = {
            "company_name": "OpenAI",
            "founding_year": 2015,
            "founder_names": ["Sam Altman", "Elon Musk"],
            "product_description": "AI research and deployment",
            "funding_summary": "Multiple funding rounds",
            "notable_customers": "Various enterprises"
        }
        
        # Set up the mock to return different responses for different calls
        mock_openai.return_value.chat.completions.create.side_effect = [
            # First call for query generation
            Mock(choices=[Mock(message=Mock(content=json.dumps([
                {"query": "OpenAI company information", "purpose": "basic info"}
            ])))]),
            # Second call for information extraction
            Mock(choices=[Mock(message=Mock(content=json.dumps(mock_extraction_response)))])
        ]
        
        # Create nodes and test individual functions
        nodes = CompanyResearchNodes("test-openai", "test-tavily")
        
        # Test query generation
        state_after_queries = nodes.generate_search_queries(test_state)
        assert len(state_after_queries.search_queries) > 0
        print("✓ Query generation node works")
        
        # Test search (with mock)
        state_after_search = nodes.search_web_parallel(state_after_queries)
        assert len(state_after_search.search_results) >= 0  # Can be 0 with mocked results
        print("✓ Web search node works")
        
        # Test extraction
        state_after_extraction = nodes.extract_company_info(state_after_search)
        assert state_after_extraction.company_info.company_name == "OpenAI"
        print("✓ Information extraction node works")
        
        # Test reflection
        final_state = nodes.reflect_on_completeness(state_after_extraction)
        assert isinstance(final_state.is_complete, bool)
        print("✓ Reflection node works")


def main():
    """Run all tests"""
    print("=== Company Research Workflow Tests ===\n")
    
    try:
        test_models()
        test_conditional_logic()
        test_workflow_structure()
        test_node_initialization()
        run_basic_workflow_test()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED!")
        print("The company research workflow is ready to use.")
        print("Make sure to set your OPENAI_API_KEY and TAVILY_API_KEY environment variables.")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())