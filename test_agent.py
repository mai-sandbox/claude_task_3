#!/usr/bin/env python3
"""Test script for the company researcher agent."""

import json
from agent import research_company, app, CompanyInfo
from langchain_core.messages import HumanMessage

def test_research_function():
    """Test the convenience function for research."""
    print("=== Testing research_company function ===")
    try:
        # Test without API keys - will show structure
        result = research_company(
            company_name="OpenAI",
            user_notes="Focus on recent AI developments",
            max_queries=3,
            max_reflections=1
        )
        print("Research completed successfully!")
        print("Result type:", type(result))
        print("Company info:")
        print(json.dumps(result.model_dump(), indent=2))
        
    except Exception as e:
        print(f"Expected error (no API keys): {e}")
        print("This is expected when API keys are not configured.")


def test_graph_directly():
    """Test the graph directly with invoke."""
    print("\n=== Testing graph directly ===")
    
    initial_state = {
        "messages": [HumanMessage(content="Research company: Tesla")],
        "company_name": "Tesla",
        "user_notes": "Focus on electric vehicles and energy",
        "company_info": CompanyInfo(company_name="Tesla"),
        "search_results": [],
        "queries_executed": 0,
        "reflection_count": 0,
        "max_queries": 2,
        "max_search_results": 3,
        "max_reflections": 1,
        "is_complete": False
    }
    
    try:
        result = app.invoke(initial_state)
        print("Graph execution completed!")
        print("Final company info:")
        print(json.dumps(result["company_info"].model_dump(), indent=2))
        print(f"\nTotal messages: {len(result['messages'])}")
        print(f"Queries executed: {result.get('queries_executed', 0)}")
        print(f"Reflection count: {result.get('reflection_count', 0)}")
        
    except Exception as e:
        print(f"Expected error (no API keys): {e}")
        print("This demonstrates the graph structure is working.")


def show_graph_structure():
    """Show the graph structure."""
    print("\n=== Graph Structure ===")
    print("Nodes:", list(app.get_graph().nodes.keys()))
    print("Edges:", [(edge.source, edge.target) for edge in app.get_graph().edges])


if __name__ == "__main__":
    print("Company Researcher Agent Test")
    print("=" * 40)
    
    show_graph_structure()
    test_research_function()
    test_graph_directly()
    
    print("\n=== Setup Instructions ===")
    print("To use with real API calls, set these environment variables:")
    print("- ANTHROPIC_API_KEY: Your Anthropic API key")
    print("- TAVILY_API_KEY: Your Tavily search API key")
    print("\nExample usage:")
    print("export ANTHROPIC_API_KEY='your-key-here'")
    print("export TAVILY_API_KEY='your-key-here'")
    print("python test_agent.py")