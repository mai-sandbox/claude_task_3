#!/usr/bin/env python3

from agent import app, ResearchState
from langchain_core.messages import HumanMessage

def test_graph_structure():
    """Test that the graph is properly structured"""
    print("Testing graph structure...")
    
    # Check that the graph has been compiled
    assert app is not None, "Graph should be compiled"
    
    # Check node names
    expected_nodes = {"generate_queries", "execute_searches", "extract_info", "reflect"}
    actual_nodes = set(app.get_graph().nodes.keys())
    
    # Remove special nodes
    actual_nodes = {node for node in actual_nodes if not node.startswith("__")}
    
    print(f"Expected nodes: {expected_nodes}")
    print(f"Actual nodes: {actual_nodes}")
    
    assert expected_nodes.issubset(actual_nodes), f"Missing nodes: {expected_nodes - actual_nodes}"
    
    print("✅ Graph structure test passed!")

def test_initial_state():
    """Test that we can create a valid initial state"""
    print("\nTesting initial state creation...")
    
    test_state = ResearchState(
        messages=[HumanMessage(content="Test message")],
        company_name="Test Company",
        user_notes="Test notes",
        search_queries=[],
        search_results=[],
        company_info={},
        reflection_count=0,
        max_reflections=3,
        max_search_queries=8,
        max_search_results=5,
        is_complete=False
    )
    
    assert test_state["company_name"] == "Test Company"
    assert test_state["reflection_count"] == 0
    assert test_state["is_complete"] == False
    
    print("✅ Initial state test passed!")

if __name__ == "__main__":
    test_graph_structure()
    test_initial_state()
    print("\n🎉 All tests passed! The company researcher is ready to use.")
    print("\nTo use the researcher:")
    print("1. Set your ANTHROPIC_API_KEY environment variable")
    print("2. Set your TAVILY_API_KEY environment variable")
    print("3. Run: python agent.py")
    print("4. Or import and use: from agent import research_company")