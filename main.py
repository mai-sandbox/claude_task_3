#!/usr/bin/env python3
"""
Company Researcher using LangGraph

A multi-node graph-based company research tool that uses LLM-generated queries
and the Tavily API to gather comprehensive company information.

Usage:
    python main.py "Company Name" ["Optional user notes"]

Example:
    python main.py "Anthropic" "AI safety company that created Claude"
"""

import asyncio
import sys
import json
import os
from typing import Optional
from workflow import CompanyResearchWorkflow
from schemas import ResearchState

async def research_company_interactive():
    """Interactive mode for researching companies"""
    print("🔍 Company Researcher - Interactive Mode")
    print("=" * 50)
    
    # Get company name
    while True:
        company_name = input("\nEnter company name to research: ").strip()
        if company_name:
            break
        print("Please enter a valid company name.")
    
    # Get optional user notes
    user_notes = input("Enter optional notes (press Enter to skip): ").strip()
    if not user_notes:
        user_notes = None
    
    # Get configuration options
    print("\n📋 Configuration Options (press Enter for defaults)")
    
    try:
        max_queries = input("Max search queries (default: 8): ").strip()
        max_queries = int(max_queries) if max_queries else 8
        
        max_results = input("Max results per query (default: 5): ").strip()
        max_results = int(max_results) if max_results else 5
        
        max_reflections = input("Max reflection steps (default: 3): ").strip()
        max_reflections = int(max_reflections) if max_reflections else 3
        
    except ValueError:
        print("Invalid input, using default values")
        max_queries, max_results, max_reflections = 8, 5, 3
    
    # Initialize and run the research
    print(f"\n🚀 Starting research for: {company_name}")
    if user_notes:
        print(f"📝 Notes: {user_notes}")
    
    researcher = CompanyResearchWorkflow()
    
    try:
        result = await researcher.research_company(
            company_name=company_name,
            user_notes=user_notes,
            max_search_queries=max_queries,
            max_search_results=max_results,
            max_reflection_steps=max_reflections
        )
        
        # Display results
        researcher.print_research_summary(result)
        
        # Ask if user wants to save results
        save_choice = input("\n💾 Save results to JSON file? (y/n): ").strip().lower()
        if save_choice in ['y', 'yes']:
            filename = f"{company_name.replace(' ', '_').lower()}_research.json"
            save_results_to_file(result, filename)
            print(f"✅ Results saved to {filename}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        return None

def save_results_to_file(state: ResearchState, filename: str):
    """Save research results to a JSON file"""
    try:
        # Convert state to dictionary for JSON serialization
        results = {
            "company_name": state.company_name,
            "user_notes": state.user_notes,
            "company_info": state.company_info.model_dump() if state.company_info else None,
            "search_queries": [q.model_dump() for q in state.search_queries],
            "search_results": [r.model_dump() for r in state.search_results],
            "statistics": {
                "queries_executed": state.queries_executed,
                "reflection_count": state.reflection_count,
                "total_search_results": len(state.search_results),
                "messages_count": len(state.messages),
                "is_complete": state.is_complete,
                "missing_fields": state.get_missing_fields()
            },
            "messages": state.messages
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
    except Exception as e:
        print(f"Failed to save results: {str(e)}")

async def research_company_cli(company_name: str, user_notes: Optional[str] = None):
    """Command line mode for researching companies"""
    print(f"🔍 Researching: {company_name}")
    if user_notes:
        print(f"📝 Notes: {user_notes}")
    
    researcher = CompanyResearchWorkflow()
    
    try:
        result = await researcher.research_company(
            company_name=company_name,
            user_notes=user_notes,
            max_search_queries=8,
            max_search_results=5,
            max_reflection_steps=3
        )
        
        # Display results
        researcher.print_research_summary(result)
        
        # Auto-save results
        filename = f"{company_name.replace(' ', '_').lower()}_research.json"
        save_results_to_file(result, filename)
        print(f"\n💾 Results automatically saved to {filename}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        return None

def check_environment():
    """Check if required environment variables are set"""
    required_vars = ["OPENAI_API_KEY", "TAVILY_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("❌ Missing required environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these variables in your .env file or environment.")
        return False
    
    return True

def print_help():
    """Print help information"""
    help_text = """
🔍 Company Researcher using LangGraph

DESCRIPTION:
    A multi-node graph-based company research tool that uses LLM-generated 
    queries and the Tavily API to gather comprehensive company information.

USAGE:
    python main.py                              # Interactive mode
    python main.py "Company Name"               # Research with company name
    python main.py "Company Name" "User notes" # Research with notes

EXAMPLES:
    python main.py "Anthropic"
    python main.py "Anthropic" "AI safety company that created Claude"
    python main.py "OpenAI" "Founded by Sam Altman, created ChatGPT"

REQUIREMENTS:
    - OPENAI_API_KEY: OpenAI API key for LLM operations
    - TAVILY_API_KEY: Tavily API key for web searching

FEATURES:
    ✅ Multi-node LangGraph workflow
    ✅ Parallel web searching with Tavily API  
    ✅ LLM-generated targeted search queries
    ✅ Structured information extraction
    ✅ Reflection and quality assessment
    ✅ Configurable search limits
    ✅ JSON export of results
    ✅ Interactive and CLI modes

The tool will research:
    - Company name and founding information
    - Founder names and founding year
    - Product/service descriptions
    - Funding history and investors
    - Notable customers and partnerships
"""
    print(help_text)

async def main():
    """Main entry point"""
    
    # Check for help flag
    if len(sys.argv) > 1 and sys.argv[1] in ['-h', '--help', 'help']:
        print_help()
        return
    
    # Check environment variables
    if not check_environment():
        sys.exit(1)
    
    try:
        if len(sys.argv) == 1:
            # Interactive mode
            await research_company_interactive()
            
        elif len(sys.argv) == 2:
            # CLI mode with just company name
            company_name = sys.argv[1]
            await research_company_cli(company_name)
            
        elif len(sys.argv) == 3:
            # CLI mode with company name and notes
            company_name = sys.argv[1]
            user_notes = sys.argv[2]
            await research_company_cli(company_name, user_notes)
            
        else:
            print("❌ Invalid arguments. Use --help for usage information.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Research interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())