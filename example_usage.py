#!/usr/bin/env python3
"""
Example usage of the Company Researcher LangGraph agent

This example shows how to use the agent to research company information.
Make sure to set your API keys in the environment:
- ANTHROPIC_API_KEY
- TAVILY_API_KEY
"""

from agent import app, CompanyResearchState
from langchain_core.messages import HumanMessage

def research_company(company_name: str, notes: str = None):
    """Research a company using the LangGraph agent"""
    
    print(f"🔍 Researching: {company_name}")
    if notes:
        print(f"📝 Notes: {notes}")
    print("-" * 50)
    
    # Create initial state
    initial_state = CompanyResearchState(
        company_name=company_name,
        notes=notes,
        messages=[HumanMessage(content=f"Research information about {company_name}")]
    )
    
    try:
        # Run the research workflow
        result = app.invoke(initial_state)
        
        # Display results
        print("\n📊 Research Summary:")
        print(f"  Queries executed: {result['queries_executed']}/{result['max_queries']}")
        print(f"  Reflections: {result['reflection_count']}/{result['max_reflections']}")
        print(f"  Total search results: {len(result.get('search_results', []))}")
        
        # Display company information
        if result.get('company_info'):
            company_info = result['company_info']
            print(f"\n🏢 {company_info.company_name}")
            print(f"  Founded: {company_info.founding_year or 'Unknown'}")
            print(f"  Founders: {', '.join(company_info.founder_names) if company_info.founder_names else 'Unknown'}")
            print(f"  Product: {company_info.product_description or 'Unknown'}")
            print(f"  Funding: {company_info.funding_summary or 'Unknown'}")
            print(f"  Notable Customers: {company_info.notable_customers or 'Unknown'}")
            
            return company_info
        else:
            print("❌ No company information extracted")
            return None
            
    except Exception as e:
        print(f"❌ Error during research: {str(e)}")
        return None


if __name__ == "__main__":
    # Example 1: Research a well-known company
    print("Example 1: Basic company research")
    research_company("Stripe")
    
    print("\n" + "="*60 + "\n")
    
    # Example 2: Research with specific notes
    print("Example 2: Research with specific focus")
    research_company(
        "Anthropic", 
        "Focus on AI safety and recent product launches"
    )
    
    print("\n" + "="*60 + "\n")
    
    # Example 3: Research a startup
    print("Example 3: Startup research")
    research_company("Vercel")