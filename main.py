#!/usr/bin/env python3

import os
import json
import argparse
from typing import Optional
from workflow import run_company_research


def get_api_keys():
    """Get API keys from environment variables"""
    openai_key = os.getenv("OPENAI_API_KEY")
    tavily_key = os.getenv("TAVILY_API_KEY")
    
    if not openai_key:
        print("Warning: OPENAI_API_KEY not found in environment variables")
        openai_key = input("Please enter your OpenAI API key: ").strip()
    
    if not tavily_key:
        print("Warning: TAVILY_API_KEY not found in environment variables")
        tavily_key = input("Please enter your Tavily API key: ").strip()
    
    return openai_key, tavily_key


def interactive_mode():
    """Run in interactive mode"""
    print("=== Company Research Tool ===")
    print("Powered by LangGraph, OpenAI, and Tavily")
    print()
    
    # Get API keys
    openai_key, tavily_key = get_api_keys()
    
    while True:
        print("\n" + "="*50)
        company_name = input("Enter company name (or 'quit' to exit): ").strip()
        
        if company_name.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        if not company_name:
            print("Please enter a valid company name.")
            continue
        
        # Get optional user notes
        user_notes = input("Enter any additional notes (optional): ").strip()
        user_notes = user_notes if user_notes else None
        
        # Get configuration options
        try:
            max_queries = int(input("Max search queries (default: 5): ") or "5")
            max_results = int(input("Max results per query (default: 3): ") or "3")
            max_reflections = int(input("Max reflection steps (default: 2): ") or "2")
        except ValueError:
            print("Using default values for configuration.")
            max_queries, max_results, max_reflections = 5, 3, 2
        
        print(f"\nResearching {company_name}...")
        print("This may take a minute or two...")
        
        try:
            result = run_company_research(
                company_name=company_name,
                user_notes=user_notes,
                openai_api_key=openai_key,
                tavily_api_key=tavily_key,
                max_search_queries=max_queries,
                max_search_results=max_results,
                max_reflection_steps=max_reflections
            )
            
            print("\n" + "="*50)
            print("RESEARCH RESULTS")
            print("="*50)
            
            # Display company information
            company_info = result['company_info']
            print(f"Company: {company_info['company_name']}")
            
            if company_info.get('founding_year'):
                print(f"Founded: {company_info['founding_year']}")
            
            if company_info.get('founder_names'):
                founders = ', '.join(company_info['founder_names'])
                print(f"Founders: {founders}")
            
            if company_info.get('product_description'):
                print(f"Product/Service: {company_info['product_description']}")
            
            if company_info.get('funding_summary'):
                print(f"Funding: {company_info['funding_summary']}")
            
            if company_info.get('notable_customers'):
                print(f"Notable Customers: {company_info['notable_customers']}")
            
            # Display research metadata
            print(f"\nResearch Metadata:")
            print(f"- Search queries used: {len(result['search_queries_used'])}")
            print(f"- Total search results: {result['total_search_results']}")
            print(f"- Reflection steps: {result['reflection_steps']}")
            print(f"- Research complete: {result['is_complete']}")
            
            # Ask if user wants to save results
            save = input("\nSave results to JSON file? (y/n): ").strip().lower()
            if save in ['y', 'yes']:
                filename = f"{company_name.replace(' ', '_').lower()}_research.json"
                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"Results saved to {filename}")
        
        except Exception as e:
            print(f"Error during research: {e}")
            print("Please check your API keys and try again.")


def cli_mode():
    """Run in command line mode"""
    parser = argparse.ArgumentParser(description="Research company information using AI")
    parser.add_argument("company_name", help="Name of the company to research")
    parser.add_argument("--notes", help="Additional notes about the company")
    parser.add_argument("--max-queries", type=int, default=5, help="Maximum search queries (default: 5)")
    parser.add_argument("--max-results", type=int, default=3, help="Maximum results per query (default: 3)")
    parser.add_argument("--max-reflections", type=int, default=2, help="Maximum reflection steps (default: 2)")
    parser.add_argument("--output", help="Output file for results (JSON format)")
    parser.add_argument("--openai-key", help="OpenAI API key (or set OPENAI_API_KEY env var)")
    parser.add_argument("--tavily-key", help="Tavily API key (or set TAVILY_API_KEY env var)")
    
    args = parser.parse_args()
    
    # Get API keys
    openai_key = args.openai_key or os.getenv("OPENAI_API_KEY")
    tavily_key = args.tavily_key or os.getenv("TAVILY_API_KEY")
    
    if not openai_key or not tavily_key:
        print("Error: Both OpenAI and Tavily API keys are required")
        print("Set them via --openai-key and --tavily-key flags or environment variables")
        return 1
    
    try:
        print(f"Researching {args.company_name}...")
        
        result = run_company_research(
            company_name=args.company_name,
            user_notes=args.notes,
            openai_api_key=openai_key,
            tavily_api_key=tavily_key,
            max_search_queries=args.max_queries,
            max_search_results=args.max_results,
            max_reflection_steps=args.max_reflections
        )
        
        # Output results
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"Results saved to {args.output}")
        else:
            print(json.dumps(result, indent=2, default=str))
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    import sys
    
    # If no arguments provided, run in interactive mode
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        exit_code = cli_mode()
        sys.exit(exit_code)