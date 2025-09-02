import asyncio
import json
from main import research_company_api

async def example_research():
    """Example of how to use the company research API"""
    
    # Example 1: Research OpenAI
    print("🔍 Researching OpenAI...")
    result = await research_company_api(
        company_name="OpenAI",
        user_notes="Focus on AI/ML capabilities and recent developments",
        max_queries=3,
        max_search_results=2,
        max_reflections=2
    )
    
    print("Results:")
    if result.company_info:
        print(json.dumps(result.company_info.model_dump(), indent=2))
    print(f"Queries executed: {result.query_count}")
    print(f"Reflections: {result.reflection_count}")
    print("Messages:")
    for msg in result.messages:
        print(f"  - {msg}")
    print()

if __name__ == "__main__":
    asyncio.run(example_research())