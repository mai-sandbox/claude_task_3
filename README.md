# Company Researcher with LangGraph

A sophisticated multi-node AI agent built with LangGraph that researches companies using web search APIs. The system generates targeted search queries, executes parallel searches, extracts structured information, and uses reflection to ensure completeness.

## Features

- **Multi-Node Architecture**: Built using LangGraph's StateGraph with specialized nodes
- **Intelligent Query Generation**: LLM generates targeted search queries based on missing information
- **Parallel Search Execution**: Searches run concurrently for improved performance
- **Structured Information Extraction**: Extracts data into a predefined CompanyInfo schema
- **Reflection & Iteration**: Evaluates completeness and performs additional searches if needed
- **Configurable Limits**: Set maximum queries, results per query, and reflection steps
- **Conversation Tracking**: Maintains detailed logs of the research process

## Architecture

The system uses a multi-node graph with the following flow:

1. **Initialize**: Sets up research parameters and initial state
2. **Generate Queries**: LLM creates targeted search queries
3. **Execute Searches**: Parallel execution of Tavily API searches
4. **Extract Information**: Structured extraction from search results
5. **Reflect & Evaluate**: Determines if more research is needed
6. **Loop or Finish**: Either continues with more queries or completes

## CompanyInfo Schema

```json
{
  "title": "CompanyInfo",
  "description": "Basic information about a company",
  "type": "object",
  "properties": {
    "company_name": {"type": "string", "description": "Official name of the company"},
    "founding_year": {"type": "integer", "description": "Year the company was founded"},
    "founder_names": {"type": "array", "items": {"type": "string"}, "description": "Names of the founding team members"},
    "product_description": {"type": "string", "description": "Brief description of the company's main product or service"},
    "funding_summary": {"type": "string", "description": "Summary of the company's funding history"},
    "notable_customers": {"type": "string", "description": "Known customers that use company's product/service"}
  },
  "required": ["company_name"]
}
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

### Basic Usage

```python
import asyncio
from company_researcher import CompanyResearcher

async def main():
    researcher = CompanyResearcher(
        openai_api_key="your_key",
        tavily_api_key="your_key",
        max_queries=5,
        max_results_per_query=3,
        max_reflections=2
    )
    
    result = await researcher.research_company(
        company_name="Anthropic",
        notes="AI safety company"
    )
    
    print(result["company_info"])

asyncio.run(main())
```

### Interactive Demo

```bash
python example_usage.py
```

## Configuration

- `max_queries`: Maximum search queries per company (default: 5)
- `max_results_per_query`: Maximum results per search (default: 3)
- `max_reflections`: Maximum reflection steps (default: 2)

## API Keys Required

- **OpenAI API Key**: For LLM-powered query generation and information extraction
- **Tavily API Key**: For web search capabilities

## Files

- `company_researcher.py`: Main implementation with CompanyResearcher class
- `example_usage.py`: Interactive demo and usage examples
- `requirements.txt`: Python dependencies
- `.env.example`: Environment variable template