# Company Researcher - LangGraph Implementation

A multi-node LangGraph workflow for researching companies using web search and LLM-powered information extraction.

## Features

- **Multi-node LangGraph Workflow**: Structured research process with query generation, web search, information extraction, and reflection
- **Parallel Web Searches**: Uses Tavily API to perform multiple searches concurrently for better performance
- **Intelligent Query Generation**: LLM generates targeted search queries to fill missing information gaps
- **Reflection & Iteration**: Automatically determines if more research is needed and iterates until sufficient information is gathered
- **Configurable Limits**: Set maximum search queries, results per query, and reflection steps
- **Structured Output**: Returns data in a consistent CompanyInfo schema

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up API keys in `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

## Usage

### Basic Usage

```python
from company_researcher import research_company

# Simple research
result = research_company(
    company_name="OpenAI",
    user_notes="Focus on recent developments and funding"
)

if result["success"]:
    company_info = result["company_info"]
    print(f"Company: {company_info['company_name']}")
    print(f"Founded: {company_info['founding_year']}")
    print(f"Founders: {company_info['founder_names']}")
    print(f"Product: {company_info['product_description']}")
    print(f"Funding: {company_info['funding_summary']}")
    print(f"Customers: {company_info['notable_customers']}")
```

### Advanced Usage

```python
from company_researcher import CompanyResearcher

# Create researcher instance
researcher = CompanyResearcher()

# Research with custom parameters
result = researcher.research_company(
    company_name="Anthropic",
    user_notes="Focus on AI safety and constitutional AI",
    max_search_queries=5,
    max_search_results=10,
    max_reflection_steps=3
)
```

### Run Examples

```bash
python example_usage.py
```

## CompanyInfo Schema

The system extracts information into the following structured format:

```json
{
  "company_name": "string (required)",
  "founding_year": "integer or null",
  "founder_names": ["array of strings or null"],
  "product_description": "string or null",
  "funding_summary": "string or null",
  "notable_customers": "string or null"
}
```

## Architecture

The workflow consists of these nodes:

1. **Query Generation**: LLM generates targeted search queries based on missing information
2. **Web Search**: Parallel execution of searches using Tavily API
3. **Information Extraction**: LLM extracts and structures information from search results
4. **Reflection**: Evaluates completeness and decides whether to continue or complete

## Configuration Parameters

- `max_search_queries`: Maximum number of search queries (default: 5)
- `max_search_results`: Maximum results per search (default: 10)
- `max_reflection_steps`: Maximum reflection iterations (default: 3)

## Testing

Run basic tests:
```bash
python test_researcher.py
```

## Files Structure

- `company_researcher.py`: Main workflow and graph definition
- `nodes.py`: Individual workflow nodes implementation
- `models.py`: Pydantic data models
- `tavily_client.py`: Tavily API client with parallel search support
- `example_usage.py`: Usage examples and demonstrations
- `test_researcher.py`: Basic tests and validation