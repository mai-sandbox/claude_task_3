# Company Researcher - LangGraph Multi-Node System

A comprehensive company research tool built with LangGraph that uses web search and LLM-based information extraction to gather structured company data.

## Features

✅ **Multi-Node Architecture**: Query generation → Web search → Information extraction → Reflection  
✅ **Parallel Web Search**: Uses Tavily API with configurable limits  
✅ **Smart Reflection**: Automatically determines if more information is needed  
✅ **Structured Output**: Returns standardized CompanyInfo schema  
✅ **Configurable Limits**: Control max queries, results, and reflection cycles  
✅ **Error Handling**: Robust error handling and fallbacks  
✅ **LangSmith Tracing**: Built-in observability support  

## Quick Start

```python
from company_researcher import create_company_researcher

# Create researcher with default settings
researcher = create_company_researcher()

# Research a company
results = researcher.research_company_sync(
    company_name="OpenAI",
    user_notes="Focus on recent developments"
)

# Access structured data
company_info = results["company_info"]
print(f"Company: {company_info['company_name']}")
print(f"Founded: {company_info['founding_year']}")
print(f"Founders: {company_info['founder_names']}")
```

## Installation

```bash
pip install -r requirements.txt
```

Required environment variables in `.env`:
```
ANTHROPIC_API_KEY=your_anthropic_key
TAVILY_API_KEY=your_tavily_key
```

## Configuration Options

### Pre-defined Configurations

```python
from company_researcher import (
    QUICK_RESEARCH_CONFIG,      # 3 queries, 2 results each, 1 reflection
    BALANCED_RESEARCH_CONFIG,   # 6 queries, 3 results each, 2 reflections  
    THOROUGH_RESEARCH_CONFIG    # 10 queries, 5 results each, 3 reflections
)
```

### Custom Configuration

```python
researcher = create_company_researcher(
    max_queries=8,              # Maximum search queries per company
    max_results_per_query=4,    # Maximum results per search
    max_reflections=3           # Maximum reflection iterations
)
```

## Output Schema

The system extracts the following structured information:

```json
{
  "company_name": "string",
  "founding_year": "integer or null", 
  "founder_names": ["array of strings or null"],
  "product_description": "string or null",
  "funding_summary": "string or null", 
  "notable_customers": "string or null"
}
```

## Graph Flow

1. **Query Generation**: LLM generates targeted search queries based on company name and user notes
2. **Web Search**: Executes searches in parallel using Tavily API with domain filtering
3. **Information Extraction**: LLM extracts structured data from search results
4. **Reflection**: Evaluates completeness and decides whether to search for more information

## Examples

### Basic Usage
```python
# Simple research
results = researcher.research_company_sync("Tesla", "Focus on autonomous driving")
```

### Batch Processing
```python
companies = ["Google", "Microsoft", "Apple"]
for company in companies:
    results = researcher.research_company_sync(company)
    print(f"Researched: {results['company_info']['company_name']}")
```

### Advanced Configuration
```python
from company_researcher.config import ResearchConfig

config = ResearchConfig(
    max_queries=12,
    max_results_per_query=5,
    temperature=0.0,  # More deterministic results
    excluded_domains=["youtube.com", "twitter.com"]
)
```

## Testing

Run the test suite:
```bash
python test_researcher.py
```

Run examples:
```bash
python example_usage.py
```

## API Reference

### `create_company_researcher()`
Factory function to create a researcher instance.

**Parameters:**
- `anthropic_api_key` (str, optional): Anthropic API key
- `tavily_api_key` (str, optional): Tavily API key  
- `max_queries` (int): Maximum queries per company (default: 6)
- `max_results_per_query` (int): Max results per query (default: 3)
- `max_reflections` (int): Max reflection cycles (default: 2)

### `research_company_sync()`
Synchronous research method.

**Parameters:**
- `company_name` (str): Name of company to research
- `user_notes` (str, optional): Additional context or focus areas

**Returns:**
- `company_info`: Extracted structured information
- `research_metadata`: Execution statistics  
- `messages`: Process log messages
- `search_results`: Raw search results

## Error Handling

The system includes comprehensive error handling:
- Invalid JSON responses from LLM are caught and logged
- Search API failures are handled gracefully
- Maximum iteration limits prevent infinite loops
- Fallback mechanisms ensure completion even with errors

## Performance

- **Parallel Search**: Web searches executed concurrently for speed
- **Token Optimization**: Search results truncated to stay within LLM limits
- **Caching**: Supports LangSmith caching for repeated queries
- **Rate Limiting**: Respects API rate limits with sequential execution fallback

## Limitations

- Relies on web search quality and availability
- Information accuracy depends on search result quality
- May be limited by API rate limits for high-volume usage
- Reflection quality depends on LLM reasoning capabilities

## Contributing

To extend the system:
1. Add new nodes by extending base node classes
2. Modify graph flow in `graph.py`
3. Update schema in `models.py` for new data fields
4. Add configuration options in `config.py`

## Support

For issues and questions:
1. Check the test suite output
2. Verify API keys are configured correctly
3. Review error messages in the process log
4. Ensure dependencies are installed correctly