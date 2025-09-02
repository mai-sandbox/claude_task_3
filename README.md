# Company Researcher

A LangGraph-powered company research tool that automatically gathers comprehensive information about companies using web search and AI extraction.

## Features

- **Multi-node LangGraph workflow** for systematic company research
- **Parallel web searches** using Tavily API for speed
- **AI-powered query generation** to find relevant information
- **Structured information extraction** into predefined schema
- **Reflection mechanism** to ensure research completeness
- **Configurable limits** for searches, results, and reflections
- **Conversation tracking** throughout the research process

## Company Information Schema

The system extracts the following information:

```json
{
  "company_name": "string (required)",
  "founding_year": "integer (optional)",
  "founder_names": ["array of strings (optional)"],
  "product_description": "string (optional)",
  "funding_summary": "string (optional)",
  "notable_customers": "string (optional)"
}
```

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env`:
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Quick Start

### Basic Usage

```python
from company_researcher import research_company

# Simple research
results = await research_company(
    company_name="OpenAI",
    user_notes="Focus on recent developments"
)

print(results['company_info'])
```

### Advanced Usage

```python
from company_researcher import CompanyResearcher
from config import DefaultConfigs

# Custom configuration
config = DefaultConfigs.thorough_research()

researcher = CompanyResearcher(
    max_search_queries=config.max_search_queries,
    max_search_results=config.max_search_results,
    max_reflections=config.max_reflections
)

results = await researcher.research_company(
    company_name="Stripe",
    user_notes="Payment processing company"
)

researcher.print_results(results)
```

## Configuration Options

### Research Presets

- **Quick Research**: 4 queries, 1 reflection (fast)
- **Balanced Research**: 8 queries, 3 reflections (default)
- **Thorough Research**: 12 queries, 5 reflections (comprehensive)

### Custom Configuration

```python
researcher = CompanyResearcher(
    model="gpt-4",                    # OpenAI model
    max_search_queries=10,            # Max search queries per company
    max_search_results=60,            # Max total search results
    max_reflections=3                 # Max reflection steps
)
```

## Workflow Architecture

The LangGraph workflow consists of four main nodes:

1. **Query Generation**: AI generates specific search queries
2. **Web Search**: Executes searches in parallel using Tavily
3. **Information Extraction**: Extracts structured data from results
4. **Reflection**: Validates completeness and decides if more research is needed

```
[Start] → Generate Queries → Web Search → Extract Info → Reflect
              ↑                                           ↓
              └─────────── Continue? ←────────────────────┘
                              │
                              ↓
                            [End]
```

## Examples

Run the example script to see different usage patterns:

```bash
python example_usage.py
```

Examples include:
- Basic company research
- Custom configurations
- Batch processing multiple companies
- Interactive research mode

## API Reference

### CompanyResearcher Class

#### Constructor
```python
CompanyResearcher(
    openai_api_key: str = None,      # Defaults to env var
    tavily_api_key: str = None,      # Defaults to env var  
    model: str = "gpt-4",            # OpenAI model
    max_search_queries: int = 8,     # Max queries per company
    max_search_results: int = 50,    # Max total results
    max_reflections: int = 3         # Max reflection steps
)
```

#### Methods
```python
async def research_company(company_name: str, user_notes: str = None) -> Dict[str, Any]
def print_results(results: Dict[str, Any]) -> None
```

### Convenience Function
```python
async def research_company(
    company_name: str,
    user_notes: str = None,
    **kwargs
) -> Dict[str, Any]
```

## Output Format

Results include:

- **company_info**: Extracted company data
- **research_summary**: Statistics about the research process
- **messages**: Detailed log of all steps taken
- **search_queries**: All queries generated and executed

Example output:
```json
{
  "company_info": {
    "company_name": "OpenAI",
    "founding_year": 2015,
    "founder_names": ["Sam Altman", "Elon Musk", "Greg Brockman"],
    "product_description": "AI research and deployment company",
    "funding_summary": "Raised over $11B from Microsoft and others",
    "notable_customers": "Microsoft, GitHub, enterprises"
  },
  "research_summary": {
    "queries_executed": 6,
    "max_queries": 8,
    "results_collected": 23,
    "reflections_done": 2,
    "research_complete": true
  }
}
```

## Configuration Files

- **config.py**: Research configuration and presets
- **models.py**: Pydantic data models
- **nodes.py**: LangGraph node implementations
- **company_researcher.py**: Main research orchestrator

## Error Handling

The system includes robust error handling:
- Missing API keys validation
- Search failures graceful degradation
- JSON parsing error recovery
- Rate limiting protection

## Limitations

- Requires OpenAI and Tavily API keys
- Search results quality depends on web content availability
- Information accuracy depends on source reliability
- Rate limits apply based on API plan

## Contributing

Feel free to submit issues and enhancement requests!

## License

MIT License