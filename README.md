# Company Research Agent

A multi-node LangGraph workflow for comprehensive company research using the Tavily API and Claude AI.

## Features

🔍 **Multi-Node Architecture**
- Query generation node for creating targeted search queries
- Parallel search execution using Tavily API
- Information extraction with structured output
- Reflection step with retry logic for completeness

📊 **Comprehensive Research**
- Company founding information (year, founders)
- Product/service descriptions
- Funding history and investors
- Notable customers and clients
- Configurable search limits and reflection loops

🚀 **LangGraph Integration**
- StateGraph workflow with conditional routing
- Message-based state management
- Deployment-ready configuration
- Error handling and retry mechanisms

## Architecture

```
┌─────────────────┐
│   User Input    │
│ (Company Name   │
│   + Notes)      │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Generate Search │
│    Queries      │
│ (Max 5 queries) │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│ Execute Parallel│
│    Searches     │
│ (Tavily API)    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Extract       │
│  Information    │
│ (Structured)    │
└─────────┬───────┘
          │
          ▼
┌─────────────────┐
│   Reflect on    │
│  Completeness   │
│ (Max 2 loops)   │
└─────────┬───────┘
          │
    ┌─────▼─────┐
    │Complete?  │
    └─────┬─────┘
          │
    ┌─────▼─────┐
    │    No     │     │    Yes    │
    └─────┬─────┘     └─────┬─────┘
          │                 │
          ▼                 ▼
┌─────────────────┐   ┌─────────────────┐
│ Generate More   │   │   Final Results │
│    Queries      │   │   (CompanyInfo) │
└─────────────────┘   └─────────────────┘
```

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables in `.env`:
   ```bash
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Usage

### Basic Usage

```python
from agent import research_company

# Research a company
result = research_company(
    company_name="Anthropic",
    notes="Focus on AI safety research"
)

print(f"Company: {result.company_name}")
print(f"Founded: {result.founding_year}")
print(f"Founders: {', '.join(result.founder_names)}")
```

### Interactive Demo

Run the interactive demo to test different companies:

```bash
python demo.py
```

### Test Suite

Run the test suite:

```bash
python test_agent.py
```

## Configuration

### Constants (in agent.py)

- `MAX_SEARCH_QUERIES = 5` - Maximum search queries per research session
- `MAX_SEARCH_RESULTS = 3` - Maximum results per search query
- `MAX_REFLECTION_STEPS = 2` - Maximum reflection/retry loops

### LangGraph Configuration

The agent is configured in `langgraph.json` for deployment:

```json
{
  "dependencies": ["."],
  "graphs": {
    "company_researcher": "./agent.py:app"
  },
  "env": ".env"
}
```

## Data Structure

The agent extracts information into a structured `CompanyInfo` object:

```python
class CompanyInfo(BaseModel):
    company_name: str
    founding_year: Optional[int]
    founder_names: List[str]
    product_description: Optional[str]
    funding_summary: Optional[str]
    notable_customers: Optional[str]
```

## Workflow Details

1. **Query Generation**: Uses Claude to generate 5 targeted search queries based on company name and user notes
2. **Parallel Search**: Executes searches using Tavily API in sequence (can be made parallel)
3. **Information Extraction**: Uses Claude with structured output to extract company information
4. **Reflection**: Analyzes completeness (60% threshold) and decides whether to continue
5. **Additional Research**: If incomplete, generates more targeted queries and repeats the process

## Error Handling

- API key validation
- Search failure tolerance
- Structured output validation
- State management error recovery

## Deployment

The agent is designed for LangGraph Platform deployment:

1. Ensure `langgraph.json` is configured
2. Set environment variables
3. Deploy using LangGraph CLI or Platform

## Limitations

- Limited to English language searches
- Depends on Tavily API availability
- Information accuracy depends on search results quality
- Rate limits apply to both Anthropic and Tavily APIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source. See LICENSE file for details.