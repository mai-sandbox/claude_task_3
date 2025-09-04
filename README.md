# Company Researcher - LangGraph Agent

A comprehensive company research agent built using LangGraph that intelligently gathers structured information about companies through web search and reflection.

## Features

- **Multi-node graph architecture** with intelligent routing
- **Parallel web searches** for improved speed using Tavily API
- **LLM-generated queries** tailored to information gaps
- **Reflection-based quality assessment** with iterative improvement
- **Structured output** with comprehensive company information
- **Configurable limits** for queries, search results, and reflection rounds
- **Deployment-ready** with proper error handling

## Architecture

The agent consists of 5 main nodes:

1. **Query Generation**: Analyzes information gaps and generates targeted search queries
2. **Web Search**: Executes searches in parallel using Tavily API
3. **Information Extraction**: Uses LLM to extract and structure data from search results
4. **Reflection**: Assesses information quality and determines if more research is needed
5. **Format Response**: Prepares final structured output

## Output Schema

The agent fills the following structured object:

```json
{
  "company_name": "string (required)",
  "founding_year": "integer",
  "founder_names": ["array of strings"],
  "product_description": "string",
  "funding_summary": "string",
  "notable_customers": "string"
}
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export TAVILY_API_KEY="your-tavily-api-key"
```

### 3. Test the Installation

```bash
python test_agent.py
```

## Usage

### Simple Usage

```python
from agent import research_company

# Research a company with default settings
company_info = research_company(
    company_name="OpenAI",
    user_notes="Focus on recent AI developments",
    max_queries=8,
    max_reflections=3
)

print(company_info.model_dump_json(indent=2))
```

### Advanced Usage

```python
from agent import app
from langchain_core.messages import HumanMessage

# Use the graph directly for more control
initial_state = {
    "messages": [HumanMessage(content="Research Tesla")],
    "company_name": "Tesla", 
    "user_notes": "Focus on electric vehicles and energy",
    "company_info": CompanyInfo(company_name="Tesla"),
    "search_results": [],
    "queries_executed": 0,
    "reflection_count": 0,
    "max_queries": 8,
    "max_search_results": 5,
    "max_reflections": 3,
    "is_complete": False
}

result = app.invoke(initial_state)
print(result["company_info"].model_dump_json(indent=2))
```

## Configuration

Default limits can be customized:

- `max_queries`: Maximum search queries per company (default: 8)
- `max_search_results`: Maximum results per query (default: 5) 
- `max_reflections`: Maximum reflection rounds (default: 3)

## Graph Flow

```
START → Generate Queries → Web Search → Extract Info → Reflect
                ↑                                          ↓
                └── Continue Research? ←─ Yes ──────────── ┘
                                    ↓ No
                              Format Response → END
```

## Key Features

### Parallel Search Execution
Multiple search queries are executed concurrently for faster results.

### Intelligent Query Generation
The LLM analyzes current information gaps and generates targeted queries.

### Quality Assessment
The reflection step ensures comprehensive information gathering before completion.

### Robust Error Handling
Graceful handling of API failures and missing configuration.

## Files

- `agent.py`: Main agent implementation with graph definition
- `langgraph.json`: LangGraph configuration for deployment
- `requirements.txt`: Python dependencies
- `test_agent.py`: Test script and usage examples

## Deployment

The agent is deployment-ready and exports the compiled graph as `app` for use with LangGraph deployment platforms.