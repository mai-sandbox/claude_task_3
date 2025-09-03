# Company Researcher with LangGraph

A multi-node company research system built using LangGraph that automatically researches companies by generating intelligent search queries and extracting structured information.

## Features

- 🔍 **Multi-Node Architecture**: Built with LangGraph for robust, stateful workflows
- 🚀 **Parallel Web Searches**: Uses Tavily API for fast, concurrent web searches  
- 🤖 **Intelligent Query Generation**: LLM generates targeted search queries based on information gaps
- 🔄 **Reflection & Iteration**: Evaluates research completeness and continues if needed
- 📊 **Structured Output**: Returns consistent CompanyInfo schema
- ⚡ **Configurable Limits**: Set max queries, results, and reflection rounds
- 🎯 **User Notes Support**: Include custom research focus areas

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your API keys:
```bash
cp .env.example .env
# Edit .env file with your actual API keys
```

## Required API Keys

- **Tavily API Key**: Get from [tavily.com](https://tavily.com/) for web searches
- **OpenAI API Key**: Get from [platform.openai.com](https://platform.openai.com/) for LLM calls

## Usage

### Basic Usage

```python
import asyncio
from company_researcher import CompanyResearcher

async def main():
    researcher = CompanyResearcher(
        tavily_api_key="your_tavily_key",
        openai_api_key="your_openai_key"
    )
    
    company_info = await researcher.research_company(
        company_name="OpenAI",
        user_notes="Focus on ChatGPT and recent developments"
    )
    
    print(company_info.model_dump_json(indent=2))

asyncio.run(main())
```

### Configuration Options

```python
researcher = CompanyResearcher(
    tavily_api_key="your_key",
    openai_api_key="your_key",
    max_queries=5,              # Max search queries per company
    max_results_per_query=3,    # Max results per search
    max_reflections=2           # Max reflection rounds
)
```

### Run Examples

```bash
python example_usage.py
```

## CompanyInfo Schema

The system extracts information into this structured format:

```json
{
  "company_name": "string (required)",
  "founding_year": "integer or null",
  "founder_names": ["array of strings"],
  "product_description": "string or null", 
  "funding_summary": "string or null",
  "notable_customers": "string or null"
}
```

## Architecture

The LangGraph workflow consists of these nodes:

1. **Initialize**: Set up research state and parameters
2. **Generate Queries**: LLM creates targeted search queries for missing information
3. **Execute Searches**: Parallel web searches using Tavily API
4. **Extract Information**: LLM extracts relevant info from search results
5. **Reflect & Evaluate**: Assess completeness and decide if more research is needed
6. **Finalize**: Prepare final structured output

## Workflow Features

- **State Management**: Tracks research progress, query count, and reflection rounds
- **Parallel Processing**: Executes multiple web searches concurrently for speed
- **Iterative Refinement**: Continues research until information is sufficient or limits reached
- **Error Handling**: Gracefully handles API failures and malformed responses
- **Message Tracking**: Maintains conversation history throughout the workflow