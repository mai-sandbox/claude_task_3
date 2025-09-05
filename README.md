# Company Research Agent

A multi-node LangGraph system for researching companies using the Tavily API. This intelligent agent conducts comprehensive company research by generating targeted search queries, performing parallel web searches, extracting structured information, and reflecting on completeness.

## Features

- **Multi-Node Graph Architecture**: Built with LangGraph for robust workflow orchestration
- **Intelligent Query Generation**: LLM generates targeted search queries based on missing information
- **Parallel Web Search**: Uses Tavily API with concurrent searches for improved speed
- **Structured Information Extraction**: Extracts data into a standardized CompanyInfo schema
- **Reflection & Iteration**: Automatically assesses completeness and re-searches if needed
- **Configurable Limits**: Set max queries, results per query, and reflection steps
- **Conversation Tracking**: Maintains a log of the research process
- **CLI Interface**: Easy-to-use command-line interface

## CompanyInfo Schema

The system extracts the following structured information:

```json
{
  "company_name": "Official name of the company",
  "founding_year": "Year the company was founded (integer)",
  "founder_names": ["Names", "of", "founding", "team"],
  "product_description": "Brief description of main product/service",
  "funding_summary": "Summary of funding history",
  "notable_customers": "Known customers using the product/service"
}
```

## Installation

1. Clone this repository
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

### CLI Interface
```bash
python company_researcher.py
```

### API Keys Required
- **OpenAI API Key**: For LLM operations (get from [OpenAI Platform](https://platform.openai.com/account/api-keys))
- **Tavily API Key**: For web search (get from [Tavily](https://tavily.com/))

### Configuration Options
- `max_queries`: Maximum search queries per research cycle (default: 5)
- `max_results_per_query`: Maximum results per search query (default: 3)
- `max_reflections`: Maximum reflection/retry cycles (default: 3)

## Graph Architecture

The system consists of four main nodes:

1. **Query Generation Node**: Analyzes missing information and generates targeted search queries
2. **Web Search Node**: Performs parallel searches using Tavily API
3. **Information Extraction Node**: Extracts structured data from search results using LLM
4. **Reflection Node**: Evaluates completeness and decides whether to continue research

## Testing

Run the test suite with mock data (no API keys required):
```bash
python test_researcher.py
```

## Example Output

```
🔍 Company Research Agent
==================================================

Enter company name to research: OpenAI
Enter optional notes (press Enter to skip): 

🚀 Starting research for: OpenAI
This may take a few moments...

=== Research Process Log ===
[USER] Research company: OpenAI
[SYSTEM] Generated queries: ['OpenAI founding history founders', 'OpenAI ChatGPT GPT products', 'OpenAI funding Microsoft investment']
[SYSTEM] Found 9 search results
[SYSTEM] Extracted information for OpenAI
[SYSTEM] Reflection 1: Completeness 100.0%. Research complete

==================================================
📊 RESEARCH RESULTS
==================================================
🏢 Company Name: OpenAI
📅 Founded: 2015
👥 Founders: Sam Altman, Elon Musk, Greg Brockman
🎯 Product/Service: AI research company developing ChatGPT and GPT models
💰 Funding: Over $11 billion raised including Microsoft investment
🤝 Notable Customers: Microsoft, enterprises using OpenAI API

💾 Results saved to: openai_research.json
```

## Files

- `company_researcher.py`: Main application with LangGraph implementation
- `test_researcher.py`: Test suite with mock data
- `requirements.txt`: Python dependencies
- `.env.example`: Environment variable template
- `README.md`: This documentation