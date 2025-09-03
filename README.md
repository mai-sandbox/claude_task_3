# Company Researcher with LangGraph

A multi-node graph-based company research system built with LangGraph that automatically gathers comprehensive company information using intelligent web searches.

## 🌟 Features

- **Multi-Node Architecture**: Uses LangGraph's graph-based workflow for systematic research
- **Intelligent Query Generation**: LLM generates targeted search queries based on missing information
- **Parallel Web Search**: Executes multiple searches concurrently using Tavily API
- **Smart Information Extraction**: Extracts and aggregates data from multiple sources
- **Reflection & Quality Control**: Assesses research completeness and determines if more data is needed
- **Configurable Limits**: Set maximum queries, results, and reflection steps
- **Structured Output**: Returns standardized CompanyInfo object

## 📋 Company Information Schema

```json
{
  "company_name": "Official name of the company",
  "founding_year": "Year the company was founded",
  "founder_names": ["Names of the founding team members"],
  "product_description": "Brief description of the company's main product or service",
  "funding_summary": "Summary of the company's funding history",
  "notable_customers": "Known customers that use company's product/service"
}
```

## 🏗️ Graph Architecture

```
START → generate_queries → parallel_search → extract_info → reflect
                ↑                                              ↓
                └─────────── (continue) ←──────────────────────┘
                                  ↓ (end)
                                 END
```

### Nodes:

1. **generate_queries**: Creates targeted search queries based on missing company information
2. **parallel_search**: Executes multiple web searches concurrently using Tavily API
3. **extract_info**: Extracts and aggregates information from search results using LLM
4. **reflect**: Assesses research completeness and decides whether to continue or finish

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- OpenAI API key
- Tavily API key

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd claude-task-3-main
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env and add your API keys:
# OPENAI_API_KEY=your_openai_api_key_here
# TAVILY_API_KEY=your_tavily_api_key_here
```

### Basic Usage

```python
from company_researcher import CompanyResearcher

# Initialize researcher
researcher = CompanyResearcher()

# Research a company
result = researcher.research_company("OpenAI", "AI research company")

# Access results
print(f"Company: {result.company_name}")
print(f"Founded: {result.founding_year}")
print(f"Founders: {result.founder_names}")
print(f"Product: {result.product_description}")
print(f"Funding: {result.funding_summary}")
print(f"Customers: {result.notable_customers}")
```

### Interactive Mode

```bash
python company_researcher.py
```

### Run Tests

```bash
python test_researcher.py
```

## ⚙️ Configuration

Customize the research behavior using the `Config` class:

```python
from company_researcher import CompanyResearcher, Config

config = Config(
    max_search_queries=5,        # Maximum web searches per company
    max_search_results_per_query=3,  # Results per search query
    max_reflection_steps=2,      # Maximum reflection iterations
    model_name="gpt-4o-mini"     # OpenAI model to use
)

researcher = CompanyResearcher(config)
```

## 🔍 How It Works

1. **Input Processing**: Accept company name and optional user notes
2. **Query Generation**: LLM analyzes missing information and generates targeted search queries
3. **Parallel Search**: Execute multiple web searches concurrently for speed
4. **Information Extraction**: LLM processes search results and extracts relevant company data
5. **Reflection**: Assess completeness and quality, decide if more research is needed
6. **Iteration**: If more data needed and within limits, generate new queries and repeat
7. **Output**: Return structured CompanyInfo object with gathered information

## 🎛️ Advanced Features

### Custom Search Strategies
- Adaptive query generation based on missing fields
- Parallel execution for improved performance
- Intelligent result filtering and ranking

### Quality Control
- Reflection mechanism to assess research completeness
- Configurable completeness thresholds
- Prevents information overwriting unless more accurate data is found

### Rate Limiting & Cost Control
- Configurable maximum queries per research session
- Configurable maximum results per query
- Reflection step limits to prevent infinite loops

## 📊 Output Example

```json
{
  "company_name": "OpenAI",
  "founding_year": 2015,
  "founder_names": ["Sam Altman", "Elon Musk", "Greg Brockman", "Ilya Sutskever"],
  "product_description": "OpenAI develops artificial general intelligence (AGI) and provides AI services including GPT models, DALL-E, and ChatGPT.",
  "funding_summary": "Raised over $11 billion in multiple funding rounds, with significant investments from Microsoft, Khosla Ventures, and others.",
  "notable_customers": "Microsoft, GitHub, Shopify, and thousands of developers through API access"
}
```

## 🔧 Troubleshooting

### Common Issues

1. **API Keys**: Ensure both OPENAI_API_KEY and TAVILY_API_KEY are set in your environment
2. **Rate Limits**: If hitting API limits, reduce max_search_queries or add delays
3. **Token Limits**: Large search results may exceed model context; results are automatically truncated
4. **Network Issues**: The system handles search failures gracefully and continues with available data

### Error Handling

- Invalid API responses are caught and handled gracefully
- Failed searches don't stop the entire research process
- Malformed LLM outputs fall back to safe defaults

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## 📄 License

This project is open source and available under the MIT License.