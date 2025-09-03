# Company Researcher - LangGraph Multi-Node System

A sophisticated company research system built with LangGraph that uses AI-powered web searches to gather comprehensive company information. The system employs a multi-node graph architecture with intelligent query generation, parallel web searching, data extraction, and reflection capabilities.

## 🏗️ Architecture

This system implements a **multi-node LangGraph** with the following components:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Generate       │    │  Web Search     │    │  Extract Data   │    │   Reflect       │
│  Queries        │───▶│  (Parallel)     │───▶│  & Structure    │───▶│  & Decide       │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                                                      │
         │                                                                      ▼
         └──────────────────────────────────────────────────────────── ┌─────────────┐
                                 "continue" decision                    │    END      │
                                                                        └─────────────┘
```

### Node Functions:

1. **Generate Queries**: AI-powered query generation targeting missing company information
2. **Web Search**: Parallel execution of searches using Tavily API 
3. **Extract Data**: Structure extraction and data consolidation using LLM
4. **Reflect**: Quality assessment and completion evaluation with retry logic

## 🎯 Features

- **Multi-node LangGraph architecture** with conditional routing
- **Parallel web searches** for improved speed
- **Intelligent query generation** based on missing information
- **Structured data extraction** to CompanyInfo schema
- **Reflection and retry logic** with configurable limits
- **Comprehensive conversation tracking** in messages array
- **Configurable search and reflection limits**
- **Async/await support** for better performance

## 📊 Output Schema

The system extracts information into a structured `CompanyInfo` object:

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

## 🚀 Quick Start

### 1. Setup

```bash
# Clone or download the repository
cd company-researcher

# Run the setup script
python setup.py
```

### 2. Configure API Keys

Create a `.env` file:
```env
OPENAI_API_KEY=your_openai_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

### 3. Run Examples

```bash
# Run interactive demo
python example_usage.py

# Or use the researcher directly in Python
python -c "
import asyncio
from company_researcher import CompanyResearcher

async def main():
    researcher = CompanyResearcher(
        openai_api_key='your_key',
        tavily_api_key='your_key'
    )
    result = await researcher.research_company('Anthropic', 'AI safety company')
    print(result)

asyncio.run(main())
"
```

## 🔧 Configuration

### Basic Configuration

```python
from company_researcher import CompanyResearcher

researcher = CompanyResearcher(
    openai_api_key="your_key",
    tavily_api_key="your_key",
    max_search_queries=5,     # Max queries per company
    max_search_results=3,     # Max results per query  
    max_reflections=2         # Max reflection steps
)
```

### Advanced Configuration

```python
from config import THOROUGH_CONFIG, FAST_CONFIG

# Use predefined configs
researcher = CompanyResearcher(**THOROUGH_CONFIG.__dict__)

# Or customize
config = ResearchConfig(
    max_search_queries=8,
    max_search_results_per_query=5,
    max_reflection_steps=3,
    llm_model="gpt-4o-mini",
    verbose_logging=True
)
```

## 🎮 Usage Examples

### Research Single Company

```python
import asyncio
from company_researcher import CompanyResearcher

async def research_company():
    researcher = CompanyResearcher(
        openai_api_key="your_key",
        tavily_api_key="your_key"
    )
    
    result = await researcher.research_company(
        company_name="Stripe",
        notes="Payment processing platform"
    )
    
    print(f"Company: {result['company_name']}")
    print(f"Founded: {result.get('founding_year', 'Unknown')}")
    print(f"Founders: {', '.join(result.get('founder_names', []))}")
    print(f"Product: {result.get('product_description', 'Unknown')}")

asyncio.run(research_company())
```

### Batch Research

```python
companies = ["OpenAI", "Anthropic", "Cohere"]

async def research_batch():
    researcher = CompanyResearcher(openai_key, tavily_key)
    
    tasks = [
        researcher.research_company(company) 
        for company in companies
    ]
    
    results = await asyncio.gather(*tasks)
    return results
```

## 📁 Project Structure

```
company-researcher/
├── company_researcher.py   # Main research engine
├── example_usage.py       # Usage examples and demos  
├── config.py             # Configuration classes
├── setup.py              # Setup and installation script
├── requirements.txt      # Python dependencies
├── .env                 # API keys (create from template)
└── README.md            # This documentation
```

## 🔍 How It Works

### 1. Query Generation Node
- Analyzes existing company information to identify gaps
- Generates targeted search queries using AI
- Respects maximum query limits per company

### 2. Web Search Node  
- Executes searches in parallel using Tavily API
- Handles search failures gracefully
- Aggregates results from multiple queries

### 3. Data Extraction Node
- Uses LLM to extract structured information
- Merges with existing data, avoiding duplicates
- Validates output against CompanyInfo schema

### 4. Reflection Node
- Evaluates information completeness (1-5 scale)
- Decides whether to continue searching or finish
- Respects maximum reflection step limits

### Termination Conditions
The graph terminates when:
- Information is deemed sufficient by reflection
- Maximum search queries reached
- Maximum reflection steps reached

## ⚙️ Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_search_queries` | 5 | Maximum web searches per company |
| `max_search_results` | 3 | Maximum results per search query |
| `max_reflections` | 2 | Maximum reflection/retry cycles |
| `llm_model` | "gpt-4o-mini" | OpenAI model for processing |
| `search_timeout` | 30s | Timeout for individual searches |

## 🎯 Key Features Implemented

✅ **Multi-node LangGraph** with conditional routing  
✅ **Parallel web searches** using asyncio.gather()  
✅ **Intelligent query generation** targeting missing data  
✅ **Structured data extraction** to specified schema  
✅ **Reflection and retry logic** with configurable limits  
✅ **Conversation tracking** in messages array  
✅ **Configurable search limits** and result limits  
✅ **Async/await architecture** for performance  

## 🛠️ Requirements

- Python 3.8+
- OpenAI API key
- Tavily API key  
- Dependencies in requirements.txt

## 📝 License

This project is open source and available under the MIT License.