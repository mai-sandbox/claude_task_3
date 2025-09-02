# Company Researcher using LangGraph

A multi-node graph-based company research tool that uses LLM-generated queries and the Tavily API to gather comprehensive company information.

## Features

✅ **Multi-node LangGraph workflow** with conditional routing  
✅ **Parallel web searching** with Tavily API for improved speed  
✅ **LLM-generated targeted search queries** based on missing information  
✅ **Structured information extraction** into CompanyInfo schema  
✅ **Reflection and quality assessment** with retry logic  
✅ **Configurable search limits** (queries, results, reflections)  
✅ **JSON export** of research results  
✅ **Interactive and CLI modes** for flexible usage  
✅ **Message tracking** throughout the research process  

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables in `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key
TAVILY_API_KEY=your_tavily_api_key
```

## Usage

### Command Line Interface
```bash
# Interactive mode
python main.py

# Research a company
python main.py "Company Name"

# Research with user notes
python main.py "Company Name" "Additional context or notes"
```

### Examples
```bash
python main.py "Anthropic" "AI safety company that created Claude"
python main.py "OpenAI" "Founded by Sam Altman, created ChatGPT"
python main.py "Stripe" "Payment processing company"
```

### Help
```bash
python main.py --help
```

## Company Information Schema

The tool extracts the following structured information:

```json
{
  "company_name": "Official name of the company",
  "founding_year": "Year the company was founded (integer)",
  "founder_names": ["Array", "of", "founder", "names"],
  "product_description": "Brief description of main product/service",
  "funding_summary": "Summary of the company's funding history",
  "notable_customers": "Known customers that use company's product/service"
}
```

## Workflow Architecture

The LangGraph workflow consists of 4 main nodes:

1. **Generate Queries** - LLM generates targeted search queries based on missing information
2. **Search** - Execute queries in parallel using Tavily API
3. **Extract** - LLM extracts structured information from search results
4. **Reflect** - Assess information quality and decide if more research is needed

### Flow Diagram
```
Start → Generate Queries → Search → Extract → Reflect
                ↑                              ↓
                └──────── (if incomplete) ─────┘
                                              ↓
                                            End
```

## Configuration

Default settings:
- **Max search queries**: 8 per company
- **Max results per query**: 5
- **Max reflection steps**: 3

These can be modified in the interactive mode or by editing the code.

## Output

The tool provides:
- **Console summary** with extracted company information
- **JSON export** with complete research data including:
  - Company information
  - Search queries used  
  - All search results
  - Research statistics
  - Message history

## Files

- `main.py` - Main execution script with CLI interface
- `workflow.py` - LangGraph workflow definition
- `nodes.py` - Individual workflow node implementations
- `schemas.py` - Pydantic data models
- `requirements.txt` - Python dependencies

## Error Handling

The tool includes robust error handling:
- Fallback queries if LLM generation fails
- Manual information extraction if JSON parsing fails
- Graceful degradation when search APIs are unavailable
- Automatic completion when limits are reached

## Examples of Successful Research

### Anthropic
- **Founded**: 2021
- **Founders**: Dario Amodei, Daniela Amodei, Chris Olah, Jack Clark, Sam McCandlish, Tom Brown, Jared Kaplan
- **Product**: Claude chatbot and family of language models focusing on AI safety
- **Funding**: $5B round at $170B valuation

### OpenAI  
- **Founded**: 2015
- **Founders**: Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremba, John Schulman
- **Product**: AI research and deployment including ChatGPT
- **Funding**: $8.3B round at $300B valuation, ~$40B total raised

## Contributing

To extend the tool:
1. Add new fields to `CompanyInfo` schema in `schemas.py`
2. Update query generation prompts in `nodes.py`
3. Modify extraction logic to handle new fields
4. Test with various companies to ensure robustness