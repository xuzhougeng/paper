# PaperCLI

A command-line tool for searching academic papers using LLM-powered query understanding and ranking.

## Features

- **Multi-source search**: PubMed, OpenAlex, Google Scholar (via SerpAPI), arXiv
- **LLM-powered query understanding**: Automatically extracts intent, expands synonyms, and translates queries
- **Smart ranking**: Coarse lexical ranking followed by LLM-based relevance scoring
- **Evidence extraction**: Returns the most relevant quote from each paper
- **Configurable models**: Use different LLM models for intent extraction vs evaluation
- **Caching**: SQLite-based caching to avoid redundant API calls

## Installation

```bash
# From source
pip install -e .

# With dev dependencies
pip install -e ".[dev]"
```

## Configuration

Set the following environment variables:

```bash
# Required: LLM API key (OpenAI or compatible)
export OPENAI_API_KEY="sk-..."
# Or use a custom key name
export LLM_API_KEY="your-key"

# Required for Google Scholar search
export SERPAPI_API_KEY="your-serpapi-key"

# Optional: Custom LLM endpoint (OpenAI-compatible)
export LLM_BASE_URL="https://api.openai.com/v1"

# Optional: Model configuration
export PAPERCLI_INTENT_MODEL="gpt-4o-mini"
export PAPERCLI_EVAL_MODEL="gpt-4o"
```

Or create a `~/.papercli.toml` configuration file:

```toml
[llm]
base_url = "https://api.openai.com/v1"
intent_model = "gpt-4o-mini"
eval_model = "gpt-4o"

[cache]
path = "~/.cache/papercli.sqlite"
enabled = true
```

## Usage

### Basic search

```bash
paper find "CRISPR gene editing for cancer therapy"
```

### With options

```bash
# Return top 5 results in JSON format
paper find "machine learning for drug discovery" --top-n 5 --format json

# Search only PubMed and OpenAlex
paper find "protein folding prediction" --sources pubmed,openalex

# Use specific models
paper find "neural networks" --intent-model gpt-4o-mini --eval-model gpt-4o

# Verbose output
paper find "single cell RNA sequencing" --verbose
```

### Output formats

- `table` (default): Rich formatted table
- `json`: JSON output for programmatic use
- `md`: Markdown format

## How it works

1. **Query Intent Extraction**: LLM analyzes your query to extract keywords, synonyms, and search intent
2. **Multi-source Search**: Searches PubMed, OpenAlex, Scholar, and optionally arXiv in parallel
3. **Deduplication**: Removes duplicate papers using DOI, source IDs, and normalized titles
4. **Coarse Ranking**: Uses lexical matching to reduce candidates to a manageable number
5. **LLM Reranking**: Each candidate is evaluated by LLM for relevance, with evidence extraction
6. **Top-N Output**: Returns the most relevant papers with supporting evidence

## License

MIT

