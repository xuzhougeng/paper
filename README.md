# PaperCLI

A command-line tool for searching academic papers using LLM-powered query understanding and ranking.

## Features

- **Multi-source search**: PubMed, OpenAlex, Google Scholar (via SerpAPI), arXiv
- **LLM-powered query understanding**: Automatically extracts intent, expands synonyms, and translates queries
- **Platform-specific query generation**: Generate optimized search queries for PubMed, Google Scholar, or Web of Science
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
export LLM_API_KEY="sk-..."

# Optional: Custom LLM endpoint (OpenAI-compatible, defaults to OpenAI)
export LLM_BASE_URL="https://api.openai.com/v1"

# Optional: Model configuration (defaults shown)
export PAPERCLI_INTENT_MODEL="gpt-4o-mini"  # For query rewriting
export PAPERCLI_EVAL_MODEL="gpt-4o"          # For paper evaluation

# Optional: Required for Google Scholar search
export SERPAPI_API_KEY="your-serpapi-key"
```

Or create a `~/.papercli.toml` configuration file:

```toml
[llm]
base_url = "https://api.openai.com/v1"
intent_model = "gpt-4o-mini"  # Default: gpt-4o-mini (for query rewriting)
eval_model = "gpt-4o"          # Default: gpt-4o (for paper evaluation)

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

### Generate platform-specific search queries

Use `gen-query` to generate an optimized search query for a specific database platform without actually searching. This is useful when you want to run the search manually or refine the query.

```bash
# Generate a PubMed search query (default)
paper gen-query "CRISPR gene editing for cancer therapy"

# Generate a Google Scholar search query
paper gen-query "single cell RNA velocity" --platform scholar

# Generate a Web of Science search query
paper gen-query "machine learning drug discovery" --platform wos

# Output as Markdown (easy to copy to notes)
paper gen-query "protein structure prediction" --platform pubmed --format md

# Output as JSON (for programmatic use)
paper gen-query "neural networks" --platform scholar --format json
```

Supported platforms:
- `pubmed` (default): PubMed/MEDLINE - uses Boolean operators, field tags like `[Title/Abstract]`, `[MeSH Terms]`
- `scholar`: Google Scholar - optimized for shorter, keyword-focused queries
- `wos`: Web of Science - uses `TS=`, `TI=` field tags and `NEAR/x` proximity operators

Platform aliases: `google_scholar` → `scholar`, `web_of_science` / `world_of_knowledge` → `wos`

## How it works

1. **Query Intent Extraction**: LLM analyzes your query to extract keywords, synonyms, and search intent
2. **Multi-source Search**: Searches PubMed, OpenAlex, Scholar, and optionally arXiv in parallel
3. **Deduplication**: Removes duplicate papers using DOI, source IDs, and normalized titles
4. **Coarse Ranking**: Uses lexical matching to reduce candidates to a manageable number
5. **LLM Reranking**: Each candidate is evaluated by LLM for relevance, with evidence extraction
6. **Top-N Output**: Returns the most relevant papers with supporting evidence

## License

MIT

