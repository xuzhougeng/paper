# PaperCLI

A command-line tool for searching academic papers using LLM-powered query understanding and ranking.

## Features

- **Multi-source search**: PubMed, OpenAlex, Google Scholar (via SerpAPI), arXiv
- **LLM-powered query understanding**: Automatically extracts intent, expands synonyms, and translates queries
- **Platform-specific query generation**: Generate optimized search queries for PubMed, Google Scholar, or Web of Science
- **PDF text extraction**: Extract text from PDFs using Doc2X API with page-level JSONL output
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

# Optional: Required for PDF extraction (paper extract command)
export DOC2X_API_KEY="sk-..."  # Get from https://open.noedgeai.com
```

注: PAPERCLI_INTENT_MODEL建议模型不能弱于gpt-4o。如果无法使用官方API，可以通过LLM_BASE_URL设置第三方代理。例如我用的是[CloseAI](https://referer.shadowai.xyz/r/12432)。

Or create a `~/.papercli.toml` configuration file:

```toml
[llm]
base_url = "https://api.openai.com/v1"
intent_model = "gpt-4o-mini"  # Default: gpt-4o-mini (for query rewriting)
eval_model = "gpt-4o"          # Default: gpt-4o (for paper evaluation)

[cache]
path = "~/.cache/papercli.sqlite"
enabled = true

[doc2x]
api_key = "sk-..."  # Optional: for PDF extraction
# base_url = "https://v2.doc2x.noedgeai.com"  # Default
```

## Usage

### Basic search

```bash
paper find "CRISPR gene editing for cancer therapy"
```

### With options

```bash
# Return top 10 results in JSON format
paper find "machine learning for drug discovery" --top-n 10 --format json

# Search only PubMed and OpenAlex
paper find "protein folding prediction" --sources pubmed,openalex

# Use specific models
paper find "neural networks" --intent-model gpt-4o-mini --eval-model gpt-4o

# Show all retrieved papers (skip LLM ranking)
paper find "CRISPR therapy" --show-all

# Verbose output
paper find "single cell RNA sequencing" --verbose
```

### Output formats

- `table` (default): Rich formatted table
- `json`: JSON output for programmatic use
- `md`: Markdown format

#### How it works

![How PaperCLI Works](assets/workflow.png)

1. **Query Intent Extraction**: LLM analyzes your query to extract keywords, synonyms, and search intent
2. **Multi-source Search**: Searches PubMed, OpenAlex, Scholar, and optionally arXiv in parallel
3. **Deduplication**: Removes duplicate papers using DOI, source IDs, and normalized titles
4. **Coarse Ranking**: Uses lexical matching to reduce candidates to a manageable number
5. **LLM Reranking**: Each candidate is evaluated by LLM for relevance, with evidence extraction
6. **Top-N Output**: Returns the most relevant papers with supporting evidence


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

### Extract text from PDF

Use `extract` to parse a PDF file using [Doc2X](https://doc2x.noedgeai.com/) and output page-level JSONL. Each line contains one page with extracted text.

```bash
# Extract to stdout (pipe to file or other tools)
paper extract paper.pdf

# Extract to a file
paper extract paper.pdf --out result.jsonl

# Include raw page data (for debugging or further processing)
paper extract paper.pdf --include-raw --out result.jsonl

# With verbose output
paper extract paper.pdf --verbose
```

Output JSONL format (one JSON object per line):
```json
{"doc2x_uid": "...", "source_path": "/path/to/paper.pdf", "page_index": 0, "page_no": 1, "text": "..."}
{"doc2x_uid": "...", "source_path": "/path/to/paper.pdf", "page_index": 1, "page_no": 2, "text": "..."}
```

Options:
- `--out PATH`: Write output to file instead of stdout
- `--poll-interval FLOAT`: Seconds between status polls (default: 2.0)
- `--timeout FLOAT`: Maximum wait time in seconds (default: 900)
- `--include-raw/--no-include-raw`: Include raw page data in output
- `--verbose/-V`: Show detailed progress
- `--quiet/-q`: Suppress progress output

### Structure `result.jsonl` for database ingestion

`paper structure` performs a **second-pass parse** on the page-level JSONL produced by `paper extract`, and outputs a single structured JSON object with fields like title/abstract/methods/results and main vs supplementary figures/tables.

```bash
# Turn result.jsonl into a single structured JSON document
paper structure result.jsonl --out structured.json

# Or output a Markdown report for easier reading
paper structure result.jsonl --out structured.md
# (equivalent) paper structure result.jsonl --format md --out structured.md
```

## License

MIT

