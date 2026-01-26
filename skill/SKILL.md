---
name: paper-cli
description: Install and use PaperCLI (papercli) for paper search, PDF processing, and slide generation. Use when the user mentions paper-cli/papercli or asks about installing or using the PaperCLI commands.
---

# Paper CLI

## Scope
Use this skill to guide installation and day-to-day usage of the `paper` CLI based on repo docs in `README.md` and `doc/how-to-use.md`.

## Quick start
Prefer `uv` for env and install.

```bash
# Create and activate a Python 3.12 venv
uv venv --python 3.12
source .venv/bin/activate

# Install from GitHub
uv pip install git+https://github.com/xuzhougeng/paper.git
```

```bash
# Verify the CLI
paper --help
```

## Configuration
Use env vars or a config file. Env vars override config.

```bash
# Required for most commands
export LLM_API_KEY="sk-..."

# Optional model overrides
export PAPERCLI_INTENT_MODEL="gpt-4o-mini"
export PAPERCLI_EVAL_MODEL="gpt-4o"

# Optional per-feature keys
export SERPAPI_API_KEY="your-serpapi-key"
export DOC2X_API_KEY="sk-..."
export UNPAYWALL_EMAIL="your@email.com"
export NCBI_API_KEY="your-ncbi-key"
export GEMINI_API_KEY="your-gemini-key"
```

```toml
# ~/.config/papercli.toml (or ~/.papercli.toml)
[llm]
api_key = "sk-..."
base_url = "https://api.openai.com/v1"
intent_model = "gpt-4o-mini"
eval_model = "gpt-4o"

[doc2x]
api_key = "sk-..."

[unpaywall]
email = "your@email.com"

[api_keys]
serpapi_key = "..."
ncbi_api_key = "..."

[gemini]
api_key = "..."
```

## Command patterns
Always present commands as short, runnable blocks with a brief note above them.

### Find papers
Finds and ranks papers across sources.

```bash
paper find "CRISPR gene editing for cancer therapy"
```

```bash
# Limit sources and output JSON
paper find "protein folding prediction" --sources pubmed,openalex --top-n 10 --format json
```

### Cite statements
Splits text with LLM and finds citations per sentence.

```bash
paper cite --text "Your statement here."
```

```bash
# From a file to a report
paper cite --in notes.txt --out report.md --top-k 3
```

### Generate platform queries
Creates platform-specific queries without searching.

```bash
paper gen-query "single cell RNA velocity" --platform scholar
```

### Extract PDF to JSONL
Uses Doc2X to parse a PDF into page-level JSONL.

```bash
paper extract paper.pdf --out result.jsonl --image-dir ./images
```

### Structure JSONL
Turns JSONL into a single structured JSON or Markdown file.

```bash
paper structure result.jsonl --out structured.md
```

### Fetch PDF by DOI
Downloads an open-access PDF via Unpaywall or PMC fallback.

```bash
paper fetch-pdf 10.1038/nature12373 --out-dir ./pdfs
```

```bash
# Only show PDF URL
paper fetch-pdf 10.1038/nature12373 --no-download --format json
```

### Generate highlight slides
Creates a single-page PNG slide from text.

```bash
paper slide --in article.txt --style academic --out summary.png
```

## Troubleshooting
Use these only when the user hits the specific issue.

```bash
# If OA PDF download returns 403, set a browser user agent
export PAPERCLI_PDF_USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
paper fetch-pdf 10.1101/2025.03.04.641200
```

## When unsure
If flags or behavior are unclear, check `README.md` and `doc/how-to-use.md`. For CLI options not documented, check `papercli/cli.py`.
