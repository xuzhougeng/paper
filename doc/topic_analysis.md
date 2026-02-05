

```bash
# Search papers published in Bioinformatics in 2025
paper find "single cell RNA-seq" --year 2025 --venue Bioinformatics

# Analyze topics from search results
paper topics "bioinformatics methods" --year 2025 --venue Bioinformatics --out topics.md

# Two-step workflow
paper find "genomics" --year 2025 --show-all --format json > papers.json
paper topics --in papers.json --format md --out analysis.md
```