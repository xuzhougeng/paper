# 如何使用

## 软件配置

安装软件 (要求能够顺利访问github)

```bash
pip install git+https://github.com/xuzhougeng/paper.git
```

配置环境变量，以[CloseAI](https://referer.shadowai.xyz/r/12432)第三方API KEY为例

```bash
# Required: LLM API key (OpenAI or compatible)
export LLM_BASE_URL="https://api.openai-proxy.org/v1"  # for OpenAI
export LLM_API_KEY="sk-..."

# Optional: Model configuration (defaults shown)
export PAPERCLI_INTENT_MODEL="gpt-5.2"  # For query rewriting
export PAPERCLI_EVAL_MODEL="gpt-4o"          # For paper evaluation

# Optional: Required for Google Scholar search
export SERPAPI_API_KEY="your-serpapi-key"

# Optional: Required for PDF extraction (paper extract command)
export DOC2X_API_KEY="sk-..."  # Get from https://open.noedgeai.com

# Optional: Required for PDF download by DOI (paper fetch-pdf command)
export UNPAYWALL_EMAIL="your@email.com"  # Required for Unpaywall API
export NCBI_API_KEY="your-ncbi-key"       # Optional, improves PMC rate limits

# Optional: Required for slide generation (paper slide command)
export GEMINI_BASE_URL="https://api.openai-proxy.org/google/v1beta"  # for Googlee
export GEMINI_API_KEY="your-gemini-key"  # Get from Google AI Studio
```

## 文献检索

寻找引文支持，这一步依赖于OpenAI API，和SERPAPI API , 都需要API KEY.

```bash
paper find "作物单细胞制备困难" 
```

输出结果

```
Found 5 relevant papers (5 highly relevant) from 107 candidates

╭─ 1. FX-Cell: a method for single-cell RNA sequencing on difficult-to-digest and cryo... ────────────────────╮
│ Score: 10.0/10 ✓ Meets need                                                                                 │
│ 2025 | Xin Ming, Mu-Chun Wan et al. | bioRxiv (Cold Spring Harbor Laboratory) | [openalex]                  │
│                                                                                                             │
│ "a method for single-cell RNA sequencing on difficult-to-digest and cryopreserved plant samples" — title    │
│                                                                                                             │
│ The paper directly addresses the difficulty of preparing single-cell samples from hard-to-digest plant      │
│ tissues, which aligns with the user's query about the difficulty of preparing single-cell crops.            │
│                                                                                                             │
│ https://doi.org/10.1101/2025.03.04.641200                                                                   │
│                                                                                                             │
│ DOI: 10.1101/2025.03.04.641200                                                                              │
╰─────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
...
```

cite 命令使用 LLM 进行智能分句，不是简单地按标点符号拆分，而是根据语句之间的语义关系（如因果、对比、延续等）推断出更合适的拆分方式。分句使用的是 `intent_model`（意图理解模型），你可以通过环境变量或命令行参数覆盖。

```bash
paper cite --text "作物单细胞制备困难。不同组织的酶解条件差异很大！"
```

从文本文件读取并输出报告：

```bash
paper cite --in notes.txt --out report.md --top-k 3
```

支持 stdin：

```bash
cat notes.txt | paper cite --top-k 2
```

指定其他模型进行分句：

```bash
paper cite --text "..." --intent-model gpt-4o
```


## 文献下载

根据DOI下载PDF，这一步基于unpaywall API(无需KEY)

```bash
paper fetch-pdf 10.1101/2025.03.04.641200
```

输出结果: 

```bash
DOI          10.1101/2025.03.04.641200                                                           
Source       unpaywall                                                                           
PDF URL      https://www.biorxiv.org/content/biorxiv/early/2025/03/07/2025.03.04.641200.full.pdf 
Landing URL  https://www.biorxiv.org/content/biorxiv/early/2025/03/07/2025.03.04.641200.full.pdf 
OA Status    green                                                                               

Error downloading PDF: HTTP 403 Forbidden downloading PDF. The host may be blocking automated downloads (anti-bot). URL: https://www.biorxiv.org/content/biorxiv/early/2025/03/07/2025.03.04.641200.full.pdf. Try setting PAPERCLI_PDF_USER_AGENT to a browser User-Agent, or download via the landing page.
```

如果出现 OA status 是 green 但是返回 403，通常是站点反爬虫拦截（例如 bioRxiv）。可以尝试设置浏览器 User-Agent 再重试：

```bash
export PAPERCLI_PDF_USER_AGENT="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
paper fetch-pdf 10.1101/2025.03.04.641200
```

如果仍无法下载，就需要打开 Landing URL 手动下载。

## 文件结构化保存

将PDF转成Markdown格式，这一步需要doc2x API, 需要API Key.

```bash
paper extract paper.pdf --out result.jsonl --image-dir ./images
paper structure result.jsonl --out structured.md
```

生成的 `structured.md` 顶部包含 YAML front matter，包括自动提取的元数据（title、author、abstract、keywords、journal、date、doi 等），便于导入 Obsidian、Hugo、Jekyll 等工具：

```yaml
---
title: "Paper Title"
author:
  - John Smith
  - Jane Doe
abstract: |
  摘要内容...
keywords:
  - keyword1
  - keyword2
journal: Nature Methods
date: 2024-03-15
doi: 10.1234/example
---
```

## 制作单页Slider

分析markdown文本输入，基于亮点制作单页的Slider.

```bash
paper slide --in structured.md --out summary_default.png
```

![default](summary_default.png)

```bash
paper slide --in structured.md --out summary.png --style 'academic'
```

![academic](summary_academic.png)