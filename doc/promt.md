# Prompt 说明（当前仓库的提示词一览与修改指南）

本文档用于说明 `papercli/` 里目前有哪些 prompt（包括 system prompt 与用户侧 prompt 模板）、各自用途、源码位置，以及如果要修改应从哪里改、如何自测验证。

> 说明：本仓库里有些 prompt 是“常量字符串”（如 `EVAL_SYSTEM_PROMPT`），也有一些是“运行时拼接的模板字符串”（如 `extract_intent()` 内的 f-string）。本文档两类都会列出。

## 总览

- **智能分句/是否需要引文判断（`paper cite`）**：`papercli/cli.py`
- **检索意图提取 & 平台 Query 生成（`paper find` 内部使用）**：`papercli/query.py`
- **候选论文相关性评估/重排（`paper find` 内部使用）**：`papercli/eval.py`
- **单页 Slide：亮点抽取(JSON) + 图片生成 prompt（`paper slide`）**：`papercli/slide.py`
- **重要公共行为：JSON-only 约束追加与重试策略**：`papercli/llm.py`（会影响多处 prompt 的最终 system instruction）

---

## 1) `paper cite`：LLM 分句 + needs_citation 判断

### 1.1 system prompt（固定字符串）

- **位置**：`papercli/cli.py` 里的 `_SEGMENTATION_SYSTEM_PROMPT`
- **用途**：指导模型按“语义/篇章关系”分段，而不是仅按标点；并判断每段是否需要引文（`needs_citation`）
- **内容**：

```text
You are an expert discourse analyst. Your task is to segment the given text into
meaningful discourse units and determine whether each segment needs a citation.

Guidelines for segmentation:
1. Segment by semantic/discourse boundaries, NOT just punctuation.
2. Group closely related statements that share a single claim or idea.
3. Separate statements that make distinct claims requiring different citations.
4. Consider discourse relations: continuation, contrast, cause-effect, elaboration.
5. Each segment should be a complete, citable unit of meaning.

Guidelines for citation judgment (needs_citation):
1. Set needs_citation=false for:
   - Author's own work/contributions ("We propose...", "We conducted...", "Our approach...")
   - Paper structure descriptions ("This paper is organized as follows...")
   - Transitional sentences that don't make factual claims
   - Common knowledge that requires no citation
2. Set needs_citation=true for:
   - Claims about others' research findings or conclusions
   - Specific methods/techniques from prior work
   - Domain facts, statistics, or consensus statements
   - Comparisons referencing other work

Return ONLY valid JSON matching the required schema.
```

### 1.2 user prompt（固定模板字符串）

- **位置**：`papercli/cli.py` 里的 `_SEGMENTATION_USER_PROMPT`
- **用途**：承载待处理文本，并约束输出字段（`segments[].text/needs_citation/...`）
- **内容**（注意其中 `{text}` 会被 `.format(text=...)` 替换）：

```text
Segment the following text into discourse units and determine if each needs citation.

Text to segment:
---
{text}
---

Return JSON with a "segments" array. Each segment must have:
- "text": the exact text of this segment (required, non-empty)
- "needs_citation": whether this segment needs a literature citation (required, boolean)
- "citation_reason": brief explanation for the citation decision (optional, 1 sentence)
- "relation_to_prev": how this relates to the previous segment (optional; MUST be one of: continuation, contrast, cause, elaboration, shift, other; if unsure use "other")
- "reason": brief explanation for this segmentation choice (optional)
```

### 1.3 修改建议与自测

- **如何修改**：直接编辑 `papercli/cli.py` 的 `_SEGMENTATION_SYSTEM_PROMPT` / `_SEGMENTATION_USER_PROMPT`。
- **注意事项**：
  - 输出会被 `LLMClient.complete_json()` 解析并做 Pydantic 校验（见下文“公共 JSON 约束”），所以别随意改字段名/结构，除非你同步改了对应的响应模型（`SegmentationResponse`）。
- **推荐自测**：
  - 运行单测：`pytest -k cite`
  - 或运行命令：`paper cite --text "..."`（需要配置 LLM）

---

## 2) `paper find`：检索意图提取（QueryIntent）

### 2.1 system prompt（固定字符串）

- **位置**：`papercli/query.py` 的 `INTENT_SYSTEM_PROMPT`
- **用途**：指导模型把用户自然语言问题抽取成“检索意图结构”（英文 query、关键词、同义词、必须短语、排除词等）
- **内容**：

```text
You are a search query expert specializing in academic literature search. Given a user's natural language query about academic papers, you must:

1. First, think step-by-step about what the user really wants to find
2. Analyze the domain, key concepts, and research context
3. Generate an optimized English search query for academic databases
4. Identify key terms, synonyms, abbreviations, and related concepts
5. Determine if any specific phrases must appear or terms should be excluded

Focus on scientific/academic terminology. Always expand abbreviations and provide common synonyms.
If the query is in Chinese or another language, translate it to English for search while preserving the original meaning.
```

### 2.2 user prompt（运行时拼接模板）

- **位置**：`papercli/query.py` 的 `extract_intent()` 内部 `prompt = f"""..."""`（会把用户 query 插进去）
- **用途**：明确要求返回字段（`reasoning/query_en/keywords/synonyms/...`），并补充中文 query 的可选字段 `query_zh`
- **内容（模板骨架）**：请直接查看源码中的该 f-string（因为包含较长的字段说明与示例）。

### 2.3 修改建议与自测

- **如何修改**：编辑 `papercli/query.py`：
  - 改 `INTENT_SYSTEM_PROMPT`：影响“整体策略/风格”
  - 改 `extract_intent()` 里的 f-string：影响“返回字段要求/格式细节/示例”
- **推荐自测**：
  - 运行单测：`pytest -k intent or pytest -k query`
  - 或运行命令：`paper find "..."`（需要配置 LLM + 数据源）

---

## 3) `paper find`：平台特定 query 生成（PubMed / Scholar / WoS）

### 3.1 system prompt（平台特定固定字符串）

- **位置**：`papercli/query.py` 的 `PLATFORM_SYSTEM_PROMPTS` 字典（键：`pubmed`/`scholar`/`wos`）
- **用途**：把“抽取后的 intent”翻译成各平台可直接粘贴使用的查询语法
- **内容**：见 `papercli/query.py` 中 `PLATFORM_SYSTEM_PROMPTS["pubmed"]` / `["scholar"]` / `["wos"]` 三段长字符串（包含平台语法规则与示例）。

### 3.2 user prompt（运行时拼接模板）

- **位置**：`papercli/query.py` 的 `generate_platform_query()` 内部 `prompt = f"""..."""`（把 intent 结构化信息展开后塞进去）
- **用途**：要求模型返回 `platform_query`（可直接复制粘贴）和 `notes`（1-3 句小建议）

### 3.3 修改建议与自测

- **如何修改**：
  - 想提升某个平台 query 的“语法准确性/可用性”，优先改 `PLATFORM_SYSTEM_PROMPTS[platform]`
  - 想改变输出字段/说明方式，改 `generate_platform_query()` 的 prompt f-string
- **推荐自测**：`pytest -k query`，以及手动跑 `paper find --sources ...` 观察生成的 query 是否可直接在平台使用。

---

## 4) `paper find`：候选论文相关性评估/证据抽取（rerank）

### 4.1 system prompt（固定字符串）

- **位置**：`papercli/eval.py` 的 `EVAL_SYSTEM_PROMPT`
- **用途**：对每篇论文给出 0-10 分、是否满足需求、并从 title/abstract 抽取“原文证据片段”（严格要求是原文子串）
- **内容**：

```text
You are an expert at evaluating academic paper relevance. Given a user's search query and a paper's title and abstract, you must:

1. Score the paper's relevance from 0-10 (10 = perfect match)
2. Determine if the paper meets the user's need (true/false)
3. Extract the most relevant quote from the title or abstract that supports why this paper is relevant
4. Identify which field the quote came from (title or abstract)
5. Provide a brief reason for your assessment

Be strict but fair. A paper should score high only if it directly addresses the user's query.
The evidence_quote must be an EXACT substring from the title or abstract - do not paraphrase.
```

### 4.2 user prompt（运行时拼接模板）

- **位置**：`papercli/eval.py` 的 `_build_eval_prompt()`（把用户 query、论文 title、abstract 拼成一段指令）
- **用途**：给模型足够上下文并要求输出固定字段（由 `PaperEvaluation` 模型约束）

### 4.3 修改建议与自测

- **如何修改**：编辑 `papercli/eval.py` 的 `EVAL_SYSTEM_PROMPT` / `_build_eval_prompt()`
- **注意事项**：
  - 代码里有 `_validate_evidence()` 会检查 `evidence_quote` 是否真的出现在 title/abstract；你若修改 prompt 让模型更“抽象总结”，会更容易触发校验失败/回退。
- **推荐自测**：`pytest -k eval` 或直接 `pytest tests/test_eval.py`

---

## 5) `paper slide`：亮点抽取(JSON) + 图片生成 prompt

### 5.1 风格 prompt（固定字典）

- **位置**：`papercli/slide.py` 的 `STYLE_PROMPTS`
- **用途**：为图片生成提供风格描述（handdrawn/minimal/academic/dark/colorful）

### 5.2 亮点抽取 system instruction（固定字符串，Gemini）

- **位置**：`papercli/slide.py` 的 `HIGHLIGHT_EXTRACTION_SYSTEM`
- **用途**：指导 Gemini 输出一个 JSON 对象：`title/subtitle/bullets/takeaway/keywords`
- **内容**：见 `papercli/slide.py` 的 `HIGHLIGHT_EXTRACTION_SYSTEM` 长字符串（包含结构与约束）。

### 5.3 亮点抽取 user prompt（运行时拼接模板）

- **位置**：`papercli/slide.py` 的 `_build_highlight_prompt(text, num_bullets)`
- **用途**：把 article text（截断到 12000 字符）塞入 prompt，并要求输出 JSON

### 5.4 Slide 图片生成 user prompt（运行时拼接模板）

- **位置**：`papercli/slide.py` 的 `_build_slide_prompt(highlights, style, aspect_ratio)`
- **用途**：把 highlights（标题、要点、takeaway、可选 keywords）变成“生成单页 slide 图”的指令，并把 `STYLE_PROMPTS[style]` 注入进去

### 5.5 修改建议与自测

- **如何修改**：编辑 `papercli/slide.py` 的 `STYLE_PROMPTS` / `HIGHLIGHT_EXTRACTION_SYSTEM` / `_build_*_prompt()`
- **推荐自测**：`pytest -k slide` 或 `pytest tests/test_slide.py`

---

## 6) 公共行为：JSON-only 约束追加（会影响多处 prompt）

在 `papercli/llm.py` 的 `LLMClient.complete_json()` 中，会在你传入的 `system_prompt` 后面**自动追加一段 JSON-only 指令**（并且在解析失败时会用更严格版本重试）。这会影响：\n\n- `query.py` 里的 `llm.intent_completion(...)`\n- `eval.py` 里的 `llm.eval_completion(...)`\n- `cli.py` 里的 `llm_client.complete_json(...)`\n\n如果你遇到“模型总是输出 markdown/代码块/解释文字”，或想统一收紧/放宽 JSON 输出约束，优先改这里，而不是每个业务 prompt 都改一遍。\n\n> Gemini 的 `generate_text_json()`（`papercli/gemini.py`）不会自动追加这种 system 约束，它是“先生成文本，再尝试从文本里解析 JSON”，因此 `slide.py` 里才需要 `HIGHLIGHT_EXTRACTION_SYSTEM` 强调“Return ONLY JSON”。\n+
