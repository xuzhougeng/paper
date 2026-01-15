# PaperCLI 模块架构文档

本文档介绍 `papercli` 目录下的各个 Python 模块的功能、职责和相互关系。

## 目录结构

```
papercli/
├── __init__.py          # 包初始化，版本信息
├── cli.py               # CLI 命令行接口
├── config.py            # 配置管理
├── cache.py             # SQLite 缓存
├── models.py            # 数据模型定义
├── llm.py               # LLM 客户端 (OpenAI 兼容)
├── gemini.py            # Gemini 客户端 (文本+图片生成)
├── query.py             # 查询意图提取
├── pipeline.py          # 主搜索流程
├── rank.py              # 去重与排序
├── eval.py              # LLM 重排序与评估
├── output.py            # 输出格式化
├── slide.py             # Slide 生成 (亮点提取+图片)
├── doc2x.py             # Doc2X PDF 解析服务
├── extract.py           # JSONL 转换工具
├── structure.py         # 结构化解析
├── pdf_fetch.py         # DOI PDF 下载 (Unpaywall/PMC)
└── sources/             # 数据源适配器
    ├── __init__.py
    ├── base.py          # 基类
    ├── pubmed.py        # PubMed
    ├── openalex.py      # OpenAlex
    ├── scholar.py       # Google Scholar
    └── arxiv.py         # arXiv
```

---

## 核心模块

### `__init__.py`
**功能**: 包初始化与版本管理

- 定义 `__version__` 版本号
- 作为 papercli 包的入口点
- 导出包的公共接口

---

### `cli.py`
**功能**: 命令行接口 (CLI) 定义

使用 [Typer](https://typer.tiangolo.com/) 框架构建命令行工具，提供以下命令：

#### 主要命令

1. **`find`** - 语义搜索学术论文
   - 解析用户查询意图
   - 多源并行搜索 (PubMed、OpenAlex、Scholar、arXiv)
   - 去重与粗排
   - LLM 重排序与证据提取
   - 格式化输出

2. **`cite`** - 按句生成引用报告
   - 读取文本输入（文件、参数或 STDIN）
   - **LLM 语义分句**: 使用 LLM 根据语篇关系（因果、对比、延续等）智能拆分文本，而非简单标点分句
   - 逐句检索并为每句寻找最合适的引用
   - 输出每句 top-K 结果，标注推荐 top-1
   - 生成 Markdown 报告（默认 `report.md`）
   - **Fail-fast**: LLM 调用失败或返回无效 JSON 时直接报错退出

3. **`gen-query`** - 生成平台特定搜索查询
   - 为 PubMed、Google Scholar、Web of Science 等平台生成优化查询
   - 不执行实际搜索，仅生成查询字符串

4. **`extract`** - PDF 转 JSONL (页面级)
   - 调用 Doc2X API 解析 PDF
   - 输出页面级别的 JSONL 文件
   - 支持进度回调和详细日志

5. **`structure`** - JSONL 结构化解析 (二次解析)
   - 将页面级 JSONL 转换为数据库友好的结构化 JSON
   - 提取 title、abstract、methods、results、references 等字段
   - 分离主图表与补充图表

6. **`fetch-pdf`** - 基于 DOI 下载 PDF
   - 通过 Unpaywall API 查找开放获取 PDF
   - PMC 作为备选数据源 (DOI → PMID → PMCID → PDF)
   - 支持仅查询 URL 或直接下载

7. **`slide`** - 生成文章亮点总结 Slide
   - 读取文本输入（文件或 STDIN）
   - 使用 Gemini 提取关键亮点（标题、要点、结论）
   - 生成单页 16:9 PNG 图片
   - 支持多种视觉风格：手绘风 (handdrawn)、极简 (minimal)、学术 (academic)、深色科技 (dark)、多彩 (colorful)

#### 辅助功能
- 丰富的命令行参数 (verbose、quiet、format、output 等)
- 进度条和状态显示
- 异常处理与用户友好的错误消息

#### LLM 语义分句模块

`cli.py` 中定义了用于 `cite` 命令的 LLM 分句功能：

1. **`Segment` 模型** - 单个语篇单元
   - `text`: 分句文本（必填，不能为空）
   - `relation_to_prev`: 与前一句的关系（continuation/contrast/cause/elaboration/shift/other）
   - `reason`: 分句理由（调试用）

2. **`SegmentationResponse` 模型** - LLM 分句响应
   - `segments`: 分句列表（必须非空）

3. **`split_sentences_llm(text, llm_client, model)`** - LLM 分句函数
   - 使用 `intent_model`（默认 `gpt-4o-mini`）进行语义分句
   - 返回分句文本列表 `list[str]`
   - 失败时抛出 `LLMError`（fail-fast）

4. **`_split_sentences_simple(text)`** - 简单标点分句（保留作为参考）
   - 基于中英文标点符号和换行符拆分
   - 不再作为主要分句方式使用

---

## 配置与缓存

### `config.py`
**功能**: 配置管理与加载

使用 [Pydantic](https://pydantic-docs.helpmanual.io/) 定义配置模型，支持：

#### 配置项

1. **`LLMConfig`** - LLM 配置
   - `base_url`: API 基础 URL (默认 OpenAI)
   - `api_key`: API 密钥
   - `intent_model`: 意图提取模型 (默认 gpt-4o-mini)
   - `eval_model`: 论文评估模型 (默认 gpt-4o)
   - `timeout`: 请求超时时间
   - `max_retries`: 最大重试次数

2. **`CacheConfig`** - 缓存配置
   - `path`: SQLite 数据库路径
   - `enabled`: 是否启用缓存
   - `ttl_hours`: 缓存过期时间 (默认 7 天)

3. **`Doc2XConfig`** - Doc2X PDF 解析服务配置
   - `base_url`: Doc2X API 地址
   - `api_key`: API 密钥
   - `timeout`: HTTP 请求超时
   - `poll_interval`: 轮询间隔
   - `max_wait`: 最大等待时间 (默认 15 分钟)

4. **`UnpaywallConfig`** - Unpaywall PDF 下载服务配置
   - `base_url`: Unpaywall API 地址 (默认 `https://api.unpaywall.org/v2`)
   - `email`: 邮箱 (必填，用于 polite pool 访问)
   - `timeout`: HTTP 请求超时 (默认 30 秒)

5. **`APIKeysConfig`** - 第三方 API 密钥
   - `serpapi_key`: SerpAPI 密钥 (用于 Google Scholar)
   - `ncbi_api_key`: NCBI API 密钥 (可选，用于 PubMed)
   - `openalex_email`: OpenAlex 邮箱 (用于礼貌池)

6. **`GeminiConfig`** - Gemini API 配置 (Slide 生成)
   - `base_url`: API 地址 (默认 `https://api.openai-proxy.org/google/v1beta`)
   - `api_key`: Gemini API 密钥
   - `text_model`: 文本模型 (默认 `gemini-3-flash-preview`)
   - `image_model`: 图片生成模型 (默认 `gemini-3-pro-image-preview`)
   - `timeout`: 请求超时 (默认 120 秒)
   - `max_retries`: 最大重试次数 (默认 3)

#### 配置加载
- 支持从 `~/.config/papercli.toml` 或 `./papercli.toml` 加载
- 环境变量覆盖：`LLM_API_KEY`、`DOC2X_API_KEY`、`UNPAYWALL_EMAIL`、`GEMINI_API_KEY` 等
- 提供 `get_settings()` 函数获取全局配置实例

---

### `cache.py`
**功能**: SQLite 缓存层

为搜索结果和 LLM 响应提供持久化缓存，减少重复请求。

#### 核心功能

- **`Cache` 类**: SQLite 缓存管理器
  - `get(key)`: 获取缓存 (自动检查过期)
  - `set(key, value, ttl)`: 设置缓存
  - `delete(key)`: 删除缓存
  - `clear_expired()`: 清理过期缓存
  - `stats()`: 缓存统计信息

#### 特性
- 自动 JSON 序列化/反序列化
- TTL (Time-To-Live) 过期机制
- 支持自定义过期时间
- 索引优化查询性能

---

## LLM 相关模块

### `llm.py`
**功能**: OpenAI 兼容 LLM 客户端

封装 OpenAI API，提供结构化输出和重试机制。

#### 核心功能

1. **`LLMClient` 类**
   - `chat(messages, model, temperature)`: 发送聊天请求
   - `chat_structured(messages, response_model)`: 结构化输出 (Pydantic 模型)
   - 自动重试机制 (使用 tenacity)
   - 错误诊断与日志

2. **`LLMError` 异常**
   - 包含模型、API URL、原始响应等诊断信息
   - 便于调试和错误追踪

#### 特性
- 支持所有 OpenAI 兼容的 API (OpenAI、DeepSeek、GLM 等)
- 指数退避重试策略
- 结构化输出使用 `response_format` (JSON Schema)
- 详细的错误上下文

---

### `query.py`
**功能**: 查询意图提取与重写

使用 LLM 分析用户自然语言查询，生成优化的搜索查询。

#### 核心功能

1. **`extract_intent(query, llm, cache)`**
   - 提取用户搜索意图
   - 生成英文/中文查询
   - 提取关键词和同义词
   - 返回 `QueryIntent` 模型

2. **`generate_platform_query(query, platform, llm, cache)`**
   - 为特定平台 (PubMed、Scholar、WOS) 生成优化查询
   - 使用平台特定的 prompt 模板
   - 返回 `PlatformQueryResult` 模型

#### 支持的平台
- **PubMed**: 支持 MeSH、字段标签、布尔运算符
- **Google Scholar**: 简洁查询，短语匹配
- **Web of Science (WOS)**: TS=、AU=、SO= 字段搜索

---

### `eval.py`
**功能**: LLM 重排序与证据提取

对候选论文进行细粒度的相关性评估。

#### 核心功能

1. **`rerank_with_llm(query, papers, llm, cache, progress_callback)`**
   - 批量评估论文相关性
   - 为每篇论文打分 (0-10)
   - 提取支持证据 (引用原文)
   - 判断是否满足需求
   - 返回 `EvalResult` 列表

2. **`PaperEvaluation` 模型**
   - `score`: 相关性分数 (0-10)
   - `meets_need`: 是否满足需求 (bool)
   - `evidence_quote`: 证据引用 (原文片段)
   - `evidence_field`: 证据来源 (title/abstract)
   - `short_reason`: 简短理由

#### 评估策略
- 使用系统 prompt 引导 LLM 严格评估
- 要求引用原文片段 (不允许改写)
- 支持并发评估 (异步批处理)
- 进度回调支持

---

### `gemini.py`
**功能**: Gemini API 客户端 (文本与图片生成)

封装 Google Gemini API，支持文本生成和图片生成。

#### 核心功能

1. **`GeminiClient` 类**
   - `generate_text(prompt, model, ...)`: 文本生成
   - `generate_text_json(prompt, ...)`: 生成并解析为 JSON
   - `generate_image(prompt, aspect_ratio, image_size, ...)`: 图片生成
   - 自动重试机制 (使用 tenacity)
   - 参数回退策略 (aspect_ratio、image_size)

2. **`GeminiError` 异常**
   - 包含模型、API URL、状态码、原始响应等诊断信息
   - 便于调试和错误追踪

#### 特性
- 支持代理 URL (默认 `https://api.openai-proxy.org/google/v1beta`)
- 兼容不同字段名变体 (`inlineData`/`inline_data`、`mimeType`/`mime_type`)
- 图片生成支持多种宽高比 (16:9、4:3、1:1) 和尺寸 (1K、2K、4K)
- 自动 base64 解码图片数据

---

### `slide.py`
**功能**: 文章亮点 Slide 生成

将文章文本转换为单页视觉 Slide (PNG 图片)。

#### 核心功能

1. **`summarize_highlights(text, client, cache, num_bullets)`**
   - 使用 Gemini 提取文章关键亮点
   - 返回 `ArticleHighlights` 模型 (标题、要点、结论、关键词)
   - 支持缓存

2. **`render_slide_image(highlights, style, client, ...)`**
   - 根据亮点内容生成 Slide 图片
   - 支持多种视觉风格
   - 返回 PNG 字节数据

3. **`generate_slide(text, style, client, ...)`**
   - 完整流程：提取亮点 → 生成图片
   - 返回 (图片字节, 亮点数据) 元组

#### 支持风格
- **`handdrawn`**: 手绘/涂鸦/记号笔风格
- **`minimal`**: 极简信息图
- **`academic`**: 学术海报感
- **`dark`**: 深色科技风
- **`colorful`**: 多彩活力风

#### 数据模型
- **`ArticleHighlights`**: 提取的亮点结构
  - `title`: 主标题
  - `subtitle`: 副标题 (可选)
  - `bullets`: 要点列表 (1-8 条)
  - `takeaway`: 一句话结论
  - `keywords`: 视觉主题关键词

---

## 搜索流程模块

### `pipeline.py`
**功能**: 主搜索流程编排

协调整个论文搜索流程的各个阶段。

#### 核心流程

1. **`run_pipeline(query, sources, top_n, ...)`**
   - 意图提取 (query.py)
   - 多源并行搜索 (sources/)
   - 去重与粗排 (rank.py)
   - LLM 重排序 (eval.py)
   - 格式化输出 (output.py)

#### 流程控制
- 使用 Rich 库显示进度条
- 支持 `verbose`/`quiet` 模式
- 支持 `show_all` 模式 (跳过 LLM 评估)
- 异步执行 (asyncio)

#### 阶段说明
```
用户查询
  ↓
意图提取 (LLM)
  ↓
多源搜索 (并行)
  ├─ PubMed
  ├─ OpenAlex
  ├─ Google Scholar
  └─ arXiv
  ↓
去重 (DOI + 标题)
  ↓
粗排 (词法匹配)
  ↓
LLM 重排序 (细粒度评估)
  ↓
Top-N 输出
```

---

### `rank.py`
**功能**: 去重与粗排序

在 LLM 评估之前进行快速过滤，减少计算成本。

#### 核心功能

1. **`deduplicate(papers)`**
   - 基于 DOI 去重 (优先)
   - 基于 source_id 去重 (同源)
   - 基于归一化标题去重 (模糊匹配)
   - 选择元数据最完整的版本

2. **`coarse_rank(papers, intent, k)`**
   - 词法匹配 (关键词命中)
   - TF-IDF 相似度
   - 返回 Top-K 候选

#### 去重策略
```
优先级：DOI > 同源 ID > 标题归一化
元数据完整性：abstract > year > authors
```

---

### `output.py`
**功能**: 输出格式化

将评估结果格式化为多种输出格式。

#### 支持格式

1. **`table`** - Rich 表格 (终端)
   - 彩色高亮
   - 分数、证据、来源等字段
   - 适合交互式查看

2. **`json`** - JSON 格式
   - 结构化数据
   - 便于程序化处理
   - 包含完整元数据

3. **`md`** - Markdown 格式
   - 适合文档嵌入
   - 包含引用链接
   - 格式优美

#### 功能
- `format_output(results, format, top_n, show_all)`: 主格式化函数
- `format_platform_query_output(result, format)`: 平台查询输出格式化
- 支持截断长文本
- 支持显示所有论文 (`show_all` 模式)

---

## 数据模型

### `models.py`
**功能**: 核心数据结构定义

使用 Pydantic 定义所有数据模型，确保类型安全和验证。

#### 主要模型

1. **`Paper`** - 论文统一模型
   - `source`: 数据源 (pubmed/openalex/scholar/arxiv)
   - `source_id`: 源内 ID (PMID、OpenAlex ID 等)
   - `title`: 标题
   - `abstract`: 摘要
   - `year`: 发表年份
   - `authors`: 作者列表
   - `url`: 论文链接
   - `doi`: DOI 标识符
   - `venue`: 期刊/会议名称
   - `normalized_title`: 归一化标题 (用于去重)

2. **`QueryIntent`** - 查询意图
   - `reasoning`: LLM 推理过程
   - `query_en`: 英文查询
   - `query_zh`: 中文查询 (可选)
   - `keywords`: 关键词列表
   - `synonyms`: 同义词列表

3. **`EvalResult`** - 评估结果
   - `paper`: 论文对象
   - `score`: 相关性分数
   - `meets_need`: 是否满足需求
   - `evidence_quote`: 证据引用
   - `evidence_field`: 证据来源
   - `short_reason`: 简短理由

4. **`PlatformQueryResult`** - 平台查询结果
   - `platform`: 平台名称
   - `query`: 优化后的查询
   - `syntax_explanation`: 语法说明

---

## PDF 处理模块

### `doc2x.py`
**功能**: Doc2X PDF 解析服务客户端

封装 [Doc2X v2 API](https://doc2x.noedgeai.com/)，将 PDF 解析为结构化数据。

#### 核心功能

1. **`Doc2XClient` 类**
   - `parse_pdf(pdf_path, output_format, progress_callback)`: 解析 PDF
   - 自动文件上传
   - 轮询任务状态
   - 下载解析结果

2. **`Doc2XError` 异常**
   - 错误码映射 (quota_limit、file_too_large 等)
   - 诊断信息 (UID、状态、详情)
   - 可操作的错误消息

#### 支持格式
- `md`: Markdown (默认)
- `json`: 结构化 JSON
- `tex`: LaTeX
- `docx`: Word 文档

#### 错误处理
- 任务限制 (`parse_task_limit_exceeded`)
- 配额不足 (`parse_quota_limit`)
- 文件过大 (`parse_file_too_large`, 最大 300MB)
- 页数过多 (`parse_page_limit_exceeded`, 最大 2000 页)
- 超时 (`parse_timeout`, 最大 15 分钟)

---

### `extract.py`
**功能**: JSONL 转换工具

将 Doc2X 解析结果转换为页面级 JSONL 格式。

#### 核心功能

1. **`extract_page_text(page_obj)`**
   - 从 Doc2X 页面对象提取文本
   - 支持多种结构 (`md`、`text`、`blocks`)
   - 最大努力提取策略

2. **`result_to_jsonl(result, source_path, doc2x_uid)`**
   - 将 Doc2X 结果转换为 JSONL
   - 每行一个页面对象
   - 包含元数据 (doc2x_uid、source_path、page_index 等)

#### JSONL 格式
```jsonl
{"doc2x_uid": "...", "source_path": "paper.pdf", "page_index": 0, "page_no": 1, "text": "..."}
{"doc2x_uid": "...", "source_path": "paper.pdf", "page_index": 1, "page_no": 2, "text": "..."}
...
```

---

### `structure.py`
**功能**: 结构化解析 (二次解析)

将页面级 JSONL 转换为数据库友好的结构化 JSON。

#### 核心功能

1. **`structure_from_jsonl_path(jsonl_path)`**
   - 读取页面级 JSONL
   - 提取 title、abstract、methods、results 等字段
   - 分离主图表与补充图表
   - 关联图表与图例
   - 返回 `StructuredPaper` 模型

2. **输出字段**
   - `title`: 标题
   - `abstract`: 摘要
   - `methods`: 方法部分
   - `results`: 结果部分
   - `references`: 参考文献
   - `appendix`: 附录
   - `main_figures`: 主图列表
   - `main_tables`: 主表列表
   - `supp_figures`: 补充图列表
   - `supp_tables`: 补充表列表
   - `warnings`: 解析警告

#### 模型定义

- **`ExtractedFigure`**: 图
  - `figure_id`: 图编号 (如 "Figure 1")
  - `caption`: 图例文本
  - `page_index`/`page_no`: 页码
  - `image_urls`: 图片 URL
  - `alt_text`: 替代文本
  - `is_supplementary`: 是否为补充图

- **`ExtractedTable`**: 表
  - `table_id`: 表编号 (如 "Table 1")
  - `caption`: 表标题
  - `page_index`/`page_no`: 页码
  - `markdown_content`: Markdown 表格内容
  - `is_supplementary`: 是否为补充表

#### 特性
- 跨页图例关联
- 补充材料识别 (S1、S2 等)
- 尽力而为策略 (best-effort)
- 警告机制 (解析异常、缺失字段等)

---

### `pdf_fetch.py`
**功能**: 基于 DOI 的 PDF 下载服务

通过 Unpaywall (主) 和 PMC (备) 两个数据源查找并下载开放获取的 PDF。

#### 核心功能

1. **`UnpaywallClient` 类** - Unpaywall API 客户端
   - `lookup(doi)`: 查询 DOI 的开放获取信息
   - 返回 PDF 直链、着陆页 URL、OA 状态、许可证等

2. **`PMCClient` 类** - NCBI PMC API 客户端
   - `doi_to_pmid(doi)`: DOI → PMID (通过 ESearch)
   - `pmid_to_pmcid(pmid)`: PMID → PMCID (通过 ELink)
   - `pmcid_to_pdf_url(pmcid)`: 构建 PMC PDF URL
   - `lookup(doi)`: 完整链路查询

3. **`fetch_pdf_url(doi, settings, ...)`** - 主入口函数
   - 先查 Unpaywall，找到 PDF 直链即返回
   - 否则回退到 PMC 链路
   - 支持跳过某个数据源

4. **`download_pdf(pdf_url, out_path)`** - 下载 PDF 到本地

#### 数据模型

- **`PDFResult`**: PDF 查询结果
  - `doi`: DOI
  - `pdf_url`: PDF 直链 (如有)
  - `landing_url`: 着陆页 URL
  - `source`: 数据来源 (`unpaywall`/`pmc`/`none`)
  - `oa_status`: OA 状态 (gold/green/hybrid/bronze/closed)
  - `license`: 许可证信息
  - `pmid`/`pmcid`: PubMed/PMC ID
  - `error`: 错误信息

#### API 流程

```
DOI
 │
 ├─ Unpaywall ──► best_oa_location.url_for_pdf ──► PDF
 │
 └─ (fallback) PMC:
      DOI → ESearch → PMID
               ↓
            ELink → PMCID
               ↓
      https://pmc.ncbi.nlm.nih.gov/articles/{PMCID}/pdf/
```

#### 错误处理
- DOI 格式验证
- 网络错误与超时
- 非 PDF 内容检测 (检查 magic bytes)
- 用户友好的错误消息

---

## 数据源适配器

### `sources/` 目录
包含各个学术搜索平台的适配器。

#### `base.py`
**功能**: 数据源基类

定义 `BaseSource` 抽象类，所有数据源适配器必须实现：
- `search(intent, max_results)`: 搜索接口
- `name`: 数据源名称

---

#### `pubmed.py`
**功能**: PubMed 搜索适配器

使用 [NCBI E-utilities API](https://www.ncbi.nlm.nih.gov/books/NBK25501/)。

##### 核心流程
1. **ESearch**: 根据查询获取 PMID 列表
2. **EFetch**: 批量获取论文详情 (XML 格式)
3. 解析 XML 提取 title、abstract、authors 等

##### 特性
- 支持 NCBI API Key (提高速率限制)
- 缓存支持
- XML 解析 (PubmedArticle)
- 自动提取 DOI、PubMed URL

---

#### `openalex.py`
**功能**: OpenAlex 搜索适配器

使用 [OpenAlex API](https://openalex.org/)。

##### 核心流程
1. 调用 `/works` 接口搜索
2. 解析 `abstract_inverted_index` (倒排索引) → 完整摘要
3. 提取作者、DOI、发表信息

##### 特性
- 支持礼貌池 (polite pool) - 提供邮箱可提速
- 开放 API，无需密钥
- 丰富的元数据 (引用数、开放访问状态等)

---

#### `scholar.py`
**功能**: Google Scholar 搜索适配器

使用 [SerpAPI](https://serpapi.com/) 代理 Google Scholar。

##### 核心流程
1. 调用 SerpAPI `/search` 接口
2. 指定 `engine=google_scholar`
3. 解析 JSON 响应

##### 特性
- **需要 SerpAPI 密钥** (商业服务)
- 无密钥时自动跳过 Scholar 搜索
- 支持片段 (snippet) 作为摘要

---

#### `arxiv.py`
**功能**: arXiv 搜索适配器

使用 [arXiv API](https://arxiv.org/help/api/)。

##### 核心流程
1. 构建 arXiv 查询 (all:term1 AND all:term2)
2. 调用 `/api/query` 接口
3. 解析 Atom XML 格式

##### 特性
- 开放 API，无需密钥
- 支持全文搜索
- 自动提取 arXiv ID、分类、作者

---

## 模块依赖关系

```
cli.py
  ├─ config.py (加载配置)
  ├─ llm.py (LLM 分句 - cite 命令)
  ├─ pipeline.py (执行搜索流程)
  ├─ doc2x.py (PDF 解析)
  ├─ extract.py (JSONL 转换)
  ├─ structure.py (结构化解析)
  ├─ pdf_fetch.py (DOI PDF 下载)
  └─ slide.py (Slide 生成)

pipeline.py
  ├─ query.py (意图提取)
  ├─ sources/ (多源搜索)
  ├─ rank.py (去重与粗排)
  ├─ eval.py (LLM 重排序)
  └─ output.py (格式化输出)

query.py / eval.py
  ├─ llm.py (LLM 客户端)
  └─ cache.py (缓存)

slide.py
  ├─ gemini.py (Gemini 客户端)
  ├─ models.py (ArticleHighlights 模型)
  └─ cache.py (缓存)

gemini.py
  └─ config.py (Gemini 配置)

sources/*
  ├─ models.py (Paper 模型)
  └─ cache.py (缓存)

pdf_fetch.py
  └─ config.py (Unpaywall/NCBI 配置)
```

---

## 关键设计理念

### 1. **模块化与可扩展性**
- 每个数据源是独立的适配器 (sources/)
- 使用 `BaseSource` 抽象基类
- 新增数据源只需实现 `search()` 方法

### 2. **异步优先**
- 所有 I/O 操作使用 `async/await`
- 多源搜索并行执行
- LLM 评估批量并发

### 3. **缓存优化**
- SQLite 持久化缓存
- 自动过期机制 (TTL)
- 减少重复 API 调用

### 4. **错误处理与诊断**
- 自定义异常类 (`LLMError`、`Doc2XError`)
- 详细的错误上下文
- 用户友好的错误消息

### 5. **类型安全**
- 全面使用 Pydantic 模型
- 静态类型提示 (Type Hints)
- 运行时验证

### 6. **用户体验**
- Rich 库实现美观的终端输出
- 进度条与状态反馈
- 多种输出格式 (table/json/md)
- `verbose`/`quiet` 模式切换

---

## 使用示例

### 1. 搜索论文
```bash
paper find "CRISPR gene editing in cancer therapy" --top-n 5
```

### 2. 生成平台查询
```bash
paper gen-query "machine learning in genomics" --platform pubmed
```

### 3. PDF 转 JSONL
```bash
paper extract paper.pdf --out result.jsonl
```

### 4. 结构化解析
```bash
paper structure result.jsonl --out structured.json
```

### 5. 基于 DOI 下载 PDF
```bash
# 下载 PDF 到当前目录
paper fetch-pdf 10.1038/nature12373

# 指定输出目录和文件名
paper fetch-pdf "10.1038/s41586-023-06291-2" --out-dir ./pdfs --filename paper.pdf

# 仅查询 URL，不下载
paper fetch-pdf 10.1000/xyz123 --no-download --format json
```

### 6. 生成文章亮点 Slide
```bash
# 从文件生成手绘风格 Slide
paper slide --in article.txt --style handdrawn --out slide.png

# 使用管道输入，极简风格
cat paper.txt | paper slide --style minimal --out summary.png

# 指定要点数量和图片尺寸
paper slide --in text.txt --bullets 3 --image-size 2K --style academic

# 显示提取的亮点内容
paper slide --in article.txt --style dark --show-highlights
```

---

## 配置文件示例

`~/.config/papercli.toml`:

```toml
[llm]
base_url = "https://api.openai.com/v1"
api_key = "sk-..."
intent_model = "gpt-4o-mini"
eval_model = "gpt-4o"

[doc2x]
base_url = "https://v2.doc2x.noedgeai.com"
api_key = "..."

[unpaywall]
email = "your@email.com"  # 必填，用于 polite pool 访问

[cache]
enabled = true
ttl_hours = 168  # 7 days

[api_keys]
serpapi_key = "..."
openalex_email = "your@email.com"
ncbi_api_key = "..."  # 可选，提高 PubMed/PMC 请求速率

[gemini]
base_url = "https://api.openai-proxy.org/google/v1beta"  # 或官方 API
api_key = "..."
text_model = "gemini-3-flash-preview"      # 用于亮点提取
image_model = "gemini-3-pro-image-preview"  # 用于 Slide 图片生成
```

---

## 总结

`papercli` 通过模块化设计，将学术论文搜索与 PDF 处理的各个环节解耦：

- **搜索流程**: cli → pipeline → query/sources/rank/eval → output
- **PDF 解析**: cli → doc2x → extract → structure
- **PDF 下载**: cli → pdf_fetch (Unpaywall/PMC)
- **Slide 生成**: cli → slide → gemini (文本提取 + 图片生成)
- **基础设施**: config、cache、llm、gemini、models

每个模块职责清晰，易于测试和维护。通过统一的数据模型 (`Paper`、`QueryIntent`、`ArticleHighlights` 等) 和接口 (`BaseSource`) 实现模块间的松耦合。

