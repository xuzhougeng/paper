# PDF格式化

## Doc2X 解析结果与 `result.jsonl`（PaperCLI 页级 JSONL）格式说明

本文档用于说明：

- **上游（Doc2X API v2）**：`/api/v2/parse/preupload` → PUT 上传 → `/api/v2/parse/status` 轮询返回的任务状态与 `result` 字段含义（以官方文档为准）
- **下游（PaperCLI）**：`paper extract` 将 Doc2X 的 `result` 规整为**按页一行**的 `result.jsonl` 格式（本仓库实际落地格式）

参考资料：

- Doc2X API v2 PDF 接口文档：[`https://doc2x.noedgeai.com/help/zh-cn/api/doc2x-api-v2-pdf-interface.html`](https://doc2x.noedgeai.com/help/zh-cn/api/doc2x-api-v2-pdf-interface.html)

## 1. Doc2X API v2：任务级返回（`parse/status`）

在 PaperCLI 中，解析流程对应 Doc2X 文档中的：

- `POST /api/v2/parse/preupload`：获取 `uid` 与预签名上传 `url`
- `PUT <url>`：把 PDF 二进制上传到 OSS 预签名地址
- `GET /api/v2/parse/status?uid=...`：轮询异步解析状态，直到完成

### 1.1 `parse/status` 的核心字段（任务级）

Doc2X 文档描述的 `status` 查询成功响应中，`data`（下称 *status_data*）通常包含：

- **`status`**（string）：任务状态
  - `processing`：处理中
  - `success`：解析成功
  - `failed`：解析失败
- **`progress`**（int，可能存在）：处理进度（一般 0~100）
- **`result`**（object / array，`status=success` 时存在）：解析结果（其内部结构**官方文档未在该页面完整枚举**，需要以实际返回为准）
- **`detail`**（string / object，`status=failed` 时存在）：失败原因/错误信息

> 注意：Doc2X 文档也强调了结果（包含图片等）仅在云端保留 **24h**；上传 URL **5min** 内有效且成功后不可复用；上传后状态更新可能有延迟（<20s）。详见官方文档中的说明与错误码表。

## 2. PaperCLI：`result.jsonl`（页级 JSONL）格式定义

### 2.1 文件格式

- **文件类型**：JSON Lines（JSONL）
- **行语义**：每一行是一个 JSON 对象，表示 **1 页**
- **编码**：UTF-8

### 2.2 行对象 Schema（默认）

`paper extract paper.pdf --out result.jsonl` 的默认输出（未开启 `--include-raw`）每行包含以下字段：

- **`doc2x_uid`**（string）
  - Doc2X 异步任务的 `uid`（来自 `preupload` 返回）
- **`source_path`**（string）
  - 输入 PDF 的路径（CLI 传入的路径原样写入）
- **`page_index`**（int）
  - 页索引，从 0 开始
- **`page_no`**（int）
  - 页码，从 1 开始
  - 约束：`page_no == page_index + 1`
- **`text`**（string）
  - 当前页抽取出的文本（“尽力而为”的规整结果）
  - 可能是 Markdown / 纯文本混合，可能包含 HTML 注释片段（取决于 Doc2X 返回）

### 2.3 行对象 Schema（开启 `--include-raw`）

当使用 `paper extract --include-raw ...` 时，每行会额外包含：

- **`raw_page`**（object）
  - Doc2X 返回结果中对应“该页”的原始 payload（用于调试或二次处理）
  - **字段不做稳定性保证**：Doc2X 可能会调整内部结构；建议上层逻辑只依赖 `text` 等稳定字段

### 2.4 `text` 字段的抽取/规整规则（实现约定）

PaperCLI 对每页文本的抽取策略是“尽力而为”，优先级大致为：

1. 如果页对象存在 `md`（string）→ 直接使用
2. 否则如果存在 `text`（string）→ 直接使用
3. 否则尝试遍历 `blocks[]`，拼接每个 block 中可用的 `text/content/md/value`（string）
4. 兜底使用 `content`（string 或 list）做串联

因此：

- **不要假设 `text` 是纯文本**（可能包含 Markdown 标记或结构化残留）
- 如果你要做下游 NLP/Embedding，建议先按需做清洗（去掉 HTML 注释、规范化空白等）

### 2.5 `result.jsonl` 的常见文件级不变量（本仓库输出）

在一次 `paper extract` 产出的同一个 `result.jsonl` 中，通常满足：

- 全部行的 `doc2x_uid` 相同
- 全部行的 `source_path` 相同
- `page_index` 从 0 连续递增
- `page_no` 从 1 连续递增，且 `page_no == page_index + 1`

### 2.6 本仓库示例 `result.jsonl`（`paper.pdf`）的观测结论

以当前仓库根目录的 `result.jsonl` 为例（对应 `paper.pdf`）：

- **总页记录数**：49 行（`page_index`: 0~48，`page_no`: 1~49）
- **字段集合**：固定为 `doc2x_uid/source_path/page_index/page_no/text`（未包含 `raw_page`）
- **`doc2x_uid`**：全文件一致（示例：`019b9d84-fe68-773d-9652-7f7dbdb9cc5d`）
- **`text` 长度分布**：约 1079~9438 字符/页（无空页）

## 3. 最小示例

（为便于阅读，这里只展示字段结构，`text` 内容已截断）

```json
{"doc2x_uid":"019b...","source_path":"/path/to/paper.pdf","page_index":0,"page_no":1,"text":"..."}
{"doc2x_uid":"019b...","source_path":"/path/to/paper.pdf","page_index":1,"page_no":2,"text":"..."}
```

## 4. 与 Doc2X 官方 `result` 的关系（为什么需要规整）

Doc2X `parse/status` 在 `status=success` 时会返回一个 `result`（其内部结构可能随版本/配置变化）。PaperCLI 为了让下游处理稳定：

- 允许 `result` 是 `{"pages":[...]}` 或直接是 `[...]` 等多种形状
- 统一转换为 **页级 JSONL**（每页一条记录）
- 把复杂结构压扁为一个稳定的 `text` 字段（可选保留 `raw_page`）

如果你需要精确定位 Doc2X 某个结构字段（例如图表、版面块信息），建议使用 `--include-raw` 并基于 `raw_page` 做二次解析。


