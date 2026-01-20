# 知识库

知识库包括如下内容

- 结构化输入
- 意图识别
- 数据库检索
- 汇总回答

涉及到的命令行

| 命令 | 说明 |
|------|------|
| `paper base create <name>` | 创建知识库 |
| `paper base add <source>` | 添加文档到知识库 |
| `paper base query <query>` | 查询知识库 |
| `paper base list` | 列出知识库/文档 |
| `paper base info` | 显示知识库统计信息 |
| `paper base remove <doc_id>` | 从知识库移除文档 |

常见问题

- 哪篇文献提到了植物的开花
- 找到 zhougeng xu 参与发表的文章
- 知识库里包括多少篇文献
- 知识库涉及到那些作者

## 创建知识库

使用子命令`create`和`add`( 会调用已有的功能 `extract`, `structure`)将各种来源的数据进行预处理，得到markdown格式作为输入源创建结构化数据库，用于后续分析。

PDF预处理
