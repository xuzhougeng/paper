

## RSS

- https://www.cell.com/cell/inpress.rss
- https://www.nature.com/ng.rss
- https://www.nature.com/nplants.rss
- 


## 保存为 Markdown

```python
import feedparser
from datetime import datetime

def rss_to_markdown(url, output_path=None):
    """解析 RSS 并保存为 Markdown 文件"""
    feed = feedparser.parse(url)
    
    # 构建 Markdown 内容
    lines = []
    lines.append(f"# {feed.feed.get('title', 'RSS Feed')}")
    lines.append(f"\n> 更新时间: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"> 来源: {url}\n")
    
    for i, entry in enumerate(feed.entries, 1):
        title = entry.get('title', 'Untitled')
        link = entry.get('link', '')
        doi = entry.get('dc_identifier', '')
        authors = entry.get('author', '')
        summary = entry.get('summary', '')
        date = entry.get('updated', '')
        section = entry.get('prism_section', '')
        
        lines.append(f"## {i}. {title}\n")
        if section:
            lines.append(f"**类型**: {section}  ")
        if date:
            lines.append(f"**日期**: {date}  ")
        if doi:
            lines.append(f"**DOI**: `{doi}`  ")
        lines.append(f"**链接**: [{link}]({link})\n")
        if authors:
            lines.append(f"**作者**: {authors}\n")
        if summary:
            lines.append(f"**摘要**: {summary}\n")
        lines.append("---\n")
    
    md_content = "\n".join(lines)
    
    # 保存文件
    if output_path is None:
        output_path = f"cell_inpress_{datetime.now().strftime('%Y%m%d')}.md"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f"已保存到: {output_path}")
    return output_path

# 使用示例
url = "https://www.cell.com/cell/inpress.rss"
rss_to_markdown(url, "cell_inpress.md")
```
