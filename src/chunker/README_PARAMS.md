# Chunker 参数配置说明

## 核心参数配置

### 1. MarkdownChunker 初始化参数

| 参数名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `chunk_size` | int | 2000 | **目标块大小**（字符数）。每个 chunk 尽量接近此大小 |
| `chunk_overlap` | int | 200 | **块重叠大小**（字符数）。相邻 chunk 之间的重叠字符数，用于保持上下文连续性 |
| `hard_limit` | int | 4000 | **硬性上限**（字符数）。单个 chunk 的最大允许字符数，超过此值会强制分块 |
| `min_chunk_size` | int | 100 | **最小块大小**（字符数）。小于此大小的内容会被合并或丢弃 |
| `preserve_tables` | bool | True | **保护表格**。是否保持表格完整性，不在表格中间切分 |

---

## 分块策略与阈值

### 2. 分块流程（优先级从高到低）

#### **第一阶段：结构感知分块**
1. **标题层级解析**
   - 识别 Markdown 标题：`#`, `##`, `###` 等（1-6 级）
   - 构建标题树结构
   - 按标题边界划分文档区块

2. **页面标记识别**
   - 识别页面标记：`**第 X 页**`、`第 X 页`（正则匹配）
   - 对于无标题的内容，按页面标记划分

#### **第二阶段：内容分块**（触发条件：单个 section > `chunk_size`）

3. **表格保护**（如果 `preserve_tables=True`）
   - 检测表格范围（Markdown 格式）
   - 将表格作为独立 chunk，不在表格内部切分
   - 表格前后的内容单独处理

4. **递归字符分块**（核心分块策略）

   **递归切分优先级：**

   | 优先级 | 分隔符 | 使用场景 | 代码位置 |
   |--------|--------|----------|----------|
   | 1 | `\n\n` (双换行) | 按段落切分 | [chunker.py:470](chunker.py#L470) |
   | 2 | `\n` (单换行) | 按行切分 | [chunker.py:475](chunker.py#L475) |
   | 3 | `[。！？.!?]` (中英文句号) | 按句子切分 | [chunker.py:480](chunker.py#L480) |
   | 4 | 固定窗口 | 最后手段，强制切分 | [chunker.py:489](chunker.py#L489) |

---

## 递归分块详细逻辑

### 3. `_split_by_delimiter` 函数参数

此函数处理按分隔符切分后的内容聚合逻辑：

**输入参数：**
- `pieces`: 分隔符切分后的文本片段列表
- `delimiter`: 分隔符（`\n\n`、`\n` 或空）
- `title_path`: 标题层级路径
- `page_range`: 页码范围
- `source_file`: 源文件路径

**聚合规则：**
```python
# 当前块 + 新片段 <= chunk_size → 继续聚合
if len(test_chunk) <= self.chunk_size:
    current_chunk = test_chunk

# 当前块已满 → 保存并处理下一片段
else:
    # 1. 保存当前块（需 >= min_chunk_size）
    if len(current_chunk) >= self.min_chunk_size:
        chunks.append(...)

    # 2. 处理超大片段（> hard_limit）
    if len(piece) > self.hard_limit:
        # 强制使用固定窗口切分
        sub_chunks = self._fixed_window_split(...)

    # 3. 处理独立有效片段
    elif len(piece) >= self.min_chunk_size:
        chunks.append(...)  # 作为独立 chunk

    # 4. 处理过小片段（< min_chunk_size）
    else:
        # 添加重叠并合并到下一个 chunk
        overlap = current_chunk[-self.chunk_overlap:]
        current_chunk = overlap + delimiter + piece
```

---

## 固定窗口切分

### 4. `_fixed_window_split` 参数

**使用场景：** 当所有语义切分失败时的最后手段

**逻辑：**
```python
start = 0
while start < content_len:
    end = start + chunk_size  # 切分点
    chunk = content[start:end]

    if len(chunk) >= min_chunk_size:
        chunks.append(chunk)

    start = end - chunk_overlap  # 滑动窗口，带重叠
```

**关键参数：**
- `chunk_size`: 每次切分的长度
- `chunk_overlap`: 窗口滑动重叠量
- `min_chunk_size`: 最小块大小过滤

---

## 表格检测参数

### 5. 表格识别规则

**正则表达式：**
```python
table_pattern = re.compile(
    r'^[| :\-+\s]*\|[\s\|:+\-]*$|^\|?[\s\-:]+\|[\s\-:]+\|?',
    re.MULTILINE
)
```

**识别逻辑：**
- 包含 `|` 符号的行
- 或匹配 Markdown 表格分隔符（`|---|`）
- 连续的表格行构成完整表格范围

**表格处理：**
- 表格不会被切分，作为独立 chunk
- 表格前后的内容分别处理

---

## 页码标记识别

### 6. 页码提取参数

**正则表达式：**
```python
page_marker_pattern = re.compile(r'\*{0,2}第\s*(\d+)\s*页\*{0,2}', re.IGNORECASE)
```

**支持的格式：**
- `第 1 页`
- `**第 1 页**`
- `***第1页***`

**页码存储：**
- 单页：`page = 1`
- 页码范围：`page = "1-5"`
- 无页码：`page = None`

---

## Chunk 元数据结构

### 7. Chunk 对象字段

| 字段名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `chunk_id` | str | 唯一标识（MD5前8位 + UUID前8位） | `chunk_a1b2c3d4_1a2b3c4d` |
| `content` | str | 块内容文本 | 实际文本内容 |
| `title_path` | List[str] | 标题层级路径 | `['第一章', '1.1 节']` |
| `page_range` | Optional[List[int]] | 页码范围 `[起始, 结束]` | `[1, 3]` 或 `[5, 5]` |
| `char_count` | int | 字符数（不含空白） | 1850 |
| `metadata` | dict | 元数据字典 | 见下表 |

### 8. metadata 字段

| 键名 | 值类型 | 说明 |
|------|--------|------|
| `source` | str | 源文件路径 |
| `page` | str/int | 页码（"1-3" 或 5） |
| `section` | str | 章节路径（"第一章 > 1.1 节"） |

---

## 完整参数配置示例

```python
from src.chunker.chunker import MarkdownChunker

# 自定义配置
chunker = MarkdownChunker(
    chunk_size=3000,          # 更大的块适合长文档
    chunk_overlap=300,        # 增加重叠保持上下文
    hard_limit=6000,          # 硬性上限
    min_chunk_size=200,       # 提高最小阈值避免碎片
    preserve_tables=True      # 保护表格
)

# 分块
result = chunker.chunk_file("data/report.md")
print(f"生成 {result.total_chunks} 个 chunks")
```

---

## 参数调优建议

### 根据文档类型调整：

| 文档类型 | 推荐配置 | 说明 |
|----------|----------|------|
| **技术文档** | `chunk_size=1500, overlap=200` | 信息密度高，小块更精确 |
| **年报/财报** | `chunk_size=2500, overlap=300` | 段落长，需要更大块 |
| **法律合同** | `chunk_size=2000, overlap=400` | 需要更多上下文重叠 |
| **新闻文章** | `chunk_size=1000, overlap=100` | 短文章，小块即可 |

### 特殊场景：

1. **表格密集文档** → `preserve_tables=True`（必须）
2. **需要精确检索** → 降低 `chunk_size`，提高 `chunk_overlap`
3. **降低碎片化** → 提高 `min_chunk_size`
4. **大文件处理** → 确保 `hard_limit` 设置合理

---

## 关键阈值总结

| 阈值名称 | 默认值 | 作用位置 |
|----------|--------|----------|
| **目标块大小** | 2000 字符 | 主要分块目标 |
| **块重叠** | 200 字符 | 保持上下文连续性 |
| **最小块大小** | 100 字符 | 过滤碎片 |
| **硬性上限** | 4000 字符 | 防止超大块 |

这些参数共同控制分块行为，可根据具体应用场景灵活调整。
