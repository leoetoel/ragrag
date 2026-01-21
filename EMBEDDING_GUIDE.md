# Embedding 入库详细步骤

## 完整流程概述

```
PDF 文件 → Markdown → 分块 JSON → BGE Embedding → Milvus 向量库 → 检索
```

---

## 步骤 1: 环境准备

### 1.1 安装 Python 依赖

```bash
pip install FlagEmbedding pymilvus
```

或在项目目录下：

```bash
pip install -r requirements.txt
```

### 1.2 安装 Docker Desktop

确保已安装 Docker Desktop（Windows/Mac）或 Docker Engine（Linux）。

下载地址：https://www.docker.com/products/docker-desktop/

---

## 步骤 2: 启动 Milvus 向量库

### 2.1 使用 Docker Compose 启动

在项目根目录下执行：

```bash
docker-compose up -d
```

### 2.2 验证 Milvus 运行状态

```bash
# 查看容器状态
docker-compose ps

# 查看日志
docker-compose logs -f milvus-standalone
```

正常输出应该包含：
```
[milvus] successfully started
```

### 2.3 服务端口说明

| 服务 | 端口 | 说明 |
|------|------|------|
| Milvus | 19530 | 向量数据库主要端口 |
| Milvus | 9091 | Metrics 端口 |
| MinIO | 9000 | 对象存储（自动管理） |
| MinIO | 9001 | Web 控制台（可选） |

---

## 步骤 3: 准备数据

### 3.1 PDF 转 Markdown（如果还没有）

```bash
cd src

# 单个 PDF 文件
python main.py parse data/report.pdf -o output/report.md

# 或批量处理
python main.py parse data/ -o output/moutai/
```

### 3.2 Markdown 分块（如果还没有）

```bash
# 分块 Markdown 文件
python main.py chunk output/moutai/2024年度报告.md -o output/chunks/

# 输出: output/chunks/2024年度报告_chunks.json
```

### 3.3 检查 JSON 格式

```json
[
  {
    "chunk_id": "chunk_xxx",
    "content": "文本内容...",
    "metadata": {
      "source": "文件路径",
      "page": 2,
      "section": "章节名"
    }
  }
]
```

---

## 步骤 4: Embedding 入库

### 4.1 基础入库命令

```bash
cd src

# 使用默认参数（BGE-large-zh-v1.5，CUDA）
python main.py embed ../output/chunks/2024年度报告_chunks.json --overwrite

# 使用 CPU（如果没有 GPU）
python main.py embed ../output/chunks/2024年度报告_chunks.json --device cpu --overwrite
```

### 4.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `BAAI/bge-large-zh-v1.5` | BGE 模型名称 |
| `--device` | `cuda` | 设备：cuda/cpu/mps |
| `--batch-size` | 32 | 批量大小 |
| `--host` | `localhost` | Milvus 地址 |
| `--port` | 19530 | Milvus 端口 |
| `--collection` | `rag_chunks` | 集合名称 |
| `--index-type` | `HNSW` | 索引类型：FLAT/IVF_FLAT/HNSW |
| `--metric-type` | `IP` | 距离度量：IP/COSINE/L2 |
| `--overwrite` | false | 是否覆盖已有集合 |

### 4.3 执行示例

```bash
# 示例 1: 完整命令
python main.py embed \
    ../output/chunks/2024年度报告_chunks.json \
    --model BAAI/bge-large-zh-v1.5 \
    --device cuda \
    --batch-size 64 \
    --collection rag_chunks \
    --index-type HNSW \
    --metric-type IP \
    --overwrite

# 示例 2: 使用 CPU（适合没有 GPU 的环境）
python main.py embed \
    ../output/chunks/2024年度报告_chunks.json \
    --device cpu \
    --batch-size 16 \
    --overwrite

# 示例 3: 使用 BGE-M3（支持超长文本）
python main.py embed \
    ../output/chunks/2024年度报告_chunks.json \
    --model BAAI/bge-m3 \
    --overwrite
```

### 4.4 预期输出

```
Loading BGE model: BAAI/bge-large-zh-v1.5
Model loaded successfully (dimension=1024)
BGEEmbedder initialized (model=BAAI/bge-large-zh-v1.5, device=cuda, dim=1024)
Connected to Milvus at localhost:19530
Created collection: rag_chunks
Created index: HNSW (IP)
Loaded collection: rag_chunks
Loaded 751 chunks from ../output/chunks/2024年度报告_chunks.json
Generating embeddings...
Inserted 751 chunks (insert_count=751)

=== Embedding Summary ===
Input File: ../output/chunks/2024年度报告_chunks.json
Collection: rag_chunks
Inserted Chunks: 751
Model: BAAI/bge-large-zh-v1.5
Dimension: 1024

Total Entities in Collection: 751
```

---

## 步骤 5: 向量检索测试

### 5.1 基础检索命令

```bash
cd src

# 简单检索
python main.py search "茅台酒2024年营业收入是多少？"

# 获取更多结果
python main.py search "茅台酒的分红政策" --top-k 10
```

### 5.2 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `query` | 必填 | 查询文本 |
| `--top-k` | 5 | 返回结果数量 |
| `--model` | `BAAI/bge-large-zh-v1.5` | 模型名称 |
| `--device` | `cuda` | 设备 |
| `--collection` | `rag_chunks` | 集合名称 |
| `--metric-type` | `IP` | 距离度量 |

### 5.3 检索示例

```bash
# 示例 1: 财务数据查询
python main.py search "2024年茅台酒的总资产是多少？"

# 示例 2: 分红查询
python main.py search "茅台酒每10股派发多少现金红利？"

# 示例 3: 业务查询
python main.py search "茅台酒的主要销售渠道是什么？"

# 示例 4: 获取更多结果
python main.py search "董事会成员" --top-k 10
```

### 5.4 预期输出

```
=== Search Results ===
Query: 茅台酒2024年营业收入是多少？
Top 5 Results:

--- Result 1 ---
Score: 0.8562
Source: D:\code\bestproject\output\moutai\2024年度报告.md
Page: 45
Section: 财务数据 > 利润表
Content Preview: 公司2024年度实现营业收入85,632,125,300.00元，同比增长18.2%...

--- Result 2 ---
Score: 0.8234
Source: D:\code\bestproject\output\moutai\2024年度报告.md
Page: 47
Section: 财务数据 > 主要财务指标
Content Preview: 报告期，公司实现营业收入856.32亿元，同比增加18.2%...
```

---

## 步骤 6: 管理和维护

### 6.1 查看 Collection 信息

使用 Python 交互式：

```python
from pymilvus import connections, Collection

connections.connect(host="localhost", port="19530")
collection = Collection("rag_chunks")

# 查看统计信息
print(f"Total Entities: {collection.num_entities}")

# 查看索引信息
indexes = collection.indexes
for index in indexes:
    print(f"Index: {index.field_name}, Type: {index.params}")
```

### 6.2 删除 Collection

```python
from pymilvus import utility

utility.drop_collection("rag_chunks")
```

### 6.3 备份数据

```bash
# 备份 MinIO 数据
docker cp milvus-minio:/minio_data ./backup/minio_data

# 或使用 docker volume
docker volume ls
docker volume inspect milvus-etcd
```

---

## 常见问题排查

### 问题 1: 连接 Milvus 失败

```
Failed to connect to Milvus: <MilvusException: (code=1, Connection failed)>
```

**解决方案**：
```bash
# 检查容器是否运行
docker-compose ps

# 重启容器
docker-compose restart

# 检查端口占用
netstat -an | grep 19530
```

### 问题 2: CUDA 内存不足

```
CUDA out of memory
```

**解决方案**：
```bash
# 使用 CPU
python main.py embed ... --device cpu

# 或减小批量大小
python main.py embed ... --batch-size 16
```

### 问题 3: 模型下载失败

```
OSError: Can't load tokenizer for 'BAAI/bge-large-zh-v1.5'
```

**解决方案**：

方法 1: 使用镜像（推荐）
```bash
export HF_ENDPOINT=https://hf-mirror.com
python main.py embed ...
```

方法 2: 手动下载
```bash
# 从 Hugging Face 下载模型
# https://huggingface.co/BAAI/bge-large-zh-v1.5

# 放到本地目录
# ~/.cache/huggingface/hub/
```

方法 3: 使用本地路径
```bash
python main.py embed ... --model /path/to/local/model
```

### 问题 4: 入库速度慢

**优化建议**：

```bash
# 1. 增加批量大小（如果显存足够）
python main.py embed ... --batch-size 64

# 2. 使用 FLAT 索引（小数据集）
python main.py embed ... --index-type FLAT

# 3. 确保向量归一化（默认已启用）
# 在 embedder.py 中已设置 normalize_embeddings=True
```

---

## 性能参考

| 操作 | 数据量 | 预期时间 | 硬件 |
|------|--------|----------|------|
| 模型加载 | - | 10-30秒 | GPU/CPU |
| Embedding 751 chunks | 751 | 2-5秒 | RTX 3060 |
| 入库 Milvus | 751 | 1-2秒 | 本地 Milvus |
| 检索 (Top 5) | 751 | <0.5秒 | 本地 Milvus |

---

## 下一步

- 集成 LLM（如 ChatGPT、Claude）实现完整 RAG
- 添加混合检索（向量 + 关键词）
- 实现重排序（Rerank）提升精度
- 添加 Web API（FastAPI）
