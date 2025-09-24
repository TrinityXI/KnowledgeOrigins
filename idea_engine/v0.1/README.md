# LightRAG MVP Docker 容器化

基于 LightRAG 的学术创新发现 API 服务的 Docker 容器化版本，提供学术论文检索和分析 API。

## 📦 项目结构

```
v0.1/
├── data/                           # RAG 数据文件（6334节点，8208边）
│   ├── graph_chunk_entity_relation.graphml
│   ├── vdb_entities.json
│   ├── vdb_relationships.json
│   ├── vdb_chunks.json
│   └── kv_store_full_docs.json
├── docker/
│   └── Dockerfile                 # 轻量级多阶段构建文件
├── src/
│   ├── app/
│   │   └── main.py                # FastAPI 主应用
│   └── requirements.txt           # Python 依赖
├── secrets/                       # API 密钥等敏感信息
├── docker-compose.simple.yml      # Docker Compose 配置
└── README.md                      # 本文档
```

## 🚀 快速启动

### 方法一：一键部署（推荐）
```bash
# 构建并启动服务（后台运行）
docker-compose -f docker-compose.simple.yml up -d --build
```

### 方法二：交互式启动
```bash
# 构建并启动服务（前台运行，查看实时日志）
docker-compose -f docker-compose.simple.yml up --build
```

**两种方法的区别**：
- 方法一（`-d`）：后台运行，适合生产环境
- 方法二：前台运行，适合开发调试，可直接查看启动日志

## ⚙️ 前置条件与配置

### 系统要求
- Docker 20.10+
- Docker Compose 2.0+
- 至少 2GB 可用内存

### 数据文件准备
确保 `data/` 目录包含以下 RAG 数据文件：
- `graph_chunk_entity_relation.graphml` - 知识图谱结构
- `vdb_entities.json` - 实体向量数据库
- `vdb_relationships.json` - 关系向量数据库
- `vdb_chunks.json` - 文本块向量数据库
- `kv_store_full_docs.json` - 完整文档存储

### API密钥配置（三种方式）

#### 方式一：本地secrets文件（推荐）
```bash
# 创建API密钥文件
echo "your_openai_api_key_here" > secrets/llm_api_key.txt
```
优点：安全，不会暴露在配置文件中

#### 方式二：环境变量
```bash
export LLM_API_KEY="your_api_key_here"
export LLM_BASE_URL="https://api.openai.com"
export LLM_MODEL="gpt-4o-mini"
```

#### 方式三：docker-compose配置
在 `docker-compose.simple.yml` 中修改：
```yaml
environment:
  - LLM_API_KEY=your_actual_api_key_here  # 直接填入
  - LLM_BASE_URL=${LLM_BASE_URL:-https://proxy.infix-ai.xyz}
  - LLM_MODEL=${LLM_MODEL:-gpt-4o-mini}
```

**优先级**：环境变量 > secrets文件 > docker-compose默认值

## 🔍 验证服务

服务启动后（通常需要60秒初始化），进行以下验证：

### 健康检查
```bash
curl http://localhost:8002/api/v1/health
```
期望返回：
```json
{
  "status": "healthy",
  "service": "ICLR RAG Service MVP",
  "version": "0.1.0",
  "rag_ready": true,
  "timestamp": "2025-09-24 13:58:43"
}
```

### 三种查询模式测试

#### 1. Hybrid模式（推荐）
```bash
curl -X POST "http://localhost:8002/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "图神经网络有哪些创新点？",
    "mode": "hybrid"
  }'
```

#### 2. Global模式（全局分析）
```bash
curl -X POST "http://localhost:8002/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "图神经网络有哪些创新点？",
    "mode": "global"
  }'
```

#### 3. Local模式（局部检索）
```bash
curl -X POST "http://localhost:8002/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "图神经网络有哪些创新点？",
    "mode": "local"
  }'
```

## 📖 API文档与查询参数

### 访问地址
- **API 服务**: http://localhost:8002
- **API 文档**: http://localhost:8002/docs (Swagger UI)
- **健康检查**: http://localhost:8002/api/v1/health
- **查询示例**: http://localhost:8002/api/v1/query/examples

### 主要端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/v1/query` | POST | 执行学术查询 |
| `/api/v1/health` | GET | 健康检查 |
| `/api/v1/stats` | GET | 系统统计 |
| `/api/v1/query/examples` | GET | 查询示例 |

### 查询参数详解

```json
{
  "query": "图神经网络有哪些创新点？",
  "mode": "hybrid",
  "top_k": 10
}
```

#### 参数说明
- **`query`** (必需): 查询文本，支持中英文
- **`mode`** (可选): 查询模式，默认 `hybrid`
  - `hybrid`: 混合模式，结合局部和全局信息（推荐）
  - `global`: 全局模式，基于整体知识图谱分析
  - `local`: 局部模式，基于相关实体检索
- **`top_k`** (可选): 返回结果数量，默认10，范围1-50

## 🔧 技术栈与性能

### 架构组件
- **API框架**: FastAPI + uvicorn
- **AI引擎**: LightRAG + OpenAI API
- **容器化**: Docker多阶段构建
- **数据存储**: 知识图谱 + 向量数据库
- **依赖管理**: 精简至10个核心包

### 性能指标
- **数据规模**: 6334个节点，8208条边
- **启动时间**: ~60秒（包含RAG引擎初始化）
- **内存使用**: 1-2GB
- **查询响应**: 平均15-25秒
- **并发支持**: 支持多用户并发查询

## 🤝 支持与故障排除

### 常见问题
1. **服务启动失败**: 检查数据文件完整性和API密钥配置
2. **查询超时**: 检查网络连接和LLM服务可用性
3. **内存不足**: 确保至少2GB可用内存

### 日志查看
```bash
# 查看实时日志
docker logs -f lightrag-mvp0.1

# 检查服务状态
curl http://localhost:8002/api/v1/health
```