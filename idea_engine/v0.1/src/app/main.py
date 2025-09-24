#!/usr/bin/env python3
"""
ICLR RAG Service MVP - Docker版本
基于已验证的MVP服务构建的容器化Web API服务
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import asyncio
import sys
import os
import time
import json
from typing import Optional, Dict, Any, List
import traceback

# 导入LightRAG相关组件
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.utils import EmbeddingFunc
    from lightrag.kg.shared_storage import initialize_pipeline_status
    import aiohttp
    import numpy as np
except ImportError as e:
    print(f"导入依赖失败: {e}")
    print("请确保LightRAG环境已正确配置")

app = FastAPI(
    title="ICLR 2025 RAG Service MVP",
    description="基于LightRAG的学术创新发现API服务",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 数据模型定义
class QueryRequest(BaseModel):
    query: str = Field(..., description="查询文本", example="图神经网络有哪些创新点？")
    mode: str = Field("hybrid", description="查询模式", example="hybrid", pattern="^(local|global|hybrid)$")
    enable_rerank: bool = Field(False, description="是否启用重排")
    top_k: int = Field(10, description="返回结果数量", ge=1, le=50)
    max_tokens: int = Field(4000, description="最大token数", ge=100, le=8000)
    debug_mode: bool = Field(False, description="是否返回检索详细信息（测试用）")

class RetrievalData(BaseModel):
    entities: List[Dict[str, Any]] = Field(default=[], description="检索到的实体")
    relationships: List[Dict[str, Any]] = Field(default=[], description="检索到的关系")
    chunks: List[Dict[str, Any]] = Field(default=[], description="检索到的文本块")
    context: str = Field(default="", description="上下文文本")

class QueryResponse(BaseModel):
    status: str = Field(..., description="请求状态")
    query: str = Field(..., description="原始查询")
    answer: str = Field(..., description="回答内容")
    metadata: Dict[str, Any] = Field(..., description="元数据信息")
    retrieval_data: Optional[RetrievalData] = Field(None, description="检索详细数据（debug模式）")

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    rag_ready: bool
    timestamp: str

class StatsResponse(BaseModel):
    working_directory: str
    data_files: List[Dict[str, Any]]
    system_info: Dict[str, Any]

# 从环境变量或文件获取API配置
def get_api_key():
    # 优先从环境变量获取
    api_key = os.getenv("LLM_API_KEY")
    if api_key and api_key != "your_api_key_here":
        return api_key

    # 从文件获取（支持容器内和本地）
    key_files = [
        "/app/secrets/llm_api_key.txt",  # 容器内路径
        "secrets/llm_api_key.txt",      # 本地开发路径
        "../secrets/llm_api_key.txt"    # 备用路径
    ]

    for key_file in key_files:
        try:
            if os.path.exists(key_file):
                with open(key_file, 'r', encoding='utf-8') as f:
                    key = f.read().strip()
                    if key and not key.startswith("your_"):
                        return key
        except Exception:
            continue

    return "your_api_key_here"

GPT_CONFIG = {
    "api_key": get_api_key(),
    "base_url": os.getenv("LLM_BASE_URL", "https://proxy.infix-ai.xyz"),
    "model": os.getenv("LLM_MODEL", "gpt-4o-mini")
}

# 全局RAG实例
rag_engine = None
service_stats = {
    "queries_count": 0,
    "total_response_time": 0.0,
    "errors_count": 0,
    "start_time": time.time()
}

async def gpt_llm_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    """GPT-4o-mini LLM调用函数"""

    headers = {
        "Authorization": f"Bearer {GPT_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }

    messages = []

    # 添加系统消息
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # 添加历史消息
    for msg in history_messages:
        messages.append(msg)

    # 添加当前用户消息
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": GPT_CONFIG["model"],
        "messages": messages,
        "max_tokens": kwargs.get("max_tokens", 4000),
        "temperature": kwargs.get("temperature", 0.0)
    }

    async with aiohttp.ClientSession() as session:
        url = f"{GPT_CONFIG['base_url']}/v1/chat/completions"

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                return result["choices"][0]["message"]["content"]
            else:
                error = await response.text()
                raise Exception(f"GPT-4o-mini API失败: {response.status}, {error}")

async def openai_embedding_func(texts: list[str]) -> np.ndarray:
    """OpenAI embedding函数"""

    headers = {
        "Authorization": f"Bearer {GPT_CONFIG['api_key']}",
        "Content-Type": "application/json"
    }

    payload = {
        "input": texts,
        "model": "text-embedding-3-small"
    }

    async with aiohttp.ClientSession() as session:
        url = f"{GPT_CONFIG['base_url']}/v1/embeddings"

        async with session.post(url, json=payload, headers=headers) as response:
            if response.status == 200:
                result = await response.json()
                embeddings = []
                for item in result.get("data", []):
                    embeddings.append(item.get("embedding", []))

                if embeddings:
                    return np.array(embeddings, dtype=np.float32)

            raise Exception(f"Embedding失败: {response.status}")

@app.on_event("startup")
async def startup_event():
    """服务启动时初始化RAG系统"""
    global rag_engine

    try:
        print("🚀 正在初始化ICLR RAG MVP服务...")

        # 容器内数据目录
        working_dir = "/app/data"
        print(f"📂 数据目录: {working_dir}")

        if not os.path.exists(working_dir):
            raise Exception(f"数据目录不存在: {working_dir}")

        rag_engine = LightRAG(
            working_dir=working_dir,
            llm_model_func=gpt_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=1536,
                func=openai_embedding_func
            )
        )

        await rag_engine.initialize_storages()
        await initialize_pipeline_status()

        print("✅ ICLR RAG MVP服务初始化完成")

    except Exception as e:
        print(f"❌ 服务初始化失败: {e}")
        print(f"错误详情: {traceback.format_exc()}")
        # 不中断启动，但标记为未就绪
        rag_engine = None

@app.post("/api/v1/query", response_model=QueryResponse, summary="执行学术查询", description="查询ICLR 2025论文数据库，支持创新点发现、科研思路等多种查询类型")
async def query_papers(request: QueryRequest):
    """核心查询接口 - 支持多种学术查询场景"""
    global service_stats

    try:
        if rag_engine is None:
            service_stats["errors_count"] += 1
            raise HTTPException(status_code=503, detail="RAG引擎未就绪，请检查服务初始化状态")

        # 记录查询开始
        start_time = time.time()
        service_stats["queries_count"] += 1

        # 构建查询参数
        query_param = QueryParam(
            mode=request.mode,
            enable_rerank=request.enable_rerank,
            top_k=request.top_k
        )

        print(f"🔍 执行查询: {request.query[:50]}...")

        # 根据debug_mode决定是否获取详细检索数据
        retrieval_data = None
        if request.debug_mode:
            # 获取检索详细数据
            raw_data = await rag_engine.aquery_data(
                request.query,
                param=query_param
            )

            # 检查 raw_data 的类型并正确处理
            if isinstance(raw_data, dict):
                retrieval_data = RetrievalData(
                    entities=raw_data.get("entities", []),
                    relationships=raw_data.get("relationships", []),
                    chunks=raw_data.get("chunks", []),
                    context=raw_data.get("context", "")
                )
            else:
                # 如果 aquery_data 返回字符串，则创建空的检索数据
                print(f"⚠️ aquery_data 返回了非字典类型: {type(raw_data)}")
                retrieval_data = RetrievalData(
                    entities=[],
                    relationships=[],
                    chunks=[],
                    context=str(raw_data) if raw_data else ""
                )

            # 同时获取LLM回答
            result = await rag_engine.aquery(
                request.query,
                param=query_param
            )
        else:
            # 只获取LLM回答
            result = await rag_engine.aquery(
                request.query,
                param=query_param
            )

        end_time = time.time()
        response_time = end_time - start_time
        service_stats["total_response_time"] += response_time

        print(f"✅ 查询完成，耗时: {response_time:.2f}秒")

        return QueryResponse(
            status="success",
            query=request.query,
            answer=result,
            metadata={
                "response_time": round(response_time, 2),
                "answer_length": len(result),
                "mode": request.mode,
                "rerank_enabled": request.enable_rerank,
                "top_k": request.top_k,
                "debug_mode": request.debug_mode,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            retrieval_data=retrieval_data
        )

    except Exception as e:
        service_stats["errors_count"] += 1
        error_msg = f"查询执行失败: {str(e)}"
        print(f"❌ {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/api/v1/health", response_model=HealthResponse, summary="健康检查", description="检查服务运行状态和RAG引擎就绪状态")
async def health_check():
    """服务健康检查"""
    return HealthResponse(
        status="healthy" if rag_engine is not None else "degraded",
        service="ICLR RAG Service MVP",
        version="0.1.0",
        rag_ready=rag_engine is not None,
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
    )

@app.get("/api/v1/stats", response_model=StatsResponse, summary="系统统计", description="获取数据文件状态和服务运行统计")
async def get_stats():
    """获取系统统计信息"""
    working_dir = "/app/data"

    stats = {
        "working_directory": working_dir,
        "data_files": [],
        "system_info": {
            "queries_count": service_stats["queries_count"],
            "errors_count": service_stats["errors_count"],
            "uptime_seconds": round(time.time() - service_stats["start_time"], 2),
            "avg_response_time": round(
                service_stats["total_response_time"] / max(service_stats["queries_count"], 1), 2
            ),
            "success_rate": round(
                (service_stats["queries_count"] - service_stats["errors_count"]) / max(service_stats["queries_count"], 1) * 100, 2
            ) if service_stats["queries_count"] > 0 else 100.0
        }
    }

    # 检查关键数据文件
    data_files = [
        "graph_chunk_entity_relation.graphml",
        "vdb_entities.json",
        "vdb_relationships.json",
        "vdb_chunks.json",
        "kv_store_full_docs.json"
    ]

    for filename in data_files:
        filepath = os.path.join(working_dir, filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            stats["data_files"].append({
                "filename": filename,
                "size_bytes": size,
                "size_mb": round(size / 1024 / 1024, 2),
                "exists": True
            })
        else:
            stats["data_files"].append({
                "filename": filename,
                "exists": False,
                "size_bytes": 0,
                "size_mb": 0
            })

    return StatsResponse(**stats)

@app.get("/api/v1/query/examples", summary="查询示例", description="获取常见查询示例")
async def get_query_examples():
    """提供查询示例"""
    examples = {
        "创新点查询": [
            "图神经网络有哪些最新创新点？",
            "自监督学习在计算机视觉领域的创新突破",
            "Transformer架构的最新改进有哪些？",
            "强化学习在多智能体系统中的创新应用"
        ],
        "科研思路查询": [
            "如何改进现有的图神经网络方法？",
            "解决长序列建模问题的研究方向",
            "提升模型可解释性的技术路线",
            "降低大模型计算成本的方法探索"
        ],
        "技术对比查询": [
            "不同注意力机制的优劣势对比",
            "监督学习与自监督学习的效果比较",
            "CNN、RNN和Transformer在NLP任务中的表现",
            "不同优化算法在深度学习中的适用场景"
        ],
        "应用场景查询": [
            "图神经网络在推荐系统中的应用",
            "大语言模型在代码生成中的使用",
            "计算机视觉技术在医疗诊断中的应用",
            "强化学习在自动驾驶中的实践"
        ],
        "趋势分析查询": [
            "当前机器学习领域的研究热点",
            "未来AI发展的主要方向",
            "新兴技术在学术界的接受度",
            "跨学科研究的发展趋势"
        ]
    }

    return {
        "status": "success",
        "examples": examples,
        "usage_tips": [
            "使用具体的技术术语可以获得更精确的结果",
            "问题可以包含多个概念，系统会自动关联相关论文",
            "支持中英文查询，建议使用中文获得更好的理解",
            "可以通过mode参数调整检索策略：local(局部)、global(全局)、hybrid(混合)",
            "设置debug_mode=true可以获取检索的实体、关系和文本块详细信息（测试功能）"
        ]
    }

@app.get("/", summary="服务根页面")
async def root():
    """服务根页面"""
    return {
        "service": "ICLR 2025 RAG Service MVP",
        "version": "0.1.0",
        "description": "基于LightRAG的学术创新发现API服务",
        "docs": "/docs",
        "health": "/api/v1/health",
        "examples": "/api/v1/query/examples"
    }

if __name__ == "__main__":
    import uvicorn

    print("🌟 启动ICLR RAG MVP服务...")
    print("📚 API文档: http://localhost:8002/docs")
    print("🔍 健康检查: http://localhost:8002/api/v1/health")
    print("💡 查询示例: http://localhost:8002/api/v1/query/examples")

    uvicorn.run(app, host="0.0.0.0", port=8002)