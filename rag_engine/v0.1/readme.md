# 操作流程
1. 本地启动Qdrant，依次运行/data下三个子文件夹中的save_to_qdrant.py，以存入向量数据。
2. 运行/src/app/main.py。
3. 运行RAG_client.py发送请求。

# 功能介绍
1. 支持多路查询的RAG系统(v0.1)，每一路可设置参数详见RAG_client.py的示例。
2. 可自行在"topic_hint"字段中提供关键词（建议为/data/embedding_keywords/keywords.json中的关键词），若该字段为空，系统将自行匹配关键词。
3. 支持多路检索结果融合，目前只支持rrf策略，参数通过"param_rrf_k"设置。
4. 支持reranker，可在一级字段设置"require_reranker": True使其在rrf后调用reranker模型。

# FastAPI
1. 端口设置为8100，可自行修改。