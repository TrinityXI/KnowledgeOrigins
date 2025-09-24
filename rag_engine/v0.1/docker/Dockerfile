FROM python:3.10-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONPATH=/app
ENV TZ=Asia/Shanghai

# 国内源文件
ADD sources.list /etc/apt/

# 安装系统依赖
RUN apt-get update && \
    apt-get install -y aptitude && \
    aptitude install -y g++ build-essential && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件并安装Python依赖
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 复制项目文件
COPY gte_model_en/ /app/gte_model_en/
COPY gte_multilingual_reranker_base/ /app/gte_multilingual_reranker_base/

COPY RAG.py /app/
COPY RAG_fastapi_server.py /app/

# 暴露FastAPI服务端口
EXPOSE 8100

# 启动RAG服务
CMD ["python", "RAG_fastapi_server.py"]



