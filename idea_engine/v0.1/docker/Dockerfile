# LightRAG MVP Service - 简化版Dockerfile
# 基于Python 3.10 Slim构建的轻量级镜像

# ==========================================
# 构建阶段 - 安装依赖
# ==========================================
FROM python:3.10-slim as builder

# 设置工作目录
WORKDIR /build

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 升级pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 复制依赖文件
COPY src/requirements.txt /tmp/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# ==========================================
# 运行阶段 - 最小化镜像
# ==========================================
FROM python:3.10-slim as runtime

# 创建应用用户（安全性）
RUN groupadd -r lightrag && useradd -r -g lightrag lightrag

# 安装运行时依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 复制应用代码
COPY --chown=lightrag:lightrag src/app/ /app/

# 创建secrets目录并复制密钥文件
RUN mkdir -p /app/secrets
COPY --chown=lightrag:lightrag secrets/ /app/secrets/

# 创建日志目录
RUN mkdir -p /app/logs && chown -R lightrag:lightrag /app

# 切换到非root用户
USER lightrag

# 设置环境变量
ENV PYTHONPATH="/app"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/api/v1/health || exit 1

# 暴露端口
EXPOSE 8002

# 启动命令
CMD ["python", "main.py"]