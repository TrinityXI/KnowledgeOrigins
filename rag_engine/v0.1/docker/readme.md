# 注意事项
1. 请在RAG.py中将localhost改为host.docker.internal以访问本地Qdrant数据库。
2. 在向容器发送请求时同样将localhost改为host.docker.internal。
3. 容器使用前确保Qdrant已启动并已将/data目录下的数据入库。