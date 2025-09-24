import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import json
import torch
import os

client = QdrantClient(host="localhost", port=6333)

def import_data_qdrant(data, data_vector, collection_name, dim_num = 1024):
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
       client.create_collection(
          collection_name=collection_name,
          vectors_config=VectorParams(size=dim_num, distance=Distance.COSINE),
       )
    batch_cnt = 0
    for i in range(0, len(data), 1000):
        points = [
            PointStruct(
                id=batch_cnt + idx,
                vector=data_vector[batch_cnt + idx],
                payload=item
            )
            for idx, item in enumerate(data[i: min(i+1000, len(data))])
        ]
        time.sleep(0.5)
        # 将向量插入集合
        client.upsert(
           collection_name=collection_name,
           points=points
        )
        batch_cnt += 1000


if not client.collection_exists("paper_abstract_collection"):
    file_path = './papers_json.json'
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    data_vector = []
    for i in range(100):
        tmp = torch.load("./embedding_abstract_" + str(i) + ".pth")
        tmp = tmp.tolist()
        data_vector += tmp
    print("数据数量: ", len(data), " 向量数量: ",  len(data_vector))
    import_data_qdrant(data, data_vector, "paper_abstract_collection")
    print("save embeddings of abstracts to qdrant, over")
else:
    print("paper_abstract_collection ready")

