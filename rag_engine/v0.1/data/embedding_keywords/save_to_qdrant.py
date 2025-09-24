import time
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
import json
import torch

client = QdrantClient(host="localhost", port=6333)

def import_data_qdrant(data, data_vector, collection_name, dim_num = 1024):
    data_json = []
    for idx in range(len(data)):
        # data_json.append([{"keyword": data[idx], "vector": data_vector[idx][:10]}, {"vector": data_vector[idx]}])
        data_json.append({"keyword": data[idx]})

    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if not client.collection_exists(collection_name):
       client.create_collection(
          collection_name=collection_name,
          vectors_config=VectorParams(size=dim_num, distance=Distance.COSINE),
       )
    batch_cnt = 0
    for i in range(0, len(data_json), 1000):
        points = [
            PointStruct(
                id=batch_cnt + idx,
                vector=data_vector[batch_cnt + idx],
                payload=item
            )
            for idx, item in enumerate(data_json[i: min(i+1000, len(data_json))])
        ]
        time.sleep(0.2)
        # 将向量插入集合
        client.upsert(
           collection_name=collection_name,
           points=points
        )
        batch_cnt += 1000

if not client.collection_exists("keywords_collection"):
    data_vector = []
    for i in range(25):
        tmp = torch.load("./embeddings_keywords_" + str(i) + ".pth").tolist()
        data_vector += tmp
    with open("./keywords.json", 'r', encoding='utf-8') as file:
        data = json.load(file)["keywords"]
    print("数据数量: ", len(data), " 向量数量: ",  len(data_vector))
    import_data_qdrant(data, data_vector, "keywords_collection")
    print("save embeddings of keywords to qdrant, over")
else:
    print("keywords_collection ready")