import time

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from qdrant_client.models import PointStruct
from qdrant_client.models import Filter, FieldCondition, MatchValue
import json
import torch
import sys

import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer



data_vector = []
for i in range(84):
    tmp = torch.load("./embeddings_keywords_" + str(i) + ".pth").tolist()
    data_vector += tmp


with open("./keywords.json", 'r', encoding='utf-8') as file:
    data = json.load(file)["keywords"]
print(len(data), len(data_vector))
for i in range(len(data)):
    if data[i] == "graph generative models":
        print(i)
        print(data_vector[i])



# id=12198 version=12 score=0.807022 payload={'keyword': 'efficient computer vision'} vector=None shard_key=None order_value=None
# id=2198 version=2 score=0.807022 payload={'keyword': 'graph generative models'} vector=None shard_key=None order_value=None
# id=3198 version=3 score=0.807022 payload={'keyword': 'auto-regressive'} vector=None shard_key=None order_value=None
# id=11198 version=11 score=0.807022 payload={'keyword': 'dialogue generation'} vector=None shard_key=None order_value=None
# id=8198 version=8 score=0.807022 payload={'keyword': 'semiring'} vector=None shard_key=None order_value=None
# id=10198 version=10 score=0.807022 payload={'keyword': 'lateral thinking'} vector=None shard_key=None order_value=None
# id=198 version=0 score=0.807022 payload={'keyword': 'embodied ai'} vector=None shard_key=None order_value=None
# id=1198 version=1 score=0.807022 payload={'keyword': 'decentralized learning'} vector=None shard_key=None order_value=None
# id=9198 version=9 score=0.807022 payload={'keyword': 'online experiment'} vector=None shard_key=None order_value=None
# id=4198 version=4 score=0.807022 payload={'keyword': 'text2video'} vector=None shard_key=None order_value=None