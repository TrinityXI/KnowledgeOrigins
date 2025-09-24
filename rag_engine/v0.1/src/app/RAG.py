import json
import torch
import sys
import math
import string
from collections import Counter
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from qdrant_client import QdrantClient


reranker_model_path = "./gte_multilingual_reranker_base"
reranker_tokenizer = AutoTokenizer.from_pretrained(reranker_model_path)
reranker_model = AutoModelForSequenceClassification.from_pretrained(
    reranker_model_path, trust_remote_code=True,
    torch_dtype=torch.float32
)

gte_model_name_or_path = './gte_model_en'
gte_tokenizer = AutoTokenizer.from_pretrained(gte_model_name_or_path)
gte_model = AutoModel.from_pretrained(gte_model_name_or_path, trust_remote_code=True)


class BM25:
    def __init__(self, file_path, k1=1.5, b=0.75):
        """
        :param docs: 分词后的文档列表，每个文档是一个包含词汇的列表
        :param k1: 调节参数k1
        :param b: 调节参数b
        """
        with open(file_path, 'r', encoding='utf-8') as file:
            papaer_meta_data = json.load(file)
        documents = []
        for item in papaer_meta_data:
            documents.append(item["keywords"])
        docs = []
        for document in documents:
            document_tokens = []
            for item in document:
                document_tokens.append(preprocess_text(item))
            docs.append(document_tokens)
        self.papaer_meta_data = papaer_meta_data
        self.document_num = len(papaer_meta_data)
        self.docs = docs
        self.k1 = k1
        self.b = b
        self.doc_len = [len(doc) for doc in docs]  # 计算每个文档的长度
        self.avgdl = sum(self.doc_len) / len(docs)  # 计算所有文档的平均长度
        self.doc_freqs = []  # 存储每个文档的词频
        self.idf = {}  # 存储每个词的逆文档频率
        self.initialize()

    def initialize(self):
        """
        初始化方法，计算所有词的逆文档频率
        """
        df = {}  # 用于存储每个词在多少不同文档中出现
        for doc in self.docs:
            # 为每个文档创建一个词频统计
            self.doc_freqs.append(Counter(doc))
            # 更新df值
            for word in set(doc):
                df[word] = df.get(word, 0) + 1
        # 计算每个词的IDF值
        for word, freq in df.items():
            self.idf[word] = math.log((len(self.docs) - freq + 0.5) / (freq + 0.5) + 1)

    def score(self, doc_idx, query):
        """
        计算文档与查询的BM25得分
        :param doc: 文档的索引
        :param query: 查询词列表
        :return: 该文档与查询的相关性得分
        """
        score = 0.0
        for word in query:
            if word in self.doc_freqs[doc_idx]:
                freq = self.doc_freqs[doc_idx][word]  # 词在文档中的频率

                score += (self.idf[word] * freq * (self.k1 + 1)) / (freq + self.k1 * (1 - self.b + self.b * self.doc_len[doc_idx] / self.avgdl))
        return score
    def query(self, query, topk):
        try:
            query_tokens = []
            for word in query:
                query_tokens.append(preprocess_text(word))

            scores = [self.score(i, query_tokens) for i in range(self.document_num)]

            results = []

            for i in range(self.document_num):
                results.append({
                    'score': scores[i],
                    'payload': self.papaer_meta_data[i]
                })

            results.sort(key=lambda x: x['score'], reverse=True)

            rt = []
            for i in range(min(topk, self.document_num)):
                if results[i]['score'] > 0:
                    rt.append(results[i]['payload'])

            return rt
        except Exception as e:
            print("Error in BM25.query(): ", e)
            return e


def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))  # 移除标点符号
    return text.replace(" ", "")

def qdrant_query(sentence, topk, collection_name, qdrant_config = {"host": "localhost", "port": 6333}):
    try:
        client = QdrantClient(host=qdrant_config["host"], port=qdrant_config["port"])
        batch_dict = gte_tokenizer([sentence], max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = gte_model(**batch_dict)
        dimension = 1024
        embeddings = outputs.last_hidden_state[:, 0][:dimension]
        embeddings = F.normalize(embeddings, p=2, dim=1).tolist()[0]

        hits = client.search(
         collection_name = collection_name,
         query_vector=embeddings,
         limit = topk  # Return 5 closest points
        )

        rt = []
        for hit in hits:
            rt.append(hit.payload)

        return rt
    except Exception as e:
        return e

def get_document_num(paper_json_file_path):
    with open(paper_json_file_path, "r", encoding="utf-8") as file:
        papaer_meta_data = json.load(file)
    return len(papaer_meta_data)

def reranker(hits_json, query_sentence):
    reranker_model.eval()
    pairs = []
    for item in hits_json:
        pairs.append([item["abstract"], query_sentence])

    with torch.no_grad():
        inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
        scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()

    scores_content_list = []
    for i in range(len(scores)):
        scores_content_list.append({
            'score': scores[i].item(),
            'payload': hits_json[i]
        })
    scores_content_list.sort(key=lambda x: x['score'], reverse=True)

    rt = []
    for item in scores_content_list:
        rt.append(item["payload"])

    return rt

def get_keywords(sentence, qdrant_config):
    try:
        client = QdrantClient(host=qdrant_config["host"], port=qdrant_config["port"])
        batch_dict = gte_tokenizer([sentence], max_length=8192, padding=True, truncation=True, return_tensors='pt')
        outputs = gte_model(**batch_dict)
        dimension = 1024
        embeddings = outputs.last_hidden_state[:, 0][:dimension]
        embeddings = F.normalize(embeddings, p=2, dim=1).tolist()[0]

        hits = client.search(
         collection_name = "keywords_collection",
         query_vector = embeddings,
         limit = 20
        )
        keywords_gte = []
        for hit in hits:
            #if hit.score > 0.5:
            keywords_gte.append(hit.payload["keyword"])

        #reranker
        pairs = []
        for item in keywords_gte:
            pairs.append([item, sentence])

        with torch.no_grad():
            inputs = reranker_tokenizer(pairs, padding=True, truncation=True, return_tensors='pt', max_length=512)
            scores = reranker_model(**inputs, return_dict=True).logits.view(-1, ).float()

        scores_content_list = []
        for i in range(len(scores)):
            scores_content_list.append({
                'score': scores[i].item(),
                'payload': keywords_gte[i]
            })
        scores_content_list.sort(key=lambda x: x['score'], reverse=True)
        scores_content_list = scores_content_list[:10]
        rt = []
        for item in scores_content_list:
            rt.append(item["payload"])

        return rt
    except Exception as e:
        return e



def search(query, log = True):
    try:
        document_num = get_document_num(query["paper_json_file_path"])
        if log:
            print("document_num: ", document_num)

        qdrant_config = query["qdrant_config"]

        each_channel_result = []
        channel_idx = 0
        for channel in query["channels"]:
            if channel["algorithm"] == "gte":
                if "search_num" in channel:
                    hit = qdrant_query(query["question"], min(document_num, channel["search_num"]), channel["collection_name"], qdrant_config)
                else:
                    hit = qdrant_query(query["question"], min(document_num, query["top_k"]), channel["collection_name"], qdrant_config)
                if "require_reranker" in channel and channel["require_reranker"]:
                    hit = reranker(hit, query["question"])
                if log:
                    print("gte检索到", len(hit),"个元素")
            elif channel["algorithm"] == "keyword_bm25":
                bm25 = BM25(query["paper_json_file_path"])
                if channel["keywords_extraction_required"] or "topic_hint" not in query or len(query["topic_hint"]) == 0:
                    query["topic_hint"] = get_keywords(query["question"], qdrant_config)
                    if log:
                        print("匹配的关键词: ", query["topic_hint"])
                if len(query["topic_hint"]) > 0:
                    if "search_num" in channel:
                        hit = bm25.query(query["topic_hint"], min(document_num, channel["search_num"]))
                    else:
                        hit = bm25.query(query["topic_hint"], min(document_num, query["top_k"]))
                    if log:
                        print("keyword_bm25检索到", len(hit), "个元素")
                else:
                    hit = []
                    if log:
                        print("keyword_bm25检索到", len(hit), "个元素")

            channel_idx += 1
            each_channel_result.append({"channel_idx": channel_idx, "hit_json": hit})

        for item in each_channel_result:
            if len(item["hit_json"]) == 0:
                each_channel_result.remove(item)
        if log:
            print("共",len(each_channel_result),"条有效检索结果")

        if "fusion_strategy" in query and len(query["channels"]) > 1 and len(each_channel_result) > 1:
            if query["fusion_strategy"] == "rrf":
                min_len = sys.maxsize
                for item in each_channel_result:
                    min_len = min(min_len, len(item["hit_json"]))
                for i in range(len(each_channel_result)):
                    each_channel_result[i]["hit_json"] = each_channel_result[i]["hit_json"][:min_len]

                if "param_rrf_k" in query:
                    rrf_k = query["param_rrf_k"]
                else:
                    rrf_k = 1 #默认为1

                rrf_scores = []
                for item in each_channel_result:
                    hit_json = item["hit_json"]

                    channel_idx = item["channel_idx"]
                    try:
                        weight = query["channels"][channel_idx]["weight"]
                    except:
                        weight = 1

                    tmp = {}
                    for idx, item in enumerate(hit_json):
                        tmp[json.dumps(item)] = weight * 1 / (idx + 1 + rrf_k)
                    rrf_scores.append(tmp)
                hits_merged = list(set(item for rrf_score in rrf_scores for item in rrf_score.keys()))

                rt_with_score = []
                for item in hits_merged:
                    score = 0
                    for rrf_score in rrf_scores:
                        if item in rrf_score:
                            score += rrf_score[item]
                    rt_with_score.append({
                        'score': score,
                        'payload': json.loads(item)
                    })
                rt_with_score.sort(key=lambda x: x['score'], reverse=True)
                fusion_result = []
                for item in rt_with_score:
                    fusion_result.append(item["payload"])
            else:
                fusion_result = each_channel_result[0]["hit_json"]
        else:
            fusion_result = each_channel_result[0]["hit_json"]



        if "require_reranker" in query and query["require_reranker"]:
            final_result = reranker(fusion_result, query["question"])
        else:
            final_result = fusion_result

        return final_result[:query["top_k"]]
    except Exception as e:
        print("Error in search():", e)
        return e

