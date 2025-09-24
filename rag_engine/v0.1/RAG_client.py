import requests
import json
import time


class QueryClient:
    def __init__(self, base_url="http://localhost:8100"):
        self.base_url = base_url

    def query(self, data: dict) -> dict:
        url = f"{self.base_url}/query"
        try:
            response = requests.post(
                url,
                json={"data": data},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                "result": {},
                "status": "error",
                "message": f"Request failed: {str(e)}"
            }

    def check_health(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            return response.status_code == 200
        except:
            return False


def main():
    client = QueryClient()

    print("Checking server health...")
    for _ in range(10):
        if client.check_health():
            print("Server is ready!")
            break
        print("Waiting for server to start...")
        time.sleep(1)
    else:
        print("Server not responding. Exiting.")
        return

    test = {
        "question": "Generate ideas related to embodied intelligence(VLA).", #询问
        "top_k": 3,
        "topic_hint": ["vision-language model", "reinforcement learning"], #关键词列表，可省略
        "channels": [
            {
                "algorithm": "gte",
                "weight": 1, #该路权重
                "collection_name": "paper_abstract_collection", #可改为"keywords_of_papers_collection"，将通过关键词序列继续检索
                "search_num": 10,
                "require_reranker": True #是否在检索后应用reranker
            },
            {
                "algorithm": "keyword_bm25",
                "weight": 1,
                "search_num": 10,
                "keywords_extraction_required": True, #是否执行关键词抽取，若"topic_hint"为空，将自动执行关键词抽取
            }
        ],
        "fusion_strategy": "rrf",
        "param_rrf_k": 1, #rrf参数
        "require_reranker": True, #是否在rrf后应用reranker
        "qdrant_config":{
            "host": "localhost",
            "port": 6333
        },
        "paper_json_file_path": './papers_json.json'
    }

    # print(f"Sending query with data: {json.dumps(query, indent=2)}")

    # 发送查询请求
    result = client.query(test)

    print(f"Received response: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    main()