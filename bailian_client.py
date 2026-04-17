"""
百炼平台客户端：封装查询改写（Query Rewriting）和重排序（Reranking）
"""
import requests
from langchain_core.documents import Document
from typing import Any
import config


class BailianClient:
    """百炼平台 API 封装"""

    def __init__(self, api_key: str, base_url: str):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def _chat_complete(self, model: str, messages: list[dict[str, str]], **kwargs) -> str:
        """通用 Chat Completion 调用，返回 assistant 的 content"""
        url = f"{self.base_url}/chat/completions"
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
        }
        payload.update(kwargs)
        resp = requests.post(url, json=payload, headers=self.headers, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"].strip()


class QueryRewriter:
    """
    查询改写：将用户原始问题改写为更适合向量检索的形式。
    改写策略：
    - 拼写纠正
    - 同义词扩展
    - 复杂问句分解
    - 补充法律领域术语
    """

    def __init__(self, client: BailianClient, model: str):
        self.client = client
        self.model = model

    REWRITE_PROMPT = """你是一个专业的法律检索查询改写助手。请将用户提出的法律问题改写为更适合向量知识库检索的形式。

改写要求：
1. 保留原问题的核心法律意图
2. 补充同义词和法律术语（如"钱"→"款项/金钱"，"签合同"→"签订合同/订立协议"）
3. 如为复杂问题，拆分为多个独立检索词
4. 用简洁、检索友好的中文表达

【示例】
原始问题：我跟人合伙开店，现在对方跑路了，我能起诉他不？
改写后：合伙合同纠纷，对方违约失联，能否提起诉讼

现在请改写以下问题，只输出改写后的查询，不要解释：

原始问题：{query}
改写后："""

    def rewrite(self, query: str) -> str:
        """返回改写后的查询字符串"""
        messages = [
            {"role": "user", "content": self.REWRITE_PROMPT.format(query=query)}
        ]
        return self.client._chat_complete(self.model, messages)


class Reranker:
    """
    重排序（Cross-Encoder）：对召回的候选文档进行精细化排序。
    使用 Qwen3-rerank 对 (query, document) 逐对打分，返回排序后的文档列表。
    """

    def __init__(self, client: BailianClient, model: str, top_n: int = 5):
        self.client = client
        self.model = model
        self.top_n = top_n

    RERANK_PROMPT = """你是一个法律文档相关性评估模型。请评估以下【查询】与【文档内容】之间的相关程度。

评分标准：
- 9-10分：高度相关，文档直接回答了查询中的法律问题
- 7-8分：相关，文档涉及查询的法律要点但不够直接
- 4-6分：弱相关，文档涉及同一法律领域但偏离查询核心
- 1-3分：不相关，文档内容与查询的法律问题无关

请对以下每一份文档给出 1-10 的相关性评分，格式为"评分 | 理由"：

查询：{query}

{docs}

评分："""

    def rerank(self, query: str, documents: list[Document]) -> list[Document]:
        """
        对 documents 按与 query 的相关性进行重排序，返回 top_n 个文档。
        """
        if not documents:
            return []

        # 构建评分 prompt
        docs_text = ""
        for i, doc in enumerate(documents, 1):
            doc_snippet = doc.page_content[:300]
            docs_text += f"\n【文档{i}】\n{doc_snippet}\n"

        messages = [
            {
                "role": "user",
                "content": self.RERANK_PROMPT.format(query=query, docs=docs_text),
            }
        ]

        try:
            response = self.client._chat_complete(self.model, messages)
            # 解析评分结果（提取每行开头的数字评分）
            scored_docs = self._parse_scores(response, documents)
            # 按评分降序，取 top_n
            scored_docs.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in scored_docs[: self.top_n]]
        except Exception as e:
            # 重排失败时降级为直接返回原始召回结果的前 top_n
            print(f"[Reranker] 重排失败，降级返回原始召回结果: {e}")
            return documents[: self.top_n]

    def _parse_scores(
        self, response: str, documents: list[Document]
    ) -> list[tuple[Document, float]]:
        """从 LLM 输出中解析评分"""
        scored = []
        lines = response.strip().split("\n")
        score_map: dict[int, float] = {}

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # 匹配 "1. 9 | 理由" 或 "【文档1】9分" 等模式
            parts = line.split("|")
            if len(parts) >= 1:
                # 尝试提取行首的数字
                prefix = parts[0].strip()
                # 去掉 【文档N】 或 评分 等前缀
                prefix = prefix.replace("【文档", "").replace("】", "").replace("评分", "")
                # 取第一个连续数字
                digits = "".join(c for c in prefix if c.isdigit())
                if digits:
                    idx = int(digits[0]) - 1  # 文档编号从1开始
                    score = float(digits[0])  # 取最高位数字作为评分（简化处理）
                    if 0 <= idx < len(documents):
                        score_map[idx] = score

        # 补全未打分的文档（默认0分）
        for i, doc in enumerate(documents):
            score = score_map.get(i, 0.0)
            scored.append((doc, score))

        return scored


# ===== 工厂函数 =====
def create_bailian_client() -> BailianClient:
    return BailianClient(api_key=config.bailian_api_key, base_url=config.bailian_base_url)


def create_query_rewriter(client: BailianClient | None = None) -> QueryRewriter:
    if client is None:
        client = create_bailian_client()
    return QueryRewriter(client, config.query_rewrite_model)


def create_reranker(client: BailianClient | None = None, top_n: int | None = None) -> Reranker:
    if client is None:
        client = create_bailian_client()
    if top_n is None:
        top_n = config.rerank_k
    return Reranker(client, config.rerank_model, top_n)
