import os

# ===== 百炼平台配置 =====
bailian_api_key = "sk-84ae0057d2974eb0a29f7cfa5dabbc58"
bailian_base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1"

# ===== 模型配置 =====
query_rewrite_model = "qwen3-4b"        # 查询改写模型
rerank_model = "qwen3-rerank"            # 重排序模型（Cross-Encoder）

# ===== Embedding / 向量检索配置（保持不变）=====
embeddings_model = "qwen3-embedding:0.6b"
txt_folder = "./subdata"
persist_dir = "./laws_chroma"

# ===== 检索配置 =====
retrieval_k = 10   # 召回阶段取 Top-K（重排前）
 rerank_k = 5      # 重排后取 Top-N（最终送生成）
