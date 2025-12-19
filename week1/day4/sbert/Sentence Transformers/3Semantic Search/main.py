"""
这是一个简单的句子嵌入应用：语义搜索

我们有一个包含各种句子的语料库。然后，对于给定的查询句子，
我们希望在这个语料库中找到最相似的句子。

此脚本会针对各种查询输出语料库中最相似的前5个句子。
"""

import torch

from sentence_transformers import SentenceTransformer

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 包含示例文档的语料库
corpus = [
    "Machine learning is a field of study that gives computers the ability to learn without being explicitly programmed.",
    "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
    "Neural networks are computing systems vaguely inspired by the biological neural networks that constitute animal brains.",
    "Mars rovers are robotic vehicles designed to travel on the surface of Mars to collect data and perform experiments.",
    "The James Webb Space Telescope is the largest optical telescope in space, designed to conduct infrared astronomy.",
    "SpaceX's Starship is designed to be a fully reusable transportation system capable of carrying humans to Mars and beyond.",
    "Global warming is the long-term heating of Earth's climate system observed since the pre-industrial period due to human activities.",
    "Renewable energy sources include solar, wind, hydro, and geothermal power that naturally replenish over time.",
    "Carbon capture technologies aim to collect CO2 emissions before they enter the atmosphere and store them underground.",
]
# 使用"convert_to_tensor=True"将张量保留在GPU上（如果可用）
corpus_embeddings = embedder.encode_document(corpus, convert_to_tensor=True)

# 查询句子:
queries = [
    "How do artificial neural networks work?",
    "What technology is used for modern space exploration?",
    "How can we address climate change challenges?",
]

# 基于余弦相似度为每个查询句子查找语料库中最接近的5个句子
top_k = min(5, len(corpus))
for query in queries:
    query_embedding = embedder.encode_query(query, convert_to_tensor=True)

    # 我们使用余弦相似度和torch.topk来找到最高的5个分数
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\n查询:", query)
    print("语料库中最相似的前5个句子:")

    for score, idx in zip(scores, indices):
        print(f"(得分: {score:.4f})", corpus[idx])

    """
    # 或者，我们也可以使用util.semantic_search来执行余弦相似度+topk
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=5)
    hits = hits[0]      #获取第一个查询的匹配结果
    for hit in hits:
        print(corpus[hit['corpus_id']], "(得分: {:.4f})".format(hit['score']))
    """