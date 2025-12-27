import numpy as np

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    基于语义相似度搜索最相关的文档块
    """
    # 为查询创建嵌入向量
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文档块的相似度
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk_embedding)
        )
        similarity_scores.append((i, similarity))

    # 按相似度排序并返回top-k结果
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]

# 示例搜索
# query = "What is artificial intelligence?"
# top_chunks = semantic_search(query, text_chunks, chunks_embeddings, k=3)
#
# print(f"查询: {query}")
# for i, chunk in enumerate(top_chunks):
#     print(f"结果 {i+1}:\n{chunk}\n" + "="*50)