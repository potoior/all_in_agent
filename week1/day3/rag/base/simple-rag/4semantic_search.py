"""
语义搜索模块
基于余弦相似度实现语义搜索功能，用于在文本块中查找与查询最相关的内容
"""

import numpy as np  # 数值计算库

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1 (array-like): 第一个向量
        vec2 (array-like): 第二个向量
        
    Returns:
        float: 两个向量的余弦相似度，值域为[-1, 1]
        A*B/(|A|*|B|)
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    基于语义相似度搜索最相关的文档块
    
    Args:
        query (str): 查询文本
        text_chunks (list): 文本块列表
        embeddings (list): 文本块对应的嵌入向量列表
        k (int): 返回最相关的k个结果，默认为5
        
    Returns:
        list: 最相关的文本块列表
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

    # 按相似度降序排序并返回top-k结果
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 上面可能是[[1,0.9],[3,0.8],[666,0.7]....]这样的
    # 获取前K个结果 结果为一个list
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]

# 示例搜索
# 注意：这里假设 query, text_chunks, chunks_embeddings 已经定义
# query = "What is artificial intelligence?"
# top_chunks = semantic_search(query, text_chunks, chunks_embeddings, k=3)

# print(f"查询: {query}")
# for i, chunk in enumerate(top_chunks):
#     print(f"结果 {i+1}:\n{chunk}\n" + "="*50)