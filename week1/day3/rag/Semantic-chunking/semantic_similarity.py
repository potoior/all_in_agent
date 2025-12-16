"""
语义相似度计算模块
计算句子间的语义相似度，为语义分块提供依据
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
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_similarities(embeddings):
    """
    计算相邻句子间的相似度
    
    Args:
        embeddings (numpy.ndarray): 句子嵌入向量矩阵
        
    Returns:
        list: 相邻句子间的相似度列表
    """
    similarities = []

    # 计算每对相邻句子的相似度
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(similarity)

    return similarities

# 计算相邻句子相似度示例
# 注意：这里假设 embeddings 已经定义
# similarities = calculate_similarities(embeddings)
# print("相邻句子相似度:", similarities)