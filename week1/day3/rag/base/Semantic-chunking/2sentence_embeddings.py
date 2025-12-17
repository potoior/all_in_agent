"""
句子嵌入模块
为句子创建向量嵌入表示，用于计算句子间的语义相似度
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型
import numpy as np  # 数值计算库

def create_sentence_embeddings(sentences, model="BAAI/bge-base-en-v1.5"):
    """
    为每个句子创建嵌入向量

    Args:
        sentences (list): 句子列表
        model (str): 使用的嵌入模型名称

    Returns:
        numpy.ndarray: 句子嵌入向量矩阵
    """
    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbedding(model_name=model)
    # 批量创建嵌入向量
    embeddings = embedding_model.get_text_embedding_batch(sentences)

    return np.array(embeddings)

# 创建句子嵌入示例
sentences = ["AI is a branch of computer science.",
             "It aims to create intelligent machines.",
             "Machine learning is a subset of AI."]

embeddings = create_sentence_embeddings(sentences)
print(f"嵌入矩阵形状: {embeddings.shape}")
# 嵌入矩阵形状: (3, 768)   768是嵌入向量的维度