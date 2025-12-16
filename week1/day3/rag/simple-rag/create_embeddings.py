"""
文本嵌入模块
使用HuggingFace的预训练模型为文本创建向量嵌入表示，用于后续的语义相似度计算
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型接口

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为文本创建向量嵌入
    
    Args:
        text (str or list): 单个文本字符串或文本列表
        model (str): 使用的嵌入模型名称，默认为"BAAI/bge-base-en-v1.5"
        
    Returns:
        嵌入向量或嵌入向量列表
    """
    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbedding(model_name=model)

    if isinstance(text, list):
        # 批量处理文本列表
        response = embedding_model.get_text_embedding_batch(text)
    else:
        # 处理单个文本
        response = embedding_model.get_text_embedding(text)

    return response

# 创建所有文本块的嵌入
# 注意：这里假设 text_chunks 已经定义
# chunks_embeddings = create_embeddings(text_chunks)
# print(f"嵌入向量维度: {len(chunks_embeddings[0])}")