from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为文本创建向量嵌入
    """
    embedding_model = HuggingFaceEmbedding(model_name=model)

    if isinstance(text, list):
        # 批量处理文本列表
        response = embedding_model.get_text_embedding_batch(text)
    else:
        # 处理单个文本
        response = embedding_model.get_text_embedding(text)

    return response

# 创建所有文本块的嵌入
# chunks_embeddings = create_embeddings(text_chunks)
# print(f"嵌入向量维度: {len(chunks_embeddings[0])}")