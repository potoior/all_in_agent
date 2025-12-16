"""
语义分块构建模块
根据分块边界将句子重新组合成语义完整的文本块
"""

def create_semantic_chunks(sentences, boundaries):
    """
    根据边界创建语义分块
    
    Args:
        sentences (list): 句子列表
        boundaries (list): 分块边界位置列表
        
    Returns:
        list: 语义分块列表
    """
    chunks = []

    # 根据边界将句子合并为语义块
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # 合并句子为一个块
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)

    return chunks

# 创建语义分块示例
# 注意：这里假设 sentences 和 boundaries 已经定义
# semantic_chunks = create_semantic_chunks(sentences, boundaries)

# print("语义分块结果:")
# for i, chunk in enumerate(semantic_chunks):
#     print(f"块 {i+1}: {chunk}")