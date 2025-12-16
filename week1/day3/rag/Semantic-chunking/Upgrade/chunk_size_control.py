"""
分块大小控制模块
在语义分块的基础上增加对分块大小的控制，确保分块既符合语义又满足大小要求
"""

def create_controlled_chunks(sentences, similarities,
                           max_chunk_size=1000, min_chunk_size=200):
    """
    控制分块大小的语义分块
    
    Args:
        sentences (list): 句子列表
        similarities (list): 相邻句子相似度列表
        max_chunk_size (int): 最大分块大小
        min_chunk_size (int): 最小分块大小
        
    Returns:
        list: 控制大小后的分块列表
    """
    chunks = []
    current_chunk = []
    current_size = 0

    # 遍历句子列表构建分块
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_size += len(sentence)

        # 检查是否需要分块
        should_break = False

        # 如果不是最后一个句子
        if i < len(similarities):
            # 如果相似度低且当前块大小合适
            if (similarities[i] < 0.5 and
                current_size >= min_chunk_size):
                should_break = True

        # 如果块太大，强制分块
        if current_size >= max_chunk_size:
            should_break = True

        # 满足分块条件或到达最后一个句子时进行分块
        if should_break or i == len(sentences) - 1:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

    return chunks