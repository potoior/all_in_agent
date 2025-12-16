"""
分块边界检测模块
基于句子间相似度确定语义分块的边界位置
"""

def find_chunk_boundaries(similarities, threshold=0.5):
    """
    基于相似度阈值确定分块边界

    Args:
        similarities (list): 相邻句子相似度列表
        threshold (float): 相似度阈值，低于此值则分块

    Returns:
        list: 分块边界位置列表
    """
    boundaries = [0]  # 第一个边界总是0

    # 遍历相似度列表，在相似度低于阈值的位置设置分块边界
    for i, similarity in enumerate(similarities):
        if similarity < threshold:
            boundaries.append(i + 1)  # 在相似度低的地方设置边界

    # 添加最后一个边界
    boundaries.append(len(similarities) + 1)

    return boundaries

# 确定分块边界示例
# 注意：这里假设 similarities 已经定义
# boundaries = find_chunk_boundaries(similarities, threshold=0.6)
# print("分块边界:", boundaries)