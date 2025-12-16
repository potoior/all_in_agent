"""
动态阈值计算模块
基于相似度分布计算动态分块阈值，提高分块的适应性
"""

import numpy as np  # 数值计算库

def calculate_dynamic_threshold(similarities, percentile=25):
    """
    基于相似度分布计算动态阈值

    Args:
        similarities (list): 相似度列表
        percentile (int): 百分位数（较低的百分位对应更严格的分块）
        
    Returns:
        float: 动态计算的阈值
    """
    # 使用numpy的percentile函数计算指定百分位数的值
    threshold = np.percentile(similarities, percentile)
    return threshold

# 使用动态阈值示例
# 注意：这里假设 similarities 已经定义
# dynamic_threshold = calculate_dynamic_threshold(similarities, percentile=30)
# print(f"动态阈值: {dynamic_threshold}")