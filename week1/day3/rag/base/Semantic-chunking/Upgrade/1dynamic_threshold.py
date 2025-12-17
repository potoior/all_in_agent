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
        
    注释:
        百分位数计算说明:
        1. 例如有8个相似度值: [0.9, 0.8, 0.3, 0.7, 0.2, 0.6, 0.1, 0.5]
        2. 排序后变为: [0.1, 0.2, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
        3. 计算25%百分位数位置: (数组长度-1) * 0.25 = (8-1) * 0.25 = 1.75
           其中8是数组元素个数，1是位置的整数部分，0.75是小数部分
        4. 通过线性插值计算最终结果: 0.2 + 0.75 * (0.3 - 0.2) = 0.275
    """
    # 使用numpy的percentile函数计算指定百分位数的值
    threshold = np.percentile(similarities, percentile)
    return threshold

# 使用动态阈值示例
# 注意：这里假设 similarities 已经定义
# dynamic_threshold = calculate_dynamic_threshold(similarities, percentile=30)
# print(f"动态阈值: {dynamic_threshold}")