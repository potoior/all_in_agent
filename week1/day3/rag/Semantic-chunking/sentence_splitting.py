"""
句子分割模块
将文本按照句子边界进行分割，为语义分块提供基础单元
"""

def split_into_sentences(text):
    """
    将文本分割为句子列表
    
    Args:
        text (str): 待分割的原始文本
        
    Returns:
        list: 句子列表
    """
    # 简单的句子分割（基于句号）
    sentences = text.split(".")

    # 清理空句子和添加句号
    sentences = [s.strip() + "." for s in sentences if s.strip()]

    return sentences

# 示例
text = "人工智能是计算机科学的分支。它研究智能的本质。AI可以模拟人类思维。"
sentences = split_into_sentences(text)
print("句子列表:", sentences)