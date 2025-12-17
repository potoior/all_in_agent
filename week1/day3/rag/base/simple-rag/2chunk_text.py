"""
文本分块模块
将长文本分割成固定大小的块，支持块之间的重叠，以便更好地保持语义完整性
"""

def chunk_text(text, n, overlap):
    """
    将文本分割成固定大小的块，支持重叠

    Args:
        text (str): 原始文本
        n (int): 每块的字符数
        overlap (int): 相邻块之间重叠的字符数
        
    Returns:
        list: 文本块列表
    """
    chunks = []

    # 计算步长 = 块大小 - 重叠大小
    step = n - overlap
    """
    举个具体例子：
    假设我们有一个文本 "abcdefghijklmnopqrstuvwxyz"（26个字母）
    设置块大小 n = 10，重叠 overlap = 3
    那么 step = 10 - 3 = 7
    第一个块：位置 0-9，文本为 "abcdefghij" 
    第二个块：位置 7-16，文本为 "hijklmnopq" 
    第三个块：位置 14-23，文本为 "opqrstuvwx"
    """

    # 按照指定步长分割文本
    for i in range(0, len(text), step):
        chunk = text[i:i + n]  # 提取文本块
        chunks.append(chunk)

    return chunks

# 使用示例
# 注意：这里假设 extracted_text 已经定义
# text_chunks = chunk_text(extracted_text, 1000, 200)
# print(f"创建了 {len(text_chunks)} 个文本块")
# print(f"第一个文本块:\n{text_chunks[0]}")