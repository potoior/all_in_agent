def chunk_text(text, n, overlap):
    """
    将文本分割成固定大小的块，支持重叠

    Args:
        text: 原始文本
        n: 每块的字符数
        overlap: 重叠的字符数
    """
    chunks = []

    # 步长 = 块大小 - 重叠大小
    step = n - overlap

    for i in range(0, len(text), step):
        chunk = text[i:i + n]
        chunks.append(chunk)

    return chunks

# 使用示例
# text_chunks = chunk_text(extracted_text, 1000, 200)
# print(f"创建了 {len(text_chunks)} 个文本块")
# print(f"第一个文本块:\n{text_chunks[0]}")