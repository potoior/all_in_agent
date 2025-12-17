"""
PDF文本提取模块
用于从PDF文件中提取文本内容，为后续的文本分块和嵌入处理做准备
"""

import fitz  # PyMuPDF - 用于处理PDF文件的库

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容
    
    Args:
        pdf_path (str): PDF文件的路径
        
    Returns:
        str: 提取出的所有文本内容
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""

    # 遍历PDF的每一页并提取文本
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")  # 获取页面文本
        all_text += text

    return all_text

# 使用示例
pdf_path = "../../basic_rag/data/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)
print(f"提取的文本长度: {len(extracted_text)} 字符")
print(f"提取的文本:\n{extracted_text[:100]}...")
# 提取的文本长度: 105563 字符