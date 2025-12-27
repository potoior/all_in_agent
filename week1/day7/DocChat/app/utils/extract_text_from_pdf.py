import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    # 遍历PDF的每一页
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

# 使用示例
# pdf_path = "data/AI_Information.pdf"
# extracted_text = extract_text_from_pdf(pdf_path)
# print(f"提取的文本长度: {len(extracted_text)} 字符")