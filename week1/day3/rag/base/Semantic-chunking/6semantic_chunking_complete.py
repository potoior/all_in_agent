"""
语义分块完整实现
基于句子语义相似度的智能文本分块系统，能够更好地保持文本的语义完整性
"""

import fitz  # PyMuPDF - 用于处理PDF文件
import numpy as np  # 数值计算库
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型

class SemanticChunker:
    """
    语义分块器类
    通过分析句子间的语义相似度来确定最佳的分块边界
    """
    
    def __init__(self, model_name="BAAI/bge-base-en-v1.5", similarity_threshold=0.5):
        """
        初始化语义分块器
        
        Args:
            model_name (str): 使用的嵌入模型名称
            similarity_threshold (float): 相似度阈值，低于此值则分块
        """
        # 初始化嵌入模型
        self.embedding_model = HuggingFaceEmbedding(model_name=model_name)
        # 设置相似度阈值
        self.similarity_threshold = similarity_threshold

    def extract_text_from_pdf(self, pdf_path):
        """
        从PDF提取文本
        
        Args:
            pdf_path (str): PDF文件路径
            
        Returns:
            str: 提取的文本内容
        """
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            all_text += page.get_text("text") + " "
        return all_text.strip()

    def split_into_sentences(self, text):
        """
        分割句子
        
        Args:
            text (str): 待分割的文本
            
        Returns:
            list: 句子列表
        """
        sentences = text.split(".")
        sentences = [s.strip() + "." for s in sentences if s.strip()]
        return sentences

    def cosine_similarity(self, vec1, vec2):
        """
        计算余弦相似度
        
        Args:
            vec1 (array-like): 第一个向量
            vec2 (array-like): 第二个向量
            
        Returns:
            float: 余弦相似度
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def calculate_similarities(self, embeddings):
        """
        计算相邻句子相似度
        
        Args:
            embeddings (numpy.ndarray): 句子嵌入向量矩阵
            
        Returns:
            list: 相邻句子相似度列表
        """
        similarities = []
        for i in range(len(embeddings) - 1):
            similarity = self.cosine_similarity(embeddings[i], embeddings[i + 1])
            similarities.append(similarity)
        return similarities
    # 这里应该是返回的list的长度为n-1  内容为i和i+1的相似度

    def find_boundaries(self, similarities):
        """
        确定分块边界
        
        Args:
            similarities (list): 相邻句子相似度列表
            
        Returns:
            list: 分块边界位置列表
        """
        boundaries = [0]

        for i, similarity in enumerate(similarities):
            if similarity < self.similarity_threshold:
                boundaries.append(i + 1)
        #         这里对应上面的边界

        boundaries.append(len(similarities) + 1)
        return boundaries

    def chunk_text(self, text):
        """
        语义分块主函数
        
        Args:
            text (str): 待分块的文本
            
        Returns:
            list: 语义分块列表
        """
        # 1. 分割句子
        sentences = self.split_into_sentences(text)
        print(f"分割了 {len(sentences)} 个句子")

        # 2. 创建句子嵌入
        embeddings = self.embedding_model.get_text_embedding_batch(sentences)
        embeddings = np.array(embeddings)

        # 3. 计算相似度
        similarities = self.calculate_similarities(embeddings)

        # 4. 确定边界
        boundaries = self.find_boundaries(similarities)

        # 5. 创建分块
        chunks = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk = " ".join(sentences[start:end])
            # 上一个分段和这个分段的
            chunks.append(chunk)

        return chunks

    def process_document(self, pdf_path):
        """
        处理整个文档
        
        Args:
            pdf_path (str): PDF文件路径
            
        Returns:
            list: 语义分块列表
        """
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)

        # 语义分块
        chunks = self.chunk_text(text)

        print(f"创建了 {len(chunks)} 个语义分块")
        return chunks

# 使用示例
if __name__ == "__main__":
    # 创建语义分块器
    chunker = SemanticChunker(similarity_threshold=0.6)

    # 处理文档
    chunks = chunker.process_document("../../basic_rag/data/AI_Information.pdf")

    # 显示前3个分块
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n=== 语义分块 {i+1} ===")
        print(chunk)
        print("-" * 50)