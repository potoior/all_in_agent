"""
简易RAG系统完整实现
集成文档加载、文本分块、向量嵌入、语义搜索和响应生成等功能的完整RAG系统
"""

import fitz  # PyMuPDF - 用于处理PDF文件
import os  # 操作系统接口
import numpy as np  # 数值计算库
import json  # JSON数据处理
from openai import OpenAI  # OpenAI API客户端
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型
from dotenv import load_dotenv  # 环境变量加载
from langchain_openai import ChatOpenAI
# 加载环境变量
load_dotenv()

class SimpleRAG:
    """
    简易RAG系统类
    实现了完整的RAG流程：文档加载 -> 文本分块 -> 向量嵌入 -> 语义搜索 -> 响应生成
    """

    def __init__(self, model_name="BAAI/bge-base-en-v1.5"):
        """
        初始化RAG系统

        Args:
            model_name (str): 使用的嵌入模型名称
        """
        # 初始化嵌入模型
        self.embedding_model = HuggingFaceEmbedding(model_name=model_name)
        # 初始化OpenAI客户端
        self.client = OpenAI(
            base_url="https://api.siliconflow.cn/v1",
            api_key=os.getenv("SILICONFLOW_API_KEY")
        )
        # 存储文本块和嵌入向量
        self.text_chunks = []
        self.embeddings = []

    def load_document(self, pdf_path):
        """
        加载并处理PDF文档

        Args:
            pdf_path (str): PDF文件路径
        """
        # 提取文本
        text = self.extract_text_from_pdf(pdf_path)

        # 分块
        # 这里分块就很巧妙了
        # 这里是n=100,overlap=10的结果
        # 已加载文档，创建了 437 个文本块
        # 问题: transformer模型在经过8个GPU训练3.5天后创下的单模型BLEU新纪录是多少？
        # 回答: 根据上下文1，Transformer模型在经过8个GPU训练3.5天后创下的单模型BLEU新纪录是41.8。

        # 这里是n=1000,overlap=200的结果
        # 已加载文档，创建了 50 个文本块
        # 问题: transformer模型在经过8个GPU训练3.5天后创下的单模型BLEU新纪录是多少？
        # 回答: 根据提供的信息，Transformer模型在经过8个P100 GPU训练3.5天后，在WMT 2014 English-to-German翻译任务上创下的单模型BLEU新纪录是28.4。
        self.text_chunks = self.chunk_text(text, 1000, 200)

        # 创建嵌入
        self.embeddings = self.create_embeddings(self.text_chunks)

        print(f"已加载文档，创建了 {len(self.text_chunks)} 个文本块")

    def extract_text_from_pdf(self, pdf_path):
        """
        从PDF文件中提取文本

        Args:
            pdf_path (str): PDF文件路径

        Returns:
            str: 提取的文本内容
        """
        mypdf = fitz.open(pdf_path)
        all_text = ""
        for page_num in range(mypdf.page_count):
            page = mypdf[page_num]
            all_text += page.get_text("text")
        return all_text

    def chunk_text(self, text, n, overlap):
        """
        将文本分割成固定大小的块

        Args:
            text (str): 原始文本
            n (int): 每块的字符数
            overlap (int): 相邻块之间重叠的字符数

        Returns:
            list: 文本块列表
        """
        chunks = []
        for i in range(0, len(text), n - overlap):
            chunks.append(text[i:i + n])
        return chunks

    def create_embeddings(self, texts):
        """
        为文本创建嵌入向量

        Args:
            texts (list): 文本列表

        Returns:
            list: 对应的嵌入向量列表
        """
        return self.embedding_model.get_text_embedding_batch(texts)

    def cosine_similarity(self, vec1, vec2):
        """
        计算两个向量的余弦相似度

        Args:
            vec1 (array-like): 第一个向量
            vec2 (array-like): 第二个向量

        Returns:
            float: 余弦相似度
        """
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def search(self, query, k=3):
        """
        搜索相关文档

        Args:
            query (str): 查询文本
            k (int): 返回最相关的k个结果

        Returns:
            list: 最相关的文本块列表
        """
        # 为查询创建嵌入向量
        query_embedding = self.embedding_model.get_text_embedding(query)
        similarities = []

        # 计算查询与每个文档块的相似度
        for i, chunk_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(
                np.array(query_embedding),
                np.array(chunk_embedding)
            )
            similarities.append((i, similarity))

        # 按相似度排序并返回top-k结果
        similarities.sort(key=lambda x: x[1], reverse=True)
        top_indices = [i for i, _ in similarities[:k]]

        return [self.text_chunks[i] for i in top_indices]

    def answer(self, query):
        """
        回答问题
        
        Args:
            query (str): 用户查询
            
        Returns:
            str: AI生成的回答
        """
        # 检索相关文档
        relevant_chunks = self.search(query, k=3)

        # 构建提示
        context = "\n".join([f"上下文 {i+1}:\n{chunk}"
                           for i, chunk in enumerate(relevant_chunks)])

        # 系统提示词
        system_prompt = """你是一个AI助手，严格基于给定的上下文回答问题。
如果答案无法从提供的上下文中直接得出，请回答："我没有足够的信息来回答这个问题。" """

        # 用户提示词
        user_prompt = f"{context}\n\n问题: {query}"

        # 生成回答
        response = self.client.chat.completions.create(
            model="Qwen/Qwen2.5-72B-Instruct",
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )

        return response.choices[0].message.content

# 使用示例
if __name__ == "__main__":
    # 创建RAG实例
    rag = SimpleRAG()

    # 加载文档
    rag.load_document("../../basic_rag/data/Attention Is All You Need.pdf")

    # 提问
    question = "transformer模型在经过8个GPU训练3.5天后创下的单模型BLEU新纪录是多少？"
    answer = rag.answer(question)

    print(f"问题: {question}")
    print(f"回答: {answer}")