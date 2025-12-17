# RAG系统实现详细文档

本文档详细介绍了在 base 目录中实现的各种RAG（Retrieval-Augmented Generation）系统的代码结构和功能模块。这些实现涵盖了从基础的RAG系统到高级语义分块技术的演进过程。

## 目录结构

```
base/
├── Semantic-chunking/
│   ├── Upgrade/
│   │   ├── 1dynamic_threshold.py
│   │   ├── 2chunk_size_control.py
│   │   └── 3evaluation.py
│   ├── 1sentence_splitting.py
│   ├── 2sentence_embeddings.py
│   ├── 3semantic_similarity.py
│   ├── 4chunking_boundaries.py
│   ├── 5semantic_chunks.py
│   └── 6semantic_chunking_complete.py
├── select_chunk/
│   ├── 1chunk_size_selector.py
│   └── 2summary.py
└── simple-rag/
    ├── 1extract_text_from_pdf.py
    ├── 2chunk_text.py
    ├── 3create_embeddings.py
    ├── 4semantic_search.py
    ├── 5generate_response.py
    └── 6simple_rag_complete.py
```

## 1. 简单RAG系统（Simple RAG）

简单RAG系统是最基础的实现，包含了完整的RAG流程：文档加载、文本分块、向量嵌入、语义搜索和响应生成。

### 1.1 模块详解

#### 1.1.1 文档提取模块 (1extract_text_from_pdf.py)

该模块负责从PDF文件中提取文本内容，为后续处理做准备。

```python
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
```

#### 1.1.2 文本分块模块 (2chunk_text.py)

该模块将长文本分割成固定大小的块，支持块之间的重叠，以保持语义完整性。

```python
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
```

#### 1.1.3 文本嵌入模块 (3create_embeddings.py)

该模块使用HuggingFace的预训练模型为文本创建向量嵌入表示。

```python
"""
文本嵌入模块
使用HuggingFace的预训练模型为文本创建向量嵌入表示，用于后续的语义相似度计算
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型接口

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为文本创建向量嵌入
    
    Args:
        text (str or list): 单个文本字符串或文本列表
        model (str): 使用的嵌入模型名称，默认为"BAAI/bge-base-en-v1.5"
        
    Returns:
        嵌入向量或嵌入向量列表
    """
    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbedding(model_name=model)

    # 判断对象是否是list类型 如果是就批量嵌入 否则就单句嵌入
    if isinstance(text, list):
        # 批量处理文本列表
        response = embedding_model.get_text_embedding_batch(text)
    else:
        # 处理单个文本
        response = embedding_model.get_text_embedding(text)

    return response

# 创建所有文本块的嵌入
# 注意：这里假设 text_chunks 已经定义
# chunks_embeddings = create_embeddings(text_chunks)
# print(f"嵌入向量维度: {len(chunks_embeddings[0])}")
```

#### 1.1.4 语义搜索模块 (4semantic_search.py)

该模块基于余弦相似度实现语义搜索功能，在文本块中查找与查询最相关的内容。

```python
"""
语义搜索模块
基于余弦相似度实现语义搜索功能，用于在文本块中查找与查询最相关的内容
"""

import numpy as np  # 数值计算库

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1 (array-like): 第一个向量
        vec2 (array-like): 第二个向量
        
    Returns:
        float: 两个向量的余弦相似度，值域为[-1, 1]
        A*B/(|A|*|B|)
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, text_chunks, embeddings, k=5):
    """
    基于语义相似度搜索最相关的文档块
    
    Args:
        query (str): 查询文本
        text_chunks (list): 文本块列表
        embeddings (list): 文本块对应的嵌入向量列表
        k (int): 返回最相关的k个结果，默认为5
        
    Returns:
        list: 最相关的文本块列表
    """
    # 为查询创建嵌入向量
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文档块的相似度
    for i, chunk_embedding in enumerate(embeddings):
        similarity = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk_embedding)
        )
        similarity_scores.append((i, similarity))

    # 按相似度降序排序并返回top-k结果
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    # 上面可能是[[1,0.9],[3,0.8],[666,0.7]....]这样的
    # 获取前K个结果 结果为一个list
    top_indices = [index for index, _ in similarity_scores[:k]]

    return [text_chunks[index] for index in top_indices]

# 示例搜索
# 注意：这里假设 query, text_chunks, chunks_embeddings 已经定义
# query = "What is artificial intelligence?"
# top_chunks = semantic_search(query, text_chunks, chunks_embeddings, k=3)

# print(f"查询: {query}")
# for i, chunk in enumerate(top_chunks):
#     print(f"结果 {i+1}:\n{chunk}\n" + "="*50)
```

#### 1.1.5 响应生成模块 (5generate_response.py)

该模块使用大语言模型基于检索到的上下文生成自然语言回答。

```python
"""
响应生成模块
使用大语言模型基于检索到的上下文生成自然语言回答
"""

from openai import OpenAI  # OpenAI API客户端
import os  # 操作系统接口

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API端点
    api_key=os.getenv("OPENROUTER_API_KEY")   # 从环境变量获取API密钥
)

def generate_response(system_prompt, user_message, model="Qwen/Qwen2.5-72B-Instruct:free"):
    """
    基于检索到的上下文生成回答
    
    Args:
        system_prompt (str): 系统提示词，定义AI助手的行为模式
        user_message (str): 用户消息，包含上下文和具体问题
        model (str): 使用的大语言模型，默认为免费的Llama3.2-3B模型
        
    Returns:
        ChatCompletion: 模型生成的响应
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # 设置为0以获得确定性回答
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# 系统提示词 - 定义AI助手的行为准则
system_prompt = """你是一个AI助手，严格基于给定的上下文回答问题。
如果答案无法从提供的上下文中直接得出，请回答："我没有足够的信息来回答这个问题。" """

# 构建用户提示 - 将检索到的上下文和用户查询组合成完整的提示
# context = "\n".join([f"上下文 {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
# user_prompt = f"{context}\n\n问题: {query}"

# 生成回答
# ai_response = generate_response(system_prompt, user_prompt)
# print(f"AI回答: {ai_response.choices[0].message.content}")
```

#### 1.1.6 完整实现 (6simple_rag_complete.py)

这是简单RAG系统的完整实现，集成了上述所有模块。

```python
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
```

## 2. 语义分块系统（Semantic Chunking）

语义分块系统通过分析句子间的语义相似度来确定最佳的分块边界，比固定大小分块更能保持文本的语义完整性。

### 2.1 基础模块

#### 2.1.1 句子分割模块 (1sentence_splitting.py)

该模块将文本按照句子边界进行分割，为语义分块提供基础单元。

```python
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
    sentences = text.split("。")

    # 清理空句子和添加句号
    sentences = [s.strip() + "。" for s in sentences if s.strip()]

    return sentences

# 示例
text = "人工智能是计算机科学的分支。它研究智能的本质。AI可以模拟人类思维。"
sentences = split_into_sentences(text)
print("句子列表:", sentences)
```

#### 2.1.2 句子嵌入模块 (2sentence_embeddings.py)

该模块为句子创建向量嵌入表示，用于计算句子间的语义相似度。

```python
"""
句子嵌入模块
为句子创建向量嵌入表示，用于计算句子间的语义相似度
"""

from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型
import numpy as np  # 数值计算库

def create_sentence_embeddings(sentences, model="BAAI/bge-base-en-v1.5"):
    """
    为每个句子创建嵌入向量

    Args:
        sentences (list): 句子列表
        model (str): 使用的嵌入模型名称

    Returns:
        numpy.ndarray: 句子嵌入向量矩阵
    """
    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbedding(model_name=model)
    # 批量创建嵌入向量
    embeddings = embedding_model.get_text_embedding_batch(sentences)

    return np.array(embeddings)

# 创建句子嵌入示例
sentences = ["AI is a branch of computer science.",
             "It aims to create intelligent machines.",
             "Machine learning is a subset of AI."]

embeddings = create_sentence_embeddings(sentences)
print(f"嵌入矩阵形状: {embeddings.shape}")
# 嵌入矩阵形状: (3, 768)   768是嵌入向量的维度
```

#### 2.1.3 语义相似度计算模块 (3semantic_similarity.py)

该模块计算句子间的语义相似度，为语义分块提供依据。

```python
"""
语义相似度计算模块
计算句子间的语义相似度，为语义分块提供依据
"""

import numpy as np  # 数值计算库

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度
    
    Args:
        vec1 (array-like): 第一个向量
        vec2 (array-like): 第二个向量
        
    Returns:
        float: 两个向量的余弦相似度，值域为[-1, 1]
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def calculate_similarities(embeddings):
    """
    计算相邻句子间的相似度
    
    Args:
        embeddings (numpy.ndarray): 句子嵌入向量矩阵
        
    Returns:
        list: 相邻句子间的相似度列表
    """
    similarities = []

    # 计算每对相邻句子的相似度
    for i in range(len(embeddings) - 1):
        similarity = cosine_similarity(embeddings[i], embeddings[i + 1])
        similarities.append(similarity)

    return similarities

# 计算相邻句子相似度示例
# 注意：这里假设 embeddings 已经定义
# similarities = calculate_similarities(embeddings)
# print("相邻句子相似度:", similarities)
```

#### 2.1.4 分块边界检测模块 (4chunking_boundaries.py)

该模块基于句子间相似度确定语义分块的边界位置。

```python
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
```

#### 2.1.5 语义分块构建模块 (5semantic_chunks.py)

该模块根据分块边界将句子重新组合成语义完整的文本块。

```python
"""
语义分块构建模块
根据分块边界将句子重新组合成语义完整的文本块
"""

def create_semantic_chunks(sentences, boundaries):
    """
    根据边界创建语义分块
    
    Args:
        sentences (list): 句子列表
        boundaries (list): 分块边界位置列表
        
    Returns:
        list: 语义分块列表
    """
    chunks = []

    # 根据边界将句子合并为语义块
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]

        # 合并句子为一个块
        chunk = " ".join(sentences[start:end])
        chunks.append(chunk)

    return chunks

# 创建语义分块示例
# 注意：这里假设 sentences 和 boundaries 已经定义
# semantic_chunks = create_semantic_chunks(sentences, boundaries)

# print("语义分块结果:")
# for i, chunk in enumerate(semantic_chunks):
#     print(f"块 {i+1}: {chunk}")
```

#### 2.1.6 完整实现 (6semantic_chunking_complete.py)

这是语义分块系统的完整实现，集成了上述所有模块。

```python
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
```

### 2.2 增强模块（Upgrade）

#### 2.2.1 动态阈值计算模块 (1dynamic_threshold.py)

该模块基于相似度分布计算动态分块阈值，提高分块的适应性。

```python
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
```

#### 2.2.2 分块大小控制模块 (2chunk_size_control.py)

该模块在语义分块的基础上增加对分块大小的控制，确保分块既符合语义又满足大小要求。

```python
"""
分块大小控制模块
在语义分块的基础上增加对分块大小的控制，确保分块既符合语义又满足大小要求
"""

def create_controlled_chunks(sentences, similarities,
                           max_chunk_size=1000, min_chunk_size=200):
    """
    控制分块大小的语义分块
    
    Args:
        sentences (list): 句子列表
        similarities (list): 相邻句子相似度列表
        max_chunk_size (int): 最大分块大小
        min_chunk_size (int): 最小分块大小
        
    Returns:
        list: 控制大小后的分块列表
    """
    chunks = []
    current_chunk = []
    current_size = 0

    # 遍历句子列表构建分块
    for i, sentence in enumerate(sentences):
        current_chunk.append(sentence)
        current_size += len(sentence)

        # 检查是否需要分块
        should_break = False

        # 如果不是最后一个句子
        if i < len(similarities):
            # 如果相似度低且当前块大小合适
            if (similarities[i] < 0.5 and
                current_size >= min_chunk_size):
                should_break = True

        # 如果块太大，强制分块
        if current_size >= max_chunk_size:
            should_break = True

        # 满足分块条件或到达最后一个句子时进行分块
        if should_break or i == len(sentences) - 1:
            chunk_text = " ".join(current_chunk)
            chunks.append(chunk_text)
            current_chunk = []
            current_size = 0

    return chunks
```

#### 2.2.3 分块质量评估模块 (3evaluation.py)

该模块评估不同分块策略的质量，包括检索准确率、分块大小等指标。

```python
"""
分块质量评估模块
评估不同分块策略的质量，包括检索准确率、分块大小等指标
"""

import numpy as np  # 数值计算库
from sentence_transformers.util import semantic_search


def evaluate_chunking_quality(chunks, queries, ground_truth):
    """
    评估分块质量

    评估指标：
    1. 检索准确率
    2. 平均分块大小
    3. 大小方差
    
    Args:
        chunks (list): 分块列表
        queries (list): 查询列表
        ground_truth (list): 真实答案列表
        
    Returns:
        dict: 包含各项评估指标的字典
    """
    # 创建嵌入（注意：这里假设create_embeddings函数已经定义）
    chunk_embeddings = create_embeddings(chunks)

    # 检索测试
    correct_retrievals = 0
    total_queries = len(queries)

    # 对每个查询进行测试
    for query, truth in zip(queries, ground_truth):
        # 进行语义搜索（注意：这里假设semantic_search函数已经定义）
        retrieved = semantic_search(query, chunks, chunk_embeddings, k=1)
        # 简化的评估方法：检查真实答案是否在检索结果中
        if truth in retrieved[0]:
            correct_retrievals += 1

    # 计算准确率
    accuracy = correct_retrievals / total_queries

    # 分块统计
    chunk_sizes = [len(chunk) for chunk in chunks]
    avg_size = np.mean(chunk_sizes)
    size_variance = np.var(chunk_sizes)

    # 返回评估结果
    return {
        'accuracy': accuracy,
        'avg_chunk_size': avg_size,
        'size_variance': size_variance,
        'num_chunks': len(chunks)
    }
```

## 3. 自适应分块选择系统（Select Chunk）

自适应分块选择系统基于文档和查询特征选择最优分块大小，以优化检索效果。

### 3.1 模块详解

#### 3.1.1 分块大小选择器 (1chunk_size_selector.py)

该模块基于查询特征选择最优分块大小。

```python
class ChunkSizeSelector:
    def __init__(self):
        self.chunk_sizes = [256, 512, 1024, 2048]
        self.indices = {}  # 多尺度索引

    def select_optimal_size(self, query):
        """基于查询特征选择最优分块大小"""
        query_complexity = self.analyze_query_complexity(query)

        if query_complexity == "simple":
            return 256
        elif query_complexity == "medium":
            return 512
        else:
            return 1024

    def analyze_query_complexity(self, query):
        """分析查询复杂度"""
        # 实现查询复杂度分析逻辑
        pass
```

#### 3.1.2 完整实现 (2summary.py)

这是自适应分块选择系统的完整实现。

关键特性：
- 分析文档特征以确定最优分块大小
- 分析查询特征以优化分块策略
- 比较不同分块大小的性能
- 选择最佳分块大小进行RAG

由于该文件内容较长，此处省略详细代码展示。该模块包含以下关键功能：

1. 文档特征分析（长度、句子长度、段落长度、词汇丰富度等）
2. 查询特征分析（长度、复杂度等）
3. 分块大小推荐算法
4. 不同分块大小性能比较
5. 自适应RAG流程

## 4. 技术要点总结

### 4.1 核心技术栈

1. **文档处理**：使用PyMuPDF (fitz) 处理PDF文档
2. **文本嵌入**：使用HuggingFace的预训练模型（如BAAI/bge-base-en-v1.5）
3. **语义搜索**：基于余弦相似度的向量检索
4. **大语言模型**：使用OpenAI API或兼容接口（如SiliconFlow）
5. **数值计算**：使用NumPy进行向量运算

### 4.2 分块策略演进

1. **固定大小分块**：简单的固定大小分块，支持重叠
2. **语义分块**：基于句子语义相似度的智能分块
3. **自适应分块**：基于文档和查询特征选择最优分块大小

### 4.3 性能优化考虑

1. **批量处理**：嵌入计算采用批量处理提高效率
2. **动态阈值**：根据相似度分布动态调整分块阈值
3. **大小控制**：在语义分块基础上控制分块大小
4. **性能评估**：通过多项指标评估分块质量

这套代码展现了从基础RAG系统到高级语义分块技术的完整演进过程，为理解和实现高效的RAG系统提供了完整的参考实现。