
import os
import numpy as np
import json
from openai import OpenAI
import fitz
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容

    Args:
        pdf_path (str): PDF文件路径

    Returns:
        str: 提取的文本内容
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
   api_key=os.getenv("OPENROUTER_API_KEY")
)

# 传入的chunk是str类型
def generate_chunk_header(chunk, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    使用LLM为给定的文本块生成标题/头部

    Args:
        chunk (str): 要生成头部的文本块
        model (str): 使用的模型名称

    Returns:
        str: 生成的头部/标题
    """
    # 定义系统提示词
    system_prompt = "为给定的文本生成简洁且信息丰富的标题。"

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": chunk}
        ]
    )

    return response.choices[0].message.content.strip()

def chunk_text_with_headers(text, n, overlap):
    """
    将文本分块并为每个块生成头部

    Args:
        text (str): 要分块的完整文本
        n (int): 块大小（字符数）
        overlap (int): 重叠字符数

    Returns:
        List[dict]: 包含'header'和'text'键的字典列表
    """
    chunks = []

    # 按指定大小和重叠度分割文本
    for i in range(0, len(text), n - overlap):
        chunk = text[i:i + n]
        header = generate_chunk_header(chunk)  # 为块生成头部
        chunks.append({"header": header, "text": chunk})

    return chunks

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建嵌入向量

    Args:
        text (str): 输入文本
        model (str): 嵌入模型名称

    Returns:
        List[float]: 嵌入向量
    """
    embedding_model = HuggingFaceEmbedding(model_name=model)

    if isinstance(text, list):
        response = embedding_model.get_text_embedding_batch(text)
    else:
        response = embedding_model.get_text_embedding(text)

    return response

def cosine_similarity(vec1, vec2):
    """
    计算两个向量的余弦相似度

    Args:
        vec1 (np.ndarray): 第一个向量
        vec2 (np.ndarray): 第二个向量

    Returns:
        float: 余弦相似度值
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def semantic_search(query, chunks, k=5):
    """
    基于查询搜索最相关的文本块

    Args:
        query (str): 用户查询
        chunks (List[dict]): 包含头部和嵌入的文本块列表
        k (int): 返回的结果数

    Returns:
        List[dict]: 最相关的k个文本块
    """
    # 为查询创建嵌入
    query_embedding = create_embeddings(query)

    similarities = []

    # 计算每个块的相似度
    for chunk in chunks:
        # 计算查询与文本内容的相似度
        sim_text = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk["embedding"])
        )
        # 计算查询与头部的相似度
        sim_header = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk["header_embedding"])
        )
        # 计算平均相似度
        avg_similarity = (sim_text + sim_header) / 2
        similarities.append((chunk, avg_similarity))

    # 按相似度降序排序
    similarities.sort(key=lambda x: x[1], reverse=True)

    # 返回前k个最相关的块
    return [x[0] for x in similarities[:k]]

def generate_response(system_prompt, user_message, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    生成AI回答

    Args:
        system_prompt (str): 系统提示词
        user_message (str): 用户消息
        model (str): 使用的模型

    Returns:
        str: AI生成的回答
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content



# 技术解析
# 头部生成策略
# def generate_chunk_header(chunk, model="meta-llama/Llama-3.2-3B-Instruct"):
#     """
#     头部生成的核心逻辑
#     - 使用LLM提炼文本的关键信息
#     - 生成简洁但信息丰富的标题
#     - 保持与原文内容的一致性
#     """
#     system_prompt = "为给定的文本生成简洁且信息丰富的标题。"
#     # 低温度确保生成稳定的、相关的标题
#     response = client.chat.completions.create(
#         model=model,
#         temperature=0,  # 确保输出的一致性
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": chunk}
#         ]
#     )
#     return response.choices[0].message.content.strip()
#
# # 双重嵌入策略
# # 为每个块创建两个嵌入向量
# text_embedding = create_embeddings(chunk["text"])        # 内容嵌入
# header_embedding = create_embeddings(chunk["header"])    # 头部嵌入
#
# # 在检索时计算平均相似度
# sim_text = cosine_similarity(query_embedding, text_embedding)
# sim_header = cosine_similarity(query_embedding, header_embedding)
# avg_similarity = (sim_text + sim_header) / 2  # 平均分配权重
#
#
# # 相似度融合方法
# # 方法1: 简单平均（推荐）
# avg_similarity = (sim_text + sim_header) / 2
#
# # 方法2: 加权平均
# weight_text = 0.7
# weight_header = 0.3
# weighted_similarity = weight_text * sim_text + weight_header * sim_header
#
# # 方法3: 最大值
# max_similarity = max(sim_text, sim_header)
#
# # 方法4: 动态权重（根据查询类型调整）
# if is_specific_query(query):
#     # 具体查询更依赖内容
#     avg_similarity = 0.8 * sim_text + 0.2 * sim_header
# else:
#     # 概念性查询更依赖头部
#     avg_similarity = 0.4 * sim_text + 0.6 * sim_header
