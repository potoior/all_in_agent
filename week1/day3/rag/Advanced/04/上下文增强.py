import fitz
import os
import numpy as np
import json
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容

    Args:
        pdf_path (str): PDF文件路径

    Returns:
        str: 提取的文本内容
    """
    # 打开PDF文件
    mypdf = fitz.open(pdf_path)
    all_text = ""

    # 遍历每一页并提取文本
    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

def chunk_text(text, n, overlap):
    """
    将文本分割成重叠的文本块

    Args:
        text (str): 要分割的文本
        n (int): 每个文本块的字符数
        overlap (int): 重叠字符数

    Returns:
        List[str]: 文本块列表
    """
    chunks = []

    # 按指定步长分割文本，步长 = 块大小 - 重叠大小
    # 这样可以确保相邻块之间有指定大小的重叠
    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks

# 初始化OpenAI客户端，用于后续生成回答
# base_url: SiliconFlow API端点
# api_key: API密钥
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
   api_key=os.getenv("OPENROUTER_API_KEY")
)

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建嵌入向量

    Args:
        text (str): 输入文本
        model (str): 嵌入模型名称

    Returns:
        List[float]: 嵌入向量
    """
    # 初始化HuggingFace嵌入模型
    embedding_model = HuggingFaceEmbedding(model_name=model)

    # 根据输入类型创建嵌入向量
    if isinstance(text, list):
        # 批量处理文本列表
        response = embedding_model.get_text_embedding_batch(text)
    else:
        # 处理单个文本
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
    # 余弦相似度公式: (A·B) / (||A|| × ||B||)
    # 分子是两向量的点积，分母是两向量模长的乘积
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def context_enriched_search(query, text_chunks, embeddings, k=1, context_size=1):
    """
    执行上下文增强检索

    Args:
        query (str): 查询问题
        text_chunks (List[str]): 文本块列表
        embeddings (List): 嵌入向量列表
        k (int): 检索的相关块数量
        context_size (int): 上下文邻居块数量

    Returns:
        List[str]: 包含上下文的相关文本块
    """
    # 将查询转换为嵌入向量，用于后续相似度计算
    query_embedding = create_embeddings(query)
    similarity_scores = []

    # 计算查询与每个文本块的相似度
    for i, chunk_embedding in enumerate(embeddings):
        # 使用余弦相似度计算查询与当前文本块的相似度
        similarity_score = cosine_similarity(
            np.array(query_embedding),
            np.array(chunk_embedding)
        )
        similarity_scores.append((i, similarity_score))

    # 按相似度降序排序，最相关的块排在前面
    similarity_scores.sort(key=lambda x: x[1], reverse=True)

    # 获取最相关块的索引
    top_index = similarity_scores[0][0]
    print(f'最相关块索引: {top_index}')

    # 确定上下文范围，确保不超出边界
    # start: 起始索引，确保不小于0
    # end: 结束索引，确保不超过文本块总数
    # 这里的context_size是1,代表获取上下各1块
    start = max(0, top_index - context_size)
    # +1 是因为左闭右开
    end = min(len(text_chunks), top_index + context_size + 1)

    # 返回相关块及其邻近上下文
    return [text_chunks[i] for i in range(start, end)]

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
    # 调用大模型API生成回答
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # 温度设为0以获得更确定性的输出
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )

    return response.choices[0].message.content


# 实际应用
# 1. 文档处理：从PDF文件中提取文本
pdf_path = "../../basic_rag/data/AI_Information.pdf"
extracted_text = extract_text_from_pdf(pdf_path)

# 2. 文本分块：将提取的文本分割成重叠的块
# 参数说明：块大小1000字符，重叠200字符
text_chunks = chunk_text(extracted_text, 1000, 200)
print(f"创建了 {len(text_chunks)} 个文本块")

# 3. 创建嵌入向量：为每个文本块创建对应的向量表示
embeddings = create_embeddings(text_chunks)

# 4. 加载测试查询：从JSON文件中读取测试问题
with open('data/val.json') as f:
    data = json.load(f)

# 获取第一个测试问题
query = data[0]['question']
print(f"查询: {query}")

# 5. 执行上下文增强检索
# context_size=1 表示包含前后各1个邻居块作为上下文
top_chunks = context_enriched_search(
    query,
    text_chunks,
    embeddings,
    k=1,
    context_size=1
)

print(f"检索到 {len(top_chunks)} 个上下文块")

# 6. 显示检索结果
for i, chunk in enumerate(top_chunks):
    print(f"上下文 {i + 1}:\n{chunk}\n" + "="*50)

# 7. 生成最终回答
# 定义系统提示词：约束AI助手的行为
system_prompt = "你是一个AI助手，严格基于给定的上下文回答问题。如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

# 组合上下文：将检索到的上下文块合并成一个字符串
context = "\n\n".join([f"上下文{i+1}: {chunk}" for i, chunk in enumerate(top_chunks)])
# 构造用户消息：包含上下文和原始问题
user_message = f"上下文:\n{context}\n\n问题: {query}"

# 调用AI生成回答
response = generate_response(system_prompt, user_message)
print(f"AI回答: {response}")