import fitz  # PyMuPDF库，用于处理PDF文件
import os  # 操作系统接口
import numpy as np  # 数值计算库，用于向量运算
import json  # JSON数据处理
from openai import OpenAI  # OpenAI API客户端
import re  # 正则表达式模块
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # HuggingFace嵌入模型

# 初始化OpenAI客户端
# base_url: SiliconFlow API端点
# api_key: API密钥
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key='sk-zqzehnidkvjxmpgoqohexqzxwnvyszxwgxucpxmtftdpgrgv'
)

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容
    
    Args:
        pdf_path (str): PDF文件的路径
        
    Returns:
        str: 提取的文本内容
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

def chunk_text(text, chunk_size=800, overlap=0):
    """
    将文本分割成指定大小的块
    
    RSE通常使用非重叠块以便能够正确重构段落
    
    Args:
        text (str): 要分割的原始文本
        chunk_size (int): 每个文本块的大小（字符数）
        overlap (int): 相邻块之间的重叠字符数
        
    Returns:
        List[str]: 文本块列表
    """
    chunks = []

    # 按指定步长分割文本，步长 = 块大小 - 重叠大小
    # 当overlap为0时，块之间没有重叠
    for i in range(0, len(text), chunk_size - overlap):
        chunk = text[i:i + chunk_size]
        # 确保块不为空
        if chunk:
            chunks.append(chunk)

    return chunks

def create_embeddings(texts, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建向量嵌入
    
    Args:
        texts (str or list): 单个文本字符串或文本列表
        model (str): 使用的嵌入模型名称，默认为"BAAI/bge-base-en-v1.5"
        
    Returns:
        嵌入向量或嵌入向量列表
    """
    # 如果texts为空，直接返回空列表
    if not texts:
        return []

    # 初始化嵌入模型
    embedding_model = HuggingFaceEmbedding(model_name=model)

    # 判断对象是否是list类型 如果是就批量嵌入 否则就单句嵌入
    if isinstance(texts, list):
        # 批量处理文本列表
        response = embedding_model.get_text_embedding_batch(texts)
    else:
        # 处理单个文本
        response = embedding_model.get_text_embedding(texts)

    return response

class SimpleVectorStore:
    """轻量级向量存储实现
    
    用于存储文档及其对应的向量嵌入，支持相似度搜索功能
    """
    def __init__(self, dimension=1536):
        """
        初始化向量存储
        
        Args:
            dimension (int): 向量维度，默认为1536
        """
        self.dimension = dimension  # 向量维度
        self.vectors = []          # 存储向量嵌入
        self.documents = []        # 存储原始文档
        self.metadata = []         # 存储元数据

    def add_documents(self, documents, vectors=None, metadata=None):
        """
        向向量存储中添加文档
        
        Args:
            documents (List[str]): 要添加的文档列表
            vectors (List[array], optional): 对应的向量嵌入列表
            metadata (List[dict], optional): 对应的元数据列表
        """
        # 如果未提供向量，则为每个文档创建一个None占位符
        if vectors is None:
            # 如果 vectors 是 None，则创建一个长度与文档数量相同的列表，列表中每个元素都是 None
            vectors = [None] * len(documents)

        # 如果未提供元数据，则为每个文档创建一个空字典
        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        # 遍历文档、向量和元数据，将它们添加到存储中
        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)   # 添加文档
            self.vectors.append(vec)     # 添加向量
            self.metadata.append(meta)   # 添加元数据

    def search(self, query_vector, top_k=5):
        """
        基于余弦相似度搜索最相似的文档
        
        Args:
            query_vector (array-like): 查询向量
            top_k (int): 返回最相关的k个结果，默认为5
            
        Returns:
            List[Dict]: 最相关的文档列表，每个元素包含document、score和metadata
        """
        # 检查是否有文档和向量
        if not self.vectors or not self.documents:
            return []

        # 将查询向量转换为numpy数组
        query_array = np.array(query_vector)

        # 计算查询向量与每个存储向量的相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            # 只处理非空向量
            if vector is not None:
                # 使用余弦相似度计算相似度
                # 余弦相似度 = (A·B) / (||A|| × ||B||)
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取前top-k个结果
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],     # 文档内容
                "score": float(score),             # 相似度得分
                "metadata": self.metadata[i]       # 元数据
            })

        return results

def calculate_chunk_values(query, chunks, vector_store, irrelevant_chunk_penalty=0.2):
    """
    计算每个文档块相对于查询的价值分数
    
    通过向量相似度搜索计算每个块的相关性，并对低分块应用惩罚

    Args:
        query (str): 用户查询
        chunks (List[str]): 文档块列表
        vector_store: 向量存储
        irrelevant_chunk_penalty (float): 不相关块的惩罚系数，低于此值的块会被重度惩罚

    Returns:
        List[float]: 每个块的价值分数列表
    """
    # 为查询创建嵌入向量
    query_embedding = create_embeddings(query)
    # 使用向量存储搜索所有块的相关性得分
    search_results = vector_store.search(query_embedding, top_k=len(chunks))

    # 创建块索引到得分的映射
    chunk_scores = {}
    for result in search_results:
        """
            result = {
        "document": "这是文档内容...",
        "score": 0.85,
        "metadata": {
            "chunk_index": 5,
            "source": "data/AI_Information.pdf"
            }
        }

        """
        chunk_index = result["metadata"]["chunk_index"]
        chunk_scores[chunk_index] = result["score"]

    # 计算每个块的价值，包括对不相关块的惩罚
    chunk_values = []
    for i in range(len(chunks)):
        if i in chunk_scores:
            base_score = chunk_scores[i]
            # 对低分数块应用惩罚
            # 如果块的得分低于惩罚阈值，则对其进行重度惩罚（乘以0.1）
            if base_score < irrelevant_chunk_penalty:
                value = base_score * 0.1  # 重度惩罚
            else:
                value = base_score
        else:
            # 未找到的块价值为0
            value = 0.0

        chunk_values.append(value)

    return chunk_values

def find_best_segments(chunk_values, max_segment_length=20, total_max_length=30, min_segment_value=0.2):
    """
    使用动态规划算法找到最佳的连续文档段落
    
    通过评估不同段落组合的价值，选择总价值最高的不重叠段落集合

    Args:
        chunk_values (List[float]): 每个块的价值分数
        max_segment_length (int): 单个段落的最大长度（块数）
        total_max_length (int): 所有段落的总最大长度（块数）
        min_segment_value (float): 段落的最小平均价值，低于此值的段落将被忽略

    Returns:
        List[Tuple[int, int]]: 最佳段落的(起始索引, 结束索引)列表
    """
    n = len(chunk_values)  # 块的总数
    segments = []  # 存储所有可能的段落

    # 动态规划找到所有可能的段落组合
    # 遍历所有可能的起始位置
    for start in range(n):
        # 遍历所有可能的段落长度
        for length in range(1, min(max_segment_length + 1, n - start + 1)):
            end = start + length - 1  # 计算结束位置

            # 计算段落的平均价值
            segment_values = chunk_values[start:end + 1]
            avg_value = sum(segment_values) / len(segment_values)

            # 只保留高于最小价值阈值的段落
            if avg_value >= min_segment_value:
                segments.append({
                    'start': start,           # 起始索引
                    'end': end,              # 结束索引
                    'length': length,         # 段落长度
                    'avg_value': avg_value,   # 平均价值
                    'total_value': sum(segment_values)  # 总价值
                })

    # 按总价值降序排序，优先考虑总价值高的段落
    segments.sort(key=lambda x: x['total_value'], reverse=True)

    # 贪心算法选择不重叠的最佳段落
    selected_segments = []      # 存储选中的段落
    used_chunks = set()         # 记录已被使用的块
    total_length = 0            # 当前选中段落的总长度

    # 遍历所有按价值排序的段落
    for segment in segments:
        # 检查是否与已选段落重叠
        segment_chunks = set(range(segment['start'], segment['end'] + 1))
        if not segment_chunks.intersection(used_chunks):
            # 检查是否超过总长度限制
            if total_length + segment['length'] <= total_max_length:
                # 添加段落到选中列表
                selected_segments.append((segment['start'], segment['end']))
                # 更新已使用的块集合
                used_chunks.update(segment_chunks)
                # 更新总长度
                total_length += segment['length']

    # 按起始位置排序以保持文档顺序
    selected_segments.sort(key=lambda x: x[0])

    return selected_segments

def reconstruct_segments(chunks, best_segments):
    """
    根据最佳段落索引重构完整的文档段落
    
    将连续的文档块合并成完整的段落，以提供更连贯的上下文

    Args:
        chunks (List[str]): 原始文档块列表
        best_segments (List[Tuple[int, int]]): 最佳段落的索引范围列表，每个元素是(起始索引, 结束索引)

    Returns:
        List[str]: 重构后的文档段落列表
    """
    reconstructed_segments = []

    # 遍历所有选中的段落
    for start, end in best_segments:
        # 合并连续的块形成完整段落
        # 使用空格连接相邻的块
        segment_text = " ".join(chunks[start:end + 1])
        reconstructed_segments.append(segment_text)

    return reconstructed_segments

def format_segments_for_context(segments):
    """
    格式化段落用于作为大语言模型的上下文
    
    为每个段落添加编号标签，使模型更容易区分不同的段落

    Args:
        segments (List[str]): 重构的文档段落列表

    Returns:
        str: 格式化的上下文字符串，段落之间用双换行分隔
    """
    formatted_context = []

    # 遍历所有段落，为每个段落添加编号
    for i, segment in enumerate(segments, 1):
        # 格式化每个段落：段落编号 + 换行 + 段落内容
        formatted_context.append(f"段落{i}:\n{segment}")

    # 使用双换行符连接所有段落
    return "\n\n".join(formatted_context)

def generate_response(query, context, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    基于给定上下文生成对查询的回答
    
    使用大语言模型基于检索到的上下文生成自然语言回答

    Args:
        query (str): 用户查询
        context (str): 检索到的相关上下文
        model (str): 使用的大语言模型，默认为"Qwen/Qwen2.5-72B-Instruct"
        
    Returns:
        str: AI生成的回答
    """
    # 系统提示词 - 定义AI助手的行为准则
    system_prompt = "你是一个AI助手，严格基于给定的上下文回答问题。如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    # 用户提示词 - 将检索到的上下文和用户查询组合成完整的提示
    user_prompt = f"""
    上下文:
    {context}

    问题: {query}

    请基于以上上下文回答问题。
    """

    # 调用大语言模型API生成回答
    response = client.chat.completions.create(
        model=model,           # 使用的模型
        temperature=0,         # 设置为0以获得确定性回答
        messages=[
            {"role": "system", "content": system_prompt},   # 系统提示
            {"role": "user", "content": user_prompt}       # 用户提示
        ]
    )

    # 返回模型生成的回答内容
    return response.choices[0].message.content

def rag_with_rse(pdf_path, query, chunk_size=800, irrelevant_chunk_penalty=0.2):
    """
    使用检索-合成-执行(RSE)架构的完整RAG流程
    
    实现了完整的RSE流程：文档处理 -> 块价值计算 -> 最佳段落查找 -> 段落重构 -> 回答生成

    Args:
        pdf_path (str): PDF文档路径
        query (str): 用户查询
        chunk_size (int): 文档块大小（字符数）
        irrelevant_chunk_penalty (float): 不相关块惩罚系数，低于此值的块会被重度惩罚

    Returns:
        dict: 包含RSE处理结果的字典
    """
    print("开始RSE处理流程...")

    # 1. 处理文档
    # 提取文本、分块、创建嵌入向量并建立向量存储
    print("1. 处理文档...")
    chunks, vector_store, doc_info = process_document(pdf_path, chunk_size)

    # 2. 计算块价值
    # 基于查询计算每个文档块的相关性价值
    print("2. 计算块价值...")
    chunk_values = calculate_chunk_values(
        query, chunks, vector_store, irrelevant_chunk_penalty
    )

    # 输出块价值的统计信息
    print(f"块价值分布: 最高={max(chunk_values):.3f}, 最低={min(chunk_values):.3f}, 平均={np.mean(chunk_values):.3f}")

    # 3. 找到最佳段落
    # 使用动态规划算法找到最有价值的不重叠段落组合
    print("3. 寻找最佳段落...")
    best_segments = find_best_segments(
        chunk_values,
        max_segment_length=20,      # 单个段落最多包含20个块
        total_max_length=30,        # 所有段落总计最多包含30个块
        min_segment_value=0.2       # 段落的最小平均价值
    )

    # 输出找到的最佳段落信息
    print(f"找到 {len(best_segments)} 个最佳段落")
    for i, (start, end) in enumerate(best_segments):
        print(f"  段落{i+1}: 块{start}-{end} (长度: {end-start+1})")

    # 4. 重构段落
    # 将选中的块合并成完整的段落
    print("4. 重构文档段落...")
    reconstructed_segments = reconstruct_segments(chunks, best_segments)

    # 5. 格式化上下文
    # 将重构的段落格式化为模型友好的上下文
    context = format_segments_for_context(reconstructed_segments)

    # 输出上下文长度信息
    print(f"生成的上下文长度: {len(context)} 字符")

    # 6. 生成回答
    # 使用大语言模型基于上下文生成回答
    print("5. 生成回答...")
    response = generate_response(query, context)

    # 返回完整的处理结果
    return {
        "query": query,                          # 用户查询
        "total_chunks": len(chunks),             # 总块数
        "chunk_values": chunk_values,            # 每个块的价值
        "best_segments": best_segments,          # 最佳段落
        "reconstructed_segments": reconstructed_segments,  # 重构的段落
        "context": context,                      # 格式化后的上下文
        "response": response,                    # 生成的回答
        "doc_info": doc_info                     # 文档信息
    }

def process_document(pdf_path, chunk_size=800):
    """
    处理文档用于RSE流程
    
    包括提取文本、分块、创建嵌入向量和建立向量存储等步骤

    Args:
        pdf_path (str): PDF文档路径
        chunk_size (int): 文档块大小

    Returns:
        Tuple[List[str], SimpleVectorStore, Dict]: 文档块列表、向量存储实例和文档信息字典
    """
    print("正在从文档提取文本...")
    # 从PDF文件中提取全部文本内容
    text = extract_text_from_pdf(pdf_path)

    print("正在分块文本为非重叠段落...")
    # 将文本分割成非重叠的块
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=0)
    print(f"创建了 {len(chunks)} 个块")

    print("正在为块生成嵌入...")
    # 为每个文本块创建向量嵌入
    chunk_embeddings = create_embeddings(chunks)

    # 创建向量存储实例
    vector_store = SimpleVectorStore()

    # 为每个块创建元数据，包括块索引和源文件路径
    # 块索引用于后续的段落重构
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    # 将文档块、嵌入向量和元数据添加到向量存储中
    vector_store.add_documents(chunks, chunk_embeddings, metadata)

    # 跟踪原始文档结构用于段落重构
    doc_info = {
        "total_chunks": len(chunks),      # 总块数
        "chunk_size": chunk_size,         # 块大小
        "total_characters": len(text),    # 总字符数
        "source": pdf_path               # 源文件路径
    }

    return chunks, vector_store, doc_info

## 实际应用示例


# RSE完整流程演示
pdf_path = "data/AI_Information.pdf"
query = "什么是深度学习的主要特点？"

print(f"查询: {query}")
print("="*60)

# 执行RSE流程
rse_result = rag_with_rse(
    pdf_path=pdf_path,
    query=query,
    chunk_size=800,
    irrelevant_chunk_penalty=0.2
)

# 显示详细结果
print(f"\n📊 RSE处理结果:")
print(f"- 总文档块数: {rse_result['total_chunks']}")
print(f"- 选择的段落数: {len(rse_result['best_segments'])}")
print(f"- 上下文总长度: {len(rse_result['context'])} 字符")

print(f"\n🎯 选择的段落:")
for i, (start, end) in enumerate(rse_result['best_segments']):
    avg_value = np.mean(rse_result['chunk_values'][start:end+1])
    print(f"段落{i+1}: 块{start}-{end}, 平均价值: {avg_value:.3f}")

print(f"\n📝 重构的段落预览:")
for i, segment in enumerate(rse_result['reconstructed_segments']):
    print(f"\n段落{i+1} (前200字符):")
    print(segment[:200] + "...")

print(f"\n🤖 生成的回答:")
print(rse_result['response'])

# 与标准RAG对比
print(f"\n" + "="*60)
print("RSE vs 标准RAG对比:")
print("="*60)

# 标准RAG
# 注意：这部分代码引用了未定义的函数，实际运行时会报错
# standard_result = standard_top_k_retrieval(pdf_path, query, k=10)
print(f"\n标准RAG:")
print(f"- 检索块数: {len(standard_result['results'])}")
print(f"- 上下文长度: {len(standard_result['context'])} 字符")
print(f"- 回答: {standard_result['response'][:200]}...")

print(f"\nRSE:")
print(f"- 智能段落数: {len(rse_result['best_segments'])}")
print(f"- 上下文长度: {len(rse_result['context'])} 字符")
print(f"- 回答: {rse_result['response'][:200]}...")

# 程序结束
