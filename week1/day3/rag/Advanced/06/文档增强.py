import fitz
import os
import numpy as np
import json
from openai import OpenAI
import re
from tqdm import tqdm
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
    api_key='sk-zqzehnidkvjxmpgoqohexqzxwnvyszxwgxucpxmtftdpgrgv'
)

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本内容
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

def chunk_text(text, n, overlap):
    """
    将文本分割成重叠的文本块
    """
    chunks = []

    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks

def generate_questions(text_chunk, num_questions=5, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    为给定的文本块生成相关问题

    Args:
        text_chunk (str): 文本块内容
        num_questions (int): 要生成的问题数量
        model (str): 使用的模型

    Returns:
        List[str]: 生成的问题列表
    """
    system_prompt = "你是一个专业的问题生成专家。根据给定文本创建简洁的问题，这些问题只能使用提供的文本来回答。专注于关键信息和概念。"

    user_prompt = f"""
    基于以下文本，生成{num_questions}个不同的问题，这些问题只能使用这段文本来回答：

    {text_chunk}

    请将回答格式化为编号列表，只包含问题，不包含额外文本。
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 提取和清理问题
    questions_text = response.choices[0].message.content.strip()
    print("提取和清理的问题为"+questions_text)
    questions = []

    # 使用正则表达式提取问题
    for line in questions_text.split('\n'):
        # 移除编号并清理空白字符
        cleaned_line = re.sub(r'^\d+\.\s*', '', line.strip())
        if cleaned_line and cleaned_line.endswith('?'):
            questions.append(cleaned_line)

    return questions

def create_embeddings(text, model="BAAI/bge-base-en-v1.5"):
    """
    为给定文本创建嵌入向量
    """
    embedding_model = HuggingFaceEmbedding(model_name=model)

    if isinstance(text, list):
        response = embedding_model.get_text_embedding_batch(text)
    else:
        response = embedding_model.get_text_embedding(text)

    return response

class SimpleVectorStore:
    """
    简单的向量存储实现
    """
    def __init__(self):
        self.vectors = []
        self.texts = []
        self.metadata = []

    def add_item(self, text, embedding, metadata=None):
        """
        向向量存储中添加项目

        Args:
            text (str): 原始文本
            embedding (List[float]): 嵌入向量
            metadata (dict, optional): 额外的元数据
        """
        self.vectors.append(np.array(embedding))
        self.texts.append(text)
        self.metadata.append(metadata or {})

    def similarity_search(self, query_embedding, k=5):
        """
        查找与查询嵌入最相似的项目

        Args:
            query_embedding (List[float]): 查询嵌入向量
            k (int): 返回结果数量

        Returns:
            List[Dict]: 最相似的k个项目及其文本和元数据
        """
        if not self.vectors:
            return []

        query_vector = np.array(query_embedding)

        # 使用余弦相似度计算相似性
        similarities = []
        for i, vector in enumerate(self.vectors):
            similarity = np.dot(query_vector, vector) / (
                np.linalg.norm(query_vector) * np.linalg.norm(vector)
            )
            similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回前k个结果
        results = []
        for i in range(min(k, len(similarities))):
            idx, score = similarities[i]
            results.append({
                "text": self.texts[idx],
                "metadata": self.metadata[idx],
                "similarity": score
            })

        return results

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200, questions_per_chunk=5):
    """
    处理文档并进行问题增强

    Args:
        pdf_path (str): PDF文件路径
        chunk_size (int): 块大小（字符数）
        chunk_overlap (int): 块重叠（字符数）
        questions_per_chunk (int): 每个块生成的问题数

    Returns:
        SimpleVectorStore: 包含文档块和生成问题的向量存储
    """
    print("正在从文档中提取文本...")
    text = extract_text_from_pdf(pdf_path)

    print("正在分割文档...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    print(f"创建了 {len(chunks)} 个文本块")

    # 初始化向量存储
    vector_store = SimpleVectorStore()

    print("正在为每个块生成问题和嵌入...")
    for i, chunk in enumerate(tqdm(chunks, desc="处理文档块")):
        try:
            # 为块生成问题
            questions = generate_questions(chunk, questions_per_chunk)

            # 为原始文本块创建嵌入
            chunk_embedding = create_embeddings(chunk)

            # 将原始块添加到向量存储
            vector_store.add_item(
                text=chunk,
                embedding=chunk_embedding,
                metadata={
                    "type": "original_chunk",
                    "chunk_index": i,
                    "source": pdf_path,
                    "questions": questions
                }
            )

            # 为每个生成的问题创建嵌入并添加到向量存储
            for j, question in enumerate(questions):
                question_embedding = create_embeddings(question)
                vector_store.add_item(
                    text=question,
                    embedding=question_embedding,
                    metadata={
                        "type": "generated_question",
                        "chunk_index": i,
                        "question_index": j,
                        "source": pdf_path,
                        "original_chunk": chunk
                    }
                )

        except Exception as e:
            print(f"处理块 {i} 时出错: {e}")
            continue

    print(f"处理完成！向量存储包含 {len(vector_store.texts)} 个项目")
    return vector_store

def semantic_search(query, vector_store, k=5):
    """
    执行语义搜索

    Args:
        query (str): 用户查询
        vector_store (SimpleVectorStore): 向量存储
        k (int): 返回结果数

    Returns:
        List[Dict]: 搜索结果
    """
    query_embedding = create_embeddings(query)
    results = vector_store.similarity_search(query_embedding, k)
    return results

def prepare_context(search_results):
    """
    准备用于生成回答的上下文

    Args:
        search_results (List[Dict]): 搜索结果

    Returns:
        str: 格式化的上下文字符串
    """
    context_parts = []

    for i, result in enumerate(search_results):
        metadata = result["metadata"]

        if metadata["type"] == "original_chunk":
            # 如果是原始块，直接使用
            context_parts.append(f"上下文 {i+1}: {result['text']}")
        else:
            # 如果是生成的问题，使用对应的原始块
            original_chunk = metadata["original_chunk"]
            context_parts.append(f"上下文 {i+1}: {original_chunk}")

    return "\n\n".join(context_parts)

def generate_response(query, context, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    基于上下文生成回答
    """
    system_prompt = "你是一个AI助手，严格基于给定的上下文回答问题。如果无法从提供的上下文中得出答案，请回答：'我没有足够的信息来回答这个问题。'"

    user_prompt = f"""
    上下文:
    {context}

    问题: {query}

    请基于以上上下文回答问题。
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content

def evaluate_response(query, response, reference_answer, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    评估生成回答的质量
    """
    evaluation_prompt = f"""
    请评估以下AI回答的质量：

    问题: {query}
    AI回答: {response}
    参考答案: {reference_answer}

    请从以下维度评分（1-5分）：
    1. 准确性：回答是否正确
    2. 完整性：回答是否完整
    3. 相关性：回答是否相关
    4. 清晰度：回答是否清晰易懂

    请提供总体评分和简短说明。
    """

    eval_response = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": "你是一个专业的回答质量评估专家。"},
            {"role": "user", "content": evaluation_prompt}
        ]
    )

    return eval_response.choices[0].message.content


# 实际案例
# 1. 处理文档
pdf_path = "../../basic_rag/data/AI_Information.pdf"
print("开始处理文档...")

vector_store = process_document(
    pdf_path=pdf_path,
    chunk_size=1000,
    chunk_overlap=200,
    questions_per_chunk=5
)

# 2. 显示处理结果
print(f"\n文档处理完成:")
print(f"- 总项目数: {len(vector_store.texts)}")

# 计算原始块和生成问题的数量
original_chunks = sum(1 for metadata in vector_store.metadata if metadata["type"] == "original_chunk")
generated_questions = sum(1 for metadata in vector_store.metadata if metadata["type"] == "generated_question")

print(f"- 原始文本块: {original_chunks}")
print(f"- 生成的问题: {generated_questions}")

# 3. 显示示例生成的问题
sample_chunk_metadata = next(metadata for metadata in vector_store.metadata if metadata["type"] == "original_chunk")
print(f"\n示例生成的问题:")
for i, question in enumerate(sample_chunk_metadata["questions"], 1):
    print(f"{i}. {question}")

# 4. 加载测试查询
with open('data/val.json') as f:
    data = json.load(f)

query = data[0]['question']
reference_answer = data[0]['answer']

print(f"\n测试查询: {query}")

# 5. 执行搜索
search_results = semantic_search(query, vector_store, k=5)

print(f"\n搜索结果 (共 {len(search_results)} 项):")
for i, result in enumerate(search_results):
    print(f"\n结果 {i+1}:")
    print(f"类型: {result['metadata']['type']}")
    print(f"相似度: {result['similarity']:.4f}")
    print(f"内容: {result['text'][:200]}...")

# 6. 准备上下文并生成回答
context = prepare_context(search_results)
response = generate_response(query, context)

print(f"\n生成的回答:")
print(response)

# 7. 评估回答质量
if reference_answer:
    evaluation = evaluate_response(query, response, reference_answer)
    print(f"\n回答质量评估:")
    print(evaluation)