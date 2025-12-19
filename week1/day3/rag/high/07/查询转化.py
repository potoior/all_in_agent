import fitz
import os
import numpy as np
import json
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://api.siliconflow.cn/v1",
   api_key=os.getenv("OPENROUTER_API_KEY")
)

def rewrite_query(original_query, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    重写查询使其更具体和详细

    Args:
        original_query (str): 原始用户查询
        model (str): 用于查询重写的模型

    Returns:
        str: 重写后的查询
    """
    system_prompt = "你是一个专门改进搜索查询的AI助手。你的任务是将用户查询重写得更具体、详细，更有可能检索到相关信息。"

    user_prompt = f"""
    将以下查询重写得更具体和详细。包含相关术语和概念，这些可能有助于检索准确信息。

    原始查询: {original_query}

    重写查询:
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,  # 低温度确保输出确定性
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def generate_step_back_query(original_query, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    生成更通用的'步退'查询以检索更广泛的上下文

    Args:
        original_query (str): 原始用户查询
        model (str): 用于步退查询生成的模型

    Returns:
        str: 步退查询
    """
    system_prompt = "你是一个专门研究搜索策略的AI助手。你的任务是为特定查询生成更广泛、更通用的版本，以检索有用的背景信息。"

    user_prompt = f"""
    为以下查询生成一个更广泛、更通用的版本，这可以帮助检索有用的背景信息。

    原始查询: {original_query}

    步退查询:
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.1,  # 稍高温度增加一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    return response.choices[0].message.content.strip()

def decompose_query(original_query, num_subqueries=4, model="Qwen/Qwen2.5-72B-Instruct"):
    """
    将复杂查询分解为更简单的子查询

    Args:
        original_query (str): 原始复杂查询
        num_subqueries (int): 要生成的子查询数量
        model (str): 用于查询分解的模型

    Returns:
        List[str]: 更简单的子查询列表
    """
    system_prompt = "你是一个专门分解复杂问题的AI助手。你的任务是将复杂查询分解为更简单的子问题，这些子问题的答案结合起来可以解决原始查询。"

    user_prompt = f"""
    将以下复杂查询分解为{num_subqueries}个更简单的子查询。每个子查询应关注原始问题的不同方面。

    原始查询: {original_query}

    生成{num_subqueries}个子查询，每行一个，格式如下：
    1. [第一个子查询]
    2. [第二个子查询]
    以此类推...
    """

    response = client.chat.completions.create(
        model=model,
        temperature=0.2,  # 稍高温度增加一些变化
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )

    # 处理响应以提取子查询
    content = response.choices[0].message.content.strip()

    # 使用简单解析提取编号查询
    lines = content.split("\n")
    sub_queries = []

    for line in lines:
        if line.strip() and any(line.strip().startswith(f"{i}.") for i in range(1, 10)):
            # 移除编号和前导空格
            query = line.strip()
            query = query[query.find(".")+1:].strip()
            sub_queries.append(query)
    return sub_queries

def extract_text_from_pdf(pdf_path):
    """
    从PDF文件中提取文本
    """
    mypdf = fitz.open(pdf_path)
    all_text = ""

    for page_num in range(mypdf.page_count):
        page = mypdf[page_num]
        text = page.get_text("text")
        all_text += text

    return all_text

def chunk_text(text, n=1000, overlap=200):
    """
    将文本分割成重叠的块
    """
    chunks = []

    for i in range(0, len(text), n - overlap):
        chunks.append(text[i:i + n])

    return chunks

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
        self.documents = []
        self.metadata = []

    def add_documents(self, documents, vectors=None, metadata=None):
        """
        向向量存储添加文档
        """
        if vectors is None:
            vectors = [None] * len(documents)

        if metadata is None:
            metadata = [{} for _ in range(len(documents))]

        for doc, vec, meta in zip(documents, vectors, metadata):
            self.documents.append(doc)
            self.vectors.append(vec)
            self.metadata.append(meta)

    def search(self, query_vector, top_k=5):
        """
        搜索最相似的文档
        """
        if not self.vectors or not self.documents:
            return []

        query_array = np.array(query_vector)

        # 计算相似度
        similarities = []
        for i, vector in enumerate(self.vectors):
            if vector is not None:
                similarity = np.dot(query_array, vector) / (
                    np.linalg.norm(query_array) * np.linalg.norm(vector)
                )
                similarities.append((i, similarity))

        # 按相似度降序排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 获取top-k结果
        results = []
        for i, score in similarities[:top_k]:
            results.append({
                "document": self.documents[i],
                "score": float(score),
                "metadata": self.metadata[i]
            })

        return results

def process_document(pdf_path, chunk_size=1000, chunk_overlap=200):
    """
    处理文档用于RAG
    """
    print("正在从PDF提取文本...")
    text = extract_text_from_pdf(pdf_path)

    print("正在分块文本...")
    chunks = chunk_text(text, chunk_size, chunk_overlap)
    print(f"创建了 {len(chunks)} 个文本块")

    print("正在为块创建嵌入...")
    chunk_embeddings = create_embeddings(chunks)

    # 创建简单向量存储
    store = SimpleVectorStore()

    # 添加每个块及其嵌入到向量存储
    metadata = [{"chunk_index": i, "source": pdf_path} for i in range(len(chunks))]
    store.add_documents(chunks, chunk_embeddings, metadata)

    print(f"已将 {len(chunks)} 个块添加到向量存储")
    return store

def transformed_search(query, vector_store, transformation_type, top_k=3):
    """
    使用查询转换执行搜索

    Args:
        query (str): 原始查询
        vector_store: 向量存储
        transformation_type (str): 转换类型 ('rewrite', 'step_back', 'decompose')
        top_k (int): 返回的结果数

    Returns:
        dict: 包含转换后查询和搜索结果的字典
    """
    if transformation_type == "rewrite":
        transformed_query = rewrite_query(query)
        print(f"重写查询: {transformed_query}")

        # 执行搜索
        query_embedding = create_embeddings(transformed_query)
        results = vector_store.search(query_embedding, top_k)

        return {
            "transformation_type": "query_rewrite",
            "original_query": query,
            "transformed_query": transformed_query,
            "results": results
        }

    elif transformation_type == "step_back":
        step_back_query = generate_step_back_query(query)
        print(f"步退查询: {step_back_query}")

        # 为原始查询和步退查询都执行搜索
        original_embedding = create_embeddings(query)
        step_back_embedding = create_embeddings(step_back_query)

        original_results = vector_store.search(original_embedding, top_k//2 + 1)
        step_back_results = vector_store.search(step_back_embedding, top_k//2 + 1)

        # 合并结果，优先原始查询结果
        combined_results = original_results + step_back_results

        # 去重（基于文档内容）
        seen_docs = set()
        unique_results = []
        for result in combined_results:
            doc_hash = hash(result["document"][:100])  # 使用前100个字符作为唯一标识
            if doc_hash not in seen_docs:
                seen_docs.add(doc_hash)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        return {
            "transformation_type": "step_back",
            "original_query": query,
            "step_back_query": step_back_query,
            "results": unique_results
        }

    elif transformation_type == "decompose":
        sub_queries = decompose_query(query, num_subqueries=4)
        print(f"子查询:")
        for i, sq in enumerate(sub_queries, 1):
            print(f"  {i}. {sq}")

        # 为每个子查询执行搜索
        all_results = []
        for sub_query in sub_queries:
            sub_embedding = create_embeddings(sub_query)
            sub_results = vector_store.search(sub_embedding, top_k//len(sub_queries) + 1)
            all_results.extend(sub_results)

        # 去重和排序
        seen_docs = set()
        unique_results = []
        for result in sorted(all_results, key=lambda x: x["score"], reverse=True):
            doc_hash = hash(result["document"][:100])
            if doc_hash not in seen_docs:
                seen_docs.add(doc_hash)
                unique_results.append(result)
                if len(unique_results) >= top_k:
                    break

        return {
            "transformation_type": "decompose",
            "original_query": query,
            "sub_queries": sub_queries,
            "results": unique_results
        }

    else:
        # 默认：无转换
        query_embedding = create_embeddings(query)
        results = vector_store.search(query_embedding, top_k)

        return {
            "transformation_type": "none",
            "original_query": query,
            "results": results
        }

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

## 实际应用示例


# 演示查询转换方法
original_query = "AI对就业自动化和就业的影响是什么？"

print("原始查询:", original_query)
print("="*60)

# 1. 查询重写
print("\n1. 查询重写:")
rewritten_query = rewrite_query(original_query)
print(f"重写后: {rewritten_query}")

# 2. 步退提示
print("\n2. 步退提示:")
step_back_query = generate_step_back_query(original_query)
print(f"步退查询: {step_back_query}")

# 3. 子查询分解
print("\n3. 子查询分解:")
sub_queries = decompose_query(original_query, num_subqueries=4)
for i, query in enumerate(sub_queries, 1):
    print(f"   {i}. {query}")

print("\n" + "="*60)

# 处理文档并建立向量存储
pdf_path = "data/AI_Information.pdf"
vector_store = process_document(pdf_path)

# 加载测试查询
with open('data/val.json') as f:
    data = json.load(f)

test_query = data[0]['question']
print(f"\n测试查询: {test_query}")

# 比较不同转换方法的效果
transformation_types = ["none", "rewrite", "step_back", "decompose"]

results = {}
for transform_type in transformation_types:
    print(f"\n{'='*20} {transform_type.upper()} {'='*20}")

    if transform_type == "none":
        # 无转换的基准测试
        query_embedding = create_embeddings(test_query)
        search_results = vector_store.search(query_embedding, 3)
        results[transform_type] = {
            "transformation_type": "none",
            "original_query": test_query,
            "results": search_results
        }
    else:
        # 使用查询转换
        results[transform_type] = transformed_search(
            test_query,
            vector_store,
            transform_type,
            top_k=3
        )

    # 显示结果
    result_data = results[transform_type]
    print(f"检索到 {len(result_data['results'])} 个结果")

    for i, result in enumerate(result_data['results'], 1):
        print(f"\n结果 {i} (相似度: {result['score']:.4f}):")
        print(f"{result['document'][:200]}...")

# 为每种方法生成回答
print(f"\n{'='*60}")
print("生成的回答比较:")
print("="*60)

for transform_type, result_data in results.items():
    print(f"\n【{transform_type.upper()}方法】")

    # 准备上下文
    context = "\n\n".join([
        f"段落{i+1}: {result['document']}"
        for i, result in enumerate(result_data['results'])
    ])

    # 生成回答
    response = generate_response(test_query, context)
    print(f"回答: {response}")
    print("-" * 40)