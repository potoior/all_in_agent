# Advanced目录代码详解

Advanced目录包含了三个高级RAG技术的实现，分别是上下文增强、上下文分块头部和文档增强。这些技术旨在提升RAG系统的检索和回答质量。

## 04 - 上下文增强 (Context Enrichment)

### 概述

上下文增强技术通过在检索到最相关的文本块后，将其相邻的文本块也纳入考虑范围，从而提供更丰富的上下文信息给大语言模型，以生成更准确、完整的回答。

### 核心思想

传统的RAG系统只使用最相关的单个文本块作为上下文，可能会丢失重要的背景信息。上下文增强技术通过扩展上下文，包含相邻的文本块，解决了这一问题。

### 关键函数详解

#### `context_enriched_search` 函数

这是上下文增强技术的核心实现：

```python
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
```

### 工作流程

1. 执行标准的语义搜索，找出与查询最相关的文本块
2. 获取最相关块的索引位置
3. 根据`context_size`参数确定需要包含的上下文范围
4. 返回包括最相关块及其相邻块在内的文本块列表

### 技术优势

1. **提供更多上下文**：相比传统方法只返回单个文本块，上下文增强技术能提供更丰富的背景信息
2. **保持连贯性**：相邻的文本块往往具有语义连贯性，有助于模型理解完整含义
3. **易于实现**：在现有RAG系统上很容易添加此功能

### 实际应用示例

```python
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
```

### 适用场景

1. 当文档内容具有较强连贯性时（如技术文档、论文等）
2. 当查询涉及需要上下文才能准确回答的问题时
3. 当希望提高回答完整性和准确性时

## 05 - 上下文分块头部 (Context Chunk Headers)

### 概述

上下文分块头部技术为每个文本块生成描述性标题，并同时对文本内容和标题创建嵌入向量。在检索过程中，同时考虑查询与文本内容和标题的相似度，从而提高检索准确性。

### 核心思想

传统的文本块只有内容信息，缺乏对其主题的概括。通过为每个文本块生成标题，可以提供额外的元信息，帮助系统更准确地判断相关性。

### 关键函数详解

#### `generate_chunk_header` 函数

该函数使用大语言模型为文本块生成标题：

```python
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
```

#### `chunk_text_with_headers` 函数

该函数将文本分块并为每个块生成头部：

```python
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
```

#### `semantic_search` 函数

该函数在检索时同时考虑文本内容和标题的相似度：

```python
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
```

### 技术亮点

1. **双重嵌入策略**：
   ```python
   # 为每个块创建两个嵌入向量
   text_embedding = create_embeddings(chunk["text"])        # 内容嵌入
   header_embedding = create_embeddings(chunk["header"])    # 头部嵌入
   
   # 在检索时计算平均相似度
   sim_text = cosine_similarity(query_embedding, text_embedding)
   sim_header = cosine_similarity(query_embedding, header_embedding)
   avg_similarity = (sim_text + sim_header) / 2  # 平均分配权重
   ```

2. **相似度融合方法**：
   - 简单平均（推荐）：`avg_similarity = (sim_text + sim_header) / 2`
   - 加权平均：可以根据需要调整文本内容和标题的权重
   - 最大值：`max_similarity = max(sim_text, sim_header)`
   - 动态权重：根据查询类型调整权重

### 工作流程

1. 对文档进行分块处理
2. 为每个文本块生成描述性标题
3. 分别为文本内容和标题创建嵌入向量
4. 在检索时同时计算查询与内容和标题的相似度
5. 使用融合策略得到最终相似度并排序

### 技术优势

1. **提升检索准确性**：标题提供了额外的语义信息，有助于更精确地判断相关性
2. **增强语义理解**：标题概括了文本块的主题，有助于系统快速识别相关内容
3. **灵活性**：可以通过调整融合策略来适应不同场景的需求

### 适用场景

1. 当文档内容较为复杂，需要更好的语义理解时
2. 当希望通过标题快速筛选相关内容时
3. 当需要提高检索精度和准确率时

## 06 - 文档增强 (Document Augmentation)

### 概述

文档增强技术通过对文档内容提出问题来增强文档表示。不仅存储原始文本块的嵌入，还存储基于这些块生成的问题的嵌入，从而提供更丰富的检索途径。

### 核心思想

传统的RAG系统只基于原始文档内容进行检索，而文档增强技术通过自动生成与文档内容相关的问题，扩展了检索的可能性。用户查询可能与原始内容不完全匹配，但可能与生成的问题匹配，从而提高召回率。

### 关键函数详解

#### `generate_questions` 函数

该函数为给定的文本块生成相关问题：

```python
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
```

#### `process_document` 函数

该函数处理文档并进行问题增强：

```python
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
```

#### `prepare_context` 函数

该函数准备用于生成回答的上下文：

```python
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
```

### 技术亮点

1. **问题生成机制**：
   - 利用大语言模型为每个文本块自动生成相关问题
   - 生成的问题只能通过对应文本块回答，确保相关性

2. **双重存储策略**：
   ```python
   # 存储原始文本块
   vector_store.add_item(
       text=chunk,
       embedding=chunk_embedding,
       metadata={
           "type": "original_chunk",
           "chunk_index": i,
           "source": pdf_path,
           "questions": questions  # 还存储了生成的问题
       }
   )

   # 存储生成的问题
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
               "original_chunk": chunk  # 指向原始块
           }
       )
   ```

3. **智能上下文准备**：
   - 无论检索到的是原始块还是生成的问题，都会返回对应的原始文本块作为上下文

### 工作流程

1. 对文档进行分块处理
2. 为每个文本块自动生成若干相关问题
3. 为原始文本块和生成的问题分别创建嵌入向量
4. 将原始块和问题都存储在向量数据库中
5. 检索时，查询可以匹配原始内容或相关问题
6. 准备上下文时，始终返回原始文本块

### 技术优势

1. **提高召回率**：通过问题生成扩展了检索的可能性
2. **增强语义覆盖**：生成的问题可以覆盖用户可能的查询方式
3. **保持回答准确性**：始终使用原始文本块作为回答上下文

### 实际应用示例

```python
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
```

### 适用场景

1. 当希望提高系统召回率时
2. 当用户查询可能以不同方式表达相同意图时
3. 当文档内容专业性强，需要多角度理解时

## 总结

Advanced目录中的三个技术代表了RAG系统优化的三个方向：

1. **上下文增强** - 通过扩展检索结果的上下文来提升回答质量
2. **上下文分块头部** - 通过为文本块添加元信息来提升检索准确性
3. **文档增强** - 通过生成问题来扩展检索可能性，提高召回率

这些技术可以单独使用，也可以组合使用，以满足不同应用场景的需求。它们共同的目标是让RAG系统更加智能、准确和全面。