# Base目录内容总结

Base目录包含了三套不同的RAG（Retrieval-Augmented Generation）系统实现，每套系统代表了不同的文本分块策略和技术复杂度。

## 1. 简单RAG系统（simple-rag）

这是最基本的RAG实现，包含了RAG系统的核心组件：

### 核心功能模块：

1. **文档提取** - 从PDF文件中提取文本内容
2. **文本分块** - 将长文本按固定大小分割成块，支持重叠
3. **向量嵌入** - 使用HuggingFace模型为文本创建向量表示
4. **语义搜索** - 基于余弦相似度进行文档检索
5. **响应生成** - 使用大语言模型基于检索内容生成回答

### 关键代码示例：

```python
# 文本分块核心逻辑
def chunk_text(text, n, overlap):
    chunks = []
    step = n - overlap
    for i in range(0, len(text), step):
        chunk = text[i:i + n]
        chunks.append(chunk)
    return chunks

# 余弦相似度计算
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
```

这套系统适合初学者了解RAG的基本工作原理。

## 2. 语义分块系统（Semantic-chunking）

这套系统实现了更智能的文本分块方法，基于句子间的语义相似度来确定分块边界，能更好地保持文本的语义完整性。

### 核心功能模块：

1. **句子分割** - 将文本按句子边界分割
2. **句子嵌入** - 为每个句子创建向量表示
3. **相似度计算** - 计算相邻句子间的语义相似度
4. **边界检测** - 基于相似度阈值确定分块边界
5. **语义分块** - 根据边界将句子重组为语义完整的块

### 关键代码示例：

```python
# 基于相似度阈值确定分块边界
def find_chunk_boundaries(similarities, threshold=0.5):
    boundaries = [0]
    for i, similarity in enumerate(similarities):
        if similarity < threshold:
            boundaries.append(i + 1)
    boundaries.append(len(similarities) + 1)
    return boundaries
```

### 增强功能（Upgrade目录）：

1. **动态阈值计算** - 基于相似度分布动态计算分块阈值
2. **分块大小控制** - 在语义分块基础上控制块大小范围
3. **分块质量评估** - 评估不同分块策略的效果

这套系统适用于需要更好语义保持的场景。

## 3. 自适应分块选择系统（select_chunk）

这是最高级的实现，能够根据文档和查询的特征自动选择最优的分块策略。

### 核心功能模块：

1. **文档特征分析** - 分析文档长度、句子长度、段落结构等
2. **查询特征分析** - 分析查询的复杂度和类型
3. **分块大小推荐** - 基于文档和查询特征推荐最优分块大小
4. **性能比较** - 比较不同分块大小的检索效果
5. **自适应RAG** - 整合上述功能的完整RAG流程

### 关键代码示例：

```python
# 推荐分块大小的核心逻辑
def recommend_chunk_size(doc_characteristics, query_characteristics):
    base_chunk_size = 1000
    # 根据文档特征调整
    if doc_characteristics['avg_paragraph_length'] > 500:
        doc_adjustment = 1.3
    elif doc_characteristics['avg_paragraph_length'] < 200:
        doc_adjustment = 0.7
    else:
        doc_adjustment = 1.0
    # ... 其他调整逻辑
    final_chunk_size = int(base_chunk_size * doc_adjustment * ...)
    return final_chunk_size, int(final_chunk_size * 0.2)
```

这套系统适用于对性能有较高要求的生产环境。

## 总结

Base目录展示了RAG系统中文本分块技术的演进历程：

1. **简单固定分块** → **语义智能分块** → **自适应分块选择**
2. **固定参数** → **动态计算** → **智能推荐**
3. **通用处理** → **语义保持** → **个性化优化**

这三种实现方式为不同需求和复杂度的项目提供了合适的解决方案。