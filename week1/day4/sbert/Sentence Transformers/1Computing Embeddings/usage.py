# 计算句子嵌入向量
# 导入SentenceTransformer类，用于加载预训练的句子变换模型
from sentence_transformers import SentenceTransformer

# 1. 加载预训练的Sentence Transformer模型
# all-MiniLM-L6-v2是一个轻量级但性能良好的预训练模型，适用于大多数句子相似度任务
model = SentenceTransformer("all-MiniLM-L6-v2")

# 需要编码的句子列表，这些句子将被转换为向量表示
sentences = [
    "The weather is lovely today.",   # 今天天气很好
    "It's so sunny outside!",         # 外面阳光明媚
    "He drove to the stadium.",       # 他开车去了体育馆
]

# 2. 通过调用model.encode()方法计算句子的嵌入向量
# 嵌入向量是高维空间中的数值表示，能够捕获句子的语义信息
embeddings = model.encode(sentences)
print(embeddings.shape)
# 输出形状为[3, 384]，表示3个句子，每个句子被表示为384维的向量

# 3. 计算嵌入向量之间的相似度
# 使用余弦相似度来衡量句子之间的语义相似性，值域为[-1, 1]，1表示完全相似
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# 输出结果为一个3x3的相似度矩阵：
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])


# 初始化Sentence Transformer模型
# 导入SentenceTransformer类
from sentence_transformers import SentenceTransformer

# 加载另一个预训练模型all-mpnet-base-v2，它通常提供更高的准确性但计算成本更高
model = SentenceTransformer("all-mpnet-base-v2")
# 或者，也可以传递本地模型目录的路径来加载本地保存的模型:





# 提示词模板
# 使用支持多种语言的模型intfloat/multilingual-e5-large
model = SentenceTransformer(
    "intfloat/multilingual-e5-large",
    # 定义不同类型任务的提示词模板
    prompts={
        "classification": "Classify the following text: ",     # 文本分类任务的提示词
        "retrieval": "Retrieve semantically similar text: ",  # 语义检索任务的提示词
        "clustering": "Identify the topic or theme based on the text: ",  # 聚类任务的提示词
    },
    default_prompt_name="retrieval",  # 设置默认使用的提示词类型
)
# 或者单独设置默认提示词类型
model.default_prompt_name="retrieval"



# 输入序列长度控制
# 导入SentenceTransformer类
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer("all-MiniLM-L6-v2")
# 查看模型的最大序列长度（默认为256）
print("Max Sequence Length:", model.max_seq_length)
# => Max Sequence Length: 256

# 修改模型的最大序列长度为200
model.max_seq_length = 200

# 再次查看修改后的最大序列长度
print("Max Sequence Length:", model.max_seq_length)
# => Max Sequence Length: 200

# 多进程/多GPU编码
# 导入SentenceTransformer类
from sentence_transformers import SentenceTransformer

def main():
    # 加载模型
    inputs = input("Press Enter to continue...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    # 启动一个多进程池，可以在多个GPU上并行处理数据
    # target_devices指定要使用的设备，这里使用两个GPU: cuda:0 和 cuda:1
    pool = model.start_multi_process_pool(target_devices=["cuda:0", "cuda:1"])
    # 使用多个GPU进行编码，提高处理大量数据的效率
    embeddings = model.encode(inputs, pool=pool)
    # 使用完毕后停止进程池，释放资源
    model.stop_multi_process_pool(pool)

if __name__ == "__main__":
    main()