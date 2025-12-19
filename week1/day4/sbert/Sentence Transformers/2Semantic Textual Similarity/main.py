# 导入SentenceTransformer类，用于加载预训练的句子变换模型
from sentence_transformers import SentenceTransformer

# 加载预训练的Sentence Transformer模型
# all-MiniLM-L6-v2是一个轻量级但性能良好的模型，适用于大多数句子相似度任务
model = SentenceTransformer("all-MiniLM-L6-v2")

# 第一个句子列表，包含需要比较的句子
sentences1 = [
    "The new movie is awesome",    # 新电影很棒
    "The cat sits outside",        # 猫坐在外面
    "A man is playing guitar",     # 一个男人在弹吉他
]

# 第二个句子列表，将与第一个列表中的句子进行相似度比较
sentences2 = [
    "The dog plays in the garden",   # 狗在花园里玩耍
    "The new movie is so great",     # 新电影太棒了
    "A woman watches TV",            # 一个女人在看电视
]

# 计算两个句子列表中所有句子的嵌入向量
# 嵌入向量是高维空间中的数值表示，能够捕获句子的语义信息
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# 计算两个句子列表之间的余弦相似度
# 余弦相似度衡量两个向量之间的夹角，值域为[-1, 1]，1表示完全相似，-1表示完全相反，0表示无关
similarities = model.similarity(embeddings1, embeddings2)

# 输出每对句子及其相似度得分
# 通过双重循环遍历所有可能的句子对
for idx_i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for idx_j, sentence2 in enumerate(sentences2):
        # 格式化输出，对齐句子并显示相似度得分（保留4位小数）
        print(f" - {sentence2: <30}: {similarities[idx_i][idx_j]:.4f}")

# --------------------------------------------------------
# 相似度计算
# 导入SentenceTransformer和SimilarityFunction类
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 加载模型并指定相似度计算函数为点积
# DOT_PRODUCT点积计算通常用于衡量向量的大小和方向关系
model = SentenceTransformer("all-MiniLM-L6-v2", similarity_fn_name=SimilarityFunction.DOT_PRODUCT)

# 再次导入必要的类（可能是为了强调依赖关系）
from sentence_transformers import SentenceTransformer, SimilarityFunction

# 加载预训练的Sentence Transformer模型
model = SentenceTransformer("all-MiniLM-L6-v2")

# 定义需要嵌入编码的句子列表
sentences = [
    "The weather is lovely today.",   # 今天天气很好
    "It's so sunny outside!",         # 外面阳光明媚
    "He drove to the stadium.",       # 他开车去了体育馆
]

# 计算句子的嵌入向量表示
embeddings = model.encode(sentences)

# 使用默认的余弦相似度计算函数计算句子间的相似度
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# 输出结果为一个3x3的相似度矩阵：
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])

# 将相似度计算函数更改为曼哈顿距离
# 曼哈顿距离是各维度差值绝对值之和，适合某些特定的应用场景
model.similarity_fn_name = SimilarityFunction.MANHATTAN
print(model.similarity_fn_name)
# => "manhattan"

# 使用曼哈顿距离重新计算相似度
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# 使用曼哈顿距离计算得到的相似度矩阵（注意：这里的值是负数，因为距离越小越相似）：
# tensor([[ -0.0000, -12.6269, -20.2167],
#         [-12.6269,  -0.0000, -20.1288],
#         [-20.2167, -20.1288,  -0.0000]])