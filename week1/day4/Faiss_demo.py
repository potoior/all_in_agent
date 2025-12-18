import numpy as np

# 定义向量的维度
d = 64                           # dimension

# 定义数据库中向量的数量
nb = 100000                      # database size

# 定义查询向量的数量
nq = 10000                       # nb of queries

# 设置随机种子以确保实验的可重复性
np.random.seed(1234)             # make reproducible

# 生成随机的数据库向量，形状为(nb, d)，数据类型为float32
xb = np.random.random((nb, d)).astype('float32')

# 修改第一列的数据，使其具有一定的规律性，便于后续验证结果
xb[:, 0] += np.arange(nb) / 1000.

# 生成随机的查询向量，形状为(nq, d)，数据类型为float32
xq = np.random.random((nq, d)).astype('float32')

# 修改第一列的数据，使其具有一定的规律性
xq[:, 0] += np.arange(nq) / 1000.

# 导入faiss库
import faiss                   # make faiss available

# 创建一个基于L2距离（欧氏距离）的平面索引
# L2距离计算公式: distance = sqrt(sum((x_i - y_i)^2))
index = faiss.IndexFlatL2(d)   # build the index

# 检查索引是否已经训练完成
# 对于IndexFlatL2来说，由于是精确匹配，不需要训练，所以始终返回True
print("索引是否已训练:", index.is_trained)

# 将数据库向量添加到索引中，以便后续进行搜索
index.add(xb)                  # add vectors to the index

# 打印索引中包含的向量总数
print("索引中的向量总数:", index.ntotal)

# 定义要返回的最近邻数量
k = 4                          # we want to see 4 nearest neighbors

# 对数据库中的前5个向量进行搜索，作为验证测试
# search方法返回两个数组：
# D: 每个查询向量与其最近邻之间的距离
# I: 每个查询向量的最近邻在数据库中的索引位置
D, I = index.search(xb[:5], k) # sanity check

# 打印前5个向量的最近邻索引
print("前5个向量的最近邻索引:")
print(I)

# 打印前5个向量与其最近邻之间的距离
print("前5个向量与其最近邻的距离:")
print(D)

# 对所有的查询向量进行实际搜索
D, I = index.search(xq, k)     # actual search

# 打印前5个查询向量的最近邻索引
print("前5个查询向量的最近邻索引:")
print(I[:5])                   # neighbors of the 5 first queries

# 打印后5个查询向量的最近邻索引
print("后5个查询向量的最近邻索引:")
print(I[-5:])                  # neighbors of the 5 last queries