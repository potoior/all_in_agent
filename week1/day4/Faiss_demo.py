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
# 第一列+（0.0, 0.001, 0.002, ..., 99.999）
print("修改第一列数据之前")
print(xb[:5, 0])
xb[:, 0] += np.arange(nb) / 1000.
print("修改第一列数据之后")
print(xb[:5, 0])

# 生成随机的查询向量，形状为(nq, d)，数据类型为float32
xq = np.random.random((nq, d)).astype('float32')

# 修改第一列的数据，使其具有一定的规律性
# 同上
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


# 定义聚类中心数量
nlist = 100

# 定义要返回的最近邻数量
k = 4

# 创建量化器索引，用于对向量进行聚类
# 这里使用IndexFlatL2作为量化器，它会对向量进行精确的L2距离计算
quantizer = faiss.IndexFlatL2(d)  # the other index

# 创建IVF（Inverted File）索引
# IndexIVFFlat是一种基于倒排文件的索引结构，它将向量空间划分成nlist个聚类
# quantizer: 用于聚类的量化器
# d: 向量维度
# nlist: 聚类中心的数量
index = faiss.IndexIVFFlat(quantizer, d, nlist)

# 检查索引是否已训练，此时应该返回False，因为还未进行训练
assert not index.is_trained

# 对数据库向量进行训练，学习聚类中心
# 训练过程会使用quantizer对xb中的向量进行聚类，得到nlist个聚类中心
index.train(xb)

# 检查索引是否已完成训练，此时应该返回True
assert index.is_trained

# 将数据库向量添加到已训练好的索引中
# 由于需要为每个向量分配到最近的聚类中心，所以添加过程可能比IndexFlatL2稍慢
index.add(xb)                  # add may be a bit slower as well

# 使用默认参数进行实际搜索（nprobe=1，即只搜索最近的1个聚类）
# 在IVF索引中，搜索时只会检查查询向量最接近的nprobe个聚类中的向量
D, I = index.search(xq, k)     # actual search

# 打印后5个查询向量的最近邻索引
print("使用默认nprobe=1时，后5个查询向量的最近邻索引:")
print(I[-5:])                  # neighbors of the 5 last queries

# 修改nprobe参数为10，表示搜索时检查最近的10个聚类
# 默认nprobe为1，增加nprobe可以提高搜索精度，但也会增加搜索时间
index.nprobe = 10              # default nprobe is 1, try a few more

# 使用修改后的参数进行搜索
D, I = index.search(xq, k)

# 打印后5个查询向量的最近邻索引（使用nprobe=10）
print("使用nprobe=10时，后5个查询向量的最近邻索引:")
print(I[-5:])                  # neighbors of the 5 last queries

nlist = 100
m = 8                             # number of subquantizers
k = 4
quantizer = faiss.IndexFlatL2(d)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, 8)
                                    # 8 specifies that each sub-vector is encoded as 8 bits
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], k) # sanity check
print(I)
print(D)
index.nprobe = 10              # make comparable with experiment above
D, I = index.search(xq, k)     # search
print(I[-5:])