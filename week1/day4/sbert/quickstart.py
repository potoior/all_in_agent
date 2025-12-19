# Import Sentence Transformers library for sentence embedding and semantic similarity calculation
from sentence_transformers import SentenceTransformer

# ========================
# Sentence Transformers
# ========================

# 1. Load a pretrained Sentence Transformer model
# all-MiniLM-L6-v2 is a lightweight but high-performance model with 384-dimensional embedding vectors
model = SentenceTransformer("all-MiniLM-L6-v2")

# Define the list of sentences to encode
sentences = [
    "The weather is lovely today.",      # 今天天气很好
    "It's so sunny outside!",            # 外面阳光明媚！
    "He drove to the stadium.",          # 他开车去了体育场
]

# 2. Calculate sentence embedding vectors by calling model.encode()
# Embedding vectors are numerical representations in high-dimensional space that can capture semantic information of sentences
embeddings = model.encode(sentences)

# Print the shape of embedding vectors: [number of sentences, embedding dimension]
# This should be [3, 384], representing 3 sentences, each represented by a 384-dimensional vector
print("Embedding vector shape:", embeddings.shape)
# [3, 384]

# 3. Calculate similarity between embedding vectors
# The similarity method calculates cosine similarity with a value range of [-1, 1], where 1 means completely similar, -1 means completely opposite, and 0 means unrelated
similarities = model.similarity(embeddings, embeddings)
print("Sentence similarity matrix:")
print(similarities)
# Output explanation:
# tensor([[1.0000, 0.6660, 0.1046],  # Similarity of sentence 1 with all sentences
#         [0.6660, 1.0000, 0.1411],  # Similarity of sentence 2 with all sentences
#         [0.1046, 0.1411, 1.0000]]) # Similarity of sentence 3 with all sentences
# The diagonal is 1 because each sentence is completely similar to itself
# Sentences 1 and 2 both describe weather, so they have high similarity (0.6660)
# Sentence 3 describes going to a stadium, which is less relevant to the other two sentences

# ========================
# Cross Encoder
# ========================

# Import the CrossEncoder class from the sentence_transformers.cross_encoder module
# CrossEncoder directly compares two sentences and is usually more accurate but slower than SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder

# 1. Load a pretrained CrossEncoder model
# cross-encoder/stsb-distilroberta-base is a model trained on semantic textual similarity tasks
model = CrossEncoder("cross-encoder/stsb-distilroberta-base")

# Define query sentence
query = "A man is eating pasta."  # 一个男人正在吃意大利面

# Define corpus sentences (set of sentences to compare against)
corpus = [
    "A man is eating food.",                            # 一个男人正在吃食物
    "A man is eating a piece of bread.",                # 一个男人正在吃一片面包
    "The girl is carrying a baby.",                     # 一个女孩抱着婴儿
    "A man is riding a horse.",                         # 一个男人正在骑马
    "A woman is playing violin.",                       # 一个女人正在拉小提琴
    "Two men pushed carts through the woods.",          # 两个男人推着车穿过树林
    "A man is riding a white horse on an enclosed ground.", # 一个男人在一个封闭的地面上骑白马
    "A monkey is playing drums.",                       # 一只猴子在打鼓
    "A cheetah is running behind its prey.",            # 一只猎豹在追捕猎物
]

# 2. Rank all sentences in the corpus by relevance to the query sentence
# The rank method returns a list sorted by relevance score in descending order, containing corpus_id and score
ranks = model.rank(query, corpus)

# Print the query sentence and ranking results
print("Query sentence: ", query)
print("Results ranked by relevance:")
for rank in ranks:
    # rank['score']: similarity score, higher means more relevant
    # rank['corpus_id']: index in the corpus
    # corpus[rank['corpus_id']]: corresponding sentence
    print(f"Relevance score {rank['score']:.2f}\t{corpus[rank['corpus_id']]}")

# Output explanation:
"""
Query:  A man is eating pasta.
0.67    A man is eating food.           # Most relevant, both describe a man eating
0.34    A man is eating a piece of bread. # Relevant, both specific descriptions of a man eating
0.08    A man is riding a horse.        # Weakly relevant, both involve men but different actions
0.07    A man is riding a white horse on an enclosed ground. # Weakly relevant
0.01    The girl is carrying a baby.    # Almost irrelevant
0.01    Two men pushed carts through the woods. # Almost irrelevant
0.01    A monkey is playing drums.      # Almost irrelevant
0.01    A woman is playing violin.      # Almost irrelevant
0.01    A cheetah is running behind its prey. # Almost irrelevant
"""

# 3. Alternative approach: manually calculate scores between two sentences
import numpy as np

# Create a list of sentence combinations, each item is [query sentence, sentence from corpus]
sentence_combinations = [[query, sentence] for sentence in corpus]

# Use the predict method to calculate similarity scores for all sentence pairs
scores = model.predict(sentence_combinations)

# Use argsort to sort scores in descending order and get corresponding corpus indices
# argsort returns ascending indices, [::-1] reverses them to descending order
ranked_indices = np.argsort(scores)[::-1]

print("Similarity scores for all sentence pairs:", scores)
print("Corpus indices sorted by descending scores:", ranked_indices)

# Output explanation:
"""
Scores: [0.6732372, 0.34102544, 0.00542465, 0.07569341, 0.00525378, 0.00536814, 0.06676237, 0.00534825, 0.00516717]
Indices: [0 1 3 6 2 5 7 4 8]
"""

# ========================
# Sparse Encoder
# ========================

# Import the SparseEncoder class from sentence_transformers
# SparseEncoder generates sparse vector representations where most elements are 0 with only a few non-zero elements
from sentence_transformers import SparseEncoder

# 1. Load a pretrained SparseEncoder model
# naver/splade-cocondenser-ensembledistil is a sparse encoding model based on SPLADE
model = SparseEncoder("naver/splade-cocondenser-ensembledistil")

# Define sentences to encode
sentences = [
    "The weather is lovely today.",      # 今天天气很好
    "It's so sunny outside!",            # 外面阳光明媚！
    "He drove to the stadium.",          # 他开车去了体育场
]

# 2. Calculate sparse embedding vectors by calling model.encode()
# The dimension of sparse embedding vectors usually equals the vocabulary size (here it's 30522)
embeddings = model.encode(sentences)

# Print the shape of embedding vectors: [number of sentences, vocabulary size]
# This should be [3, 30522], representing 3 sentences, each represented by a 30522-dimensional sparse vector
print("Sparse embedding vector shape:", embeddings.shape)
# [3, 30522] - sparse representation, dimension equals vocabulary size

# 3. Calculate similarity between embedding vectors (dot product by default)
# Since these are sparse vectors, dot product is typically used instead of cosine similarity
similarities = model.similarity(embeddings, embeddings)
print("Similarity matrix of sparse vectors:")
print(similarities)
# Output explanation:
# tensor([[   35.629,     9.154,     0.098],  # Similarity of sentence 1 with all sentences
#         [    9.154,    27.478,     0.019],  # Similarity of sentence 2 with all sentences
#         [    0.098,     0.019,    29.553]]) # Similarity of sentence 3 with all sentences

# 4. Check sparsity statistics
# Calculate sparsity metrics of embedding vectors
stats = SparseEncoder.sparsity(embeddings)
print(f"Sparsity rate: {stats['sparsity_ratio']:.2%}")  # Usually >99% of elements are 0
print(f"Average number of non-zero dimensions per embedding vector: {stats['active_dims']:.2f}")