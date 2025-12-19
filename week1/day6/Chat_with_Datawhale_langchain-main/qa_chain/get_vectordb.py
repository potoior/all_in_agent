"""
向量数据库获取模块
Vector Database Retrieval Module

这个模块提供了一个统一的接口来获取向量数据库实例，支持多种嵌入模型。
负责向量数据库的创建、加载和管理工作。

This module provides a unified interface to get vector database instances, supporting multiple embedding models.
Responsible for vector database creation, loading, and management.
"""

import sys 
# sys.path.append("../embedding") 
# sys.path.append("../database") 

from langchain.embeddings.openai import OpenAIEmbeddings    # 调用 OpenAI 的 Embeddings 模型 / Call OpenAI Embeddings model
import os
from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from database.create_db import create_db,load_knowledge_db
from embedding.call_embedding import get_embedding

def get_vectordb(file_path:str=None, persist_path:str=None, embedding = "openai",embedding_key:str=None):
    """
    获取向量数据库对象
    Get vector database object
    
    功能 / Function:
    - 根据文件路径和持久化路径获取向量数据库
    - Get vector database based on file path and persistence path
    - 如果持久化目录存在且非空，则加载现有向量数据库
    - If persistence directory exists and is not empty, load existing vector database
    - 如果目录不存在或为空，则创建新的向量数据库
    - If directory doesn't exist or is empty, create new vector database
    
    参数 / Parameters:
        file_path: 知识库文件路径 / Knowledge base file path
        persist_path: 向量数据库持久化路径 / Vector database persistence path
        embedding: 嵌入模型类型 / Embedding model type
        embedding_key: 嵌入模型API密钥 / Embedding model API key
        
    返回 / Returns:
        vectordb: 向量数据库对象 / Vector database object
    """
    embedding = get_embedding(embedding=embedding, embedding_key=embedding_key)
    if os.path.exists(persist_path):  #持久化目录存在
        contents = os.listdir(persist_path)
        if len(contents) == 0:  #但是下面为空
            #print("目录为空")
            vectordb = create_db(file_path, persist_path, embedding)
            #presit_knowledge_db(vectordb)
            vectordb = load_knowledge_db(persist_path, embedding)
        else:
            #print("目录不为空")
            vectordb = load_knowledge_db(persist_path, embedding)
    else: #目录不存在，从头开始创建向量数据库
        vectordb = create_db(file_path, persist_path, embedding)
        #presit_knowledge_db(vectordb)
        vectordb = load_knowledge_db(persist_path, embedding)

    return vectordb