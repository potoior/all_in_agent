"""
嵌入模型调用模块 - Embedding Model Call Module

这个模块提供了一个统一的接口来获取不同类型的嵌入模型实例。
支持OpenAI、智谱AI和M3E等多种嵌入模型。

This module provides a unified interface to get different types of embedding model instances.
Supports multiple embedding models including OpenAI, Zhipu AI, and M3E.
"""

import os
import sys

# 添加父目录到系统路径 - Add parent directory to system path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 导入嵌入模型类 - Import embedding model classes
from embedding.zhipuai_embedding import ZhipuAIEmbeddings
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.embeddings.openai import OpenAIEmbeddings
from llm.call_llm import parse_llm_api_key

def get_embedding(embedding: str, embedding_key: str=None, env_file: str=None):
    """
    获取指定类型的嵌入模型实例
    Get embedding model instance of specified type
    
    参数 / Parameters:
        embedding: 嵌入模型类型 ('m3e', 'openai', 'zhipuai') / Embedding model type
        embedding_key: API密钥（可选）/ API key (optional)
        env_file: 环境变量文件路径（可选）/ Environment variable file path (optional)
        
    返回 / Returns:
        嵌入模型实例 / Embedding model instance
        
    异常 / Exceptions:
        ValueError: 当指定的嵌入模型不支持时 / When specified embedding model is not supported
    """
    
    # M3E模型使用HuggingFace本地模型 - M3E model uses HuggingFace local model
    if embedding == 'm3e':
        return HuggingFaceEmbeddings(model_name="moka-ai/m3e-base")
    
    # 如果没有提供API密钥，尝试从环境变量获取 - If no API key provided, try to get from environment variables
    if embedding_key == None:
        embedding_key = parse_llm_api_key(embedding)
    
    # 根据模型类型返回对应的嵌入模型实例 - Return corresponding embedding model instance based on model type
    if embedding == "openai":
        # OpenAI嵌入模型 - OpenAI embedding model
        return OpenAIEmbeddings(openai_api_key=embedding_key)
    elif embedding == "zhipuai":
        # 智谱AI嵌入模型 - Zhipu AI embedding model
        return ZhipuAIEmbeddings(zhipuai_api_key=embedding_key)
    else:
        # 不支持的嵌入模型类型 - Unsupported embedding model type
        raise ValueError(f"embedding {embedding} not support ")
