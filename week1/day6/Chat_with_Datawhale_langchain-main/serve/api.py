#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   spark_api.py
@Time    :   2023/09/24 11:00:46
@Author  :   Logan Zou 
@Version :   1.0
@Contact :   loganzou0421@163.com
@License :   (C)Copyright 2017-2018, Liugroup-NLPR-CASIA
@Desc    :   启动服务为本地 API
@Desc_EN:   Start service as local API
'''

"""
FastAPI服务模块 - FastAPI Service Module

这个模块提供了一个基于FastAPI的RESTful API接口，用于知识库问答功能。
支持多种大语言模型和配置参数。

This module provides a RESTful API interface based on FastAPI for knowledge base Q&A functionality.
Supports multiple large language models and configuration parameters.
"""

from fastapi import FastAPI
from pydantic import BaseModel
import os
import sys
# 导入功能模块目录 - Import functional module directory
sys.path.append("../")
from qa_chain.QA_chain_self import QA_chain_self

# os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
# os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"

app = FastAPI() # 创建 api 对象 / Create API object

# 默认提示模板 - Default prompt template
template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说"谢谢你的提问！"。
{context}
问题: {question}
有用的回答:"""

# 定义一个数据模型，用于接收POST请求中的数据
# Define a data model for receiving data in POST requests
class Item(BaseModel):
    """
    API请求数据模型
    API Request Data Model
    
    用于定义FastAPI接口接收的参数结构和类型验证。
    Defines parameter structure and type validation for FastAPI interface.
    """
    prompt : str                                          # 用户问题 / User question
    model : str = "gpt-3.5-turbo"                        # 使用的模型 / Model to use
    temperature : float = 0.1                              # 温度系数 / Temperature coefficient
    if_history : bool = False                              # 是否使用历史对话功能 / Whether to use conversation history
    api_key: str = None                                    # API密钥 / API key
    secret_key : str = None                                # 密钥（百度文心）/ Secret key (Baidu Wenxin)
    access_token: str = None                               # 访问令牌 / Access token
    appid : str = None                                     # 应用ID（讯飞星火）/ App ID (iFlytek Spark)
    Spark_api_secret : str = None                          # 讯飞星火密钥 / iFlytek Spark secret key
    Wenxin_secret_key : str = None                         # 百度文心密钥 / Baidu Wenxin secret key
    db_path : str = "/Users/lta/Desktop/llm-universe/data_base/vector_db/chroma"  # 数据库路径 / Database path
    file_path : str = "/Users/lta/Desktop/llm-universe/data_base/knowledge_db"    # 源文件路径 / Source file path
    prompt_template : str = template                     # 提示模板 / Prompt template
    input_variables : list = ["context","question"]        # 模板变量 / Template variables
    embedding : str = "m3e"                                # 嵌入模型 / Embedding model
    top_k : int = 5                                        # 返回前k个结果 / Return top k results
    embedding_key : str = None                             # 嵌入模型密钥 / Embedding model key

@app.post("/")
async def get_response(item: Item):
    """
    处理问答请求的API端点
    API endpoint for processing Q&A requests
    
    参数 / Parameters:
        item: Item数据模型实例，包含所有请求参数 / Item data model instance containing all request parameters
        
    返回 / Returns:
        str: 生成的答案 / Generated answer
        
    功能 / Function:
    - 根据if_history参数决定是否使用历史对话功能
    - Decide whether to use conversation history based on if_history parameter
    - 创建问答链实例并生成答案
    - Create Q&A chain instance and generate answer
    - 目前API不支持历史链功能
    - Currently API does not support history chain functionality
    """

    # 首先确定需要调用的链 / First determine which chain to call
    if not item.if_history:
        # 调用不带历史记录的问答链 / Call Q&A chain without history
        # return item.embedding_key
        if item.embedding_key == None:
            item.embedding_key = item.api_key
        
        # 创建问答链实例 / Create Q&A chain instance
        chain = QA_chain_self(model=item.model, temperature=item.temperature, top_k=item.top_k, file_path=item.file_path, persist_path=item.db_path, 
                                appid=item.appid, api_key=item.api_key, embedding=item.embedding, template=template, Spark_api_secret=item.Spark_api_secret, Wenxin_secret_key=item.Wenxin_secret_key, embedding_key=item.embedding_key)

        # 生成答案 / Generate answer
        response = chain.answer(question = item.prompt)
    
        return response
    
    # 由于 API 存在即时性问题，不能支持历史链
    # Due to real-time issues with API, history chain is not supported
    else:
        return "API 不支持历史链"