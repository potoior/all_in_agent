"""
带历史记录的问答链模块
Q&A Chain with Conversation History Module

这个模块实现了支持对话历史记录的检索增强生成(RAG)问答链。
支持多种大语言模型和向量数据库，提供上下文感知的问答功能。

This module implements a Retrieval-Augmented Generation (RAG) Q&A chain 
with conversation history support. Supports multiple large language models 
and vector databases, providing context-aware Q&A functionality.
"""

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
import sys
import os

# 添加项目根目录到系统路径
# Add project root directory to system path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import re

class Chat_QA_chain_self:
    """
    带历史记录的问答链类
    Q&A Chain with Conversation History Class
    
    功能 / Features:
    - 支持对话历史记录的RAG问答链
    - RAG Q&A chain with conversation history support
    - 集成多种LLM API和向量数据库
    - Integration of multiple LLM APIs and vector databases
    - 提供上下文感知的问答功能
    - Provides context-aware Q&A functionality
    
    参数 / Parameters:
        model: 调用的模型名称 / Model name to use
        temperature: 温度系数，控制生成的随机性 / Temperature coefficient controlling generation randomness
        top_k: 返回检索的前k个相似文档 / Return top k most similar documents
        chat_history: 历史记录，输入一个列表，默认是一个空列表 / Chat history as a list, empty by default
        history_len: 控制保留的最近 history_len 次对话 / Control retaining the most recent history_len conversations
        file_path: 建库文件所在路径 / Path to knowledge base files
        persist_path: 向量数据库持久化路径 / Vector database persistence path
        appid: 星火API的APP ID / Spark API APP ID
        api_key: API密钥（星火、百度文心、OpenAI、智谱）/ API key for various platforms
        Spark_api_secret: 星火API密钥 / Spark API secret key
        Wenxin_secret_key: 文心API密钥 / Wenxin API secret key
        embedding: 使用的embedding模型 / Embedding model to use
        embedding_key: embedding模型的API密钥 / API key for embedding model
    """
    def __init__(self, model:str, temperature:float=0.0, top_k:int=4, chat_history:list=[], 
                 file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, 
                 Spark_api_secret:str=None, Wenxin_secret_key:str=None, 
                 embedding:str = "openai", embedding_key:str=None):
        
        # 初始化模型参数 - Initialize model parameters
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.chat_history = chat_history
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key

        # 初始化向量数据库 - Initialize vector database
        # 通过get_vectordb函数获取向量数据库实例
        # Get vector database instance through get_vectordb function
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding, self.embedding_key)
        
    
    def clear_history(self):
        """
        清空历史记录
        Clear conversation history
        
        功能 / Function:
        - 重置问答链的历史记录，开始新的对话会话
        - Reset Q&A chain history to start a new conversation session
        """
        # 清空聊天历史列表 - Clear chat history list
        self.chat_history = []
        
        # 如果向量数据库存在，重新初始化检索器以清除历史状态
        # If vector database exists, reinitialize retriever to clear historical state
        return self.chat_history.clear()

    
    def change_history_length(self,history_len:int=1):
        """
        保存指定对话轮次的历史记录
        输入参数：
        - history_len ：控制保留的最近 history_len 次对话
        - chat_history：当前的历史对话记录
        输出：返回最近 history_len 次对话
        """
        n = len(self.chat_history)
        return self.chat_history[n-history_len:]

 
    def answer(self, question:str=None,temperature = None, top_k = 4):
        """"
        核心方法，调用问答链
        arguments: 
        - question：用户提问
        """
        
        if len(question) == 0:
            return "", self.chat_history
        
        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
        llm = model_to_llm(self.model, temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)

        #self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': top_k})  #默认similarity，k=4

        qa = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever = retriever
        )
        # 这里的qa只是一个类似于llm的东西
        
        #print(self.llm)
        result = qa({"question": question,"chat_history": self.chat_history})     
        # 这里类似于llm里面把question和chat_history放进去
        #   #result里有question、chat_history、answer
        """
        # 结果格式类似：
        {
            'question': '什么是机器学习？',
            'chat_history': [('什么是机器学习？', '机器学习是人工智能的一个分支...')],
            'answer': '机器学习是人工智能的一个分支...'
        }
        """
        answer =  result['answer']
        answer = re.sub(r"\\n", '<br/>', answer)
        self.chat_history.append((question,answer)) #更新历史记录

        return self.chat_history  #返回本次回答和更新后的历史记录
        
















