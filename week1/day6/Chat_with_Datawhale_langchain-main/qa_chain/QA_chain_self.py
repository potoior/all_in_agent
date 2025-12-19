"""
不带历史记录的问答链模块
Q&A Chain without History Module

这个模块实现了不带历史记录的基础RAG问答链功能。
支持多种大语言模型和向量数据库，提供简洁的问答接口。

This module implements basic RAG Q&A chain functionality without history support.
Supports multiple large language models and vector databases, providing a simple Q&A interface.
"""

from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import sys
sys.path.append("../")
from qa_chain.model_to_llm import model_to_llm
from qa_chain.get_vectordb import get_vectordb
import sys
import re

class QA_chain_self():
    """
    不带历史记录的问答链类
    Q&A Chain Class without History
    
    功能 / Features:
    - 实现基础的RAG问答功能，不带对话历史记录
    - Implement basic RAG Q&A functionality without conversation history
    - 支持多种大语言模型和嵌入模型
    - Support multiple large language models and embedding models
    - 提供可自定义的提示模板
    - Provide customizable prompt templates
    
    参数 / Parameters:
        model: 调用的模型名称 / Model name to use
        temperature: 温度系数，控制生成的随机性 / Temperature coefficient controlling generation randomness
        top_k: 返回检索的前k个相似文档 / Return top k most similar documents
        file_path: 建库文件所在路径 / Path to knowledge base files
        persist_path: 向量数据库持久化路径 / Vector database persistence path
        appid: 讯飞星火模型需要的应用ID / App ID for iFlytek Spark model
        api_key: 所有模型都需要的API密钥 / API key required by all models
        Spark_api_secret: 讯飞星火模型需要的密钥 / API secret for iFlytek Spark model
        Wenxin_secret_key: 百度文心模型需要的密钥 / Secret key for Baidu Wenxin model
        embedding: 使用的嵌入模型类型 / Embedding model type to use
        embedding_key: 使用的嵌入模型的密钥 / Embedding model key to use
        template: 自定义提示模板，没有输入则使用默认模板 / Custom prompt template, uses default if not provided
    """

    # 基于召回结果和query结合起来构建的prompt使用的默认提示模板
    # Default prompt template for building prompts based on retrieved results and query
    default_template_rq = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。最多使用三句话。尽量使答案简明扼要。总是在回答的最后说"谢谢你的提问！"。
    {context}
    问题: {question}
    有用的回答:"""

    def __init__(self, model:str, temperature:float=0.0, top_k:int=4,  file_path:str=None, persist_path:str=None, appid:str=None, api_key:str=None, Spark_api_secret:str=None,Wenxin_secret_key:str=None, embedding = "openai",  embedding_key = None, template=default_template_rq):
        """
        初始化问答链
        Initialize Q&A chain
        
        功能 / Function:
        - 初始化向量数据库和LLM模型
        - Initialize vector database and LLM model
        - 配置检索器和问答链
        - Configure retriever and Q&A chain
        """
        self.model = model
        self.temperature = temperature
        self.top_k = top_k
        self.file_path = file_path
        self.persist_path = persist_path
        self.appid = appid
        self.api_key = api_key
        self.Spark_api_secret = Spark_api_secret
        self.Wenxin_secret_key = Wenxin_secret_key
        self.embedding = embedding
        self.embedding_key = embedding_key
        self.template = template
        
        # 获取向量数据库实例 / Get vector database instance
        self.vectordb = get_vectordb(self.file_path, self.persist_path, self.embedding,self.embedding_key)
        
        # 获取LLM模型实例 / Get LLM model instance
        self.llm = model_to_llm(self.model, self.temperature, self.appid, self.api_key, self.Spark_api_secret,self.Wenxin_secret_key)

        # 创建问答链提示模板 / Create Q&A chain prompt template
        self.QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=self.template)
        
        # 配置检索器，使用相似度搜索，返回top_k个结果 / Configure retriever using similarity search, return top_k results
        self.retriever = self.vectordb.as_retriever(search_type="similarity",   
                                        search_kwargs={'k': self.top_k})  # 默认similarity，k=4 / Default similarity, k=4
        
        # 创建自定义问答链 / Create custom Q&A chain
        self.qa_chain = RetrievalQA.from_chain_type(llm=self.llm,
                                        retriever=self.retriever,
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":self.QA_CHAIN_PROMPT})

    #基于大模型的问答 prompt 使用的默认提示模版
    #default_template_llm = """请回答下列问题:{question}"""
           
    def answer(self, question:str=None, temperature = None, top_k = 4):
        """
        核心方法，调用问答链生成答案
        Core method to call Q&A chain and generate answer
        
        参数 / Parameters:
            question: 用户提问 / User question
            temperature: 温度系数（可选，使用实例默认值）/ Temperature coefficient (optional, uses instance default)
            top_k: 返回检索的前k个文档（可选，使用实例默认值）/ Return top k documents (optional, uses instance default)
            
        返回 / Returns:
            str: 生成的答案，换行符替换为HTML换行标签 / Generated answer with newlines replaced by HTML line break tags
            
        处理逻辑 / Processing Logic:
        1. 检查问题是否为空 / Check if question is empty
        2. 使用实例默认值处理可选参数 / Use instance defaults for optional parameters
        3. 调用问答链生成答案 / Call Q&A chain to generate answer
        4. 格式化答案（替换换行符）/ Format answer (replace newlines)
        """

        if len(question) == 0:
            return ""
        
        if temperature == None:
            temperature = self.temperature
            
        if top_k == None:
            top_k = self.top_k

        # 调用问答链生成答案 / Call Q&A chain to generate answer
        result = self.qa_chain({"query": question, "temperature": temperature, "top_k": top_k})
        answer = result["result"]
        
        # 将换行符替换为HTML换行标签，便于Web界面显示 / Replace newlines with HTML line break tags for web interface display
        answer = re.sub(r"\\n", '<br/>', answer)
        return answer   
