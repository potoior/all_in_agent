"""
模型到LLM实例转换模块
Model to LLM Instance Conversion Module

这个模块提供了一个统一的接口来将模型名称转换为对应的LLM实例。
支持OpenAI、百度文心、讯飞星火和智谱AI等多种大语言模型。

This module provides a unified interface to convert model names to corresponding LLM instances.
Supports multiple large language models including OpenAI, Baidu Wenxin, iFlytek Spark, and Zhipu AI.
"""

import sys 
sys.path.append("../llm")
from llm.wenxin_llm import Wenxin_LLM
from llm.spark_llm import Spark_LLM
from llm.zhipuai_llm import ZhipuAILLM
from langchain.chat_models import ChatOpenAI
from llm.call_llm import parse_llm_api_key


def model_to_llm(model:str=None, temperature:float=0.0, appid:str=None, api_key:str=None,
Spark_api_secret:str=None,Wenxin_secret_key:str=None):
        """
        将模型名称转换为对应的LLM实例
        Convert model name to corresponding LLM instance
        
        参数 / Parameters:
            model: 模型名称 / Model name
            temperature: 温度系数（控制生成随机性）/ Temperature coefficient (controls generation randomness)
            appid: 讯飞星火模型需要的应用ID / App ID for iFlytek Spark model
            api_key: API密钥 / API key
            Spark_api_secret: 讯飞星火模型需要的密钥 / API secret for iFlytek Spark model
            Wenxin_secret_key: 百度文心模型需要的密钥 / Secret key for Baidu Wenxin model
            
        返回 / Returns:
            LLM实例 / LLM instance
            
        异常 / Exceptions:
            ValueError: 当指定的模型不支持时 / When specified model is not supported
            
        支持的模型 / Supported models:
            - OpenAI: gpt-3.5-turbo, gpt-3.5-turbo-16k-0613, gpt-3.5-turbo-0613, gpt-4, gpt-4-32k
            - 百度文心 / Baidu Wenxin: ERNIE-Bot, ERNIE-Bot-4, ERNIE-Bot-turbo
            - 讯飞星火 / iFlytek Spark: Spark-1.5, Spark-2.0
            - 智谱AI / Zhipu AI: chatglm_pro, chatglm_std, chatglm_lite
        """
        if model in ["gpt-3.5-turbo", "gpt-3.5-turbo-16k-0613", "gpt-3.5-turbo-0613", "gpt-4", "gpt-4-32k"]:
            if api_key == None:
                api_key = parse_llm_api_key("openai")
            llm = ChatOpenAI(model_name = model, temperature = temperature , openai_api_key = api_key)
        elif model in ["ERNIE-Bot", "ERNIE-Bot-4", "ERNIE-Bot-turbo"]:
            if api_key == None or Wenxin_secret_key == None:
                api_key, Wenxin_secret_key = parse_llm_api_key("wenxin")
            llm = Wenxin_LLM(model=model, temperature = temperature, api_key=api_key, secret_key=Wenxin_secret_key)
        elif model in ["Spark-1.5", "Spark-2.0"]:
            if api_key == None or appid == None and Spark_api_secret == None:
                api_key, appid, Spark_api_secret = parse_llm_api_key("spark")
            llm = Spark_LLM(model=model, temperature = temperature, appid=appid, api_secret=Spark_api_secret, api_key=api_key)
        elif model in ["chatglm_pro", "chatglm_std", "chatglm_lite"]:
            if api_key == None:
                api_key = parse_llm_api_key("zhipuai")
            llm = ZhipuAILLM(model=model, zhipuai_api_key=api_key, temperature = temperature)
        else:
            raise ValueError(f"model{model} not support!!!")
        return llm