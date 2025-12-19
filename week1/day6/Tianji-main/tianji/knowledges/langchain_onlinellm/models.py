# =============================================================================
# Tianji项目 - 在线LLM和嵌入模型实现
# =============================================================================
# 功能：提供对多种在线LLM API和嵌入模型的封装
# 支持的模型：
# - ZhipuAI GLM系列模型
# - SiliconFlow平台上的开源模型
# 支持的嵌入模型：
# - ZhipuAI Embedding模型
# - SiliconFlow Embedding模型
# =============================================================================

# LangChain核心组件
from langchain_core.language_models.llms import LLM  # 基础LLM类
from langchain_core.callbacks.manager import CallbackManagerForLLMRun  # 回调管理
from langchain.embeddings.base import Embeddings  # 基础嵌入类

# 类型注解和工具
from typing import Any, Dict, List, Optional
import os  # 环境变量操作
import requests  # HTTP请求（备用）

# 第三方API客户端
from zhipuai import ZhipuAI  # 智谱AI客户端
from langchain.pydantic_v1 import BaseModel, root_validator  # 数据验证
import loguru  # 日志记录

# =============================================================================
# ZhipuAI LLM实现类
# =============================================================================
# 功能：封装智谱AI的GLM系列模型，使其兼容LangChain框架
# 特点：
# - 使用glm-4-flash模型
# - 支持标准的LangChain LLM接口
# - 自动从环境变量获取API密钥
# =============================================================================

class ZhipuLLM(LLM):
    """智谱AI自定义聊天模型类"""

    client: Any = None  # ZhipuAI客户端实例

    def __init__(self):
        """初始化ZhipuAI客户端"""
        super().__init__()
        print("正在初始化ZhipuAI模型...")
        # 从环境变量获取API密钥并创建客户端
        self.client = ZhipuAI(api_key=os.environ.get("ZHIPUAI_API_KEY"))
        print("ZhipuAI模型初始化完成")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用LLM生成回答
        
        参数：
        - prompt: 用户输入的提示词
        - stop: 可选的停止词列表
        - run_manager: 可选的运行管理器
        - **kwargs: 其他参数
        
        返回：
        - str: 模型生成的回答内容
        """
        # 调用智谱AI API创建聊天完成
        response = self.client.chat.completions.create(
            model="glm-4-flash",  # 使用glm-4-flash模型
            messages=[
                {"role": "user", "content": prompt},  # 用户消息
            ],
        )
        # 返回生成的回答内容
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型的标识参数
        
        用于LangChain内部识别和缓存管理
        """
        return {"model_name": "ZhipuAI"}

    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识
        
        用于区分不同类型的语言模型
        """
        return "ZhipuAI"


# =============================================================================
# SiliconFlow LLM实现类
# =============================================================================
# 功能：封装SiliconFlow平台上的开源模型，使其兼容LangChain框架
# 特点：
# - 支持多种开源模型（默认使用Qwen2.5-7B-Instruct）
# - 使用OpenAI兼容的API格式
# - 内置系统提示词，定义模型角色为"人情世故大师"
# - 支持自定义温度、最大token数等参数
# =============================================================================

class SiliconFlowLLM(LLM):
    """SiliconFlow自定义聊天模型类"""

    model_name: str = "Qwen/Qwen2.5-7B-Instruct"  # 默认使用的模型
    base_url: str = "https://api.siliconflow.cn/v1"  # SiliconFlow API地址
    token: Optional[str] = None  # API密钥
    client: Any = None  # OpenAI客户端实例

    def __init__(self):
        """初始化SiliconFlow客户端"""
        super().__init__()
        print("正在初始化SiliconFlow模型...")
        from openai import OpenAI
        
        # 从环境变量获取API密钥
        self.token = os.getenv("OPENAI_API_KEY")
        if not self.token:
            raise ValueError("环境变量中未找到OPENAI_API_KEY")
            
        # 创建OpenAI客户端（兼容SiliconFlow API）
        self.client = OpenAI(
            api_key=self.token,
            base_url=self.base_url
        )
        print("SiliconFlow模型初始化完成")

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """调用LLM生成回答
        
        参数：
        - prompt: 用户输入的提示词
        - stop: 可选的停止词列表
        - run_manager: 可选的运行管理器
        - **kwargs: 其他参数
        
        返回：
        - str: 模型生成的回答内容
        
        说明：
        - 使用系统提示词定义模型角色为"人情世故大师"
        - 最大token数：4096
        - 温度参数：0.7（平衡创造性和确定性）
        """
        # 调用SiliconFlow API创建聊天完成
        response = self.client.chat.completions.create(
            model=self.model_name,  # 使用的具体模型
            messages=[
                # 系统消息：定义模型角色和行为
                {"role": "system", "content": "你是 SocialAI 组织开发的人情世故大师，叫做天机，你将解答用户有关人情世故的问题。"},
                # 用户消息
                {"role": "user", "content": prompt}
            ],
            max_tokens=4096,  # 最大生成token数
            temperature=0.7,  # 温度参数，控制随机性
            response_format={"type": "text"},  # 响应格式为文本
        )
        # 返回生成的回答内容
        return response.choices[0].message.content

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """返回模型的标识参数
        
        用于LangChain内部识别和缓存管理
        """
        return {"model_name": self.model_name}

    @property
    def _llm_type(self) -> str:
        """返回LLM类型标识
        
        用于区分不同类型的语言模型
        """
        return "SiliconFlow"
# =============================================================================
# 嵌入模型实现部分
# =============================================================================
# 功能：提供文本嵌入功能，将文本转换为向量表示
# 支持的嵌入模型：
# - ZhipuAI Embedding模型（embedding-3）
# - SiliconFlow Embedding模型（BAAI/bge-m3等）
# =============================================================================

# =============================================================================
# ZhipuAI嵌入模型实现类
# =============================================================================
# 功能：封装智谱AI的嵌入模型，生成文本向量表示
# 特点：
# - 使用embedding-3模型
# - 支持单文本和多文本嵌入
# - 自动错误处理和日志记录
# =============================================================================

class ZhipuAIEmbeddings(BaseModel, Embeddings):
    """智谱AI嵌入模型类"""

    zhipuai_api_key: Optional[str] = None  # API密钥

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境配置并初始化客户端
        
        参数：
        - values: 配置参数字典
        
        返回：
        - Dict: 更新后的配置字典
        
        异常：
        - ValueError: 当zhipuai包未安装时
        """
        # 获取API密钥（优先使用传入值，其次使用环境变量）
        values["zhupuai_api_key"] = values.get("zhupuai_api_key") or os.getenv(
            "ZHIPUAI_API_KEY"
        )
        try:
            import zhipuai
            # 设置API密钥
            zhipuai.api_key = values["zhupuai_api_key"]
            # 创建客户端实例
            values["client"] = zhipuai.ZhipuAI()
        except ImportError:
            raise ValueError(
                "未找到zhipuai包，请使用 `pip install zhipuai` 安装"
            )
        return values
    def _embed(self, texts: str) -> List[float]:
        """嵌入单个文本
        
        参数：
        - texts: 要嵌入的文本字符串
        
        返回：
        - List[float]: 文本的向量表示（浮点数列表）
        
        异常：
        - ValueError: 当API调用失败时
        """
        try:
            # 调用智谱AI嵌入API
            resp = self.client.embeddings.create(
                model="embedding-3",  # 使用embedding-3模型
                input=texts,  # 输入文本
            )
        except Exception as e:
            # 记录错误日志并抛出异常
            loguru.logger.error(f"推理端点出错: {e}")
            raise ValueError(f"推理端点出错: {e}")
        # 提取嵌入向量
        embeddings = resp.data[0].embedding
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本（单文本）
        
        参数：
        - text: 要嵌入的查询文本
        
        返回：
        - List[float]: 文本的向量表示
        
        说明：
        这是LangChain的标准接口方法，内部调用embed_documents处理
        """
        try:
            # 调用embed_documents处理单个文本
            resp = self.embed_documents([text])
            return resp[0]  # 返回第一个（也是唯一一个）结果
        except Exception as e:
            # 记录错误日志并抛出异常
            loguru.logger.error(f"推理端点出错: {e}")
            raise ValueError(f"推理端点出错: {e}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档文本
        
        参数：
        - texts: 要嵌入的文本列表
        
        返回：
        - List[List[float]]: 每个文本对应的向量表示列表
        
        说明：
        这是LangChain的标准接口方法，用于批量处理多个文本
        """
        try:
            # 对每个文本调用_embed方法进行嵌入
            return [self._embed(text) for text in texts]
        except Exception as e:
            # 记录错误日志并抛出异常
            loguru.logger.error(f"推理端点出错: {e}")
            raise ValueError(f"推理端点出错: {e}")


# =============================================================================
# SiliconFlow嵌入模型实现类
# =============================================================================
# 功能：封装SiliconFlow平台上的开源嵌入模型，生成文本向量表示
# 特点：
# - 支持多种开源嵌入模型（默认使用BAAI/bge-m3）
# - 使用OpenAI兼容的API格式
# - 支持单文本和多文本嵌入
# =============================================================================

class SiliconFlowEmbeddings(BaseModel, Embeddings):
    """SiliconFlow嵌入模型类"""

    openai_api_key: Optional[str] = None  # OpenAI兼容的API密钥
    model_name: str = "BAAI/bge-m3"  # 默认嵌入模型

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """验证环境配置并初始化客户端
        
        参数：
        - values: 配置参数字典
        
        返回：
        - Dict: 更新后的配置字典
        
        异常：
        - ValueError: 当API密钥未找到或openai包未安装时
        """
        # 获取API密钥（优先使用传入值，其次使用环境变量）
        values["openai_api_key"] = values.get("openai_api_key") or os.getenv(
            "OPENAI_API_KEY"
        )
        if not values["openai_api_key"]:
            raise ValueError("未找到OpenAI API密钥")
            
        try:
            from openai import OpenAI
            # 创建OpenAI客户端（兼容SiliconFlow API）
            values["client"] = OpenAI(
                api_key=values["openai_api_key"],
                base_url="https://api.siliconflow.cn/v1"
            )
        except ImportError:
            raise ValueError(
                "未找到OpenAI包，请使用 `pip install openai` 安装"
            )
        return values

    def _embed(self, texts: str) -> List[float]:
        """嵌入单个文本
        
        参数：
        - texts: 要嵌入的文本字符串
        
        返回：
        - List[float]: 文本的向量表示（浮点数列表）
        
        异常：
        - ValueError: 当API调用失败时
        """
        try:
            # 调用SiliconFlow嵌入API
            response = self.client.embeddings.create(
                model=self.model_name,  # 使用的嵌入模型
                input=texts,  # 输入文本
                encoding_format="float"  # 编码格式为浮点数
            )
            # 返回嵌入向量
            return response.data[0].embedding
        except Exception as e:
            # 抛出异常
            raise ValueError(f"推理端点出错: {e}")

    def embed_query(self, text: str) -> List[float]:
        """嵌入查询文本（单文本）
        
        参数：
        - text: 要嵌入的查询文本
        
        返回：
        - List[float]: 文本的向量表示
        
        说明：
        这是LangChain的标准接口方法，内部调用embed_documents处理
        """
        # 调用embed_documents处理单个文本
        resp = self.embed_documents([text])
        return resp[0]  # 返回第一个（也是唯一一个）结果

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """嵌入多个文档文本
        
        参数：
        - texts: 要嵌入的文本列表
        
        返回：
        - List[List[float]]: 每个文本对应的向量表示列表
        
        说明：
        这是LangChain的标准接口方法，用于批量处理多个文本
        """
        # 对每个文本调用_embed方法进行嵌入
        return [self._embed(text) for text in texts]