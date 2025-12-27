# 从pydantic导入BaseModel用于创建数据模型
from pydantic import BaseModel
# 从typing导入Optional和List用于类型提示
from typing import Optional, List

class Document(BaseModel):
    """
    文档数据模型
    
    Attributes:
        id (str): 文档唯一标识符
        name (str): 文档名称
        content (Optional[str]): 文档内容，可选字段，默认为None
    """
    id: str
    name: str
    content: Optional[str] = None

class ChatMessage(BaseModel):
    """
    聊天消息数据模型
    
    Attributes:
        role (str): 消息发送者角色（如"user"或"assistant"）
        content (str): 消息内容
    """
    role: str
    content: str

class ChatRequest(BaseModel):
    """
    聊天请求数据模型
    
    Attributes:
        document_id (str): 关联的文档ID
        messages (List[ChatMessage]): 聊天消息历史列表
        session_id (Optional[str]): 会话ID，可选字段，默认为'default'
    """
    document_id: str
    messages: List[ChatMessage]
    session_id: Optional[str] = "default"

class ChatResponse(BaseModel):
    """
    聊天响应数据模型
    
    Attributes:
        response (str): AI回复的内容
        sources (List[str]): 引用的文档来源列表
    """
    response: str
    sources: List[str]


class QueryRequest(BaseModel):
    """
    查询请求数据模型（用于适配前端请求）
    
    Attributes:
        query (str): 用户查询内容
        session_id (Optional[str]): 会话ID，可选字段，默认为'default'
    """
    query: str
    session_id: Optional[str] = "default"


class QueryResponse(BaseModel):
    """
    查询响应数据模型（用于适配前端请求）
    
    Attributes:
        response (str): AI回复的内容
        content (str): 响应内容
    """
    response: str
    content: str = ""