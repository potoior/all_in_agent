# 导入标准库中的List类型
import os

# 从fastapi导入APIRouter用于定义路由，HTTPException用于处理HTTP异常
from fastapi import APIRouter, HTTPException

from DocChat.app.models.chat_chain import ChatChain
# 从本地模块schemas导入数据模型
from DocChat.app.schemas.schemas import QueryRequest, QueryResponse

# 创建API路由器实例
router = APIRouter()


@router.post("/chat", response_model=QueryResponse)
async def query_endpoint(query_request: QueryRequest):
    """
    查询接口，适配前端发送的查询请求
    此端点用于接收前端直接发送的query参数
    
    Args:
        query_request (QueryRequest): 包含查询内容的数据模型
        
    Returns:
        QueryResponse: 返回AI回复内容
    """
    try:
        # 这里可以实现具体的查询逻辑，暂时返回接收到的查询内容
        response_content = f"后端已收到查询: {query_request.query}"
        print(response_content)

        # 使用单例模式的ChatChain，而不是每次都创建新实例
        chat_chain = ChatChain(
            chat_history=[],
            api_key=os.getenv("OPENAI_API_KEY"),
            api_router=os.getenv("OPENAI_API_ROUTER")
        )
        
        # 使用请求中的会话ID，如果没有提供则使用默认值
        session_id = query_request.session_id
        print("session_id:", session_id)
        # 调用answer方法，该方法会打印最相关的文本块
        result = chat_chain.answer(query_request.query, session_id=session_id)
        # 从结果中获取最新的回答
        if result and len(result) > 0:
            latest_answer = result[-1][1]  # 获取最新回答内容
            
            return QueryResponse(response=latest_answer)
        else:
            return QueryResponse(response="抱歉，无法生成回答")
    except Exception as e:
        # 如果出现异常，抛出400错误
        print(e)
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/chat/history/{session_id}", response_model=list[QueryResponse])
async def get_chat_history(session_id: str):
    """
    获取指定会话的聊天历史记录
    
    Args:
        session_id: 会话ID
        
    Returns:
        聊天历史记录列表
    """
    try:
        chat_chain = ChatChain(
            chat_history=[],
            api_key=os.getenv("OPENAI_API_KEY"),
            api_router=os.getenv("OPENAI_API_ROUTER")
        )
        
        # 获取指定会话的历史记录
        if session_id in chat_chain.session_histories:
            history = chat_chain.session_histories[session_id]
            
            # 将历史记录转换为响应格式
            responses = [QueryResponse(response=pair[1]) for pair in history]
            
            return responses
        else:
            return []
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))