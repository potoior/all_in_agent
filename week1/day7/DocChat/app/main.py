# 从fastapi导入FastAPI主应用类
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from DocChat.app.models.chat_chain import ChatChain
# 从本地模块routes导入已定义的路由
from DocChat.app.router.routes import router
# 从本地模块schemas导入数据模型


# 创建FastAPI应用实例，并配置API元数据
app = FastAPI(
    title="DocChat API",                 # API标题
    description="Document Chat Application API",  # API描述
    version="0.1.0",                     # API版本
)

# 添加CORS中间件，允许跨域请求
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源，生产环境中应指定具体域名
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

# 将路由模块挂载到应用，所有路由都加上/api/v1前缀
app.include_router(router, prefix="/api")


os.environ["OPENAI_API_KEY"] = 'sk-zqzehnidkvjxmpgoqohexqzxwnvyszxwgxucpxmtftdpgrgv'
os.environ["OPENAI_API_ROUTER"] = 'https://api.siliconflow.cn/v1'

@app.get("/")
async def root():
    """
    根路径访问接口
    
    Returns:
        dict: 欢迎信息
    """
    return {"message": "Welcome to DocChat API"}

@app.get("/health")
async def health_check():
    """
    健康检查接口
    
    Returns:
        dict: 应用健康状态
    """
    return {"status": "healthy"}

# 当直接运行此脚本时启动开发服务器
if __name__ == "__main__":
    # 导入uvicorn ASGI服务器
    import uvicorn

    chat_chain = ChatChain(
        chat_history=[],
        api_key=os.getenv("OPENAI_API_KEY"),
        api_router=os.getenv("OPENAI_API_ROUTER")
    )
    # 启动服务器，监听所有网络接口的8000端口
    uvicorn.run(app, host="0.0.0.0", port=5000)