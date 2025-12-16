"""
FastAPI 基础路由示例
演示了基本的路由操作、路径参数和路由顺序的重要性
"""

# 导入 FastAPI 框架
from fastapi import FastAPI

# 创建 FastAPI 应用实例
app = FastAPI()


@app.get("/")
async def root():
    """
    根路径处理函数
    返回一个简单的欢迎消息
    
    Returns:
        dict: 包含欢迎消息的字典
    """
    return {"message": "Hello World"}


@app.get("/items/{item_id}")
async def read_item(item_id: int):
    """
    读取特定项目的处理函数
    将路径参数 item_id 的值传递给函数参数，并返回该项目ID
    
    Args:
        item_id (int): 项目唯一标识符
    
    Returns:
        dict: 包含项目ID的字典
    """
    return {"item_id": item_id}


# 路由顺序重要性示例
# 有时，路径操作中的路径是写死的。
# 比如要使用 /users/me 获取当前用户的数据。
# 然后还要使用 /users/{user_id}，通过用户 ID 获取指定用户的数据。
# 由于路径操作是按顺序依次运行的，因此，一定要在 /users/{user_id} 之前声明 /users/me

@app.get("/users/me")
async def read_user_me():
    """
    获取当前用户信息
    注意：此路由必须在 /users/{user_id} 之前声明，否则会被误识别为 user_id="me"
    
    Returns:
        dict: 当前用户的信息
    """
    return {"user_id": "the current user"}


@app.get("/users/{user_id}")
async def read_user(user_id: str):
    """
    根据用户ID获取用户信息
    注意：此路由应该在固定路径路由之后声明
    
    Args:
        user_id (str): 用户唯一标识符
    
    Returns:
        dict: 指定用户的信息
    """
    return {"user_id": user_id}

