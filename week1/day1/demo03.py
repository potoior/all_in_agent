"""
FastAPI 查询参数示例
演示了查询参数、可选参数、布尔参数和多参数路径的使用
"""

# 导入 FastAPI 框架
from fastapi import FastAPI

# 创建 FastAPI 应用实例
app = FastAPI()

# 模拟数据库中的项目数据
fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]


# 发送的请求http://127.0.0.1:8000/items/?skip=0&limit=10
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    """
    分页读取项目列表
    使用查询参数控制分页，skip 表示跳过的项目数，limit 表示返回的项目数
    
    Args:
        skip (int): 跳过的项目数量，默认为 0
        limit (int): 返回的项目数量，默认为 10
    
    Returns:
        list: 项目列表的一部分
    """
    return fake_items_db[skip : skip + limit]


# 可选查询参数示例
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None):
    """
    根据项目ID读取项目信息，并支持可选的查询参数
    
    Args:
        item_id (str): 项目唯一标识符
        q (str | None): 可选的查询字符串参数
    
    Returns:
        dict: 项目信息，如果有查询参数则一并返回
    """
    # 这里可以把默认值设置为none
    if q:
        return {"item_id": item_id, "q": q}
    return {"item_id": item_id}


# 布尔查询参数示例
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str | None = None, short: bool = False):
    """
    根据项目ID读取项目信息，支持查询参数和简略显示选项
    
    Args:
        item_id (str): 项目唯一标识符
        q (str | None): 可选的查询字符串参数
        short (bool): 是否简略显示，默认为 False
    
    Returns:
        dict: 项目信息，根据short参数决定是否包含详细描述
    """
    # bool自动设置
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item


# 多参数路径示例
# 如果是必须要的参数就不要设置默认值
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str | None = None, short: bool = False
):
    """
    获取特定用户的特定项目信息
    展示了多路径参数的使用方式
    
    Args:
        user_id (int): 用户唯一标识符
        item_id (str): 项目唯一标识符
        q (str | None): 可选的查询字符串参数
        short (bool): 是否简略显示，默认为 False
    
    Returns:
        dict: 包含项目信息和所有者信息的字典
    """
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item

