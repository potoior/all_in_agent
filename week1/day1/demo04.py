"""
FastAPI 请求体处理示例
演示了如何使用 Pydantic 模型处理请求体数据
"""

# 导入 FastAPI 框架和 Pydantic 基础模型
from fastapi import FastAPI
from pydantic import BaseModel


class Item(BaseModel):
    """
    定义项目数据模型
    继承自 Pydantic 的 BaseModel，用于验证和序列化请求体数据
    """
    name: str                # 项目名称，必需字段
    description: str | None = None  # 项目描述，可选字段，默认为 None
    price: float             # 项目价格，必需字段
    tax: float | None = None # 税费，可选字段，默认为 None


# 创建 FastAPI 应用实例
app = FastAPI()

"""
一般的json是
{
    "name": "Foo",
    "description": "An optional description",
    "price": 45.2,
    "tax": 3.5
}
但是由于 description 和 tax 是可选的（默认值为 None），下面的 JSON 对象也有效：
{
    "name": "Foo",
    "price": 45.2
}
"""


@app.post("/items/")
async def create_item(item: Item):
    """
    创建新项目
    接收一个 Item 模型的实例作为请求体
    
    Args:
        item (Item): 包含项目信息的请求体数据
    
    Returns:
        Item: 创建的项目信息
    """
    return item


@app.post("/items/")
async def create_item(item: Item):
    """
    创建新项目并计算含税价格
    将请求体数据转换为字典，并在有税费时计算含税价格
    
    Args:
        item (Item): 包含项目信息的请求体数据
    
    Returns:
        dict: 包含项目信息和可能的含税价格的字典
    """
    item_dict = item.dict()
    if item.tax is not None:
        # 这里允许直接访问
        price_with_tax = item.price + item.tax
        item_dict.update({"price_with_tax": price_with_tax})
    return item_dict


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    """
    更新指定项目
    FastAPI 支持同时声明请求体和路径参数
    
    Args:
        item_id (int): 项目唯一标识符（路径参数）
        item (Item): 包含更新信息的请求体数据
    
    Returns:
        dict: 包含项目ID和更新信息的字典
    """
    return {"item_id": item_id, **item.dict()}


# FastAPI 支持同时声明请求体、路径参数和查询参数。
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item, q: str | None = None):
    """
    更新指定项目并支持查询参数
    展示了同时使用请求体、路径参数和查询参数的方式
    
    Args:
        item_id (int): 项目唯一标识符（路径参数）
        item (Item): 包含更新信息的请求体数据
        q (str | None): 可选的查询参数
    
    Returns:
        dict: 包含项目ID、更新信息和可能的查询参数的字典
    """
    result = {"item_id": item_id, **item.dict()}
    if q:
        result.update({"q": q})
    return result