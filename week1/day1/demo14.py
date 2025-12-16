"""FastAPI 部分更新示例
演示了如何使用PATCH方法实现资源的部分更新
"""

from typing import List, Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# 创建 FastAPI 应用实例
app = FastAPI()


class Item(BaseModel):
    """项目数据模型"""
    name: Union[str, None] = None      # 项目名称，可选字段
    description: Union[str, None] = None  # 项目描述，可选字段
    price: Union[float, None] = None   # 项目价格，可选字段
    tax: float = 10.5                  # 税费，默认值为10.5
    tags: List[str] = []               # 标签列表，默认为空列表


# 模拟数据库中的项目数据
items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


@app.get("/items/{item_id}", response_model=Item)
async def read_item(item_id: str):
    """读取指定项目信息
    
    Args:
        item_id (str): 项目ID
        
    Returns:
        Item: 项目信息
    """
    return items[item_id]


@app.patch("/items/{item_id}", response_model=Item)
async def update_item(item_id: str, item: Item):
    """部分更新项目信息
    
    使用PATCH方法实现资源的部分更新，只更新请求中提供的字段
    
    Args:
        item_id (str): 项目ID
        item (Item): 包含要更新字段的项目数据
        
    Returns:
        Item: 更新后的项目信息
    """
    # 获取存储中的原始项目数据
    stored_item_data = items[item_id]
    # 将原始数据转换为Item模型实例
    stored_item_model = Item(**stored_item_data)
    # 获取更新数据，exclude_unset=True确保只获取已设置的字段
    update_data = item.dict(exclude_unset=True)
    # 创建更新后的项目模型实例
    updated_item = stored_item_model.copy(update=update_data)
    # 将更新后的数据存储回数据库（转换为可JSON序列化的格式）
    items[item_id] = jsonable_encoder(updated_item)
    return updated_item