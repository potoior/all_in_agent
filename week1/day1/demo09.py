"""FastAPI 嵌套模型和示例数据示例
演示了如何使用嵌套的 Pydantic 模型以及如何为模型添加示例数据
"""

from fastapi import FastAPI
from pydantic import BaseModel

# 创建 FastAPI 应用实例
app = FastAPI()


class Image(BaseModel):
    """图片信息模型"""
    url: str    # 图片URL地址
    name: str   # 图片名称


class Item(BaseModel):
    """项目数据模型，包含嵌套的图片信息"""
    name: str                    # 项目名称
    description: str | None = None  # 项目描述，可选字段
    price: float                 # 项目价格
    tax: float | None = None     # 税费，可选字段
    tags: set[str] = set()       # 标签集合，使用set确保标签不重复
    image: Image | None = None   # 图片信息，嵌套的Image模型


"""
希望收到下面的请求
{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2,
    "tags": ["rock", "metal", "bar"],
    "image": {
        "url": "http://example.com/baz.jpg",
        "name": "The Foo live"
    }
}
"""
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    """更新项目信息，支持嵌套的图片信息
    
    Args:
        item_id (int): 项目ID
        item (Item): 包含项目信息和图片信息的请求体数据
        
    Returns:
        dict: 包含项目ID和项目信息的字典
    """
    results = {"item_id": item_id, "item": item}
    return results


# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from fastapi import FastAPI
from pydantic import BaseModel, HttpUrl

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


class Image(BaseModel):
    """增强版图片信息模型，使用HttpUrl进行URL验证"""
    url: HttpUrl  # 使用HttpUrl类型确保URL格式正确
    name: str     # 图片名称


class Item(BaseModel):
    """增强版项目数据模型，支持多个图片"""
    name: str                    # 项目名称
    description: str | None = None  # 项目描述，可选字段
    price: float                 # 项目价格
    tax: float | None = None     # 税费，可选字段
    tags: set[str] = set()       # 标签集合，使用set确保标签不重复
    images: list[Image] | None = None  # 图片列表，支持多个图片


"""
这将期望（转换，校验，记录文档等）下面这样的 JSON 请求体
{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2,
    "tags": [
        "rock",
        "metal",
        "bar"
    ],
    "images": [
        {
            "url": "http://example.com/baz.jpg",
            "name": "The Foo live"
        },
        {
            "url": "http://example.com/dave.jpg",
            "name": "The Baz"
        }
    ]
}
"""
@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    """更新项目信息，支持多个图片信息
    
    Args:
        item_id (int): 项目ID
        item (Item): 包含项目信息和多个图片信息的请求体数据
        
    Returns:
        dict: 包含项目ID和项目信息的字典
    """
    results = {"item_id": item_id, "item": item}
    return results


# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from fastapi import FastAPI
from pydantic import BaseModel

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


# 提供一个正确的示例
class Item(BaseModel):
    """带示例数据的项目模型"""
    name: str                    # 项目名称
    description: str | None = None  # 项目描述，可选字段
    price: float                 # 项目价格
    tax: float | None = None     # 税费，可选字段

    # 为模型配置示例数据，这些数据会出现在自动生成的API文档中
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "Foo",
                    "description": "A very nice Item",
                    "price": 35.4,
                    "tax": 3.2,
                }
            ]
        }
    }


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Item):
    """更新项目信息，使用带示例数据的模型
    
    Args:
        item_id (int): 项目ID
        item (Item): 包含项目信息的请求体数据，带有示例数据
        
    Returns:
        dict: 包含项目ID和项目信息的字典
    """
    results = {"item_id": item_id, "item": item}
    return results