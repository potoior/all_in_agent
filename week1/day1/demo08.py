"""FastAPI 请求体处理高级示例
演示了多种请求体处理方式，包括嵌套模型、多模型和字段验证
"""

from typing import Annotated
from fastapi import FastAPI, Path, Body
from pydantic import BaseModel

# 创建 FastAPI 应用实例
app = FastAPI()


class Item(BaseModel):
    """项目数据模型"""
    name: str                    # 项目名称
    description: str | None = None  # 项目描述，可选字段
    price: float                 # 项目价格
    tax: float | None = None     # 税费，可选字段


"""
期望的请求体
{
    "name": "Foo",
    "description": "The pretender",
    "price": 42.0,
    "tax": 3.2
}
"""
@app.put("/items/{item_id}")
async def update_item(
    # 路径参数：项目ID，值必须在 0-1000 之间
    item_id: Annotated[int, Path(title="The ID of the item to get", ge=0, le=1000)],
    # 查询参数：可选的查询字符串
    q: str | None = None,
    # 请求体参数：Item 模型实例
    item: Item | None = None,
):
    """更新指定项目信息
    
    Args:
        item_id (int): 项目ID，路径参数，值必须在 0-1000 之间
        q (str | None): 可选的查询参数
        item (Item | None): 包含项目信息的请求体数据
        
    Returns:
        dict: 包含项目ID、查询参数和项目信息的字典
    """
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if item:
        results.update({"item": item})
    return results


"""
期望的请求体:
{
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    },
    "user": {
        "username": "dave",
        "full_name": "Dave Grohl"
    }
}
"""
class User(BaseModel):
    """用户数据模型"""
    username: str               # 用户名
    full_name: str | None = None  # 用户全名，可选字段

# 也可以声明多个请求体
@app.put("/items/{item_id}")
async def update_item(
    item_id: int,      # 路径参数：项目ID
    item: Item,        # 请求体参数：项目信息
    user: User         # 请求体参数：用户信息
):
    """更新项目信息并关联用户信息
    
    当声明多个请求体模型时，FastAPI 会自动为每个模型创建对应的键
    
    Args:
        item_id (int): 项目ID
        item (Item): 项目信息
        user (User): 用户信息
        
    Returns:
        dict: 包含项目ID、项目信息和用户信息的字典
    """
    results = {"item_id": item_id, "item": item, "user": user}
    return results


"""
期望的请求体
{
    "item": {
        "name": "Foo",
        "description": "The pretender",
        "price": 42.0,
        "tax": 3.2
    },
    "user": {
        "username": "dave",
        "full_name": "Dave Grohl"
    },
    "importance": 5
}
"""

@app.put("/items/{item_id}")
async def update_item(
    item_id: int, 
    item: Item, 
    user: User, 
    # 使用 Body() 明确指定 importance 是一个独立的请求体参数
    importance: Annotated[int, Body()]
):
    """更新项目信息，关联用户信息并指定重要性等级
    
    使用 Body() 可以将单独的值作为请求体的一部分发送
    
    Args:
        item_id (int): 项目ID
        item (Item): 项目信息
        user (User): 用户信息
        importance (int): 重要性等级
        
    Returns:
        dict: 包含项目ID、项目信息、用户信息和重要性等级的字典
    """
    results = {"item_id": item_id, "item": item, "user": user, "importance": importance}
    return results


"""
这种就是直接嵌入,不用再先使用result = Item这种了,而是直接用
"""
@app.put("/items/{item_id}")
async def update_item(
    item_id: int, 
    # 使用 Body(embed=True) 将模型直接嵌入到请求体的根级别
    item: Annotated[Item, Body(embed=True)]
):
    """更新项目信息，使用嵌入模式
    
    当使用 embed=True 时，客户端必须将模型数据包装在键中发送
    
    Args:
        item_id (int): 项目ID
        item (Item): 项目信息，需要包装在 "item" 键中
        
    Returns:
        dict: 包含项目ID和项目信息的字典
    """
    results = {"item_id": item_id, "item": item}
    return results


# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from typing import Annotated
from fastapi import Body, FastAPI
from pydantic import BaseModel, Field

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


class Item(BaseModel):
    """增强版项目数据模型，使用 Field 进行字段验证"""
    name: str
    # 使用 Field 添加字段描述和最大长度限制
    description: str | None = Field(
        default=None, title="The description of the item", max_length=300
    )
    # 使用 Field 添加数值验证和字段描述
    price: float = Field(gt=0, description="The price must be greater than zero")
    tax: float | None = None


@app.put("/items/{item_id}")
async def update_item(item_id: int, item: Annotated[Item, Body(embed=True)]):
    """更新项目信息，使用带字段验证的模型
    
    Args:
        item_id (int): 项目ID
        item (Item): 带验证规则的项目信息
        
    Returns:
        dict: 包含项目ID和项目信息的字典
    """
    results = {"item_id": item_id, "item": item}
    return results