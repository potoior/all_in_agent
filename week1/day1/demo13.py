"""FastAPI 异常处理和元数据示例
演示了自定义异常处理、API标签、弃用标记以及JSON编码器的使用
"""

from fastapi import FastAPI, HTTPException

# 创建 FastAPI 应用实例
app = FastAPI()

# 模拟数据库中的项目数据
items = {"foo": "The Foo Wrestlers"}


@app.get("/items-header/{item_id}")
async def read_item_header(item_id: str):
    """读取项目信息，自定义HTTP异常和响应头
    
    Args:
        item_id (str): 项目ID
        
    Returns:
        dict: 项目信息
        
    Raises:
        HTTPException: 当项目不存在时抛出404异常，并添加自定义响应头
    """
    if item_id not in items:
        # 抛出自定义HTTP异常，包含状态码、详细信息和自定义响应头
        raise HTTPException(
            status_code=404,
            detail="Item not found",
            headers={"X-Error": "There goes my error"},
        )
    return {"item": items[item_id]}


# 自定义异常处理示例
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


class UnicornException(Exception):
    """自定义异常类"""
    def __init__(self, name: str):
        self.name = name


# 创建 FastAPI 应用实例（实际项目中不应重复创建）
app = FastAPI()


@app.exception_handler(UnicornException)
async def unicorn_exception_handler(request: Request, exc: UnicornException):
    """处理UnicornException异常的处理器
    
    Args:
        request (Request): 请求对象
        exc (UnicornException): UnicornException异常实例
        
    Returns:
        JSONResponse: 自定义的JSON响应
    """
    return JSONResponse(
        status_code=418,  # 使用418状态码（I'm a teapot）
        content={"message": f"Oops! {exc.name} did something. There goes a rainbow..."},
    )


@app.get("/unicorns/{name}")
async def read_unicorn(name: str):
    """读取独角兽信息，可能抛出自定义异常
    
    Args:
        name (str): 独角兽名称
        
    Returns:
        dict: 独角兽名称
        
    Raises:
        UnicornException: 当名称为"yolo"时抛出此异常
    """
    if name == "yolo":
        raise UnicornException(name=name)
    return {"unicorn_name": name}


# API元数据示例
from fastapi import FastAPI

# 创建 FastAPI 应用实例（实际项目中不应重复创建）
app = FastAPI()


@app.get("/items/", tags=["items"])
async def read_items():
    """读取项目列表，标记为"items"标签
    
    Returns:
        list: 项目列表
    """
    return [{"name": "Foo", "price": 42}]


@app.get("/users/", tags=["users"])
async def read_users():
    """读取用户列表，标记为"users"标签
    
    Returns:
        list: 用户列表
    """
    return [{"username": "johndoe"}]


@app.get("/elements/", tags=["items"], deprecated=True)
async def read_elements():
    """读取元素列表，标记为"items"标签并标记为已弃用
    
    Returns:
        list: 元素列表
    """
    return [{"item_id": "Foo"}]


# JSON编码器示例
from datetime import datetime
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

# 模拟数据库
fake_db = {}


class Item(BaseModel):
    """项目数据模型"""
    title: str                  # 标题
    timestamp: datetime         # 时间戳
    description: str | None = None  # 描述，可选字段


# 创建 FastAPI 应用实例（实际项目中不应重复创建）
app = FastAPI()


@app.put("/items/{id}")
def update_item(id: str, item: Item):
    """更新项目信息，使用JSON编码器处理数据
    
    Args:
        id (str): 项目ID
        item (Item): 项目数据
        
    Returns:
        None: 无返回值，数据存储在fake_db中
    """
    # 使用jsonable_encoder将Pydantic模型转换为可JSON序列化的数据
    json_compatible_item_data = jsonable_encoder(item)
    fake_db[id] = json_compatible_item_data