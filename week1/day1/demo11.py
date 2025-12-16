"""FastAPI 响应模型和数据处理示例
演示了如何使用响应模型控制输出数据、处理输入输出分离以及模型继承
"""

from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

# 创建 FastAPI 应用实例
app = FastAPI()


class UserIn(BaseModel):
    """用户输入模型，包含敏感信息"""
    username: str      # 用户名
    password: str      # 密码（敏感信息）
    email: EmailStr    # 邮箱，使用EmailStr确保邮箱格式正确
    full_name: str | None = None  # 用户全名，可选字段


class UserOut(BaseModel):
    """用户输出模型，不包含敏感信息"""
    username: str      # 用户名
    email: EmailStr    # 邮箱
    full_name: str | None = None  # 用户全名，可选字段


@app.post("/user/", response_model=UserOut)
async def create_user(user: UserIn) -> Any:
    """创建用户，使用响应模型过滤敏感信息
    
    通过 response_model=UserOut 确保响应中不包含密码等敏感信息
    
    Args:
        user (UserIn): 包含用户信息的请求体数据
        
    Returns:
        UserOut: 不包含敏感信息的用户信息
    """
    return user


# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from typing import List, Union
from fastapi import FastAPI
from pydantic import BaseModel

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


class Item(BaseModel):
    """项目数据模型"""
    name: str                      # 项目名称
    description: Union[str, None] = None  # 项目描述，可选字段
    price: float                   # 项目价格
    tax: float = 10.5              # 税费，默认值为10.5
    tags: List[str] = []           # 标签列表，默认为空列表


# 模拟数据库中的项目数据
items = {
    "foo": {"name": "Foo", "price": 50.2},
    "bar": {"name": "Bar", "description": "The bartenders", "price": 62, "tax": 20.2},
    "baz": {"name": "Baz", "description": None, "price": 50.2, "tax": 10.5, "tags": []},
}


"""
这时候的响应值为
{
    "name": "Foo",
    "price": 50.2
}
那些默认值的都不返回(如果是none的话)
"""
@app.get("/items/{item_id}", response_model=Item, response_model_exclude_unset=True)
async def read_item(item_id: str):
    """读取项目信息，排除未设置的字段
    
    使用 response_model_exclude_unset=True 确保只返回已设置的字段，
    默认值和None值不会包含在响应中
    
    Args:
        item_id (str): 项目ID
        
    Returns:
        dict: 项目信息，只包含已设置的字段
    """
    return items[item_id]


"""
这时候就只包含name 和 description
"""
@app.get(
    "/items/{item_id}/name",
    response_model=Item,
    response_model_include={"name", "description"},
)
async def read_item_name(item_id: str):
    """读取项目的名称和描述信息
    
    使用 response_model_include 只返回指定的字段
    
    Args:
        item_id (str): 项目ID
        
    Returns:
        dict: 项目信息，只包含name和description字段
    """
    return items[item_id]


"""
这时候就排除tax
"""
@app.get("/items/{item_id}/public", response_model=Item, response_model_exclude={"tax"})
async def read_item_public_data(item_id: str):
    """读取项目公开信息，排除税费字段
    
    使用 response_model_exclude 排除指定的字段（如税费等敏感信息）
    
    Args:
        item_id (str): 项目ID
        
    Returns:
        dict: 项目信息，不包含tax字段
    """
    return items[item_id]


# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from fastapi import FastAPI
from pydantic import BaseModel, EmailStr

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


class UserBase(BaseModel):
    """用户基础模型，包含公共字段"""
    username: str              # 用户名
    email: EmailStr            # 邮箱
    full_name: str | None = None  # 用户全名，可选字段


class UserIn(UserBase):
    """用户输入模型，继承自UserBase，增加密码字段"""
    password: str  # 密码


class UserOut(UserBase):
    """用户输出模型，继承自UserBase，不包含密码字段"""
    pass


class UserInDB(UserBase):
    """数据库用户模型，继承自UserBase，增加哈希密码字段"""
    hashed_password: str  # 哈希后的密码


def fake_password_hasher(raw_password: str):
    """模拟密码哈希函数
    
    Args:
        raw_password (str): 原始密码
        
    Returns:
        str: 哈希后的密码
    """
    return "supersecret" + raw_password


def fake_save_user(user_in: UserIn):
    """模拟保存用户到数据库
    
    Args:
        user_in (UserIn): 用户输入数据
        
    Returns:
        UserInDB: 保存到数据库的用户数据
    """
    # 对密码进行哈希处理
    hashed_password = fake_password_hasher(user_in.password)
    # 创建数据库用户模型实例
    user_in_db = UserInDB(**user_in.dict(), hashed_password=hashed_password)
    print("User saved! ..not really")
    return user_in_db


@app.post("/user/", response_model=UserOut)
async def create_user(user_in: UserIn):
    """创建用户，实现输入输出模型分离
    
    将用户输入数据转换为数据库模型，再转换为输出模型返回
    
    Args:
        user_in (UserIn): 用户输入数据
        
    Returns:
        UserOut: 用户输出数据，不包含密码信息
    """
    user_saved = fake_save_user(user_in)
    return user_saved