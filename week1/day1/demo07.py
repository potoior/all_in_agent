"""FastAPI 查询参数依赖注入和 Pydantic 模型验证示例
演示了如何使用 Pydantic 模型作为查询参数以及字段验证功能
"""

from typing import Annotated, Literal
from fastapi import FastAPI, Query
from pydantic import BaseModel, Field

# 创建 FastAPI 应用实例
app = FastAPI()


class FilterParams(BaseModel):
    """过滤参数模型类
    使用 Pydantic 的 Field 类对字段进行验证和设置默认值
    """
    # limit 字段：默认值为 100，必须大于 0 且小于等于 100
    limit: int = Field(100, gt=0, le=100)
    # offset 字段：默认值为 0，必须大于等于 0
    offset: int = Field(0, ge=0)
    # order_by 字段：只能是 "created_at" 或 "updated_at" 中的一个，默认为 "created_at"
    order_by: Literal["created_at", "updated_at"] = "created_at"
    # tags 字段：字符串列表，默认为空列表
    tags: list[str] = []

    # Pydantic 模型配置：禁止接收未定义的额外字段
    # 当设置为 "forbid" 时，如果传入模型中未定义的字段，将会引发验证错误
    # 这增强了数据的严格性，防止意外的数据污染
    # 当访问https://example.com/items/?limit=10&tool=plumbus时
    """
    会出现这个
    {
        "detail": [
            {
                "type": "extra_forbidden",
                "loc": ["query", "tool"],
                "msg": "Extra inputs are not permitted",
                "input": "plumbus"
            }
        ]
    }
    """
    model_config = {"extra": "forbid"}


@app.get("/items/")
async def read_items(filter_query: Annotated[FilterParams, Query()]):
    """读取项目列表，支持复杂查询参数
    
    使用 Pydantic 模型作为查询参数，FastAPI 会自动将查询参数映射到模型字段
    例如：GET /items/?limit=50&offset=10&order_by=updated_at&tags=tag1&tags=tag2
    
    Args:
        filter_query (FilterParams): 包含过滤参数的 Pydantic 模型
        
    Returns:
        FilterParams: 返回接收到的过滤参数
    """
    return filter_query