"""FastAPI 路径参数和查询参数高级用法示例
演示了使用 Annotated、Path 和 Query 类进行参数验证和元数据设置
"""

from typing import Annotated
from fastapi import FastAPI, Path, Query

# 创建 FastAPI 应用实例
app = FastAPI()

@app.get("/items/{item_id}")
async def read_items(
    # 使用 Annotated 为路径参数添加元数据
    # Path(title="The ID of the item to get") 设置参数标题，会在 API 文档中显示
    item_id: Annotated[int, Path(title="The ID of the item to get")],
    # 使用 Annotated 为查询参数添加元数据
    # Query(alias="item-query") 设置查询参数的别名为 "item-query"
    # 这意味着在 URL 中需要使用 ?item-query=xxx 来传递参数
    q: Annotated[str | None, Query(alias="item-query")] = None,
):
    """根据项目ID获取项目信息，支持带别名的查询参数
    
    Args:
        item_id (int): 项目ID，路径参数，会在API文档中显示标题
        q (str | None): 查询参数，使用 "item-query" 作为URL中的参数名
    
    Returns:
        dict: 包含项目ID和查询参数的字典
    """
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    return results


@app.get("/items/{item_id}")
async def read_items(
    *,  # 使用 * 强制后续参数为关键字参数，提高代码可读性
    # Path 参数验证：ge=0 表示大于等于0，le=1000 表示小于等于1000
    item_id: int = Path(title="The ID of the item to get", ge=0, le=1000),
    # 必需的查询参数
    q: str,
    # Query 参数验证：gt=0 表示大于0，lt=10.5 表示小于10.5
    size: float = Query(gt=0, lt=10.5),
):
    """根据项目ID获取项目信息，包含数值验证的参数
    
    Args:
        item_id (int): 项目ID，路径参数，值必须在 0-1000 之间
        q (str): 必需的查询参数
        size (float): 尺寸参数，值必须在 0-10.5 之间
    
    Returns:
        dict: 包含项目ID、查询参数和尺寸的字典
    """
    results = {"item_id": item_id}
    if q:
        results.update({"q": q})
    if size:
        results.update({"size": size})
    return results