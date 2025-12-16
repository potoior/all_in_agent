"""
FastAPI 查询参数高级用法示例
演示了 Query 类的各种特性和用法
"""

# 导入类型提示所需的模块和 FastAPI 组件
from typing import Union, List
from fastapi import FastAPI, Query

# 创建 FastAPI 应用实例
app = FastAPI()


@app.get("/items/")
async def read_items(
    q: Union[str, None] = Query(default="1234", min_length=3, max_length=50),
):
    """
    读取项目列表，支持带约束条件的查询参数
    使用 Query 类为查询参数添加默认值和长度限制
    
    Args:
        q (Union[str, None]): 查询字符串，最小长度3，最大长度50，默认值为"1234"
    
    Returns:
        dict: 包含项目列表和查询参数的字典
    """
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results


"""
如果访问http://localhost:8000/items/?q=foo&q=bar
响应的是:
{
  "q": [
    "foo",
    "bar"
  ]
}
"""
@app.get("/items/")
async def read_items(q: Union[List[str], None] = Query(default=None)):
    """
    读取项目列表，支持列表形式的查询参数
    允许同一个查询参数名对应多个值
    
    Args:
        q (Union[List[str], None]): 字符串列表形式的查询参数，默认为 None
    
    Returns:
        dict: 包含查询参数列表的字典
    """
    query_items = {"q": q}
    return query_items


@app.get("/items/")
async def read_items(
    q: Union[str, None] = Query(
        default=None,
        title="Query string",
        description="Query string for the items to search in the database that have a good match",
        min_length=3,
    ),
):
    """
    读取项目列表，使用带元数据的查询参数
    使用 Query 类为查询参数添加标题和描述信息，这些信息会显示在自动生成的 API 文档中
    
    Args:
        q (Union[str, None]): 查询字符串
            - title: 查询字符串的标题
            - description: 查询字符串的详细描述
            - min_length: 最小长度限制
    
    Returns:
        dict: 包含项目列表和查询参数的字典
    """
    results = {"items": [{"item_id": "Foo"}, {"item_id": "Bar"}]}
    if q:
        results.update({"q": q})
    return results