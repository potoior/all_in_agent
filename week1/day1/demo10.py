"""FastAPI 特殊数据类型和Cookie参数示例
演示了如何处理日期时间、UUID等特殊数据类型以及如何使用Cookie参数
"""

from datetime import datetime, time, timedelta
from typing import Annotated
from uuid import UUID

from fastapi import Body, FastAPI

# 创建 FastAPI 应用实例
app = FastAPI()


@app.put("/items/{item_id}")
async def read_items(
    # UUID类型参数：确保接收到的是有效的UUID格式
    item_id: UUID,
    # datetime类型参数：日期时间
    start_datetime: Annotated[datetime, Body()],
    # datetime类型参数：结束日期时间
    end_datetime: Annotated[datetime, Body()],
    # timedelta类型参数：时间间隔
    process_after: Annotated[timedelta, Body()],
    # time类型参数：可选的时间参数
    repeat_at: Annotated[time | None, Body()] = None,
):
    """处理项目时间相关的信息
    
    Args:
        item_id (UUID): 项目唯一标识符，使用UUID类型确保唯一性
        start_datetime (datetime): 开始日期时间
        end_datetime (datetime): 结束日期时间
        process_after (timedelta): 处理延迟时间间隔
        repeat_at (time | None): 重复执行的时间点，可选参数
        
    Returns:
        dict: 包含所有时间相关信息的字典
    """
    # 计算实际开始处理时间：开始时间 + 延迟时间
    start_process = start_datetime + process_after
    # 计算处理持续时间：结束时间 - 实际开始处理时间
    duration = end_datetime - start_process
    
    return {
        "item_id": item_id,
        "start_datetime": start_datetime,
        "end_datetime": end_datetime,
        "process_after": process_after,
        "repeat_at": repeat_at,
        "start_process": start_process,
        "duration": duration,
    }


# 重新导入必要的模块（因为在前面已经导入过，此处为示例完整性）
from typing import Annotated
from fastapi import Cookie, FastAPI

# 重新创建应用实例（实际项目中不应重复创建）
app = FastAPI()


@app.get("/items/")
async def read_items(
    # Cookie参数：从请求的Cookie中提取ads_id的值
    ads_id: Annotated[str | None, Cookie()] = None
):
    """从Cookie中读取广告ID信息
    
    Args:
        ads_id (str | None): 从Cookie中获取的广告ID，如果不存在则为None
        
    Returns:
        dict: 包含广告ID的字典
    """
    return {"ads_id": ads_id}