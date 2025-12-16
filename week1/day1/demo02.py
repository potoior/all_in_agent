"""
FastAPI 枚举类型与路径参数示例
演示了如何使用枚举类型和特殊路径参数
"""

# 导入枚举模块和FastAPI框架
from enum import Enum
from fastapi import FastAPI


class ModelName(str, Enum):
    """
    定义模型名称枚举类
    继承自 str 和 Enum，使枚举值既是字符串又是枚举成员
    """
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"


# 创建 FastAPI 应用实例
app = FastAPI()


@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    """
    根据模型名称获取模型信息
    展示了枚举类型的使用方法以及不同条件判断方式
    
    Args:
        model_name (ModelName): 模型名称枚举值
    
    Returns:
        dict: 包含模型名称和相关信息的字典
    """
    # 直接比较枚举成员
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}
    
    # 支持比较操作，通过枚举的 value 属性获取实际字符串值
    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}
    
    # 返回给客户端之前，要把枚举元素转换为对应的值（本例中为字符串）
    return {"model_name": model_name, "message": "Have some residuals"}


# 特殊路径参数示例
# 本例中，参数名为 file_path，结尾部分的 :path 说明该参数应匹配路径。
# 注意，包含 /home/johndoe/myfile.txt 的路径参数要以斜杠（/）开头。
# 本例中的 URL 是 /files//home/johndoe/myfile.txt。注意，files 和 home 之间要使用双斜杠（//）。
@app.get("/files/{file_path:path}")
async def read_file(file_path: str):
    """
    读取文件路径信息
    使用 :path 转换器接收完整的路径参数，包括其中的斜杠
    
    Args:
        file_path (str): 文件完整路径
    
    Returns:
        dict: 包含文件路径的字典
    """
    return {"file_path": file_path}