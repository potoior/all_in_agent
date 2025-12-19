"""
辅助函数模块

该模块提供了各种工具函数，用于处理JSON数据、时间戳生成、场景标签提取等功能。
"""
from tianji import TIANJI_PATH
import json
import os
from datetime import datetime


def timestamp_str():
    """
    生成时间戳字符串
    
    生成格式化的当前时间戳，用于文件命名等场景。
    格式：年月日_时分秒_毫秒
    
    Returns:
        str: 格式化的时间戳字符串
    """
    now = datetime.now()
    timestamp_str = now.strftime("%Y-%m-%d %H:%M:%S.%f")
    # 移除特殊字符，生成适合文件名的格式
    timestamp_str = (
        timestamp_str[:-3]  # 保留毫秒前3位
        .replace(" ", "_")   # 空格替换为下划线
        .replace(":", "")     # 移除冒号
        .replace("-", "")    # 移除连字符
        .replace(".", "")    # 移除点
    )
    return timestamp_str


def load_json(file_name):
    """
    加载JSON文件内容
    
    Args:
        file_name (str): JSON文件名
        
    Returns:
        dict: JSON文件内容
    """
    with open(
        os.path.join(
            TIANJI_PATH, "tianji", "agents", "metagpt_agents", "utils", file_name
        ),
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def extract_all_types(json_data):
    """
    提取所有场景标签
    
    从JSON数据中提取所有场景类型标签。
    
    Args:
        json_data (dict): 包含场景类型的JSON数据
        
    Returns:
        list: 所有场景类型标签的列表
    """
    scene_types = json_data.get("scene_types", {})
    types = [scene_types[key]["type"] for key in scene_types]
    return types


def extract_single_type_attributes_and_examples(json_data, type_number):
    """
    提取指定场景的细化要素以及例子
    
    Args:
        json_data (dict): 包含场景信息的JSON数据
        type_number (str): 场景类型编号
        
    Returns:
        tuple: (场景类型, 属性列表, 示例) 如果找到，否则返回(None, None, None)
    """
    scene_types = json_data.get("scene_types", {})
    for key, value in scene_types.items():
        if value["type"].startswith(type_number + "："):
            return value["type"], value["attributes"], value["example"]
    return None, None, None


def extract_all_types_and_examples(json_data):
    """
    提取所有场景标签以及对应的例子
    
    Args:
        json_data (dict): 包含场景信息的JSON数据
        
    Returns:
        dict: 场景类型到示例的映射字典
    """
    scene_types = json_data.get("scene_types", {})
    types_and_examples = {
        value["type"]: value["example"] for key, value in scene_types.items()
    }
    return types_and_examples


def extract_attribute_descriptions(json_data, attributes):
    """
    提取指定场景要素的要素描述
    
    Args:
        json_data (dict): 包含属性描述的JSON数据
        attributes (list): 属性名称列表
        
    Returns:
        dict: 属性到描述的映射字典
    """
    attribute_descriptions = json_data.get("attribute_descriptions", {})
    descriptions = {attr: attribute_descriptions.get(attr) for attr in attributes}
    return descriptions


def has_empty_values(dict):
    """
    判断字典中是否有空值
    
    Args:
        dict (dict): 要检查的字典
        
    Returns:
        bool: 如果字典中有空字符串或None值则返回True，否则返回False
    """
    for value in dict.values():
        if value == "" or value is None:
            return True
    return False


def is_number_in_types(json_data, number):
    """
    判断场景标签编号是否在JSON文件中
    
    Args:
        json_data (dict): 包含场景类型的JSON数据
        number (int/str): 要检查的场景编号
        
    Returns:
        bool: 如果找到对应的场景类型则返回True，否则返回False
    """
    scene_types = json_data.get("scene_types", {})
    for key, value in scene_types.items():
        if value.get("type", "").startswith(str(number) + "："):
            return True
    return False
