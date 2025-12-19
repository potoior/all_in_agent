"""
这是一个单例模式的共享数据类，用于在多个智能体之间共享数据。
主要功能:
1. 维护每个用户会话的独立数据实例
2. 存储场景标签、属性、搜索结果等会话状态
3. 保存用户的聊天历史和消息列表
4. 通过uuid区分不同用户的数据
"""

import streamlit as st


class SharedDataSingleton:
    """
    共享数据单例类 - 用于在多智能体系统中管理共享数据
    
    设计目的:
    1. 确保所有智能体访问同一份数据副本
    2. 支持多用户会话隔离
    3. 维护用户交互历史和状态信息
    
    数据分类:
    - 用户会话数据: 每个用户独立的数据实例
    - 智能体通信数据: 消息列表和交互历史
    - 场景相关数据: 场景标签、属性、搜索结果
    - 配置数据: 用户输入的基本信息
    """
    
    # 单例实例引用
    _instance = None
    
    # 共享的核心数据结构 - 存储用户的基本信息和需求
    json_from_data = None  # 这是要共享的变量，包含用户的基本信息
    
    # 智能体通信相关数据
    message_list_for_agent = []  # 存储用户与智能体之间的消息列表
    filter_weblist = []  # 过滤后的网页列表，用于搜索结果筛选
    
    # 场景识别相关数据
    scene_label = ""  # 当前识别的场景标签，如"生日祝福"、"节日问候"等
    scene_attribute = {}  # 场景属性字典，存储场景所需的各个要素
    extra_query = []  # 额外的查询条件，用于补充场景信息
    search_results = {}  # 搜索结果字典，存储不同查询的搜索结果
    
    # 用户交互历史
    chat_history = []  # 聊天历史记录，用于上下文理解
    
    # 多用户会话管理
    uuid_obj = {}  # 用户UUID到数据实例的映射，支持多用户并发
    ask_num = 0  # 询问次数计数器，用于控制对话流程

    @classmethod
    def get_instance(cls):
        """
        获取单例实例的类方法
        
        功能说明:
        1. 检查当前用户的会话状态，获取用户ID
        2. 如果用户ID不存在，创建新的数据实例
        3. 如果用户ID存在但对应实例不存在，创建新的实例并存储
        4. 如果用户ID和实例都存在，返回现有的实例
        
        返回值:
            SharedDataSingleton: 当前用户的共享数据实例
            
        设计特点:
        - 支持多用户并发访问，每个用户有独立的数据空间
        - 使用Streamlit的session_state管理用户会话
        - 延迟初始化，只在需要时创建实例
        """
        # 从Streamlit会话状态获取用户ID
        user_id = ""
        if "user_id" in st.session_state:
            user_id = st.session_state["user_id"]
            
        # 初始化返回对象
        ret_cls_obj = {}
        
        # 情况1: 没有用户ID（首次访问或匿名用户）
        if user_id == "":
            # 创建新的实例并进行初始化
            ret_cls_obj = cls()
            ret_cls_obj = SharedDataSingleton._new_init(ret_cls_obj)
        else:
            # 情况2: 有用户ID
            # 检查该用户是否已有实例
            if user_id in SharedDataSingleton.uuid_obj:
                # 用户实例已存在，直接使用
                pass
            else:
                # 用户实例不存在，创建新实例并存储
                ret_cls_obj = cls()
                ret_cls_obj = SharedDataSingleton._new_init(ret_cls_obj)
                SharedDataSingleton.uuid_obj[user_id] = ret_cls_obj

            # 获取用户的实例
            ret_cls_obj = SharedDataSingleton.uuid_obj[user_id]
        return ret_cls_obj

    def _new_init(cls):
        """
        初始化类变量的静态方法
        
        功能说明:
        1. 重置所有类变量到初始状态
        2. 清空所有数据集合和字典
        3. 将实例引用设为None，允许重新创建
        
        参数:
            cls: 类对象本身
            
        返回值:
            cls: 返回初始化后的类对象
            
        设计目的:
        - 提供一个统一的方法来重置所有共享数据
        - 确保数据清理的一致性
        - 支持实例的重新创建
        """
        # 重置单例实例引用
        cls._instance = None
        
        # 重置核心数据结构
        cls.json_from_data = None  # 这是要共享的变量
        
        # 重置智能体通信数据
        cls.message_list_for_agent = []  # 清空消息列表
        cls.scene_attribute = {}  # 清空场景属性
        cls.scene_label = ""  # 重置场景标签
        cls.extra_query = []  # 清空额外查询
        cls.search_results = {}  # 清空搜索结果
        
        # 重置用户交互历史
        cls.chat_history = []  # 清空聊天历史
        
        # 重置多用户会话管理
        cls.uuid_obj = {}  # 清空用户实例映射
        return cls

    def __init__(self):
        """
        构造函数 - 初始化共享数据实例
        
        功能说明:
        1. 确保单例模式的正确性，防止重复实例化
        2. 初始化用户基本信息的数据结构
        3. 设置所有字段的默认值为空字符串
        
        异常处理:
            Exception: 如果尝试创建第二个实例，抛出异常
            
        数据结构说明:
        json_from_data包含以下字段，用于存储用户的基本信息:
        - requirement: 用户的需求描述
        - scene: 具体场景信息
        - festival: 节日相关场景
        - role: 用户的角色（如送礼人、收礼人等）
        - age: 相关人员的年龄信息
        - career: 职业信息
        - state: 当前状态
        - character: 性格特征
        - time: 时间信息（如节日时间、事件时间等）
        - hobby: 兴趣爱好
        - wish: 祝福或愿望内容
        """
        # 单例模式保护 - 确保只能创建一个实例
        if SharedDataSingleton._instance is not None:
            raise Exception("This class is a singleton!")
            
        # 初始化共享变量 - 用户基本信息模板
        # 这个字典将作为所有智能体访问用户信息的统一入口
        SharedDataSingleton.json_from_data = {
            "requirement": "",      # 用户需求：描述具体想要什么帮助
            "scene": "",           # 场景信息：具体的社交场合
            "festival": "",      # 节日信息：如果是节日相关的场景
            "role": "",            # 角色信息：用户在场景中的身份
            "age": "",             # 年龄信息：相关人员年龄段
            "career": "",          # 职业信息：相关职业背景
            "state": "",           # 状态信息：当前的情感或状态
            "character": "",       # 性格特征：相关人员的性格特点
            "time": "",            # 时间信息：事件发生的时间
            "hobby": "",           # 爱好信息：相关人员的兴趣爱好
            "wish": "",            # 愿望信息：用户想要表达的内容
        }

    # 可以添加更多方法来操作 shared_variable
    
    def get_user_data(self):
        """
        获取用户基本数据的方法
        
        返回值:
            dict: 包含用户基本信息的字典
        """
        return SharedDataSingleton.json_from_data
    
    def update_user_data(self, key, value):
        """
        更新用户基本数据的方法
        
        参数:
            key (str): 要更新的字段名
            value (str): 新的值
        """
        if key in SharedDataSingleton.json_from_data:
            SharedDataSingleton.json_from_data[key] = value
    
    def add_message(self, message):
        """
        添加消息到智能体消息列表
        
        参数:
            message (dict): 消息字典，格式如{"user": "用户消息"}或{"assistant": "助手回复"}
        """
        SharedDataSingleton.message_list_for_agent.append(message)
    
    def get_chat_history(self):
        """
        获取聊天历史记录
        
        返回值:
            list: 聊天历史列表
        """
        return SharedDataSingleton.chat_history
    
    def add_to_chat_history(self, user_input, assistant_response):
        """
        添加对话到聊天历史
        
        参数:
            user_input (str): 用户输入
            assistant_response (str): 助手回复
        """
        SharedDataSingleton.chat_history.append({
            "user": user_input,
            "assistant": assistant_response
        })
    
    def clear_all_data(self):
        """
        清空所有数据的方法
        用于重置用户会话或开始新的对话
        """
        SharedDataSingleton.message_list_for_agent.clear()
        SharedDataSingleton.filter_weblist.clear()
        SharedDataSingleton.scene_label = ""
        SharedDataSingleton.scene_attribute.clear()
        SharedDataSingleton.extra_query.clear()
        SharedDataSingleton.search_results.clear()
        SharedDataSingleton.chat_history.clear()
        SharedDataSingleton.ask_num = 0
        
        # 重置用户基本信息为默认值
        for key in SharedDataSingleton.json_from_data:
            SharedDataSingleton.json_from_data[key] = ""
