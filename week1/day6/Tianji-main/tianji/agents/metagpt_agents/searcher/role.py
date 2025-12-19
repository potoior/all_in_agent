"""
网络搜索助手角色类 - 多智能体系统中的搜索专家

功能概述:
Searcher角色是Tianji多智能体系统中的核心组件之一，专门负责从互联网获取和筛选相关信息。
它通过一系列有序的动作来执行完整的搜索流程，确保为用户提供准确、相关的搜索结果。

设计理念:
1. 模块化设计: 将复杂的搜索过程分解为5个独立的动作，每个动作负责特定的功能
2. 顺序执行: 采用BY_ORDER模式确保动作按照预定顺序执行，保证搜索流程的连贯性
3. 智能扩展: 通过查询扩展技术生成多样化的搜索查询，提高信息检索的覆盖面
4. 结果筛选: 多层筛选机制确保只保留最相关、最有价值的搜索结果

工作流程:
1. 查询扩展: 基于用户输入和场景信息生成多个相关的搜索查询
2. 网络搜索: 使用多个搜索引擎并行搜索，获取网页片段
3. 结果选择: 分析网页片段，识别需要进一步获取的网页
4. 内容获取: 爬取选定网页的完整内容
5. 结果过滤: 对获取的内容进行智能过滤，提取有价值的信息

技术特点:
- 并发搜索: 支持多个搜索引擎同时工作，提高搜索效率
- 智能筛选: 基于内容相关性和质量进行多轮筛选
- 错误处理: 完善的异常处理机制，确保搜索过程的稳定性
- 结果缓存: 支持搜索结果的缓存和管理，避免重复搜索
"""

from dotenv import load_dotenv

# 加载环境变量配置
load_dotenv()

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from .action import (
    QueryExpansion,
    WebSearch,
    SelectResult,
    SelectFetcher,
    FilterSelectedResult,
)

"""
网络搜索助手 agent，将会使用以下行动：
action 1 （QueryExpansion）：基于用户以及大模型的对话记录，用户所面对的场景，进行查询扩展。
action 2 （WebSearch）：进行网络搜索（duckduckgo_Search），返回网页片段帮助决策。
action 3 （SelectResult）：基于返回的网页片段，判断哪些网页需要进一步查询。
action 4 （SelectFetcher）：通过 requests 模块爬取网页里的内容。
action 5 （FilterSelectedResult）：对爬取的网页内容进行过滤，并且加入到结果中。
"""


class Searcher(Role):
    """
    搜索者角色类 - 专门负责网络信息检索的智能体
    
    角色定位:
    - 名称: Searcher
    - 简介: Get extra result from search engine
    - 职责: 从搜索引擎获取额外的相关信息，为系统提供外部知识支持
    
    核心能力:
    1. 查询扩展: 能够将用户意图转化为多个相关的搜索查询
    2. 多源搜索: 支持多种搜索引擎，提高信息获取的全面性
    3. 智能筛选: 具备内容质量评估和相关性判断能力
    4. 深度爬取: 能够获取网页的完整内容，不只是摘要信息
    5. 结果精炼: 对获取的信息进行过滤和整理，提取核心价值
    """
    
    # 角色基本信息
    name: str = "Searcher"  # 角色名称，用于标识和日志记录
    profile: str = "Get extra result from search engine"  # 角色简介，描述其主要职责

    def __init__(self, **kwargs):
        """
        初始化Searcher角色
        
        参数:
            **kwargs: 传递给父类Role的其他参数
            
        功能:
        1. 调用父类构造函数完成基础初始化
        2. 设置该角色能够执行的动作列表
        3. 配置角色的反应模式为顺序执行
        
        动作配置:
        - QueryExpansion: 查询扩展，生成多样化的搜索查询
        - WebSearch: 网络搜索，获取网页片段
        - SelectResult: 结果选择，识别有价值的网页
        - SelectFetcher: 内容获取，爬取完整网页内容
        - FilterSelectedResult: 结果过滤，精炼获取的信息
        
        执行模式:
        - BY_ORDER: 严格按照动作列表的顺序执行，确保搜索流程的连贯性
        """
        super().__init__(**kwargs)
        
        # 设置可执行的动作列表
        self.set_actions(
            [
                QueryExpansion,    # 查询扩展 - 第一步
                WebSearch,         # 网络搜索 - 第二步
                SelectResult,      # 结果选择 - 第三步
                SelectFetcher,     # 内容获取 - 第四步
                FilterSelectedResult,  # 结果过滤 - 第五步
            ]
        )
        
        # 设置角色的反应模式为顺序执行
        # 这确保了搜索的各个步骤按照预定的顺序执行
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)

    async def _act(self) -> Message:
        """
        执行当前待处理的动作
        
        这是Searcher角色的核心执行方法，负责：
        1. 获取当前需要执行的动作
        2. 从记忆系统中获取最新的消息
        3. 执行动作并获取结果
        4. 创建新的消息对象并添加到记忆系统
        
        返回值:
            Message: 包含执行结果的消息对象，可用于后续处理或传递给其他角色
            
        工作流程:
        1. 记录日志信息，标识当前执行的角色和动作
        2. 获取当前待执行的动作对象
        3. 从记忆系统中获取最近的一条消息（k=1表示只取一条）
        4. 调用动作的run方法执行具体的搜索操作
        5. 创建新的消息对象，包含执行结果、角色信息和动作类型
        6. 将新消息添加到记忆系统，供后续使用
        
        异常处理:
        - 如果记忆系统为空，get_memories(k=1)[0]可能会抛出异常
        - 动作执行失败时，todo.run()可能会返回错误信息
        - 消息创建和添加过程可能会因为系统状态异常而失败
        """
        # 记录当前执行的角色和动作信息，用于调试和监控
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")

        # 获取当前待执行的动作
        todo = self.rc.todo
        
        # 从记忆系统中获取最近的一条消息
        # k=1表示只获取最近的一条消息，确保使用最新的上下文
        msg = self.get_memories(k=1)[0]
        
        # 执行动作并获取结果
        # 将消息内容作为输入传递给动作的run方法
        result = await todo.run(msg.content)

        # 创建新的消息对象，包含执行结果和元数据
        # content: 动作执行的结果内容
        # role: 执行该动作的角色标识
        # cause_by: 产生该结果的动作类型，用于追踪和调试
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        
        # 将新消息添加到记忆系统
        # 这样后续的动作或其他角色就可以访问到这个结果
        self.rc.memory.add(msg)
        
        # 返回新创建的消息对象
        return msg
