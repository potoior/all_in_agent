"""
回答助手Agent角色模块

该模块定义了AnswerBot类，作为人情世故问答系统的核心角色，
负责基于用户对话历史、场景信息和搜索结果生成专业回答。
"""
from dotenv import load_dotenv

load_dotenv()

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from .action import AnswerQuestion


class AnswerBot(Role):
    """
    回答机器人角色类
    
    该角色负责处理用户关于人情世故的问题，基于对话历史、
    场景标签、场景要素描述和搜索结果生成专业回答。
    """
    
    # 角色名称
    name: str = "Answer Bot"
    
    # 角色描述
    profile: str = "Answer User question"

    def __init__(self, **kwargs):
        """
        初始化回答机器人
        
        Args:
            **kwargs: 传递给父类的参数
        """
        super().__init__(**kwargs)
        # 设置可用的动作
        self.set_actions([AnswerQuestion])
        # 设置反应模式为按顺序执行
        self._set_react_mode(react_mode=RoleReactMode.BY_ORDER.value)

    async def _act(self) -> Message:
        """
        执行回答动作
        
        获取最新的用户消息，调用AnswerQuestion动作生成回答，
        并将结果存储到记忆中。
        
        Returns:
            Message: 包含回答内容的消息对象
        """
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")

        # 获取当前待执行的动作
        todo = self.rc.todo
        
        # 获取最新的用户消息
        msg = self.get_memories(k=1)[0]
        
        # 执行回答动作
        result = await todo.run(msg.content)

        # 创建回复消息
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        
        # 将回复添加到记忆中
        self.rc.memory.add(msg)
        return msg
