"""
场景细化智能体 (SceneRefinement Role) - 核心功能模块

设计目的:
这个智能体是4个智能体工作流中的第二步，负责细化和完善用户场景信息。
当IntentReg识别出用户场景后，SceneRefinement会分析还需要哪些具体信息。

核心功能:
1. 信息抽取分析 (sceneRefineAnalyze): 
   - 基于用户对话历史，分析当前场景需要哪些要素
   - 参考scene_attribute.json中的预定义场景要素模板
   - 抽取出已知的场景要素，识别缺失的信息

2. 智能提问 (RaiseQuestion):
   - 对于缺失的场景要素，生成合适的提问
   - 以自然对话方式引导用户补充必要信息
   - 确保收集完整的场景信息以便提供精准建议

工作流程:
- 接收IntentReg识别出的场景标签
- 分析该场景需要哪些要素（如预算、关系、对象等）
- 检查已收集的要素是否完整
- 如果要素缺失，通过提问引导用户补充
- 如果要素完整，跳过提问环节，进入AnswerBot环节

依赖组件:
- SharedDataSingleton: 获取共享的用户数据和场景信息
- helper_func: 工具函数，如has_empty_values检查要素完整性
- scene_attribute.json: 预定义的各种场景要素模板
"""

from dotenv import load_dotenv
load_dotenv()

from metagpt.logs import logger
from metagpt.roles.role import Role, RoleReactMode
from metagpt.schema import Message
from .action import sceneRefineAnalyze, RaiseQuestion
from tianji.agents.metagpt_agents.utils.json_from import SharedDataSingleton
from tianji.agents.metagpt_agents.utils.helper_func import *


class SceneRefine(Role):
    """
    场景细化智能体类 - 继承自MetaGPT的Role基类
    
    属性说明:
    - name: 智能体名称，用于标识和日志记录
    - profile: 智能体简介，描述其主要功能
    
    核心能力:
    - 场景信息分析：识别缺失的场景要素
    - 智能提问：引导用户补充必要信息
    - 状态管理：跟踪要素收集进度
    
    工作模式:
    采用REACT模式：_react -> _think -> _act 的循环执行流程
    """
    name: str = "sceneRefinement"
    profile: str = "Scene Refinement Analyze"

    def __init__(self, **kwargs):
        """
        构造函数 - 初始化场景细化智能体
        
        功能:
        1. 设置智能体可执行的动作列表
        2. 配置智能体的工作模式为REACT
        
        动作列表:
        - sceneRefineAnalyze: 分析场景要素完整性
        - RaiseQuestion: 对缺失要素进行提问
        """
        super().__init__(**kwargs)
        # 设置智能体的动作序列：先分析，再提问（如果需要）
        self.set_actions([sceneRefineAnalyze, RaiseQuestion])
        # 配置为REACT模式：思考-行动循环
        self._set_react_mode(react_mode=RoleReactMode.REACT.value)

    async def _act(self) -> Message:
        """
        执行动作方法 - 核心执行逻辑
        
        功能说明:
        1. 记录当前执行的动作类型（日志）
        2. 获取最新的消息内容
        3. 执行当前动作（分析或提问）
        4. 处理执行结果，决定是否返回消息
        
        返回值:
            Message: 如果执行的是提问动作，返回包含问题的消息
                   如果执行的是分析动作，返回空内容消息（继续后续流程）
                   
        工作流程:
        - 如果是RaiseQuestion动作：返回问题给用户
        - 如果是sceneRefineAnalyze动作：返回空消息，继续后续处理
        """
        # 记录日志：显示当前智能体正在执行的动作
        logger.info(f"{self._setting}: to do {self.rc.todo}({self.rc.todo.name})")

        # 获取当前要执行的动作
        todo = self.rc.todo
        # 获取最新的用户消息（用于分析）
        msg = self.get_memories(k=1)[0]
        # 执行动作：分析场景或生成提问
        result = await todo.run(msg.content)

        # 创建回复消息
        msg = Message(content=result, role=self.profile, cause_by=type(todo))
        # 将消息添加到智能体记忆
        self.rc.memory.add(msg)
        
        # 根据动作类型决定返回值
        if type(todo) is RaiseQuestion:
            # 如果是提问动作，返回问题内容给用户
            return msg
        else:
            # 如果是分析动作，返回空内容（继续后续的智能体流程）
            return Message(content="", role=self.profile, cause_by=type(todo))

    async def _think(self) -> None:
        """
        思考方法 - 智能决策逻辑
        
        功能说明:
        1. 检查当前场景要素是否完整收集
        2. 如果要素完整，跳过提问环节，直接结束
        3. 如果要素缺失，继续执行提问动作
        4. 管理智能体的状态转换
        
        决策逻辑:
        - 场景要素完整 → 跳过提问，准备进入AnswerBot环节
        - 场景要素缺失 → 继续提问，补充缺失信息
        
        关键函数:
        - has_empty_values(): 检查字典中是否有空值（缺失的要素）
        - sharedData.scene_attribute: 存储当前已收集的场景要素
        """
        # 获取共享数据实例，访问当前用户的场景信息
        sharedData = SharedDataSingleton.get_instance()
        
        # 关键决策点：检查场景要素是否完整
        if not has_empty_values(sharedData.scene_attribute):
            # 如果所有要素都已收集（没有空值），跳过提问环节
            # 这意味着用户已经提供了足够的信息，可以进入答案生成阶段
            self.rc.todo = None
            return
            
        # 如果有缺失的要素，继续执行下一个动作（通常是提问）
        if self.rc.state + 1 < len(self.states):
            # 还有未执行的动作，继续下一个状态
            self._set_state(self.rc.state + 1)
        else:
            # 所有动作都执行完毕，结束智能体执行
            self.rc.todo = None

    async def _react(self) -> Message:
        """
        反应循环方法 - 智能体的主要执行入口
        
        功能说明:
        1. 实现思考-行动的循环执行模式
        2. 根据_think的决策结果执行相应动作
        3. 直到所有需要执行的动作都完成
        
        执行流程:
        while循环:
        - 调用_think()进行决策（确定下一个动作或结束）
        - 如果rc.todo为None，结束循环
        - 否则调用_act()执行动作
        - 保存动作返回的结果
        
        返回值:
            Message: 最后一个动作返回的消息（通常是提问消息或空消息）
            
        设计特点:
        - 采用状态机模式，根据当前状态决定下一步动作
        - 支持动态决策，根据数据完整性调整执行流程
        - 异步执行，支持并发操作
        """
        # 初始化消息变量，用于存储最后一个动作的结果
        msg = None
        
        # 思考-行动循环：直到没有需要执行的动作
        while True:
            # 思考阶段：决定下一个要执行的动作
            await self._think()
            
            # 决策结果：如果todo为None，表示没有更多动作需要执行
            if self.rc.todo is None:
                break
                
            # 行动阶段：执行当前决定的動作
            msg = await self._act()
            
        # 返回最后一个动作的結果（可能是提问消息或空消息）
        return msg
