"""
场景细化智能体动作模块 - SceneRefinement Agent Actions

功能描述:
本模块定义了场景细化智能体的核心动作，负责分析对话历史、提取场景要素，
并在要素缺失时智能生成提问以收集必要信息。

主要功能:
1. 场景分析: 从历史对话中提取和更新场景要素
2. 智能提问: 基于缺失的场景要素生成针对性问题
3. 要素管理: 维护和更新场景属性状态

核心组件:
- sceneRefineAnalyze: 场景分析动作，提取和更新场景要素
- RaiseQuestion: 提问生成动作，基于缺失要素智能提问

设计特点:
- 动态要素提取: 能够识别对话中场景要素的变化并更新
- 智能提问策略: 只针对缺失的要素提问，避免重复询问
- 多轮对话支持: 支持复杂的对话历史和要素演变
"""

from dotenv import load_dotenv

# 加载环境变量（API密钥等配置）
load_dotenv()

import json
from metagpt.actions import Action
from metagpt.logs import logger
from tianji.agents.metagpt_agents.utils.json_from import SharedDataSingleton
from tianji.agents.metagpt_agents.utils.agent_llm import ZhipuApi as LLMApi
from tianji.agents.metagpt_agents.utils.helper_func import extract_single_type_attributes_and_examples, extract_attribute_descriptions, load_json


class sceneRefineAnalyze(Action):
    """
    场景分析动作类 - 核心场景要素提取和更新
    
    功能说明:
    1. 接收用户与大模型的历史对话记录
    2. 从中提取当前场景所需的场景要素
    3. 识别场景要素的变化并动态更新
    4. 将提取结果存储到共享数据中
    
    设计特点:
    - 动态要素识别: 能够识别对话中要素的实时变化
    - 智能更新机制: 自动替换过时的要素值
    - 多轮对话支持: 处理复杂的对话历史和上下文
    - 结构化输出: 生成标准化的JSON格式结果
    
    工作流程:
    1. 获取共享数据中的场景标签和当前要素
    2. 加载场景属性配置和描述信息
    3. 构建LLM提示词，包含对话历史和要素要求
    4. 调用LLM进行要素提取和分析
    5. 解析和清理LLM响应
    6. 更新共享数据中的场景要素
    """
    
    PROMPT_TEMPLATE: str = """
    #Role:
    - 场景细化小助手

    ## Background:
    - 作为一个专业的{scene}场景分析助手。接下来，我将向你展示一段用户与大模型的历史对话记录，user 表示用户，assistant 表示大模型，你需要从中提取相对应的场景要素并组装成json。

    ## Goals:
    - 我将提供给你需要提取的场景要素，你的任务是从历史对话记录中的内容分析并提取对应场景的场景要素。

    ## Constraints:
    - 你只需要返回单个 json 对象，不需要回复其他任何内容！，不要返回我提供以外的场景要素。
    - 如果没有提取到对应的场景要素请用空字符串表示，例如："对象角色": ""
    - 你需要根据最新的对话记录判断场景要素是否发生改变，如果是，把旧的要素替换成新的（例如"对象角色"从"爸爸"变成"妈妈"）。

    ## Attention:
    - 你可以通过查看我所提供的场景要素的描述以及例子来辅助提取对应场景的场景要素。

    ## Input:
    - 历史对话记录：```{instruction}```
    - 需要提取的场景要素: ```{scene_attributes}``
    - 每个场景要素的描述以及例子:```{scene_attributes_description}```

    ## Workflows:
    ### Step 1: 通过查看场景要素的描述以及例子，思考对话记录中有没有对应的场景要素。
    ### Step 2: 继续查看对话记录以判断已经提取的场景要素有没有发生更新，如果有，就把旧的替换掉。

    ## Example:
    ### example 1:
    #### 历史对话记录：```[user:"我想为我的朋友送上祝福",assistant:"请问你想在哪个节日上送上祝福",user:"我朋友的生日",assistant:"请问你朋友的年龄段是？",user:"18岁",
    assistant:"你想以什么样的语言风格为他送上祝福？",user:"小红书风格"]```
    #### step 1(思考过程): 通过分析每个场景要素的描述以及例子，以及查看要提取的场景要素，分析对话中拥有以下场景要素{{"节日":"生日","对象角色":"朋友","对象年龄段":"少年","语言风格":"小红书"}}。
    #### step 2(思考过程): 已经提取的场景要素并没有发生更新。

    ### example 2:
    #### 历史对话记录：```[user:"我正在相亲，好尴尬",assistant:"请问你目前的聊天场合是？",user:"我和他正在餐厅吃饭",assistant:"请问对象目前的情绪是什么，例如乏累",user:"他看起来有点生气"]```
    #### step 1(思考过程): 通过分析每个场景要素的描述以及例子，以及查看要提取的场景要素，分析对话中拥有以下场景要素{{"语言场景":"餐厅用餐","对象角色":"相亲对象","对象情绪":"乏累","对象状态":"","时间":""}}。
    #### step 2(思考过程): 已经提取的场景要素并没有发生更新。

    ### example 3:
    #### 历史对话记录：```[user:"我想送礼给我的老板",assistant:"请问你老板的性格是怎么样的呢？",user:"开朗",assistant:"好的，正在为你搜寻合适的方法",user:"我说错了，他有点严肃"]```
    #### step 1(思考过程): 通过分析每个场景要素的描述以及例子，以及查看要提取的场景要素，分析对话中拥有以下场景要素{{"对象角色":"老板","对象性格":"开朗","对象职业":""}}。
    #### step 2(思考过程): 已经提取的场景要素发生更新，"对象性格" 从 "开朗" 改变成 "严肃"，返回{{"对象角色":"老板","对象性格":"严肃","对象职业":""}}。
    """

    name: str = "sceneRefineAnalyze"

    async def run(self, instruction: str):
        """
        执行场景分析的主要方法
        
        参数:
            instruction: 用户与大模型的历史对话记录字符串
            
        返回值:
            str: LLM分析后的场景要素JSON字符串
            
        工作流程:
        1. 从共享数据获取当前场景标签和已有的场景要素
        2. 加载场景属性配置文件，获取场景描述和要素定义
        3. 构建包含对话历史、场景信息和要素要求的LLM提示词
        4. 调用LLM API进行场景要素提取和分析
        5. 清理和解析LLM响应，提取JSON格式的场景要素
        6. 更新共享数据中的场景要素状态
        7. 返回分析结果供后续处理
        
        错误处理:
        - 最多重试5次，处理LLM响应格式问题
        - 清理响应中的特殊字符和格式标记
        - 确保最终输出是有效的JSON格式
        """
        # 获取共享数据实例，访问当前用户的场景信息
        sharedData = SharedDataSingleton.get_instance()
        
        # 获取当前场景标签（如"送祝福"、"化解尴尬"等）
        scene_label = sharedData.scene_label
        
        # 获取当前已收集的场景要素（可能包含空值）
        scene_attributes = sharedData.scene_attribute

        # 加载场景属性配置文件，包含所有预定义的场景信息
        json_data = load_json("scene_attribute.json")
        
        # 提取当前场景的具体信息：场景名称、要素列表、示例等
        scene, scene_attributes, _ = extract_single_type_attributes_and_examples(
            json_data, scene_label
        )

        # 提取场景要素的详细描述和示例，用于LLM理解
        scene_attributes_description = extract_attribute_descriptions(
            json_data, scene_attributes
        )

        # 构建LLM提示词，包含所有必要信息
        prompt = self.PROMPT_TEMPLATE.format(
            instruction=instruction,                    # 对话历史记录
            scene=scene,                                 # 场景名称
            scene_attributes=scene_attributes,           # 需要提取的要素列表
            scene_attributes_description=scene_attributes_description,  # 要素描述和示例
        )

        # 调用LLM进行场景分析，最多重试5次处理格式问题
        max_retry = 5
        for attempt in range(max_retry):
            try:
                # 调用LLM API，使用较高温度增加多样性
                rsp = await LLMApi()._aask(prompt=prompt, temperature=1.00)
                logger.info("机器人分析需求：\n" + rsp)
                
                # 清理LLM响应，移除格式标记和特殊字符
                rsp = (
                    rsp.replace("```json", "")  # 移除JSON代码块标记
                    .replace("```", "")        # 移除代码块结束标记
                    .replace("[", "")         # 移除可能的数组括号
                    .replace("]", "")
                    .replace("“", '"')          # 替换中文引号为英文引号
                    .replace("”", '"')
                    .replace("，", ",")       # 替换中文逗号为英文逗号
                )
                
                # 解析JSON响应并更新共享数据
                sharedData.scene_attribute = json.loads(rsp)
                logger.info("机器人分析需求：\n" + rsp)
                return rsp
                
            except Exception as e:
                # 格式解析失败，记录错误并继续重试
                logger.warning(f"第{attempt+1}次尝试失败：{str(e)}")
                pass
                
        # 所有重试都失败，抛出异常
        raise Exception("sceneRefinement agent failed to response after 5 attempts")


class RaiseQuestion(Action):
    """
    提问生成动作类 - 基于缺失场景要素智能生成问题
    
    功能说明:
    1. 分析当前场景要素的完整性
    2. 识别缺失的场景要素（值为空的要素）
    3. 基于缺失要素智能生成针对性提问
    4. 当所有要素都已收集时返回完成标记
    
    设计特点:
    - 智能识别: 自动检测值为空的场景要素
    - 针对性提问: 基于场景和要素描述生成相关问题
    - 单问题策略: 每次只提问一个问题，避免信息过载
    - 完成检测: 所有要素收集完成时返回"Full"标记
    
    工作流程:
    1. 获取当前场景信息和要素状态
    2. 检查哪些场景要素为空值
    3. 基于场景类型和要素描述构建提问提示词
    4. 调用LLM生成针对性问题
    5. 返回生成的提问或完成标记
    
    提问策略:
    - 优先提问最关键的缺失要素
    - 结合场景上下文生成自然的问题
    - 提供具体的例子帮助用户理解
    """
    
    PROMPT_TEMPLATE: str = """
    #Role:
    - 提问小助手

    ## Goals:
    - 作为一个专业的提问小助手。接下来，我将提供你用户面对的场景，场景要素，以及每个场景要素的描述以及例子.
    - 你需要结合当前场景以及为空的场景要素,进行提问的返回
    - 例如,场景是送祝福，空的场景要素为 "对象角色": "" ，此时你才需要提问：请问你想要送祝福给谁呢？是妈妈吗？ 如果不为空,你不需要做任何事情.

    ## Constraints:
    - 如果所有场景要素都有，则不需要提问，直接返回字段"Full"。
    - 你每次只能提问一个问题。
    - 你无需输出思考过程，直接返回提问即可。

    ## Input:
    - 用户面对的场景：```{scene}```
    - 当前场景要素: ```{scene_attributes}``
    - 每个场景要素的描述以及例子:```{scene_attributes_description}```

    ## Workflows:
    ### Step 1: 判断空的场景要素是什么。
    ### Step 2: 结合用户面对的场景以及场景要素的描述以及例子进行提问。

    ## Example:
    ### example 1:
    #### 用户面对的场景："4：送祝福"
    #### 场景要素：```{{"节日":"生日","对象角色":"朋友","对象年龄段":"少年","语言风格":"小红书"}}```
    #### step 1(思考过程): 没有为空的场景要素，返回字段"Full"。

    ### example 2:
    ### 用户面对的场景："6：化解尴尬场合"
    #### 场景要素：```{{"语言场景":"餐厅用餐","对象角色":"相亲对象","对象情绪":"乏累","对象状态":"","时间":""}}```
    #### step 1(思考过程): 空的场景要素为"对象状态":""以及"时间":""。
    #### step 2(思考过程): 结合场景要素的描述以及例子，返回提问："请问目前相亲对象状态是怎么样的？，他也表现得很尴尬吗？，还是？"。

    ### example 3:
    #### 用户面对的场景："3：送礼礼仪文化"
    #### 历史对话记录：```{{"对象角色":"老板","对象性格":"开朗","对象职业":""}}```
    #### step 1(思考过程): 空的场景要素为"对象职业":""。
    #### step 2(思考过程): 结合场景要素的描述以及例子，返回提问："请问您的老板从事哪个行业？，例如科技行业还是教育行业？"
    """
    name: str = "RaiseQuestion"

    async def run(self, instruction: str):
        """
        执行提问生成的主要方法
        
        参数:
            instruction: 用户指令（当前未使用，预留接口）
            
        返回值:
            str: 生成的问题字符串，或"Full"表示所有要素已收集完成
            
        工作流程:
        1. 从共享数据获取当前场景标签和场景要素状态
        2. 加载场景属性配置文件，获取场景描述信息
        3. 提取场景要素的详细描述和示例
        4. 构建包含场景信息和要素状态的LLM提示词
        5. 调用LLM API生成针对性问题或完成标记
        6. 记录和返回生成的提问结果
        
        决策逻辑:
        - 如果有空值要素 → 生成针对性提问
        - 如果所有要素都已填写 → 返回"Full"表示完成
        - 每次只提问一个缺失的要素，确保对话自然流畅
        """
        # 获取共享数据实例，访问当前用户的场景信息
        sharedData = SharedDataSingleton.get_instance()
        
        # 获取当前场景标签（如"送祝福"、"化解尴尬"等）
        scene_label = sharedData.scene_label
        
        # 获取当前场景要素状态（包含已收集和缺失的要素）
        scene_attributes = sharedData.scene_attribute

        # 加载场景属性配置文件，包含所有预定义的场景信息
        json_data = load_json("scene_attribute.json")
        
        # 提取当前场景的具体信息：场景名称、要素列表、示例等
        scene, _, _ = extract_single_type_attributes_and_examples(
            json_data, scene_label
        )

        # 提取场景要素的详细描述和示例，用于LLM理解如何提问
        scene_attributes_description = extract_attribute_descriptions(
            json_data, scene_attributes
        )

        # 构建LLM提示词，包含场景信息和当前要素状态
        prompt = self.PROMPT_TEMPLATE.format(
            scene=scene,                                      # 场景名称
            scene_attributes=scene_attributes,              # 当前要素状态（含空值）
            scene_attributes_description=scene_attributes_description,  # 要素描述和示例
        )
        
        # 调用LLM API生成提问或完成标记，使用较高温度增加多样性
        rsp = await LLMApi()._aask(prompt=prompt, temperature=1.00)
        logger.info("机器人分析需求：\n" + rsp)
        return rsp
