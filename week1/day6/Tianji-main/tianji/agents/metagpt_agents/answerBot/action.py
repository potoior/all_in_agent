"""
回答助手Agent的动作模块

该模块定义了AnswerQuestion类，负责根据用户输入和场景信息生成专业的回答。
"""
from dotenv import load_dotenv

load_dotenv()

from metagpt.actions import Action
from tianji.agents.metagpt_agents.utils.json_from import SharedDataSingleton
from tianji.agents.metagpt_agents.utils.agent_llm import OpenaiApi as LLMApi
from tianji.agents.metagpt_agents.utils.helper_func import extract_single_type_attributes_and_examples, extract_attribute_descriptions, load_json
from metagpt.logs import logger


class AnswerQuestion(Action):
    """
    回答问题的动作类
    
    该类负责根据用户的问题、场景信息和搜索结果生成专业的回答。
    它会整合历史对话、场景要素和搜索引擎结果来提供个性化的回复。
    """
    
    # 提示模板，定义了AI助手的角色和行为规范
    PROMPT_TEMPLATE: str = """
    #Role:
    - {scene}小助手

    ## Goals:
    - 作为一个专业的{scene}小助手。你需要参考用户与大模型的历史对话记录（user 表示用户，assistant 表示大模型），场景细化要素，以及每个场景要素的描述以及例子，做出回答，帮助用户解决在该场景下所面对的问题。

    ## Constraints:
    - 你的回答需要基于提供的场景要素进行定制化，提供详细的例子，避免使用概括性或泛泛而谈的表述。

    ## Attention:
    - 场景要素里的详细描述可以参考所提供的场景要素的描述以及例子。
    - 如果搜索引擎结果不为空，不需要使用你的先验知识做出回答，反之你需要完全以搜索引擎结果为基础作为上下文，并做出尽可能详细的回答。

    ## Input:
    - 历史对话记录：```{instruction}```
    - 场景要素: ```{scene_attributes}``
    - 场景要素的描述以及例子:```{scene_attributes_description}```
    - 搜索引擎结果：```{search_result}```
    """
    
    # 动作名称
    name: str = "AnswerQuestion"

    async def run(self, instruction: str):
        """
        执行回答问题的动作
        
        Args:
            instruction (str): 用户的指令或问题
            
        Returns:
            str: AI助手生成的回答
        """
        # 获取共享数据实例
        sharedData = SharedDataSingleton.get_instance()
        scene_label = sharedData.scene_label
        scene_attributes = sharedData.scene_attribute

        # 加载场景属性JSON数据
        json_data = load_json("scene_attribute.json")
        scene, _, _ = extract_single_type_attributes_and_examples(
            json_data, scene_label
        )

        # 提取场景属性描述
        scene_attributes_description = extract_attribute_descriptions(
            json_data, scene_attributes
        )

        # 处理搜索结果
        search_results = sharedData.search_results
        filtered_dict = {}

        # 提取过滤后的内容
        for index, item in search_results.items():
            if "filtered_content" in item:
                filtered_dict[index] = item["filtered_content"]

        logger.info("AnswerQuestion 最后的回复 agent ：scene_attributes scene_attributes_description")
        
        # 构建提示词
        prompt = self.PROMPT_TEMPLATE.format(
            scene=scene,
            scene_attributes=scene_attributes,
            scene_attributes_description=scene_attributes_description,
            instruction=instruction,
            search_result=filtered_dict
            if filtered_dict is not None and filtered_dict
            else "",
        )

        # 调用LLM API生成回答
        rsp = await LLMApi()._aask(prompt=prompt, temperature=0.7)
        return rsp
