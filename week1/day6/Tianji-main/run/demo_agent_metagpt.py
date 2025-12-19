"""
Tianji多智能体演示程序

该程序实现了一个基于Streamlit的人情世故问答系统，集成了多个AI Agent：
- 意图识别Agent：识别用户问题的场景类型
- 场景细化Agent：收集必要的场景要素
- 回答助手Agent：生成专业回答
- 搜索助手Agent：提供网络搜索支持
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio
import streamlit as st
import uuid
from streamlit_chat import message
from metagpt.logs import logger
import os

import sys
module_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  #当前文件夹路径
sys.path.insert(0, module_dir)

from tianji.agents.metagpt_agents.intentRecognition import IntentReg
from tianji.agents.metagpt_agents.answerBot import AnswerBot
from tianji.agents.metagpt_agents.sceneRefinement import SceneRefine
from tianji.agents.metagpt_agents.searcher import Searcher
from tianji.agents.metagpt_agents.utils.json_from import SharedDataSingleton
from tianji.agents.metagpt_agents.utils.helper_func import has_empty_values, is_number_in_types, timestamp_str, extract_single_type_attributes_and_examples, load_json, extract_all_types
from tianji.agents.metagpt_agents.utils.agent_llm import OpenaiApi as LLMApi
import time


def initialize_session():
    """
    初始化Streamlit会话状态
    
    为新用户生成唯一ID，初始化聊天历史等状态变量。
    """
    if "user_id" not in st.session_state:
        # 为新用户会话生成一个唯一的UUID
        logger.log(0, "add uuid")
        st.session_state["user_id"] = str(uuid.uuid4())


def on_btn_click(sharedData):
    """
    清除聊天历史的回调函数
    
    Args:
        sharedData: 共享数据实例
    """
    # 清除所有共享数据
    sharedData.message_list_for_agent.clear()
    sharedData.chat_history.clear()
    sharedData.scene_label = ""
    sharedData.scene_attribute = {}
    sharedData.extra_query.clear()
    sharedData.search_results = {}
    
    # 清除会话状态
    st.session_state["generated"].clear()
    st.session_state["past"].clear()
    st.session_state["scene_label"] = ""
    st.session_state["scene_attr"] = {}


def flip():
    """
    切换网络搜索启用状态的回调函数
    """
    if st.session_state["check"]:
        st.session_state["enable_se"] = True
    else:
        st.session_state["enable_se"] = False


def initialize_sidebar(scenes, sharedData):
    """
    初始化侧边栏界面
    
    Args:
        scenes: 支持的场景列表
        sharedData: 共享数据实例
    """
    with st.sidebar:
        st.markdown("我是由人情世故大模型团队开发的多智能体应用，专注于理解您的意图并进一步提问，以提供精准答案。目前，我支持以下场景：")
        container_all_scenes = st.container(border=True)
        for item in scenes:
            container_all_scenes.write(item)
        st.markdown("用户当前意图：")
        container_current_scene = st.container(border=True)
        container_current_scene.write(st.session_state["scene_label"])
        st.markdown("当前场景要素：")
        container_scene_attribute = st.container(border=True)
        container_scene_attribute.write(st.session_state["scene_attr"])
        st.button("Clear Chat History", on_click=lambda: on_btn_click(sharedData))
        st.checkbox(
            "启用网络搜索（确保填写密钥）", value=st.session_state["enable_se"], key="check", on_change=flip
        )


async def process_user_input(user_input, sharedData, json_data, roles):
    """
    处理用户输入的主逻辑
    
    Args:
        user_input: 用户输入
        sharedData: 共享数据实例
        json_data: 场景属性JSON数据
        roles: 包含所有Agent角色的字典
        
    Returns:
        bool: 是否成功处理
    """
    # 添加用户消息到历史记录
    st.session_state["past"].append(user_input)
    sharedData.message_list_for_agent.append({"user": user_input})
    sharedData.chat_history.append({
        "message": user_input,
        "is_user": True,
        "keyname": "user" + str(timestamp_str()),
    })
    
    # 运行意图识别Agent
    # 这一步是整个多智能体系统的第一步：理解用户想要什么样的帮助
    # IntentReg智能体分析用户输入，判断属于哪种社交场景
    # 
    # 参数说明:
    # - roles["intent"]: 获取IntentReg意图识别智能体实例
    # - sharedData.message_list_for_agent: 包含用户当前和历史消息的对话记录
    # - str(...): 将消息列表转换为字符串格式供智能体处理
    # - await: 异步等待智能体处理完成（可能需要调用LLM API）
    # - .content: 提取智能体返回的核心内容（场景标签）
    #
    # 返回值说明:
    # intent_ans 可能的值:
    # - "生日祝福": 用户想准备生日礼物或祝福
    # - "节日问候": 节日相关的拜访或问候  
    # - "探病慰问": 生病探望相关的礼仪
    # - "结婚祝贺": 婚礼或结婚相关的祝贺
    # - "生育祝贺": 生孩子相关的祝贺
    # - "搬家祝贺": 乔迁之喜的祝贺
    # - "None": 不属于人情世故的场景，无法处理
    #
    # 工作流程:
    # 1. IntentReg智能体接收用户对话历史
    # 2. 分析关键词和语义（如"妈妈"、"生日"、"礼物"）
    # 3. 匹配预定义的场景模板
    # 4. 返回最匹配的场景标签或None
    intent_ans = (await roles["intent"].run(str(sharedData.message_list_for_agent))).content
    
    # 处理不支持的场景
    if intent_ans == "None":
        st.warning("此模型只支持回答关于人情世故的事项，已调用 API 为你进行单轮回答。")
        rsp = await LLMApi()._aask(prompt=user_input)
        sharedData.message_list_for_agent.clear()
        st.session_state["generated"].append(rsp)
        sharedData.chat_history.append({
            "message": rsp,
            "is_user": False,
            "keyname": "assistant" + str(timestamp_str()),
        })
        return True
    
    # 处理未知的场景标签
    elif not is_number_in_types(json_data, int(intent_ans)):
        st.warning("模型发生幻觉，请重新提问")
        sharedData.message_list_for_agent.clear()
        time.sleep(3)
        return True
    
    # 处理有效的场景
    else:
        return await handle_valid_intent(intent_ans, sharedData, json_data, roles)


async def handle_valid_intent(intent_ans, sharedData, json_data, roles):
    """
    处理有效的场景意图
    
    Args:
        intent_ans: 意图识别结果
        sharedData: 共享数据实例
        json_data: 场景属性JSON数据
        roles: 包含所有Agent角色的字典
        
    Returns:
        bool: 是否成功处理
    """
    # 确认用户意图后，更新场景标签
    if not sharedData.scene_label or sharedData.scene_label != intent_ans:
        sharedData.scene_label = intent_ans
        st.session_state["scene_label"] = sharedData.scene_label
        # 提取对应场景所需要的场景要素
        _, scene_attributes, _ = extract_single_type_attributes_and_examples(
            json_data, sharedData.scene_label
        )
        sharedData.scene_attribute = {attr: "" for attr in scene_attributes}
    
    # 运行场景细化Agent
    refine_ans = (await roles["scene"].run(str(sharedData.message_list_for_agent))).content
    st.session_state["scene_attr"] = sharedData.scene_attribute
    
    # 用户提供的场景要素不全，场景细化Agent进行提问
    if refine_ans != "":
        st.session_state["generated"].append(refine_ans)
        sharedData.message_list_for_agent.append({"assistant": refine_ans})
        sharedData.chat_history.append({
            "message": refine_ans,
            "is_user": False,
            "keyname": "assistant" + str(timestamp_str()),
        })
        return True
    
    # 用户提供的场景要素齐全，运行回答助手Agent
    if not has_empty_values(sharedData.scene_attribute):
        await generate_final_answer(sharedData, roles)
        return True
    
    return False


async def generate_final_answer(sharedData, roles):
    """
    生成最终回答
    
    Args:
        sharedData: 共享数据实例
        roles: 包含所有Agent角色的字典
    """
    # 运行回答助手Agent
    final_ans = (await roles["answer"].run(str(sharedData.message_list_for_agent))).content
    st.session_state["generated"].append(final_ans)
    sharedData.chat_history.append({
        "message": final_ans,
        "is_user": False,
        "keyname": "assistant" + str(timestamp_str()),
    })
    
    # 如果开启网络搜索助手Agent，运行搜索Agent
    if st.session_state["enable_se"] is True:
        await handle_search_functionality(sharedData, roles)
    
    # 回答完成，清除所有Agent环境中的数据
    cleanup_shared_data(sharedData)


async def handle_search_functionality(sharedData, roles):
    """
    处理网络搜索功能
    
    Args:
        sharedData: 共享数据实例
        roles: 包含所有Agent角色的字典
    """
    with st.spinner("启用搜索引擎，请稍等片刻... 如有报错，请检查密钥是否填写正确"):
        await roles["search"].run(str(sharedData.message_list_for_agent))
    
    # 显示生成的额外查询
    sa_res1 = "生成的额外查询：" + str(sharedData.extra_query)
    st.session_state["generated"].append(sa_res1)
    sharedData.chat_history.append({
        "message": sa_res1,
        "is_user": False,
        "keyname": "assistant" + str(timestamp_str()),
    })
    time.sleep(0.01)
    
    # 显示网页网址
    urls = []
    for item in sharedData.search_results.values():
        if "url" in item:
            urls.append(item["url"])
    urls = " ".join(urls)
    sa_res2 = "搜索引擎返回的网页为：\n" + urls
    st.session_state["generated"].append(sa_res2)
    sharedData.chat_history.append({
        "message": sa_res2,
        "is_user": False,
        "keyname": "assistant" + str(timestamp_str()),
    })
    time.sleep(0.01)
    
    # 显示需要进一步查询的网页
    sa_res3 = "判断需要进一步查询的网页为" + str(sharedData.filter_weblist)
    st.session_state["generated"].append(sa_res3)
    sharedData.chat_history.append({
        "message": sa_res3,
        "is_user": False,
        "keyname": "assistant" + str(timestamp_str()),
    })
    time.sleep(0.01)
    
    # 基于搜索结果再次运行回答助手Agent
    final_ans_sa = (await roles["answer"].run(str(sharedData.message_list_for_agent))).content
    final_ans_sa = "基于搜素引擎的回答：" + final_ans_sa
    st.session_state["generated"].append(final_ans_sa)
    sharedData.chat_history.append({
        "message": final_ans_sa,
        "is_user": False,
        "keyname": "assistant" + str(timestamp_str()),
    })


def cleanup_shared_data(sharedData):
    """
    清理共享数据
    
    Args:
        sharedData: 共享数据实例
    """
    sharedData.message_list_for_agent.clear()
    sharedData.scene_label = ""
    sharedData.scene_attribute = {}
    sharedData.extra_query.clear()
    sharedData.search_results = {}


async def main():
    """
    主函数：运行Streamlit应用
    """
    # 初始化会话
    initialize_session()
    
    # 初始化Agent角色
    roles = {
        "intent": IntentReg(),
        "scene": SceneRefine(),
        "answer": AnswerBot(),
        "search": Searcher()
    }
    
    # 显示会话ID和标题
    st.write(f"您的会话ID是: {st.session_state['user_id']}")
    st.title("人情世故大模型")
    
    # 加载场景属性数据
    json_data = load_json("scene_attribute.json")
    
    # 初始化会话状态变量
    if "generated" not in st.session_state:
        st.session_state["generated"] = []
    if "past" not in st.session_state:
        st.session_state["past"] = []
    if "enable_se" not in st.session_state:
        st.session_state["enable_se"] = False
    if "scene_label" not in st.session_state:
        st.session_state["scene_label"] = ""
    if "scene_attr" not in st.session_state:
        st.session_state["scene_attr"] = {}
    
    # 获取共享数据实例
    sharedData = SharedDataSingleton.get_instance()
    
    # 初始化侧边栏
    initialize_sidebar(extract_all_types(json_data), sharedData)
    
    # 显示历史对话记录
    for first_status_message in sharedData.chat_history:
        message(
            first_status_message["message"],
            is_user=first_status_message["is_user"],
            key=first_status_message["keyname"],
        )
    
    # 处理用户输入
    if user_input := st.chat_input():
        success = await process_user_input(user_input, sharedData, json_data, roles)
        if success:
            st.rerun()


# 运行主函数
if __name__ == "__main__":
    asyncio.run(main())
