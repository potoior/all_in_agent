# =============================================================================
# 天机人情世故大模型系统 - Prompt版本主程序
# =============================================================================
# 功能概述：
# - 基于Gradio构建Web界面
# - 集成智谱AI GLM-4模型
# - 提供7大人情世故场景的智能对话
# - 支持场景切换、示例选择、对话历史管理
# =============================================================================

# === 基础库导入 ===
import gradio as gr                 # Web界面构建框架
import json                        # JSON数据处理
import random                      # 随机选择功能
from dotenv import load_dotenv     # 环境变量加载
import argparse                    # 命令行参数解析

# === 加载环境变量 (.env文件) ===
load_dotenv()

# === AI模型相关导入 ===
from zhipuai import ZhipuAI        # 智谱AI SDK
import os                          # 系统操作
from tianji import TIANJI_PATH     # 天机项目路径

# =============================================================================
# 命令行参数配置
# =============================================================================
# 支持自定义部署参数：
# --listen: 监听所有网络接口 (0.0.0.0)
# --port: 指定服务端口
# --root_path: 设置服务根路径
# =============================================================================
parser = argparse.ArgumentParser(description='Launch Gradio application')
parser.add_argument('--listen', action='store_true', help='Specify to listen on 0.0.0.0')
parser.add_argument('--port', type=int, default=None, help='The port the server should listen on')
parser.add_argument('--root_path', type=str, default=None, help='The root path of the server')
args = parser.parse_args()

# =============================================================================
# 全局配置和初始化
# =============================================================================
# 提示词模板文件路径 - 包含所有场景的AI提示词配置
file_path = os.path.join(TIANJI_PATH, "tianji/prompt/yiyan_prompt/all_yiyan_prompt.json")
# d:\Desktop\test\all_in_Agent\week1\day6\Tianji-main\tianji\prompt\yiyan_prompt\all_yiyan_prompt.json
# 智谱AI API密钥 (从环境变量获取)
API_KEY = os.environ["ZHIPUAI_API_KEY"]

# 七大核心场景分类 (对应ID: 1-7)
CHOICES = ["敬酒", "请客", "送礼", "送祝福", "人际交流", "化解尴尬", "矛盾应对"]

# 加载提示词模板数据
with open(file_path, "r", encoding="utf-8") as file:
    json_data = json.load(file)


# =============================================================================
# 核心功能函数
# =============================================================================

def get_names_by_id(id):
    """
    根据场景ID获取对应的所有子场景名称
    
    比如：如何委婉地表达自己对婚姻的想法，当被父母直接催婚时怎么回应等等

    Args:
        id: 场景分类ID (1-7)
    
    Returns:
        list: 该分类下的所有子场景名称（去重后）
    """
    names = []
    for item in json_data:
        if "id" in item and item["id"] == id:
            names.append(item["name"])

    return list(set(names))  # Remove duplicates


def get_system_prompt_by_name(name):
    """
    根据场景名称获取对应的系统提示词
    
    系统提示词定义了AI在该场景下的行为模式和回答风格
    
    Args:
        name: 场景名称
    
    Returns:
        str: 系统提示词内容，如果未找到返回None
    """
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    
    for item in data:
        if item["name"] == name:
            return item["system_prompt"]
    return None  # If the name is not found


def change_example(name, cls_choose_value, chatbot):
    """
    切换场景时更新示例数据集并清空聊天历史
    
    Args:
        name: 新选择的场景名称
        cls_choose_value: 当前分类的所有场景数据
        chatbot: 聊天机器人组件
    
    Returns:
        tuple: 更新的示例数据集和清空的聊天历史

   --------------------------
    cls_choose_value例子：
        cls_choose_value = [
        {
            "name": "如何委婉地表达自己对婚姻的想法",
            "example": [{"input": "称谓：阿姨...", "output": "“阿姨，我明白您对我的关心..."}],
            "system_prompt": "你现在是一个大约30岁的单身{我的性别}青年..."
        },
        {
            "name": "当被父母直接催婚时怎么回应", 
            "example": [{"input": "我的性别：男...", "output": "“爸妈，我明白你们的心意..."}],
            "system_prompt": "你现在是一个大约30岁的单身{我的性别}青年..."
        },
        {
            "name": "当祖父母以传统观念催婚时如何沟通",
            "example": [{"input": "回应时长：50s内...", "output": "爷爷奶奶，我知道你们很期待..."}],
            "system_prompt": "你现在是一个大约30岁的单身{我的性别}青年..."
        }
        # ... 更多催婚相关的场景
    ]
    """
    now_example = []
    
    # 清空聊天历史，避免不同场景间的上下文混淆
    if chatbot is not None:
        print("切换场景清理bot历史")
        chatbot.clear()
    
    # 从当前分类数据中找到匹配的场景示例
    for i in cls_choose_value:
        if i["name"] == name:
            now_example = [[j["input"], j["output"]] for j in i["example"]]
    """
    # 转换前：
        i["example"] = [
            {"input": "称谓：阿姨...", "output": "阿姨，我明白..."},
            {"input": "称谓：叔叔...", "output": "叔叔，我知道..."}
        ]

        # 转换后：
        now_example = [
            ["称谓：阿姨...", "阿姨，我明白..."],
            ["称谓：叔叔...", "叔叔，我知道..."]
        ]
    """
    
    if now_example is []:
        raise gr.Error("获取example出错！")
    
    return gr.update(samples=now_example), chat_history


def random_button_click(chatbot):
    """
    随机选择一个场景的功能
    
    为用户提供随机探索不同人情世故场景的功能
    
    Args:
        chatbot: 聊天机器人组件
    
    Returns:
        tuple: (选择的分类名称, 对应分类数据, 更新的下拉菜单)
    """
    # 随机选择0-6之间的数字，对应7个场景分类
    choice_number = random.randint(0, 6)
    now_id = choice_number + 1  # ID从1开始
    cls_choose = CHOICES[choice_number]
    now_json_data = _get_id_json_id(choice_number)
    random_name = [i["name"] for i in now_json_data]
    
    # 清空聊天历史
    if chatbot is not None:
        print("切换场景清理bot历史")
        chatbot.clear()
    
    return (
        cls_choose,  # 更新单选按钮选择
        now_json_data,  # 更新当前分类数据
        gr.update(choices=get_names_by_id(now_id), value=random.choice(random_name)),  # 更新下拉菜单
    )


def example_click(dataset, name, now_json):
    system = ""
    for i in now_json:
        if i["name"] == name:
            system = i["system_prompt"]

    if system_prompt == "":
        print(name, now_json)
        raise "遇到代码问题，清重新选择场景"
    return dataset[0], system


def _get_id_json_id(idx):
    now_id = idx + 1  # index + 1
    now_id_json_data = []
    for item in json_data:
        if int(item["id"]) == int(now_id):
            temp_dict = dict(
                name=item["name"],
                example=item["example"],
                system_prompt=item["system_prompt"],
            )
            now_id_json_data.append(temp_dict)
    return now_id_json_data


def cls_choose_change(idx):
    now_id = idx + 1
    return _get_id_json_id(idx), gr.update(
        choices=get_names_by_id(now_id), value=get_names_by_id(now_id)[0]
    )


def combine_message_and_history(message, chat_history):
    # 将聊天历史中的每个元素（假设是元组）转换为字符串
    history_str = "\n".join(f"{sender}: {text}" for sender, text in chat_history)

    # 将新消息和聊天历史结合成一个字符串
    full_message = f"{history_str}\nUser: {message}"
    return full_message


def respond(system_prompt, message, chat_history):
    """
    核心对话函数 - 处理用户输入并生成AI回复
    
    这是整个应用的核心，负责：
    1. 管理对话历史长度
    2. 构建包含历史的提示词
    3. 调用智谱AI API
    4. 更新对话历史
    
    Args:
        system_prompt: 系统提示词（定义AI行为）
        message: 用户输入消息
        chat_history: 对话历史记录
    
    Returns:
        tuple: (清空的消息输入框, 更新的对话历史)
    """
    # 防止对话历史过长（超过11轮对话后重新开始）
    if len(chat_history) > 11:
        chat_history.clear()  # 清空聊天历史
        chat_history.append(["请注意", "对话超过 已重新开始"])
    
    # 合并当前消息和历史对话，提供上下文
    message1 = combine_message_and_history(message, chat_history)
    print(f"发送给AI的完整消息: {message1}")
    
    # 创建智谱AI客户端并发送请求
    client = ZhipuAI(api_key=API_KEY)
    response = client.chat.completions.create(
        model="glm-4-flash",  # 使用GLM-4-FLASH模型（快速版）
        messages=[
            {"role": "system", "content": system_prompt},  # 系统提示词
            {"role": "user", "content": message1},       # 用户消息（含历史）
        ],
    )

    # 提取AI回复内容
    bot_message_text = response.choices[0].message.content
    
    # 更新对话历史（用户消息 -> AI回复）
    chat_history.append([message, bot_message_text])

    return "", chat_history  # 清空输入框，返回更新后的历史


def clear_history(chat_history):
    chat_history.clear()
    return chat_history


def regenerate(chat_history, system_prompt):
    if chat_history:
        # 提取上一条输入消息
        last_message = chat_history[-1][0]
        # 移除最后一条记录
        chat_history.pop()
        # 使用上一条输入消息调用 respond 函数以生成新的回复
        msg, chat_history = respond(system_prompt, last_message, chat_history)
    # 返回更新后的聊天记录
    return msg, chat_history


# =============================================================================
# Gradio界面构建
# =============================================================================

TITLE = """
# Tianji 人情世故大模型系统——prompt版 欢迎star！\n
## 💫开源项目地址：https://github.com/SocialAI-tianji/Tianji
### 我们的愿景是构建一个从数据收集开始的大模型全栈垂直领域开源实践。\n
## 我们支持不同模型进行对话，你可以选择你喜欢的模型进行对话。
## 使用方法：选择或随机一个场景，输入提示词（或者点击上面的Example自动填充），随后发送！
"""

# =============================================================================
# 主界面构建 - 使用Gradio Blocks API
# =============================================================================
# 界面布局说明：
# 1. 左侧：场景选择和系统提示词显示
# 2. 右侧：聊天界面和控制按钮
# 3. 事件绑定：处理用户交互
# =============================================================================

with gr.Blocks() as demo:
    # === 状态变量定义 ===
    chat_history = gr.State()           # 存储对话历史
    now_json_data = gr.State(value=_get_id_json_id(0))  # 当前分类数据
    now_name = gr.State()               # 当前场景名称
    
    # === 标题显示 ===
    gr.Markdown(TITLE)
    
    # === 场景分类选择 (单选按钮) ===
    cls_choose = gr.Radio(
        label="请选择任务大类", 
        choices=CHOICES, 
        type="index", 
        value="敬酒"  # 默认选择"敬酒"
    )
    
    # === 示例数据集 (显示当前场景的示例对话) ===
    input_example = gr.Dataset(
        components=["text", "text"],
        samples=[
            ["请先选择合适的场景", "请先选择合适的场景"],
        ],
        label="示例对话"
    )
    
    # === 主界面布局 (左右分栏) ===
    with gr.Row():
        # === 左侧控制面板 ===
        with gr.Column(scale=1):
            # 子场景选择下拉菜单
            dorpdown_name = gr.Dropdown(
                choices=get_names_by_id(1),
                label="场景",
                info="请选择合适的场景",
                interactive=True,
            )
            
            # 系统提示词显示区域
            system_prompt = gr.TextArea(
                label="系统提示词", 
                placeholder="选择场景后这里会显示对应的AI提示词"
            )
            
            # 随机选择按钮
            random_button = gr.Button("🪄点我随机一个试试！", size="lg")
            
            # 绑定下拉菜单变化事件
            dorpdown_name.change(
                fn=get_system_prompt_by_name,
                inputs=[dorpdown_name],
                outputs=[system_prompt],
            )
        
        # === 右侧聊天区域 ===
        with gr.Column(scale=4):
            # 聊天机器人组件
            chatbot = gr.Chatbot(
                label="聊天界面", 
                value=[
                    ["如果喜欢，请给我们一个⭐，谢谢", "不知道选哪个？试试点击随机按钮把！"]
                ],
                height=400  # 设置聊天区域高度
            )
            
            # 用户输入框
            msg = gr.Textbox(
                label="输入信息",
                placeholder="在这里输入你的问题...",
                lines=3  # 多行输入
            )
            
            # 消息发送事件 (支持回车发送)
            msg.submit(
                respond, inputs=[system_prompt, msg, chatbot], outputs=[msg, chatbot]
            )
            
            # 发送按钮
            submit = gr.Button("发送", variant="primary").click(
                respond, inputs=[system_prompt, msg, chatbot], outputs=[msg, chatbot]
            )
            
            # === 控制按钮行 ===
            with gr.Row():
                # 清除历史按钮
                clear = gr.Button("清除历史记录").click(
                    clear_history, inputs=[chatbot], outputs=[chatbot]
                )
                # 重新生成按钮
                regenerate = gr.Button("重新生成").click(
                    regenerate, inputs=[chatbot, system_prompt], outputs=[msg, chatbot]
                )

    # === 事件绑定配置 ===
    
    # 1. 分类切换事件
    cls_choose.change(
        fn=cls_choose_change, 
        inputs=cls_choose, 
        outputs=[now_json_data, dorpdown_name]
    )
    
    # 2. 场景切换事件
    dorpdown_name.change(
        fn=change_example,
        inputs=[dorpdown_name, now_json_data, chatbot],
        outputs=[input_example, chat_history],
    )
    
    # 3. 示例点击事件
    input_example.click(
        fn=example_click,
        inputs=[input_example, dorpdown_name, now_json_data],
        outputs=[msg, system_prompt],
    )
    
    # 4. 随机按钮点击事件
    random_button.click(
        fn=random_button_click,
        inputs=chatbot,
        outputs=[cls_choose, now_json_data, dorpdown_name],
    )

# =============================================================================
# 应用启动配置
# =============================================================================
# 启动参数说明：
# - server_name: None表示localhost，0.0.0.0表示监听所有网络接口
# - server_port: 服务端口，None表示Gradio自动选择
# - root_path: 服务根路径，用于反向代理部署
# =============================================================================

if __name__ == "__main__":
    # 根据命令行参数配置服务器
    server_name = '0.0.0.0' if args.listen else None  # 是否监听所有网络接口
    server_port = args.port                           # 自定义端口
    root_path = args.root_path                        # 根路径（用于部署）
    
    # 启动Gradio应用
    demo.launch(
        server_name=server_name,
        server_port=server_port, 
        root_path=root_path,
        share=False,  # 不创建公共链接
        show_error=True,  # 显示错误信息
    )
