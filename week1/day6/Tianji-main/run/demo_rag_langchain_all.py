# =============================================================================
# 天机人情世故大模型系统 - RAG完整版主程序
# =============================================================================
# 功能概述：
# - 基于LangChain + Chroma构建RAG知识库系统
# - 集成SiliconFlow API (LLM + 嵌入模型)
# - 支持6大人情世故场景的知识检索和问答
# - 使用HuggingFace数据集作为知识来源
# - 提供Gradio Web界面交互
# =============================================================================

# === 基础库导入 ===
import os                            # 操作系统接口
import gradio as gr                  # Web界面构建框架
from dotenv import load_dotenv       # 环境变量加载

# === LangChain相关导入 ===
from tianji.knowledges.langchain_onlinellm.models import SiliconFlowEmbeddings, SiliconFlowLLM  # SiliconFlow模型
from langchain_chroma import Chroma                           # 向量数据库
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # 文档加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter         # 文本分割器
from langchain_core.runnables import RunnablePassthrough    # 链式处理组件
from langchain_core.output_parsers import StrOutputParser     # 输出解析器
from langchain import hub                                    # 提示词中心

# === 项目相关导入 ===
from tianji import TIANJI_PATH      # 天机项目路径
import argparse                       # 命令行参数解析
from huggingface_hub import snapshot_download  # HuggingFace模型下载
import requests                       # HTTP请求库
import loguru                        # 日志记录

# 加载环境变量（API密钥等）
load_dotenv()

# =============================================================================
# 命令行参数配置
# =============================================================================
# 参数说明：
# --listen: 监听所有网络接口 (0.0.0.0)
# --port: 服务端口，默认自动选择
# --root_path: 服务根路径，用于反向代理
# --force: 强制重新创建数据库
# --chunk_size: 文本分割块大小，默认896字符
# =============================================================================

parser = argparse.ArgumentParser(description='Launch Gradio application')
parser.add_argument('--listen', action='store_true', help='Specify to listen on 0.0.0.0')
parser.add_argument('--port', type=int, default=None, help='The port the server should listen on')
parser.add_argument('--root_path', type=str, default=None, help='The root path of the server')
parser.add_argument('--force', action='store_true', help='Force recreate the database')
parser.add_argument('--chunk_size', type=int, default=896, help='Chunk size for text splitting')
args = parser.parse_args()

# =============================================================================
# SiliconFlow API 功能测试
# =============================================================================
# 测试目的：确保LLM和嵌入模型API正常工作
# 测试内容：
# 1. LLM聊天功能测试 - 发送"你好"测试响应
# 2. 嵌入模型功能测试 - 文本向量化测试
# 测试结果：成功/失败，失败则抛出异常终止程序
# =============================================================================

# 开始前检查功能是否正常
try:
    llm = SiliconFlowLLM()
    test_response = llm._call("你好")
    loguru.logger.info("SiliconFlow聊天功能测试成功")
except Exception as e:
    loguru.logger.error("SiliconFlow聊天功能测试失败: {}", str(e))
    raise e
try:
    embeddings = SiliconFlowEmbeddings()
    test_text = "测试文本"
    test_embedding = embeddings.embed_query(test_text)
    if len(test_embedding) > 0:
        loguru.logger.info("SiliconFlow嵌入功能测试成功")
    else:
        raise ValueError("嵌入向量长度为0")
except Exception as e:
    loguru.logger.error("SiliconFlow嵌入功能测试失败: {}", str(e))
    raise e

# =============================================================================
# HuggingFace 数据集下载
# =============================================================================
# 数据源：sanbu/tianji-chinese 数据集
# 存储路径：TIANJI_PATH/temp/tianji-chinese/
# 下载策略：
# - 检查网络连接，失败则使用镜像源
# - 最多重试5次
# - 支持断点续传
# =============================================================================

def check_internet_connection(url='http://www.google.com/', timeout=5):
    """检查网络连接状态"""
    try:
        _ = requests.head(url, timeout=timeout)
        return True
    except requests.ConnectionError:
        return False
    
# 设置数据下载目标路径
destination_folder = os.path.join(TIANJI_PATH, "temp", "tianji-chinese")

# 如果数据不存在则下载
if not os.path.exists(destination_folder):
    # 检查网络，如果连接失败则使用国内镜像源
    if not check_internet_connection():
        os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    
    # 重试下载（最多5次）
    for _ in range(5):
        try:
            snapshot_download(
                repo_id="sanbu/tianji-chinese",
                local_dir=destination_folder,
                repo_type="dataset",
                local_dir_use_symlinks=False,
                endpoint=os.environ.get('HF_ENDPOINT', None),
            )
            break
        except Exception as e:
            loguru.logger.error("Download failed, retrying... Error message: {}", str(e))
    else:
        loguru.logger.error("Download failed, maximum retry count reached.")


# =============================================================================
# 向量数据库创建函数
# =============================================================================
# 功能：创建或加载Chroma向量数据库
# 参数：
# - data_path: 文档数据路径
# - persist_directory: 数据库持久化路径
# - embedding_func: 嵌入函数
# - chunk_size: 文本分割块大小
# - force: 是否强制重建数据库
# 返回：Chroma向量数据库实例
# =============================================================================

def create_vectordb(
    data_path: str,
    persist_directory: str,
    embedding_func,
    chunk_size: int,
    force: bool = False,
):
    # 如果数据库已存在且不强制重建，直接加载现有数据库
    if os.path.exists(persist_directory) and not force:
        loguru.logger.info("使用现有的向量数据库: {}", persist_directory)
        return Chroma(
            persist_directory=persist_directory, embedding_function=embedding_func
        )
    
    # 如果强制重建，删除现有数据库
    if force and os.path.exists(persist_directory):
        loguru.logger.info("强制重建向量数据库: {}", persist_directory)
        if os.path.isdir(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)
        else:
            os.remove(persist_directory)
    
    # 加载文档数据（只处理.txt文件）
    loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    
    # 文本分割配置
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=200  # 块大小和重叠度
    )
    
    # 分割文档
    split_docs = text_splitter.split_documents(loader.load())
    
    # 检查分割结果
    if len(split_docs) == 0:
        loguru.logger.error("Invalid knowledge data, processing data results in empty, check if data download failed, can be downloaded manually")
        raise gr.Error("Invalid knowledge data, processing data results in empty, check if data download failed, can be downloaded manually")
    
    # 创建向量数据库
    try:
        vector_db = Chroma.from_documents(
            documents=split_docs,
            embedding=embedding_func,
            persist_directory=persist_directory,
        )
        loguru.logger.info("向量数据库创建成功，文档数量: {}", len(split_docs))
    except Exception as e:
        loguru.logger.error("创建数据库失败: {}", str(e))
        raise e
    
    return vector_db


# =============================================================================
# RAG链初始化函数
# =============================================================================
# 功能：创建完整的RAG处理链
# 流程：
# 1. 创建向量数据库
# 2. 设置检索器
# 3. 配置提示词模板
# 4. 构建处理链
# 参数：
# - chunk_size: 文本分割块大小
# - persist_directory: 数据库持久化路径
# - data_path: 文档数据路径
# - force: 是否强制重建数据库
# 返回：完整的RAG处理链
# =============================================================================

def initialize_chain(chunk_size: int, persist_directory: str, data_path: str, force=False):
    """初始化RAG处理链"""
    loguru.logger.info("初始化数据库开始，当前数据路径为：{}", data_path)
    
    # 创建向量数据库
    vectordb = create_vectordb(data_path, persist_directory, embeddings, chunk_size, force)
    
    # 创建检索器
    retriever = vectordb.as_retriever()
    
    # 获取并自定义RAG提示词模板
    prompt = hub.pull("rlm/rag-prompt")
    prompt.messages[0].prompt.template = """
    您是一名用于问答任务的助手。使用检索到的上下文来回答问题。如果没有高度相关上下文 你就自由回答。\
    根据检索到的上下文，结合我的问题,直接给出最后的回答，要只紧扣问题围绕着回答，尽量根据涉及几个关键点用完整非常详细的几段话回复。。\
    \n问题：{question} \n上下文：{context} \n回答：
    """
    
    loguru.logger.info("初始化数据库结束")
    
    # 构建完整的RAG处理链
    # =============================================================================
    # RAG链式处理流程说明
    # =============================================================================
    # 这是一个LangChain的链式处理结构，使用管道操作符"|"连接各个处理步骤
    # 数据流从左到右，每个步骤的输出作为下一个步骤的输入
    # 
    # 处理流程：
    # 1. 输入准备：创建包含context和question的字典
    # 2. 提示词构建：使用模板组合context和question
    # 3. LLM生成：调用大语言模型生成回答
    # 4. 输出解析：提取并格式化最终回答
    # =============================================================================
    
    return (
        # 步骤1: 输入准备
        # - context: 通过retriever检索相关文档，再经format_docs格式化为字符串
        # - question: 使用RunnablePassthrough()直接传递用户问题
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        
        # 步骤2: 提示词构建
        # - 接收前面的字典（包含context和question）
        # - 使用预定义的RAG提示词模板
        # - 模板会将context和question组合成完整的提示词
        | prompt
        
        # 步骤3: 大语言模型生成
        # - 接收格式化后的提示词
        # - 调用LLM（如GLM-4）生成回答
        # - 模型会结合检索到的上下文和用户问题
        | llm
        
        # 步骤4: 输出解析
        # - 解析模型的输出响应
        # - 提取纯文本内容，去除多余格式
        # - 返回最终的字符串回答
        | StrOutputParser()
    )


# =============================================================================
# 文档格式化函数
# =============================================================================
# 功能：将检索到的文档列表格式化为字符串
# 参数：docs - 文档对象列表
# 返回：格式化的文档内容字符串
# =============================================================================

def format_docs(docs):
    """格式化文档内容"""
    return "\n\n".join(doc.page_content for doc in docs)


# =============================================================================
# 问题处理函数
# =============================================================================
# 功能：处理用户问题并生成回答
# 参数：
# - chain: RAG处理链
# - question: 用户问题
# - chat_history: 聊天历史
# 返回：更新后的输入框内容和聊天历史
# =============================================================================

def handle_question(chain, question: str, chat_history):
    """处理用户问题"""
    # 检查问题是否为空
    if not question:
        return "", chat_history
    
    try:
        # 调用RAG链生成回答
        result = chain.invoke(question)
        # 添加到聊天历史
        chat_history.append((question, result))
        return "", chat_history
    except Exception as e:
        # 错误处理
        loguru.logger.error("处理问题时发生错误: {}", str(e))
        return str(e), chat_history


# =============================================================================
# 场景定义和RAG链初始化
# =============================================================================
# 场景映射：中文场景名称 -> 数据文件夹名称
# 数据路径：TIANJI_PATH/temp/tianji-chinese/RAG/{scenario_folder}/
# 数据库路径：TIANJI_PATH/temp/chromadb_{scenario_folder}/
# =============================================================================

# 定义6大人情世故场景  注意这里没有4
scenarios = {
    "敬酒礼仪文化": "1-etiquette",      # 敬酒相关礼仪知识
    "请客礼仪文化": "2-hospitality",    # 请客相关礼仪知识
    "送礼礼仪文化": "3-gifting",        # 送礼相关礼仪知识
    "如何说对话": "5-communication",     # 沟通技巧知识
    "化解尴尬场合": "6-awkwardness",     # 化解尴尬的方法
    "矛盾&冲突应对": "7-conflict",       # 冲突处理技巧
}

# 为每个场景初始化RAG处理链
chains = {}
for scenario_name, scenario_folder in scenarios.items():
    # 构建数据路径
    data_path = os.path.join(
        TIANJI_PATH, "temp", "tianji-chinese", "RAG", scenario_folder
    )
    
    # 检查数据路径是否存在
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data path does not exist: {data_path}")
    
    # 构建数据库持久化路径
    persist_directory = os.path.join(TIANJI_PATH, "temp", f"chromadb_{scenario_folder}")
    
    # 初始化该场景的RAG链
    chains[scenario_name] = initialize_chain(args.chunk_size, persist_directory, data_path, args.force)
    loguru.logger.info("场景 '{}' 的RAG链初始化完成", scenario_name)

# =============================================================================
# Gradio界面配置
# =============================================================================
# 界面标题和说明
# 功能：
# - 多场景Tab页切换
# - 每个场景独立的知识库和示例
# - 实时聊天交互
# - 一键清除历史记录
# =============================================================================

# 界面标题
TITLE = """
# Tianji 人情世故大模型系统完整版(基于知识库实现) 欢迎star！\n
## 💫开源项目地址：https://github.com/SocialAI-tianji/Tianji
## 使用方法：选择你想提问的场景，输入提示，或点击Example自动填充
## 如果觉得回答不满意,可补充更多信息重复提问。
### 我们的愿景是构建一个从数据收集开始的大模型全栈垂直领域开源实践.
"""


# =============================================================================
# 获取场景示例问题函数
# =============================================================================
# 功能：为每个场景提供预设的示例问题
# 参数：scenario - 场景名称
# 返回：该场景的示例问题列表
# 说明：每个场景都有针对性的示例问题，帮助用户快速理解如何使用系统
# =============================================================================

def get_examples_for_scenario(scenario):
    # 定义各场景的示例问题
    examples_dict = {
        "敬酒礼仪文化": [
            "喝酒座位怎么排",
            "喝酒的先后顺序流程是什么",
            "喝酒需要注意什么",
            "推荐的敬酒词怎么说",
            "宴会怎么点菜",
            "喝酒容易醉怎么办",
            "喝酒的规矩是什么",
        ],
        "请客礼仪文化": ["请客有那些规矩", "如何选择合适的餐厅", "怎么请别人吃饭"],
        "送礼礼仪文化": ["送什么礼物给长辈好", "怎么送礼", "回礼的礼节是什么"],
        "如何说对话": [
            "怎么和导师沟通",
            "怎么提高情商",
            "如何读懂潜台词",
            "怎么安慰别人",
            "怎么和孩子沟通",
            "如何与男生聊天",
            "如何与女生聊天",
            "职场高情商回应技巧",
        ],
        "化解尴尬场合": ["怎么回应赞美", "怎么拒绝借钱", "如何高效沟通", "怎么和对象沟通", "聊天技巧", "怎么拒绝别人", "职场怎么沟通"],
        "矛盾&冲突应对": [
            "怎么控制情绪",
            "怎么向别人道歉",
            "和别人吵架了怎么办",
            "如何化解尴尬",
            "孩子有情绪怎么办",
            "夫妻吵架怎么办",
            "情侣冷战怎么办",
        ],
    }
    return examples_dict.get(scenario, [])


# =============================================================================
# Gradio界面创建和配置
# =============================================================================
# 功能：创建完整的Gradio界面
# 结构：
# 1. 标题和说明
# 2. 初始化状态显示
# 3. Tab页布局（每个场景一个Tab）
# 4. 每个Tab包含：
#    - 聊天窗口
#    - 输入框
#    - 示例问题
#    - 聊天按钮和清除按钮
# =============================================================================

with gr.Blocks() as demo:
    gr.Markdown(TITLE)

    # 显示初始化状态
    init_status = gr.Textbox(label="初始化状态", value="数据库已初始化", interactive=False)

    # 创建Tab页布局
    with gr.Tabs() as tabs:
        # 为每个场景创建一个Tab页
        for scenario_name in scenarios.keys():
            with gr.Tab(scenario_name):
                # 聊天窗口，高度450px，支持复制按钮
                chatbot = gr.Chatbot(height=450, show_copy_button=True)
                # 用户输入框
                msg = gr.Textbox(label="输入你的疑问")

                # 示例问题组件
                examples = gr.Examples(
                    label="快速示例",
                    examples=get_examples_for_scenario(scenario_name),
                    inputs=[msg],
                )

                # 按钮行布局
                with gr.Row():
                    # 聊天按钮
                    chat_button = gr.Button("聊天")
                    # 清除聊天记录按钮
                    clear_button = gr.ClearButton(components=[chatbot], value="清除聊天记录")

                # 定义调用RAG链的函数
                # 注意：使用默认参数scenario_name捕获当前循环的场景名称
                def invoke_chain(question, chat_history, scenario=scenario_name):
                    """调用指定场景的RAG链处理用户问题"""
                    loguru.logger.info(question)  # 记录用户问题
                    return handle_question(chains[scenario], question, chat_history)

                # 绑定聊天按钮点击事件
                chat_button.click(
                    invoke_chain,
                    inputs=[msg, chatbot],
                    outputs=[msg, chatbot],
                )


# =============================================================================
# 主函数入口
# =============================================================================
# 功能：启动Gradio应用
# 配置：
# - server_name: 监听地址（0.0.0.0表示允许外部访问）
# - server_port: 服务端口
# - root_path: 根路径（用于反向代理）
# =============================================================================

if __name__ == "__main__":
    # 根据listen参数设置监听地址
    server_name = '0.0.0.0' if args.listen else None
    server_port = args.port
    
    # 启动Gradio应用
    demo.launch(server_name=server_name, server_port=server_port, root_path=args.root_path)
