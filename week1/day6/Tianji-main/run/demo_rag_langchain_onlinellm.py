# =============================================================================
# 可配置RAG演示系统 - 在线LLM版本
# =============================================================================
# 功能：提供一个可配置的RAG（检索增强生成）系统演示
# 特点：
# - 支持多种嵌入模型选择（HuggingFace和智谱AI）
# - 支持多种数据源（本地文件夹和网页）
# - 提供交互式Gradio界面
# - 使用智谱AI的GLM-4作为语言模型
# =============================================================================

import os
import gradio as gr
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from tianji.knowledges.langchain_onlinellm.models import ZhipuAIEmbeddings, ZhipuLLM
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    TextLoader,
    DirectoryLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from tianji import TIANJI_PATH

# 加载环境变量（包含API密钥等配置）
load_dotenv()


# =============================================================================
# 嵌入模型创建函数
# =============================================================================
# 功能：根据用户选择创建相应的嵌入模型实例
# 参数：
# - embedding_choice: 嵌入模型选择 ('huggingface' 或 'zhipuai')
# - cache_folder: 缓存文件夹路径（用于HuggingFace模型）
# 返回：嵌入模型实例
# 说明：
# - HuggingFace选项使用BAAI/bge-base-zh-v1.5中文嵌入模型
# - ZhipuAI选项使用智谱AI的嵌入服务
# - 两个选项都针对中文文本优化
# =============================================================================

def create_embeddings(embedding_choice: str, cache_folder: str):
    """
    根据选择创建嵌入模型
    :param embedding_choice: 嵌入模型选择 ('huggingface' 或 'zhipuai')
    :param cache_folder: 缓存文件夹路径
    :return: 嵌入模型实例
    """
    if embedding_choice == "huggingface":
        print("正在创建HuggingFace嵌入模型...")
        return HuggingFaceEmbeddings(
            model_name="BAAI/bge-base-zh-v1.5",  # 中文优化嵌入模型
            model_kwargs={"device": "cpu"},  # 使用CPU进行推理
            encode_kwargs={"normalize_embeddings": True},  # 标准化嵌入向量
            cache_folder=cache_folder,  # 模型缓存路径
        )
    print("正在创建智谱AI嵌入模型...")
    return ZhipuAIEmbeddings()  # 使用智谱AI的嵌入API


# =============================================================================
# 向量数据库创建函数
# =============================================================================
# 功能：创建或加载向量数据库，支持多种数据源类型
# 参数：
# - data_type: 数据类型 ('folder' 或 'web')
# - data_path: 数据路径（文件夹路径或网页URL）
# - persist_directory: 向量数据库持久化目录
# - embedding_func: 嵌入函数
# - chunk_size: 文本分割块大小
# - force: 是否强制重建数据库（默认True）
# 返回：Chroma向量数据库实例
# 说明：
# - 支持从本地文件夹加载txt文件
# - 支持从网页URL抓取内容
# - 自动进行文本分割和向量化
# - 支持数据库持久化和重用
# =============================================================================

def create_vectordb(
    data_type: str,
    data_path: str,
    persist_directory: str,
    embedding_func,
    chunk_size: int,
    force: bool = True,
):
    """
    创建或加载向量数据库
    :param data_type: 数据类型 ('folder' 或 'web')
    :param data_path: 数据路径
    :param persist_directory: 持久化目录
    :param embedding_func: 嵌入函数
    :param chunk_size: 文本块大小
    :param force: 是否强制重建数据库
    :return: Chroma 向量数据库实例
    """
    # 步骤1: 检查是否使用现有数据库
    if os.path.exists(persist_directory) and not force:
        print(f"使用现有的向量数据库: {persist_directory}")
        return Chroma(
            persist_directory=persist_directory, embedding_function=embedding_func
        )

    # 步骤2: 如果需要强制重建，先删除旧数据库
    if force and os.path.exists(persist_directory):
        print(f"强制重建向量数据库: {persist_directory}")
        if os.path.isdir(persist_directory):
            import shutil
            shutil.rmtree(persist_directory)  # 删除整个目录
        else:
            os.remove(persist_directory)  # 删除单个文件

    # 步骤3: 根据数据类型创建相应的加载器
    if data_type == "folder":
        print(f"从文件夹加载数据: {data_path}")
        loader = DirectoryLoader(data_path, glob="*.txt", loader_cls=TextLoader)
    elif data_type == "web":
        print(f"从网页加载数据: {data_path}")
        loader = WebBaseLoader(web_paths=(data_path,))
    else:
        raise gr.Error("不支持的数据类型。请选择 'folder' 或 'web'。")

    # 步骤4: 创建文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,  # 每个文本块的最大字符数
        chunk_overlap=200  # 文本块之间的重叠字符数，保持上下文连贯性
    )

    # 步骤5: 加载并分割文档
    documents = loader.load()  # 加载原始文档
    split_docs = text_splitter.split_documents(documents)  # 分割文档
    
    if len(split_docs) == 0:
        raise gr.Error("当前知识数据无效,处理数据后为空")

    print(f"成功处理 {len(split_docs)} 个文档片段")

    # 步骤6: 创建向量数据库
    vector_db = Chroma.from_documents(
        documents=split_docs,  # 文档片段列表
        embedding=embedding_func,  # 嵌入函数
        persist_directory=persist_directory,  # 持久化目录
    )
    
    print(f"向量数据库创建完成: {persist_directory}")
    return vector_db


# =============================================================================
# RAG链初始化函数
# =============================================================================
# 功能：创建完整的RAG（检索增强生成）处理链
# 参数：
# - embedding_choice: 嵌入模型选择 ('huggingface' 或 'zhipuai')
# - chunk_size: 文本分割块大小
# - cache_folder: 缓存文件夹路径
# - persist_directory: 向量数据库持久化目录
# - data_type: 数据类型 ('folder' 或 'web')
# - data_path: 数据路径（文件夹路径或网页URL）
# 返回：完整的RAG处理链
# 说明：
# - 集成嵌入模型、向量数据库、检索器、提示词和LLM
# - 使用LangChain的链式处理架构
# - 针对中文问答场景优化提示词模板
# =============================================================================

def initialize_chain(
    embedding_choice: str,
    chunk_size: int,
    cache_folder: str,
    persist_directory: str,
    data_type: str,
    data_path: str,
):
    """
    初始化检索增强生成（RAG）链
    :param embedding_choice: 嵌入模型选择
    :param chunk_size: 文本块大小
    :param cache_folder: 缓存文件夹路径
    :param persist_directory: 持久化目录
    :param data_type: 数据类型
    :param data_path: 数据路径
    :return: RAG 链
    """
    print("开始初始化RAG系统...")
    
    # 步骤1: 创建嵌入模型
    print(f"创建嵌入模型: {embedding_choice}")
    embeddings = create_embeddings(embedding_choice, cache_folder)
    
    # 步骤2: 创建向量数据库
    print("创建向量数据库...")
    vectordb = create_vectordb(
        data_type, data_path, persist_directory, embeddings, chunk_size
    )
    
    # 步骤3: 创建检索器
    print("创建文档检索器...")
    retriever = vectordb.as_retriever()
    
    # 步骤4: 获取并自定义RAG提示词模板
    print("配置RAG提示词模板...")
    prompt = hub.pull("rlm/rag-prompt")  # 从LangChain Hub获取标准RAG提示词
    
    # 自定义提示词模板，针对中文问答场景优化
    prompt.messages[0].prompt.template = """
    您是一名用于问答任务的助手。使用检索到的上下文来回答问题。如果您不知道答案，就直接说不知道。\
    1.根据我的提问,总结检索到的上下文中与提问最接近的部分,将相关部分浓缩为一段话返回;
    2.根据语料结合我的问题,给出建议和解释。\
    \n问题：{question} \n上下文：{context} \n答案：
    """
    
    # 步骤5: 创建语言模型
    print("创建智谱AI语言模型...")
    llm = ZhipuLLM()  # 使用ZhipuLLM作为默认LLM
    
    print("RAG系统初始化完成")
    
    # 步骤6: 构建完整的RAG处理链
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
        # 输入准备：
        # - context: 通过retriever检索相关文档，再经format_docs格式化为字符串
        # - question: 使用RunnablePassthrough()直接传递用户问题
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        
        # 提示词构建：接收前面的字典，使用预定义的RAG提示词模板
        | prompt
        
        # 大语言模型生成：接收格式化后的提示词，调用LLM生成回答
        | llm
        
        # 输出解析：解析模型的输出响应，提取纯文本内容
        | StrOutputParser()
    )


# =============================================================================
# 辅助函数
# =============================================================================

def format_docs(docs):
    """
    格式化文档列表为字符串
    :param docs: 文档对象列表
    :return: 格式化的文档内容字符串
    说明：将多个文档的内容用双换行符连接，便于后续处理
    """
    return "\n\n".join(doc.page_content for doc in docs)


def handle_question(chain, question: str, chat_history):
    """
    处理用户问题并更新聊天历史
    :param chain: RAG处理链
    :param question: 用户问题
    :param chat_history: 聊天历史列表
    :return: (清空的问题字符串, 更新后的聊天历史)
    说明：
    - 如果问题为空，直接返回
    - 调用RAG链生成回答
    - 将问答对添加到聊天历史
    - 异常处理：显示错误信息
    """
    if not question:
        return "", chat_history
    try:
        print(f"处理用户问题: {question}")
        result = chain.invoke(question)  # 调用RAG链生成回答
        chat_history.append((question, result))  # 添加到聊天历史
        print(f"生成回答: {result[:100]}...")  # 打印回答的前100个字符
        return "", chat_history
    except Exception as e:
        error_msg = f"处理问题时发生错误: {str(e)}"
        print(error_msg)
        return error_msg, chat_history


def update_settings(
    embedding_choice: str,
    chunk_size: int,
    cache_folder: str,
    persist_directory: str,
    data_type: str,
    data_path: str,
):
    """
    更新设置并重新初始化RAG系统
    :param embedding_choice: 嵌入模型选择
    :param chunk_size: 文本块大小
    :param cache_folder: 缓存文件夹路径
    :param persist_directory: 持久化目录
    :param data_type: 数据类型
    :param data_path: 数据路径
    :return: (新的RAG链, 示例问题)
    说明：当用户在界面中更改配置时调用此函数重新初始化整个RAG系统
    """
    print("用户更新设置，重新初始化RAG系统...")
    chain = initialize_chain(
        embedding_choice,
        chunk_size,
        cache_folder,
        persist_directory,
        data_type,
        data_path,
    )
    print("RAG系统重新初始化完成")
    return chain, "什么是春节?"  # 返回示例问题供用户测试


def update_data_path(data_type: str):
    """
    根据数据类型更新默认数据路径
    :param data_type: 数据类型 ('folder' 或 'web')
    :return: 相应的默认数据路径
    说明：
    - folder类型：使用项目中的测试数据文件夹
    - web类型：使用百度百科的春节页面（通过jina.ai代理访问）
    """
    if data_type == "web":
        print("数据类型切换为网页，使用默认URL")
        return (
            "https://r.jina.ai/https://baike.baidu.com/item/%E6%98%A5%E8%8A%82/136876"
        )
    print("数据类型切换为文件夹，使用默认文件夹路径")
    return os.path.join(TIANJI_PATH, "test", "knowledges", "langchain", "db_files")


def update_chat_history(msg: str, chat_history):
    """
    更新聊天历史的辅助函数
    :param msg: 消息内容
    :param chat_history: 聊天历史
    :return: (消息内容, 聊天历史)
    说明：主要用于界面中的回调函数，保持聊天历史的同步更新
    """
    return str(msg), chat_history


# =============================================================================
# Gradio界面创建
# =============================================================================
# 功能：构建交互式的RAG系统Web界面
# 界面组件：
# - 模型配置区：嵌入模型选择、文本块大小等参数
# - 数据源配置区：数据类型、路径等设置
# - 聊天区域：显示对话历史和输入问题
# - 控制按钮：初始化数据库、发送消息、清除记录
# =============================================================================

with gr.Blocks(title="可配置RAG演示系统") as demo:
    # 页面标题和使用说明
    gr.Markdown(
        """# 可配置RAG演示系统
        
        **使用说明：**<br>
        1. 🔄 首先配置参数并点击"初始化数据库"按钮（可能需要一些时间）<br>
        2. 💬 初始化完成后，在输入框中输入问题并点击"聊天"按钮<br>
        3. ⚠️ 如果过程中出现异常，错误信息会显示在输入框中<br>
        """
    )
    
    # 配置区域：模型和数据源参数
    with gr.Row():
        # 嵌入模型选择
        embedding_choice = gr.Radio(
            ["huggingface", "zhipuai"], 
            label="选择嵌入模型", 
            value="zhipuai",
            info="选择用于文档嵌入的模型：HuggingFace（本地）或智谱AI（在线API）"
        )
        
        # 文本块大小滑块
        chunk_size = gr.Slider(
            256, 2048, 
            step=256, 
            label="选择文本块大小", 
            value=512,
            info="控制文档分割的粒度，影响检索精度和处理速度"
        )
        
        # 缓存文件夹路径
        cache_folder = gr.Textbox(
            label="缓存文件夹路径", 
            value=os.path.join(TIANJI_PATH, "temp"),
            info="HuggingFace模型的缓存路径"
        )
        
        # 向量数据库持久化路径
        persist_directory = gr.Textbox(
            label="持久化数据库路径", 
            value=os.path.join(TIANJI_PATH, "temp", "chromadb_spring"),
            info="向量数据库的存储路径"
        )
        
        # 数据类型选择
        data_type = gr.Radio(
            ["folder", "web"], 
            label="数据类型", 
            value="folder",
            info="选择数据源类型：本地文件夹或网页URL"
        )
        
        # 数据路径输入
        data_path = gr.Textbox(
            label="数据路径",
            value=os.path.join(TIANJI_PATH, "test", "knowledges", "langchain", "db_files"),
            info="文件夹路径或网页URL",
            lines=2
        )
        
        # 初始化按钮
        update_button = gr.Button("🔄 初始化数据库", variant="primary")

    # 聊天区域
    with gr.Row():
        with gr.Column(scale=3):
            # 聊天历史显示
            chatbot = gr.Chatbot(
                height=450, 
                show_copy_button=True,
                label="对话历史",
                bubble_full_width=False
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 💡 提示")
            gr.Markdown(
                """
                **优化建议：**
                - 文本块大小：512-1024适合大多数场景
                - 嵌入模型：智谱AI速度更快，HuggingFace免费
                - 数据源：网页数据需要网络连接
                """
            )

    # 输入和控制区域
    with gr.Row():
        msg = gr.Textbox(
            label="问题/提示",
            placeholder="输入您的问题，例如：什么是春节？",
            lines=2,
            scale=4
        )
        
        with gr.Column(scale=1):
            chat_button = gr.Button("💬 发送", variant="primary")
            clear_button = gr.ClearButton(
                components=[chatbot], 
                value="🗑️ 清除聊天记录",
                variant="secondary"
            )

    # 事件处理绑定
    # 数据类型改变时自动更新默认路径
    data_type.change(
        update_data_path, 
        inputs=[data_type], 
        outputs=[data_path],
        show_progress="hidden"
    )

    # 状态管理：存储RAG链实例
    model_chain = gr.State()

    # 初始化按钮事件
    update_button.click(
        update_settings,
        inputs=[
            embedding_choice,
            chunk_size,
            cache_folder,
            persist_directory,
            data_type,
            data_path,
        ],
        outputs=[model_chain, msg],
        show_progress="full"  # 显示完整进度条
    )

    # 发送按钮事件
    chat_button.click(
        handle_question,
        inputs=[model_chain, msg, chatbot],
        outputs=[msg, chatbot],
    ).then(
        update_chat_history, 
        inputs=[msg, chatbot], 
        outputs=[msg, chatbot],
        show_progress="hidden"
    )

    # 示例问题快速选择
    gr.Examples(
        examples=[
            ["什么是春节？"],
            ["春节有哪些传统习俗？"],
            ["春节的历史起源是什么？"],
            ["春节期间人们通常会做什么？"]
        ],
        inputs=msg,
        label="📝 示例问题"
    )

# =============================================================================
# 应用启动
# =============================================================================

if __name__ == "__main__":
    print("正在启动可配置RAG演示系统...")
    demo.launch(
        server_name="0.0.0.0",  # 监听所有网络接口
        server_port=7860,  # 默认端口
        share=False,  # 不创建公开链接
        inbrowser=False,  # 不自动打开浏览器
        show_api=True,  # 显示API文档
        show_error=True,  # 显示错误信息
    )
    print("可配置RAG演示系统已启动！")
