"""
知识库创建模块 - Knowledge Base Creation Module

这个模块负责处理知识库文档的加载、分割、嵌入和向量数据库存储。
支持多种文档格式（PDF、Markdown、文本文件）和多种嵌入模型。

This module handles knowledge base document loading, splitting, embedding, 
and vector database storage. Supports multiple document formats 
(PDF, Markdown, text files) and various embedding models.
"""

# 导入操作系统接口模块 / Import operating system interface module
import os
# 导入系统相关参数和函数 / Import system-specific parameters and functions
import sys
# 导入正则表达式模块 / Import regular expression module
import re
# 导入临时文件模块 / Import temporary file module
import tempfile
# 从python-dotenv库导入环境变量加载函数 / Import environment variable loading functions from python-dotenv library
from dotenv import load_dotenv, find_dotenv

# 添加父目录到系统路径 - 使得可以导入父目录中的模块 / Add parent directory to system path - allows importing modules from parent directory
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# 从嵌入模块导入获取嵌入函数 / Import get embedding function from embedding module
from embedding.call_embedding import get_embedding
# 从LangChain导入非结构化文件加载器 / Import unstructured file loader from LangChain
from langchain.document_loaders import UnstructuredFileLoader
# 从LangChain导入非结构化Markdown加载器 / Import unstructured markdown loader from LangChain
from langchain.document_loaders import UnstructuredMarkdownLoader
# 从LangChain导入递归字符文本分割器 / Import recursive character text splitter from LangChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
# 从LangChain导入PyMuPDF加载器 / Import PyMuPDF loader from LangChain
from langchain.document_loaders import PyMuPDFLoader
# 从LangChain导入Chroma向量数据库 / Import Chroma vector database from LangChain
from langchain.vectorstores import Chroma

# 加载环境变量 - 自动查找.env文件并加载其中的环境变量 / Load environment variables - automatically find .env file and load environment variables
_ = load_dotenv(find_dotenv())

# 默认配置 - 定义知识库的默认路径配置 / Default configurations - define default path configurations for knowledge base
DEFAULT_DB_PATH = "./knowledge_db"          # 默认知识库路径 / Default knowledge base path
DEFAULT_PERSIST_PATH = "./vector_db"        # 默认向量数据库持久化路径 / Default vector database persistence path


def get_files(dir_path):
    """
    递归获取目录下所有文件路径
    Recursively get all file paths in directory
    
    参数 / Parameters:
        dir_path: 目标目录路径 / Target directory path
        
    返回 / Returns:
        list: 文件路径列表 / List of file paths
    """
    # 初始化文件列表 - Initialize file list
    file_list = []
    
    # 使用os.walk遍历目录及其子目录 / Use os.walk to traverse directory and subdirectories
    # filepath: 当前目录路径 / Current directory path
    # dirnames: 子目录名称列表 / Subdirectory name list  
    # filenames: 文件名称列表 / File name list
    for filepath, dirnames, filenames in os.walk(dir_path):
        # 遍历当前目录下的所有文件 / Traverse all files in current directory
        for filename in filenames:
            # 使用os.path.join构建完整文件路径 / Use os.path.join to build complete file path
            # 这样可以正确处理不同操作系统的路径分隔符 / This correctly handles path separators for different operating systems
            file_list.append(os.path.join(filepath, filename))
    
    # 返回收集到的所有文件路径 / Return all collected file paths
    return file_list


def file_loader(file, loaders):
    """
    根据文件类型加载文档到加载器列表
    Load documents into loader list based on file type
    
    参数 / Parameters:
        file: 文件路径或临时文件对象 / File path or temporary file object
        loaders: 文档加载器列表 / Document loader list
        
    功能 / Function:
    - 支持递归处理目录 - Support recursive directory processing
    - 根据文件扩展名选择适当的加载器 - Choose appropriate loader based on file extension
    - 过滤掉包含敏感词汇的markdown文件 - Filter out markdown files containing sensitive words
    """
    # 处理临时文件对象 - Handle temporary file objects
    if isinstance(file, tempfile._TemporaryFileWrapper):
        file = file.name
    
    # 如果是目录，递归处理目录下的所有文件
    # If it's a directory, recursively process all files in the directory
    if not os.path.isfile(file):
        [file_loader(os.path.join(file, f), loaders) for f in os.listdir(file)]
        return
    
    # 获取文件扩展名 - Get file extension
    file_type = file.split('.')[-1]
    
    # 根据文件类型选择加载器 - Choose loader based on file type
    if file_type == 'pdf':
        # PDF文件使用PyMuPDFLoader - Use PyMuPDFLoader for PDF files
        loaders.append(PyMuPDFLoader(file))
    elif file_type == 'md':
        # Markdown文件需要过滤敏感内容 - Filter sensitive content for markdown files
        pattern = r"不存在|风控"  # 过滤包含"不存在"或"风控"的文件 / Filter files containing "不存在" or "风控"
        match = re.search(pattern, file)
        if not match:
            # 如果文件名不包含敏感词汇，添加到加载器
            # Add to loader if filename doesn't contain sensitive words
            loaders.append(UnstructuredMarkdownLoader(file))
    elif file_type == 'txt':
        # 文本文件使用UnstructuredFileLoader - Use UnstructuredFileLoader for text files
        loaders.append(UnstructuredFileLoader(file))
    return


def create_db_info(files=DEFAULT_DB_PATH, embeddings="openai", persist_directory=DEFAULT_PERSIST_PATH):
    """
    创建数据库信息函数 - Create database info function
    
    根据嵌入模型类型创建相应的向量数据库。
    Create corresponding vector database based on embedding model type.
    
    参数 / Parameters:
        files: 文件路径 / File path
        embeddings: 嵌入模型类型 / Embedding model type
        persist_directory: 持久化目录 / Persistence directory
    """
    # 检查嵌入模型类型是否受支持 / Check if embedding model type is supported
    if embeddings == 'openai' or embeddings == 'm3e' or embeddings =='zhipuai':
        # 创建向量数据库 / Create vector database
        vectordb = create_db(files, persist_directory, embeddings)
    # 返回空字符串（函数可能用于扩展功能）/ Return empty string (function may be used for extended functionality)
    return ""


def create_db(files=DEFAULT_DB_PATH, persist_directory=DEFAULT_PERSIST_PATH, embeddings="openai"):
    """
    创建向量数据库主函数 - Main function to create vector database
    
    该函数用于加载文件，切分文档，生成文档的嵌入向量，创建向量数据库。
    This function loads files, splits documents, generates document embeddings, and creates vector database.

    参数 / Parameters:
        files: 存放文件的路径 / Path to files
        embeddings: 用于生成Embedding的模型 / Model for generating embeddings
        persist_directory: 持久化目录 / Directory for persistence

    返回 / Returns:
        vectordb: 创建的向量数据库 / Created vector database
    """
    # 检查文件参数是否为空 / Check if files parameter is empty
    if files == None:
        return "can't load empty file"
    
    # 确保files是列表格式 / Ensure files is in list format
    if type(files) != list:
        files = [files]
    
    # 初始化文档加载器列表 / Initialize document loader list
    loaders = []
    
    # 遍历所有文件，使用file_loader函数加载 / Iterate through all files, load using file_loader function
    [file_loader(file, loaders) for file in files]
    
    # 初始化文档列表 / Initialize document list
    docs = []
    
    # 遍历所有加载器，加载文档内容 / Iterate through all loaders, load document content
    for loader in loaders:
        # 检查加载器是否有效 / Check if loader is valid
        if loader is not None:
            # 扩展文档列表（合并所有加载器的文档）/ Extend document list (merge documents from all loaders)
            docs.extend(loader.load())
    
    # 创建文本分割器，设置块大小和重叠大小 / Create text splitter, set chunk size and overlap size
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # 每个文本块的最大字符数 / Maximum character count per text chunk
        chunk_overlap=150)   # 文本块之间的重叠字符数 / Overlapping character count between chunks
    
    # 分割文档 / Split documents
    split_docs = text_splitter.split_documents(docs)
    
    # 如果embeddings参数是字符串，获取对应的嵌入模型 / If embeddings parameter is string, get corresponding embedding model
    if type(embeddings) == str:
        embeddings = get_embedding(embedding=embeddings)
    
    # 定义持久化路径（这里硬编码了，可能需要改进）/ Define persistence path (hardcoded here, may need improvement)
    persist_directory = './vector_db/chroma'
    
    # 创建Chroma向量数据库 / Create Chroma vector database
    vectordb = Chroma.from_documents(
        documents=split_docs,           # 分割后的文档 / Split documents
        embedding=embeddings,           # 嵌入模型 / Embedding model
        persist_directory=persist_directory  # 持久化目录，允许将数据库保存到磁盘 / Persistence directory, allows saving database to disk
    )
    
    # 持久化向量数据库 / Persist vector database
    vectordb.persist()
    
    # 返回创建的向量数据库 / Return created vector database
    return vectordb


def presit_knowledge_db(vectordb):
    """
    该函数用于持久化向量数据库。

    参数:
    vectordb: 要持久化的向量数据库。
    """
    vectordb.persist()


def load_knowledge_db(path, embeddings):
    """
    该函数用于加载向量数据库。

    参数:
    path: 要加载的向量数据库路径。
    embeddings: 向量数据库使用的 embedding 模型。

    返回:
    vectordb: 加载的数据库。
    """
    vectordb = Chroma(
        persist_directory=path,
        embedding_function=embeddings
    )
    return vectordb


if __name__ == "__main__":
    create_db(embeddings="m3e")
