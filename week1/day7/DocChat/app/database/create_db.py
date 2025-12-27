import os
import tempfile

from langchain_community.document_loaders import PyMuPDFLoader, UnstructuredMarkdownLoader, UnstructuredFileLoader
from langchain_community.document_loaders import Docx2txtLoader, UnstructuredExcelLoader
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter

from DocChat.app.database.call_embedding import get_embedding


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
    - 支持多种文件格式: PDF, Markdown, TXT, DOCX, XLSX - Support multiple file formats
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
    supported_extensions = {'.pdf', '.md', '.txt', '.docx', '.xlsx', '.xls'}
    file_ext = '.' + file_type.lower()
    
    if file_ext in {'.pdf'}:
        # PDF文件使用PyMuPDFLoader - Use PyMuPDFLoader for PDF files
        loaders.append(PyMuPDFLoader(file))

        print(f"加载PDF文件: {file}")
    elif file_ext in {'.md'}:
        # Markdown文件加载 - Load markdown files
        loaders.append(UnstructuredMarkdownLoader(file))
        print(f"加载Markdown文件: {file}")
    elif file_ext in {'.txt'}:
        # 文本文件使用UnstructuredFileLoader - Use UnstructuredFileLoader for text files
        loaders.append(UnstructuredFileLoader(file))
        print(f"加载TXT文件: {file}")
    elif file_ext in {'.docx'}:
        # Word文档加载
        loaders.append(Docx2txtLoader(file))
        print(f"加载Word文档: {file}")
    elif file_ext in {'.xlsx', '.xls'}:
        # Excel文件加载
        loaders.append(UnstructuredExcelLoader(file, mode="elements"))
        print(f"加载Excel文件: {file}")
    else:
        print(f"跳过不支持的文件类型: {file} (.{file_type})")
    return


def create_db(vectordb_path: str,embedding,file_path:str=None):
    if vectordb_path is None:
        print("错误: 向量数据库路径未指定，使用默认路径 './myVectordb'")
        vectordb_path = './myVectordb'
    # 创建一个新的embedding向量数据库
    if embedding is None:
        embedding = get_embedding()
    if file_path is None:
        raise ValueError("错误: 文件路径未指定")
    if type(file_path) != list:
        file_path = [file_path]
    # 加载文档
    loaders = []
    # 加载文件
    [file_loader(file,loaders) for file in file_path]

    # 初始化文档列表
    docs = []
    for loader in loaders:
        # 检查加载器是否有效
        if loader is not None:
            # 扩展文档列表（合并所有加载器的文档）
            docs.extend(loader.load())

        # 创建文本分割器，设置块大小和重叠大小
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # 每个文本块的最大字符数
        chunk_overlap=300 # 文本块之间的重叠字符数
    )
    print(f"已加载{len(docs)}个文档")


    # 分割文档
    split_docs = text_splitter.split_documents(docs)

    # 使用Chroma创建向量数据库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=vectordb_path
    )


    return vectordb


def local_knowledge_db(vectordb_path, embedding):
    print("本地向量数据库开始加载")
    try:
        vectordb = Chroma(
            persist_directory=vectordb_path,
            embedding_function=embedding)
        print("本地向量数据库加载完成")
        return vectordb
    except Exception as e:
        print(f"加载向量数据库时发生错误: {e}")
        raise e
