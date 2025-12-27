import os

from DocChat.app.database.create_db import local_knowledge_db, create_db
from DocChat.app.database.call_embedding import get_embedding
def get_vectordb(vectordb_path: str,file_path:str):
    # 获取embedding
    embedding = get_embedding()

    # 检查向量数据库是否存在（通过检查特定的chroma文件）
    chroma_index_files = ["chroma.sqlite3", "index_metadata.json"]
    
    if os.path.exists(vectordb_path):
        # 检查是否包含向量数据库的必要文件
        has_chroma_files = any(
            os.path.exists(os.path.join(vectordb_path, f)) for f in chroma_index_files
        )
        
        if has_chroma_files:
            # 向量数据库已存在，直接加载
            print("向量数据库已存在，正在加载")
            vectordb = local_knowledge_db(vectordb_path, embedding)
        else:
            # 路径存在但没有向量数据库
            print("创建新的向量数据库")
            create_db(vectordb_path, embedding, file_path)
            vectordb = local_knowledge_db(vectordb_path, embedding)
    else:
        # 路径不存在，创建路径并创建向量数据库
        os.makedirs(vectordb_path, exist_ok=True)
        print("创建新的向量数据库")
        create_db(vectordb_path, embedding, file_path)
        vectordb = local_knowledge_db(vectordb_path, embedding)

    return vectordb



