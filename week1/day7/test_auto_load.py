import os
from DocChat.app.utils.get_vectordb import get_vectordb

# 测试自动加载功能
def test_auto_load():
    # 使用相对路径
    import pathlib
    current_dir = pathlib.Path(__file__).parent
    database_path = current_dir / "app" / "database"
    doc_path = current_dir / "app" / "doc"
    
    print(f"数据库路径: {database_path}")
    print(f"文档路径: {doc_path}")
    
    # 尝试获取向量数据库，这将自动加载doc目录下的所有文档
    vectordb = get_vectordb(str(database_path), str(doc_path))
    
    print("向量数据库创建/加载成功!")
    print(f"向量数据库中包含 {len(vectordb._collection.count())} 个文档片段")
    
    return vectordb

if __name__ == "__main__":
    test_auto_load()