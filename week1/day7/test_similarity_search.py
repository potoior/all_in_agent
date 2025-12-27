#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试向量数据库的相似性搜索功能
"""

from DocChat.app.models.chat_chain import ChatChain


def test_similarity_search():
    """测试相似性搜索功能"""
    print("初始化ChatChain...")
    chat_chain = ChatChain()
    
    print("\n输入一个问题来查找最相似的文档片段：")
    question =  "什么是AI Agent"
    
    print(f"\n查找与问题 '{question}' 最相似的文档片段...")
    relevant_docs = chat_chain.get_relevant_documents(question, top_k=5)
    
    print(f"\n找到 {len(relevant_docs)} 个最相关的文档片段：")
    print("="*60)
    
    for i, doc in enumerate(relevant_docs, 1):
        print(f"\n文档 {i}:")
        print(f"内容预览: {doc.page_content}")  # 显示前200个字符
        print(f"元数据: {doc.metadata}")
        print("-" * 40)


if __name__ == "__main__":
    test_similarity_search()