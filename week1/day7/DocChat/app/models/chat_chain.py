import re

from typing import List

from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.llm import LLMChain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from llama_index.core.base.embeddings.base import similarity

from DocChat.app.utils.get_vectordb import get_vectordb
from pathlib import Path
from DocChat.app.prompt_template.prompt_template import COMBINE_PROMPT, CONDENSE_QUESTION_PROMPT
class ChatChain:
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, chat_history: list = [],
                 api_key: str = None, api_router: str = None):
        # 防止重复初始化
        if self._initialized:
            return
        if api_key is None:
            api_key = "sk-zqzehnidkvjxmpgoqohexqzxwnvyszxwgxucpxmtftdpgrgv"
        if api_router is None:
            api_router = "https://api.siliconflow.cn/v1"
        print("初始化开始")

        self.llm = ChatOpenAI(
            base_url=api_router,  # OpenRouter API端点
            api_key=api_key,   # 从环境变量获取API密钥
            model="Qwen/Qwen2.5-72B-Instruct",
            streaming=True,
        )
        # 使用相对路径，基于当前文件位置
        current_dir = Path(__file__).parent
        # print("当前文件位置:", current_dir)
        database_path = current_dir.parent / "embedding_db"
        doc_path = current_dir.parent / "doc"
        # print("数据库位置:", database_path)
        # print("文档位置:", doc_path)
        self.vectordb = get_vectordb(
            str(database_path.resolve()),
            str(doc_path.resolve())
        )
        
        # 为每个会话维护单独的历史记录
        self.session_histories = {}
        self._initialized = True

    # 清空对话
    def clean_history(self, session_id: str = "default"):
        if session_id in self.session_histories:
            self.session_histories[session_id] = []
        else:
            self.session_histories[session_id] = []
        return self.session_histories[session_id]

    def get_relevant_documents(self, question: str, top_k: int = 5):
        """
        获取与问题最相关的文档片段
        """
        # 使用search_kwargs直接设置参数
        relevant_docs = self.vectordb.similarity_search_with_score(
            question,
            k=top_k
        )

        # 打印最相似的前几个文本块
        print("get_relevant_documents----------------------")
        print(f"\n---- 最相关的 {min(top_k, len(relevant_docs))} 个文本块 ----")
        for i, (doc, score) in enumerate(relevant_docs):
            source = doc.metadata.get('source', 'Unknown')
            content_preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
            print(f"【文本块 {i+1}】来自: {source}")
            print(f"【相似度得分】: {score}")
            print(f"【内容】{content_preview}\n")
        print("------------------------------------------")
        # 只返回文档，不返回分数
        return [doc for doc, _ in relevant_docs]

    def answer(self, question: str = None, top_k: int = 2, session_id: str = "default"):
        if not question:
            return "请输入问题"

        # 保证历史存在
        hist = self.session_histories.setdefault(session_id, [])

        retriever = self.vectordb.as_retriever(
            search_kwargs={"k": top_k},
            similarity="similarity",
            similarity_threshold=0.5
        )
        self.get_relevant_documents( question)


        # 1. 历史感知 retriever
        contextualize_q_system_prompt = (
            "根据上述对话记录和用户的最新问题，"
            "请将问题改写为独立、完整的表述，并保持原语言不变。"
        )
        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        history_aware_retriever = create_history_aware_retriever(
            self.llm, retriever, contextualize_q_prompt
        )

        # 2. 生成回答
        qa_system_prompt = (
            "你是助手 DocBot，请始终用中文回答。"
            "请使用下方提供的上下文来回答问题。"
            "如果无法从中得出答案，请直接说“我不知道”，不要编造内容。\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        question_answer_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

        # 3. 调用链
        result = rag_chain.invoke({
            "input": question,
            "chat_history": hist,  # 现在是 HumanMessage/ AIMessage 列表
        })

        answer = result["answer"].replace("\\n", "<br/>")

        # 4. 把本轮 Q&A 以消息对象追加到历史
        hist.append(HumanMessage(content=question))
        hist.append(AIMessage(content=answer))

        # 5. 返回给前端（如果你仍想保留字符串格式，可以转一下）
        return [(msg.content,) if isinstance(msg, HumanMessage) else (None, msg.content)
                for msg in hist]