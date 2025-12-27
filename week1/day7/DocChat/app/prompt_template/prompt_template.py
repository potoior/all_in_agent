from langchain_core.prompts import PromptTemplate

COMBINE_PROMPT = PromptTemplate(
    input_variables=["summaries", "question", "chat_history"],
    template="""You are a helpful assistant named DocBot, you always answer in Chinese.
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say “我不知道”, don't try to make up an answer.

Context:
{summaries}

Current conversation:
{chat_history}

Question: {question}
Answer in Chinese:"""
)

# 2. 自定义“根据历史对话改写用户问题”的提示词（可选）
CONDENSE_QUESTION_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.

Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
)