"""
响应生成模块
使用大语言模型基于检索到的上下文生成自然语言回答
"""

from openai import OpenAI  # OpenAI API客户端
import os  # 操作系统接口

# 初始化OpenAI客户端
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",  # OpenRouter API端点
    api_key=os.getenv("OPENROUTER_API_KEY")   # 从环境变量获取API密钥
)

def generate_response(system_prompt, user_message, model="Qwen/Qwen2.5-72B-Instruct:free"):
    """
    基于检索到的上下文生成回答
    
    Args:
        system_prompt (str): 系统提示词，定义AI助手的行为模式
        user_message (str): 用户消息，包含上下文和具体问题
        model (str): 使用的大语言模型，默认为免费的Llama3.2-3B模型
        
    Returns:
        ChatCompletion: 模型生成的响应
    """
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # 设置为0以获得确定性回答
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response

# 系统提示词 - 定义AI助手的行为准则
system_prompt = """你是一个AI助手，严格基于给定的上下文回答问题。
如果答案无法从提供的上下文中直接得出，请回答："我没有足够的信息来回答这个问题。" """

# 构建用户提示 - 将检索到的上下文和用户查询组合成完整的提示
# context = "\n".join([f"上下文 {i+1}:\n{chunk}" for i, chunk in enumerate(top_chunks)])
# user_prompt = f"{context}\n\n问题: {query}"

# 生成回答
# ai_response = generate_response(system_prompt, user_prompt)
# print(f"AI回答: {ai_response.choices[0].message.content}")