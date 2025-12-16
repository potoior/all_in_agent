# test_setup.py - 环境测试脚本
# 用于测试RAG系统所需的各项组件是否正常工作

# 导入必要的库
from llama_index.embeddings.huggingface import HuggingFaceEmbedding  # 用于文本嵌入的HuggingFace模型
from llama_index.llms.google_genai import GoogleGenAI  # Google Gemini大语言模型接口
import os  # 操作系统接口
from dotenv import load_dotenv  # 用于加载.env文件中的环境变量

# 加载.env文件中的环境变量
load_dotenv()

# 测试嵌入模型
print("Testing embedding model...")
# 初始化HuggingFace嵌入模型
embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
# 获取测试文本的嵌入向量
test_embedding = embed_model.get_text_embedding("test")
print(f"✅ Embedding model working! Vector dimension: {len(test_embedding)}")

# 测试LLM (如果配置了API密钥)
if os.getenv("GOOGLE_API_KEY"):
    print("Testing Google Gemini...")
    # 初始化Google Gemini模型
    llm = GoogleGenAI(model="gemini-1.5-pro")
    # 发送测试请求
    response = llm.complete("Hello, how are you?")
    print(f"✅ Google Gemini working! Response: {response}")
else:
    print("⚠️  Google API key not found, skipping LLM test")

print("🎉 Environment setup complete!")