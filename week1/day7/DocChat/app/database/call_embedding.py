from langchain_huggingface import HuggingFaceEmbeddings


def get_embedding():
    embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    return  embedding