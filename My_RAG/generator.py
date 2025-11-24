from ollama import Client
import os

def generate_answer(query, context_chunks, language, ollama_url="http://ollama-gateway:11434"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    client = Client(host=ollama_url)

    if language == "en":
        prompt = f"""You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\n\nQuestion: {query} \nContext: {context} \nAnswer:\n"""
        try:
            response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
            return response.get("response", "No response from model.")
        except Exception as e:
            return f"Error using Ollama Python client: {e}"
    else:
        prompt = f"""你是一个问答任务的助手。使用以下检索到的上下文片段来回答问题。如果你不知道答案，就说你不知道。请使用最多三句话，并保持回答简洁。\n\n问题: {query} \n上下文: {context} \n回答:\n"""
        try:
            response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
            return response.get("response", "模型没有响应。")
        except Exception as e:
            return f"使用 Ollama Python 客户端时出错: {e}"
    


if __name__ == "__main__":
    # test the function
    query = "What is the capital of France?"
    context_chunks = [
        {"page_content": "France is a country in Europe. Its capital is Paris."},
        {"page_content": "The Eiffel Tower is located in Paris, the capital city of France."}
    ]
    answer = generate_answer(query, context_chunks, "en")
    print("Generated Answer:", answer)