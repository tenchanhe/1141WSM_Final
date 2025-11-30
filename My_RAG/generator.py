from ollama import Client
import os

def generate_answer(query, context_chunks, language, ollama_url="http://ollama-gateway:11434"):
    context = "\n\n".join([chunk['page_content'] for chunk in context_chunks])
    client = Client(host=ollama_url)

    if language == "en":
        prompt = f"""You are a professional assistant for fact-based question-answering tasks. Your task is to answer the question based *solely* on the provided context.
- Synthesize the information from the context to provide a comprehensive and detailed answer.
- If the context does not contain the information needed to answer the question, you must respond with exactly "Unable to answer". Do not add any other words or explanations.
- Do not use any external knowledge.

Question: {query}
Context: {context}
Answer:
"""
        try:
            response = client.generate(model="granite4:3b", prompt=prompt, stream=False)
            return response.get("response", "No response from model.")
        except Exception as e:
            return f"Error using Ollama Python client: {e}"
    else:
        prompt = f"""你是一个专业的、基于事实的问答任务助手。你的任务是*仅*根据提供的上下文来回答问题。
- 综合上下文中的信息，以提供一个全面且详细的回答。
- 如果上下文中不包含回答问题所需的信息，你必须仅回答“无法回答”，不要添加任何其他词语或解释。
- 不要使用任何外部知识。

问题: {query}
上下文: {context}
回答:
"""
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