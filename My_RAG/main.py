from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer

def main():
    query = "When did the board of directors launch a formal investigation into the internal audit findings?"

    # 1. Load Data
    print("Loading documents...")
    docs_path = './dragonball_dataset/dragonball_docs.jsonl'
    query_path = './dragonball_dataset/test_queries.jsonl'
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    # 2. Chunk Documents
    print("Chunking documents...")
    zh_chunks, en_chunks = chunk_documents(docs_for_chunking)
    print(f"Created {len(zh_chunks)} chunks.")
    print(f"Created {len(en_chunks)} chunks.")

    # 3. Create Retriever
    print("Creating retriever...")
    zh_retriever = create_retriever(zh_chunks, language="zh")
    en_retriever = create_retriever(en_chunks, language="en")
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        # 4. Retrieve relevant chunks
        query_text = query['query']['content']
        query_lang = query['language']
        print(f"\nRetrieving chunks for query: '{query_text}'")
        if query_lang == "zh":
            retrieved_chunks = zh_retriever.retrieve(query_text)
        else:
            retrieved_chunks = en_retriever.retrieve(query_text)
        print(f"Retrieved {len(retrieved_chunks)} chunks.")

        # 5. Generate Answer
        print("Generating answer...")
        answer = generate_answer(query_text, retrieved_chunks)
        # print("="*30)
        # print("Answer:")
        # print(answer)
        # print("="*30)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = retrieved_chunks[0] # Testing: only store the top chunk as reference

    save_jsonl('./result/predictions.jsonl', queries)
    print("Predictions saved")

if __name__ == "__main__":
    main()
