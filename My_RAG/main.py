from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
import argparse

OLLAMA_URL="http://ollama-gateway:11434"
# OLLAMA_URL="http://localhost:11435"

def main(query_path, docs_path, language, output_path):
    print("Loading documents...")
    docs_for_chunking = load_jsonl(docs_path)
    queries = load_jsonl(query_path)
    print(f"Loaded {len(docs_for_chunking)} documents.")
    print(f"Loaded {len(queries)} queries.")

    print("Chunking documents...")
    chunks = chunk_documents(docs_for_chunking, language)
    print(f"Created {len(chunks)} chunks.")

    print("Creating retriever...")
    # retriever = create_retriever("bm25", chunks, language)
    retriever = create_retriever("embedding", chunks, language, embedding_model="embeddinggemma:300m", ollama_url=OLLAMA_URL)
    print("Retriever created successfully.")


    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query['query']['content']
        query_language = language
        retrieved_chunks = retriever.retrieve(query_text)

        answer = generate_answer(query_text, retrieved_chunks, query_language, ollama_url=OLLAMA_URL)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [chunk['page_content'] for chunk in retrieved_chunks]

    save_jsonl(output_path, queries)
    print("Predictions saved at '{}'".format(output_path))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')
    args = parser.parse_args()
    main(args.query_path, args.docs_path, args.language, args.output)
