from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from chunker import chunk_documents
from retriever import create_retriever
from generator import generate_answer
import argparse
import os
from dotenv import load_dotenv

if os.path.exists(".env"):
    print("Loading environment variables from .env file")
    load_dotenv(".env")
    OLLAMA_URL = os.getenv("OLLAMA_URL")
else:
    OLLAMA_URL = "http://ollama-gateway:11434"
print(f"Using OLLAMA_URL: {OLLAMA_URL}")

def main(args):
    print("Arguments:", args.language, args.retriever, args.chunk_size, args.topk, args.embedding_model)
    docs_for_chunking = load_jsonl(args.docs_path)
    queries = load_jsonl(args.query_path)

    chunks = chunk_documents(docs_for_chunking, args.language, chunk_size=int(args.chunk_size))

    retriever = create_retriever(args.retriever, chunks, args.language, embedding_model=args.embedding_model, ollama_url=OLLAMA_URL)


    for query in tqdm(queries, desc="Processing Queries"):
        query_text = query['query']['content']
        query_language = args.language
        retrieved_chunks = retriever.retrieve(query_text, top_k=int(args.topk))

        answer = generate_answer(query_text, retrieved_chunks, query_language, ollama_url=OLLAMA_URL)

        query["prediction"]["content"] = answer
        query["prediction"]["references"] = [chunk['page_content'] for chunk in retrieved_chunks]

    save_jsonl(args.output, queries)
    print("Predictions saved at '{}'".format(args.output))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', help='Path to the query file')
    parser.add_argument('--docs_path', help='Path to the documents file')
    parser.add_argument('--language', help='Language to filter queries (zh or en), if not specified, process all')
    parser.add_argument('--output', help='Path to the output file')

    parser.add_argument('--retriever', help='Retriever type: bm25 or embedding', default='bm25')
    parser.add_argument('--chunk_size', help='Chunk size for document chunking', default=1024)
    parser.add_argument('--topk', help='Number of top chunks to retrieve', default=5)
    parser.add_argument('--embedding_model', help='Embedding model to use for embedding retriever', default='qwen3-embedding:0.6b')
    args = parser.parse_args()
    main(args)
