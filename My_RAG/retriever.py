from rank_bm25 import BM25Okapi
import jieba
from ollama import Client
# import ollama
import numpy as np
import os


class BM25Retriever:
    def __init__(self, chunks, language="en"):
        self.chunks = chunks
        self.language = language
        self.corpus = [chunk['page_content'] for chunk in chunks]
        if language == "zh":
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in self.corpus]
        else:
            self.tokenized_corpus = [doc.split(" ") for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def retrieve(self, query, top_k=3):
        if self.language == "zh":
            tokenized_query = list(jieba.cut(query))
        else:
            tokenized_query = query.split(" ")
        top_chunks = self.bm25.get_top_n(tokenized_query, self.chunks, n=top_k)
        return top_chunks

class EmbeddingRetriever:
    def __init__(self, chunks, language="en", embedding_model="embeddinggemma:300m", ollama_url="http://localhost:11435"):
        self.chunks = chunks
        self.embedding_model = embedding_model
        self.language = language
        self.client = Client(host=ollama_url)
        self.embeddings = self.client.embed(
            self.embedding_model, [chunk['page_content'] for chunk in chunks]
        )

    def retrieve(self, query, top_k=3):
        query_embedding = self.client.embed(self.embedding_model, query)
        query_embedding_vector = np.array(query_embedding.embeddings)[0]
        # breakpoint()
        chunk_embedding_vectors = np.array(self.embeddings.embeddings)
        similarities = np.dot(chunk_embedding_vectors, query_embedding_vector) / (
            np.linalg.norm(chunk_embedding_vectors, axis=1) * np.linalg.norm(query_embedding_vector) + 1e-10
        )
        top_indices = np.argsort(similarities)[::-1][:top_k]
        top_chunks = [self.chunks[i] for i in top_indices]
        return top_chunks

def create_retriever(retriever_type, chunks, language, embedding_model=None, ollama_url="http://localhost:11435"):
    if retriever_type == "bm25":
        return BM25Retriever(chunks, language)
    elif retriever_type == "embedding":
        return EmbeddingRetriever(chunks, language, embedding_model, ollama_url)
    else:
        raise ValueError(f"Unknown retriever type: {retriever_type}")