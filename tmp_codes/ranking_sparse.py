import argparse
from search import *
import json
from pyserini.search.lucene import LuceneSearcher

def read_query(filepath):
    queries = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            qid = data.get('id')
            query_text = data.get('contents')
            if qid and query_text:
                queries[qid] = query_text
    return queries

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=str)
    parser.add_argument("--query", type=str)
    parser.add_argument("--method", default="bm25", type=str)
    parser.add_argument("--k", default=100, type=int)
    parser.add_argument("--output", default='runs/bm25.run', type=str)
    
    args = parser.parse_args()

    if args.method == "bm25":
        searcher = LuceneSearcher(args.index)
        searcher.set_bm25(k1=2, b=0.75)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    # query = read_title(args.query)
    query = read_query(args.query)
    search(searcher, query, args)
