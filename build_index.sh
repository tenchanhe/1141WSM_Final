python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input pyserini_data/ \
  --index indexes/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw