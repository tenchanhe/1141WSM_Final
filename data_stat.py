import jsonlines
from collections import Counter, defaultdict

# file_path = "./dragonball_dataset/dragonball_docs.jsonl"
file_path = "./dragonball_dataset/dragonball_queries.jsonl"

domain_counter = Counter()
language_counter = Counter()
query_type_counter = Counter()
domain_examples = defaultdict(list)
language_examples = defaultdict(list)
query_type_examples = defaultdict(list)
max_examples = 3

with jsonlines.open(file_path, 'r') as reader:
    for obj in reader:
        # domain
        domain = obj.get('domain', None)
        if domain is not None:
            domain_counter[domain] += 1
            if len(domain_examples[domain]) < max_examples:
                domain_examples[domain].append(obj)
        # language
        language = obj.get('language', None)
        if language is not None:
            language_counter[language] += 1
            if len(language_examples[language]) < max_examples:
                language_examples[language].append(obj)
        # query_type
        query = obj.get('query', {})
        query_type = query.get('query_type', None)
        if query_type is not None:
            query_type_counter[query_type] += 1
            if len(query_type_examples[query_type]) < max_examples:
                query_type_examples[query_type].append(obj)

print("domain 統計：")
for k, v in domain_counter.items():
    print(f"  {k}: {v}")

print("\nlanguage 統計：")
for k, v in language_counter.items():
    print(f"  {k}: {v}")

print("\nquery_type 統計：")
for k, v in query_type_counter.items():
    print(f"  {k}: {v}")

# # 從每個 query_type 中各抽一題作為測試集
# import jsonlines
# test_samples = []
# for qtype, samples in query_type_examples.items():
#     if samples:
#         test_samples.append(samples[0])  # 取第一筆作為代表

# test_path = "./dragonball_dataset/test.jsonl"
# with jsonlines.open(test_path, mode='w') as writer:
#     for item in test_samples:
#         writer.write(item)
# print(f"\n已將每個 query_type 各抽一題存入 {test_path}")
