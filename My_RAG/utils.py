import jsonlines

def load_jsonl(file_path):
    """Loads a JSONL file into a list of dictionaries."""
    docs = []
    with jsonlines.open(file_path, 'r') as reader:
        for obj in reader:
            docs.append(obj)
    return docs

def save_jsonl(file_path, data):
    """Saves a list of dictionaries into a JSONL file."""
    with jsonlines.open(file_path, mode='w') as writer:
        for item in data:
            writer.write(item)