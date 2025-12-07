import re

def parse_cisi_docs(path):
    with open(path, 'r', errors='ignore') as f:
        raw = f.read()
    parts = re.split(r'\n\.I\s+(\d+)\n', raw)
    docs = {}
    for i in range(1, len(parts), 2):
        doc_id = int(parts[i])
        block = parts[i+1]
        def extract(field):
            m = re.search(rf'\.{field}\n(.*?)(?=\n\.[TAW]\n|\Z)', block, flags=re.S)
            return m.group(1).strip() if m else ''
        docs[doc_id] = {
            'title': extract('T'),
            'authors': extract('A'),
            'abstract': extract('W'),
            'text': f"{extract('T')} {extract('W')}".strip()
        }
    return docs

def parse_cisi_queries(path):
    with open(path, 'r', errors='ignore') as f:
        raw = f.read()
    parts = re.split(r'\n\.I\s+(\d+)\n', raw)
    queries = {}
    for i in range(1, len(parts), 2):
        qid = int(parts[i])
        block = parts[i+1]
        m = re.search(r'\.W\n(.*?)(?=\n\.I|\Z)', block, flags=re.S)
        queries[qid] = m.group(1).strip() if m else block.strip()
    return queries

def parse_cisi_qrels(path):
    qrels = {}
    with open(path, 'r', errors='ignore') as f:
        for line in f:
            parts = re.split(r'\s+', line.strip())
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                qid, docid = int(parts[0]), int(parts[1])
                qrels.setdefault(qid, set()).add(docid)
    return qrels
