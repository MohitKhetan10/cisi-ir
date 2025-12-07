from collections import defaultdict
import math
import numpy as np

class BM25:
    def __init__(self, docs, k1=1.2, b=0.75):
        self.docs = docs
        self.N = len(docs)
        self.k1 = k1
        self.b = b
        self.avgdl = sum(len(d) for d in docs) / self.N
        self.df = defaultdict(int)
        self.tf = [defaultdict(int) for _ in docs]
        for i, doc in enumerate(docs):
            seen = set()
            for term in doc:
                self.tf[i][term] += 1
                if term not in seen:
                    self.df[term] += 1
                    seen.add(term)

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def score(self, query_tokens):
        scores = [0.0] * self.N
        for i, doc in enumerate(self.docs):
            dl = len(doc)
            denom = 1 - self.b + self.b * (dl / self.avgdl)
            score = 0.0
            for term in query_tokens:
                tf = self.tf[i].get(term, 0)
                if tf == 0:
                    continue
                idf = self.idf(term)
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * denom)
            scores[i] = score
        return scores

def bm25_rank(query_tokens, bm25_model, doc_ids, topk=10):
    scores = bm25_model.score(query_tokens)
    ranked = np.argsort(-np.array(scores))
    return [(doc_ids[i], float(scores[i])) for i in ranked[:topk]]
