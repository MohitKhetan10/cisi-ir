from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def build_tfidf_matrix(doc_tokens):
    docs = [' '.join(toks) for toks in doc_tokens]
    vectorizer = TfidfVectorizer(tokenizer=lambda s: s.split(), preprocessor=lambda s: s, lowercase=False)
    matrix = vectorizer.fit_transform(docs)
    return vectorizer, matrix

def tfidf_rank(query_tokens, vectorizer, matrix, doc_ids, topk=10):
    query = ' '.join(query_tokens)
    qvec = vectorizer.transform([query])
    scores = (matrix @ qvec.T).toarray().ravel()
    ranked = np.argsort(-scores)
    return [(doc_ids[i], float(scores[i])) for i in ranked[:topk]]
