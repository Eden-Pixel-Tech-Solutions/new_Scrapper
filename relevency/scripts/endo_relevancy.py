#!/usr/bin/env python3
import os, json, numpy as np, re, math
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data/embeddings/endo_index.json")
EMB_PATH = os.path.join(ROOT, "data/embeddings/endo_embeddings.npy")

EMB_WEIGHT = 1.0
TOKEN_WEIGHT = 0.35
TITLE_WEIGHT = 0.5

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

def tokenize(q):
    return re.findall(r"[a-zA-Z0-9]+", q.lower())

def token_overlap(q, target):
    q_tokens = set(tokenize(q))
    t_tokens = set(tokenize(target))
    q_tokens = {t for t in q_tokens if len(t) > 2}
    t_tokens = {t for t in t_tokens if len(t) > 2}
    if not q_tokens:
        return 0.0
    return len(q_tokens & t_tokens) / len(q_tokens)

def predict_endo(query, top_k=5):
    with open(INDEX_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    emb = np.load(EMB_PATH)
    q_emb = model.encode([query], normalize_embeddings=True)[0]

    sims = np.dot(emb, q_emb)

    results = []
    for i, item in enumerate(data):
        emb_score = float(sims[i])
        tok = token_overlap(query, item["merged_text"])
        title_tok = token_overlap(query, item["title"])
        raw = EMB_WEIGHT * emb_score + TOKEN_WEIGHT * tok + TITLE_WEIGHT * title_tok

        results.append({
            "index": i,
            "title": item["title"],
            "product_code": item["product_code"],
            "type": item["type"],
            "brand": item["brand"],
            "sub_brand": item["sub_brand"],
            "spec": item["detailed_spec"],
            "emb_score": emb_score,
            "token_score": tok,
            "title_overlap": title_tok,
            "raw_score": raw,
            "relevancy": float(1 / (1 + math.exp(-raw)))
        })

    results.sort(key=lambda x: x["raw_score"], reverse=True)
    return {
        "query": query,
        "best_match": results[0],
        "top_matches": results[:top_k],
        "relevancy_score": results[0]["relevancy"],
        "relevant": results[0]["relevancy"] >= 0.5
    }

if __name__ == "__main__":
    print(json.dumps(predict_endo("Silk black 3-0 x 90 - 25mm half circlr round body"), indent=2))
