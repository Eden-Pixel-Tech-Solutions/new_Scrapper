#!/usr/bin/env python3
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data/embeddings/endo_index.json")
OUT_PATH = os.path.join(ROOT, "data/embeddings/endo_embeddings.npy")

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

print("Loading index:", INDEX_PATH)
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

texts = [it["merged_text"] for it in items]

print("Encoding", len(texts), "items...")
emb = model.encode(texts, normalize_embeddings=True)

np.save(OUT_PATH, emb)

print("Saved:", OUT_PATH)
