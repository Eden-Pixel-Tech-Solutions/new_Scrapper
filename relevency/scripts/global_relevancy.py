#!/usr/bin/env python3
"""
global_relevancy.py
- Single entrypoint for relevancy search across the global catalog
- Uses embedding similarity + token overlap + category-specific boosts
- Routes "Analyser" queries to analyser_relevancy, "Endo" to endo_relevancy (if present)
- CLI: python global_relevancy.py "your query here"
"""
import os
import json
import numpy as np
import re
import math
import argparse
import unicodedata
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
#  SAFE IMPORTS FOR SPECIAL MODELS
# ------------------------------------------------------------------
HAS_ANALYSER_MODEL = False
HAS_ENDO_MODEL = False
analyser_predict = None
predict_endo = None

try:
    # analyser_relevancy.predict_relevancy(query, top_k=...)
    from analyser_relevancy import predict_relevancy as analyser_predict
    HAS_ANALYSER_MODEL = True
except Exception as e:
    # not fatal; fallback to global index
    print("Warning: analyser_relevancy not loaded:", e)

try:
    # endo_relevancy.predict_endo(query, top_k=...)
    from endo_relevancy import predict_endo as predict_endo
    HAS_ENDO_MODEL = True
except Exception as e:
    print("Warning: endo_relevancy not loaded:", e)

# ------------------------------------------------------------------

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INDEX_PATH = os.path.join(ROOT, "data", "embeddings", "global_index.json")
EMB_PATH = os.path.join(ROOT, "data", "embeddings", "global_embeddings.npy")
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# Tunable weights
EMB_WEIGHT = 1.0
TOKEN_WEIGHT = 0.35
TITLE_WEIGHT = 0.5
CATEGORY_BOOST = 0.25
TOP_K = 5

# keyword->category map (extendable)
CATEGORY_KEYWORDS = {
    "pipette": "Pipettes",
    "pipettes": "Pipettes",
    "fixed volume": "Pipettes",
    "variable": "Pipettes",
    "dengue": "Elisa",
    "ns1": "Elisa",
    "hiv": "Elisa",
    "hbsag": "Elisa",
    "crp": "Turbidimetry",
    "rf": "Nephelometry",
    "aso": "Nephelometry",
    "control": "Controls",
    "control kit": "Controls",
    "system pack": "System Packs",
    "albumin": "System Packs",
    "anti a": "BloodGroup",
    "anti b": "BloodGroup",
    "anti d": "BloodGroup",
    "anti ab": "BloodGroup",
    "blood grouping": "BloodGroup",
    "reagent": "Reagents",
    "reagents": "Reagents",
    "analyser": "Analyser",
    "analyzer": "Analyser",
    "hematology": "Analyser",
    "hb": "Analyser",
    "meriscreen": "Meriscreen",
    "rapid": "Rapids",
    "elisa": "Elisa",
    "nephelometry": "Nephelometry",
    "turbidimetry": "Turbidimetry",
    "5 part": "Analyser",
    "3 part": "Analyser",
    "cbc": "Analyser",
    "celquant": "Analyser",
    "autoloader": "Analyser",
    # Endo keywords
    "bonewax": "Endo",
    "bone wax": "Endo",
    "catgut": "Endo",
    "suture": "Endo",
    "endo": "Endo",
    "aspiron": "Endo",
}

# ----------------- utility helpers -----------------
def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[\u200B-\u200F\u202A-\u202E\u00A0]", " ", s)
    s = s.replace("\n", " ").replace("\r", " ")
    s = s.strip()
    return s

def norm_token_list(s: str):
    s = normalize_text(s).lower()
    tokens = re.findall(r"[a-z0-9]+", s)
    return [t for t in tokens if len(t) > 2]

def token_set(s: str):
    return set(norm_token_list(s))

def token_overlap(query: str, target: str) -> float:
    q = token_set(query)
    t = token_set(target)
    if not q:
        return 0.0
    return len(q & t) / len(q)

def detect_category_from_query(q: str, index_items=None):
    ql = normalize_text(q).lower()

    hits = []
    for kw, cat in CATEGORY_KEYWORDS.items():
        if kw in ql:
            hits.append((len(kw), kw, cat))

    if hits:
        hits.sort(reverse=True)
        return hits[0][2]

    if index_items:
        q_tokens = token_set(q)
        best = None
        best_score = 0
        for it in index_items:
            combined = " ".join([
                it.get("title") or "",
                it.get("category") or "",
                it.get("type") or "",
                it.get("merged_text") or ""
            ])
            score = len(q_tokens & token_set(combined))
            if score > best_score:
                best_score = score
                best = it.get("category") or it.get("type")
        if best_score > 0:
            return best

    return None

def safe_product_code(item):
    candidates = []
    for k in ("product_code", "product code", "productcode", "code", "product"):
        v = item.get(k)
        if v:
            candidates.append(str(v).strip())

    v0 = str(item.get("product_code") or "").strip()
    if v0:
        candidates.insert(0, v0)

    for c in candidates:
        if re.search(r"[A-Za-z]", c) and re.search(r"[0-9]", c):
            return c

    for c in candidates:
        low = c.lower()
        if low in ("regular", "no slab") or low.startswith("slab"):
            continue
        if c:
            return c
    return ""

def sanitize_match(raw: dict) -> dict:
    """Ensure match dict has expected keys and Python-native numeric types."""
    if not raw:
        return {
            "index": None,
            "product_code": "",
            "title": "",
            "type": "",
            "category": "",
            "specification": "",
            "emb_score": 0.0,
            "token_score": 0.0,
            "title_overlap": 0.0,
            "raw_score": 0.0,
            "relevancy": 0.0
        }
    out = {}
    out["index"] = int(raw.get("index")) if raw.get("index") is not None else None
    out["product_code"] = str(raw.get("product_code") or "") 
    out["title"] = str(raw.get("title") or "")
    out["type"] = str(raw.get("type") or "")
    out["category"] = str(raw.get("category") or "")
    out["specification"] = str(raw.get("specification") or raw.get("spec") or raw.get("specification_text") or "")
    # numeric fields -> plain python float
    out["emb_score"] = float(raw.get("emb_score") or raw.get("emb") or 0.0)
    out["token_score"] = float(raw.get("token_score") or raw.get("token") or 0.0)
    out["title_overlap"] = float(raw.get("title_overlap") or raw.get("title_tok") or 0.0)
    out["raw_score"] = float(raw.get("raw_score") or 0.0)
    out["relevancy"] = float(raw.get("relevancy") or raw.get("relevancy_score") or raw.get("relevancy_local") or 0.0)
    return out

# ----------------- load global index & embeddings -----------------
print("Loading index and embeddings...")
with open(INDEX_PATH, "r", encoding="utf-8") as f:
    INDEX_RAW = json.load(f)

INDEX = []
for it in INDEX_RAW:
    title = normalize_text(it.get("title") or it.get("Title") or "")
    prod = safe_product_code(it)
    spec = it.get("specification") or it.get("spec") or it.get("specification_text") or ""
    spec = normalize_text(spec)

    if "SLABS" in spec:
        if "kit_price" not in spec:
            spec += " kit_price: —"
        if "test_price" not in spec:
            spec += " test_price: —"

    merged = normalize_text(it.get("merged_text") or it.get("mergedText") or title or spec)

    item = {
        "index": int(it.get("index")) if it.get("index") not in (None, "") else None,
        "product_code": prod,
        "title": title,
        "type": normalize_text(it.get("type") or it.get("Type") or it.get("category") or ""),
        "category": normalize_text(it.get("category") or it.get("Category") or ""),
        "specification": spec,
        "merged_text": merged,
    }
    INDEX.append(item)

EMB = np.load(EMB_PATH)
MODEL = SentenceTransformer(MODEL_NAME)

# ------------------------------------------------------------------
#                  MAIN PREDICT FUNCTION
# ------------------------------------------------------------------
def predict(query, top_k=TOP_K):
    query = normalize_text(query)
    detected_category = detect_category_from_query(query, INDEX)

    # -------------- route to analyser model if detected ----------------
    if detected_category and detected_category.lower() == "analyser" and HAS_ANALYSER_MODEL:
        # call analyser model and normalize result
        try:
            print("Routing query to analyser_relevancy.py ...")
            r = analyser_predict(query, top_k=top_k)
            # analyser_predict returns a dict with keys: relevancy_score, relevant, best_match, top_matches
            # Ensure sanitization and return consistent structure
            best = sanitize_match(r.get("best_match") if isinstance(r.get("best_match"), dict) else r.get("best_match") or {})
            top_matches = [sanitize_match(m) for m in (r.get("top_matches") or [])]
            return {
                "query": query,
                "detected_category": "Analyser",
                "relevancy_score": float(r.get("relevancy_score") or r.get("relevancy") or 0.0),
                "relevant": bool(r.get("relevant") or False),
                "best_match": best,
                "top_matches": top_matches,
                "model_used": "analyser_relevancy"
            }
        except Exception as e:
            # fallback to global if analyser fails
            print("Error running analyser_relevancy:", e)

    # -------------- route to endo model if detected ----------------
    if detected_category and detected_category.lower() == "endo" and HAS_ENDO_MODEL:
        try:
            print("Routing query to endo_relevancy.py ...")
            r = predict_endo(query, top_k=top_k)
            # predict_endo returns best_match + top_matches + relevancy_score etc
            best = sanitize_match(r.get("best_match") or {})
            top_matches = [sanitize_match(m) for m in (r.get("top_matches") or [])]
            return {
                "query": query,
                "detected_category": "Endo",
                "relevancy_score": float(r.get("relevancy_score") or r.get("relevancy") or best.get("relevancy") or 0.0),
                "relevant": bool(r.get("relevant") or False),
                "best_match": best,
                "top_matches": top_matches,
                "model_used": "endo_relevancy"
            }
        except Exception as e:
            print("Error running endo_relevancy:", e)

    # -------------------- global fallback ----------------------------
    q_emb = MODEL.encode([query], normalize_embeddings=True)[0]
    sims = np.dot(EMB, q_emb)

    results = []
    q_lower = query.lower()

    for i, item in enumerate(INDEX):
        emb_score = float(sims[i]) if i < len(sims) else 0.0
        tok = float(token_overlap(query, item.get("merged_text", "")))
        title_tok = float(token_overlap(query, item.get("title", "")))

        raw = EMB_WEIGHT * emb_score + TOKEN_WEIGHT * tok + TITLE_WEIGHT * title_tok

        if detected_category:
            item_cat = (item.get("category") or "").lower()
            item_type = (item.get("type") or "").lower()
            if detected_category.lower() in item_cat or detected_category.lower() in item_type:
                raw += CATEGORY_BOOST

        pc = (item.get("product_code") or "").lower()
        if pc and re.search(r"\b" + re.escape(pc) + r"\b", q_lower):
            raw += 0.5

        match = {
            "index": int(item.get("index")) if item.get("index") is not None else int(i),
            "product_code": item.get("product_code") or "",
            "title": item.get("title"),
            "type": item.get("type"),
            "category": item.get("category"),
            "specification": item.get("specification"),
            "emb_score": float(emb_score),
            "token_score": float(tok),
            "title_overlap": float(title_tok),
            "raw_score": float(raw),
            "relevancy": float(1.0 / (1.0 + math.exp(-raw)))
        }
        results.append(match)

    # sort descending by raw_score
    results.sort(key=lambda x: x["raw_score"], reverse=True)

    top = results[:top_k]
    best = top[0] if top else None
    final_score = float(best["relevancy"]) if best else 0.0

    return {
        "query": query,
        "detected_category": detected_category,
        "relevancy_score": final_score,
        "relevant": bool(final_score >= 0.5),
        "best_match": best or sanitize_match({}),
        "top_matches": top
    }

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Global relevancy search")
    parser.add_argument("query", nargs="*", help="Query text")
    parser.add_argument("--top", type=int, default=5, help="Top K results")
    args = parser.parse_args()
    q = " ".join(args.query) if args.query else "5 Part Automated Hematology Analyser (V2) (Q2)"
    res = predict(q, top_k=args.top)
    print(json.dumps(res, indent=2, ensure_ascii=False))
