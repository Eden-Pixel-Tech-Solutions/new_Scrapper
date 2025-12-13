#!/usr/bin/env python3
"""
global_relevancy.py
- Multi-query processing system for relevancy search across the global catalog
- Handles complex queries with multiple products separated by commas, semicolons, or newlines
- Uses embedding similarity + token overlap + category-specific boosts
- Routes "Analyser" queries to analyser_relevancy, "Endo" to endo_relevancy (if present)
- CLI: python global_relevancy.py "query1, query2, query3"
"""
import os
import json
import numpy as np
import re
import math
import argparse
import unicodedata
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------
#  SAFE IMPORTS FOR SPECIAL MODELS
# ------------------------------------------------------------------
HAS_ANALYSER_MODEL = False
HAS_ENDO_MODEL = False
analyser_predict = None
predict_endo = None

try:
    from analyser_relevancy import predict_relevancy as analyser_predict
    HAS_ANALYSER_MODEL = True
except Exception as e:
    print("Warning: analyser_relevancy not loaded:", e)

try:
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
    "5 part": "Analyser",
    "3 part": "Analyser",
    "6 part": "Analyser",
    "hematology": "Analyser",
    "cell counter": "Analyser",
    "automated analy": "Analyser",
    "cbc": "Analyser",
    "celquant": "Analyser",
    "autoloader": "Analyser",

    "biochemistry": "Analyser",
    "bio chemistry": "Analyser",
    "chemistry analy": "Analyser",
    "fully automatic biochemistry analyzer": "Analyser",
    "semi automatic bio chemistry analyser": "Analyser",
    "veterinary biochemistry analyzer": "Analyser",

    "elisa reader": "Analyser",
    "elisa washer": "Analyser",
    "elisa plate washer": "Analyser",
    "elisa test": "Analyser",
    "immunoassay analyzer": "Analyser",

    "coagulation": "Analyser",
    "coagulation analyzer": "Analyser",
    "semi automated coagulation analyser": "Analyser",

    "electrolyte": "Analyser",
    "electrolyte analy": "Analyser",
    "electrolyte analyzer": "Analyser",

    "pcr machine": "Analyser",
    "real time pcr": "Analyser",
    "rt-pcr": "Analyser",
    "rtpcr": "Analyser",
    "qpcr": "Analyser",
    "thermal cycler": "Analyser",

    "dna extraction system": "Analyser",
    "rna extraction": "Analyser",
    "dna extraction": "Analyser",

    "gel doc": "Analyser",
    "gel documentation system": "Analyser",

    "hplc analy": "Analyser",
    "hplc system": "Analyser",
    "liquid chromatograph": "Analyser",

    "immunoassay": "Analyser",
    "immunoassay analyzer reagents": "Analyser",
    "electrolyte analyzer reagents": "Analyser",
    "coagulation analyzer reagents": "Analyser",
    "biochemistry reagent kit": "Analyser",

    "poct": "Analyser",
    "point of care": "Analyser",
    "glucometer": "Analyser",
    # Endo
    "bonewax": "Endo",
    "bone wax": "Endo",
    "catgut": "Endo",
    "suture": "Endo",
    "sutures": "Endo",
    "endo": "Endo",
    "aspiron": "Endo",
    "Polyglactine": "Endo",
    "endo": "Endo",
    "endoscope": "Endo",
    "endoscopes": "Endo",
    "endoscopic": "Endo",
    "endoscopic equipment": "Endo",
    "endoscopic accessories": "Endo",

    "trocar": "Endo",
    "endocutter": "Endo",
    "endo cutter": "Endo",
    "reload linear cutter": "Endo",
    "circular stapler": "Endo",
    "hemorrhoid stapler": "Endo",
    "skin stapler": "Endo",
    "stapler": "Endo",

    "suture": "Endo",
    "suture item": "Endo",
    "ligation clip": "Endo",
    "fixation device": "Endo",
    "powered fixation": "Endo",

    "hernia": "Endo",
    "hernia mesh": "Endo",
    "herniamesh": "Endo",
    "anatomical mesh": "Endo",

    "haemostat": "Endo",
    "haemostatics": "Endo",
    "hemostat": "Endo",
    "gelatin sponge": "Endo",
    "oxidised cellulose": "Endo",
    "oxidised regenerated cellulose": "Endo",
    "bone wax": "Endo",
    "umbilical cotton tape": "Endo",

    "laparoscop": "Endo",
    "minimal invasive": "Endo",
    "ultrasonic surg": "Endo",
    "surgical system": "Endo",

    "diode laser": "Endo",
    "laser diode": "Endo",
    "laser fiber": "Endo",
    "fibre laser": "Endo",
    "fiber laser": "Endo",
    "laser ablation": "Endo",

    "robotics": "Endo",
    "robot machines": "Endo",
    "robot components": "Endo",
    "robotic assisted": "Endo",
    "robotic surg": "Endo",
    "surg robot": "Endo",
    "robot for surg": "Endo",
    "ras": "Endo",

    "joint replace": "Endo",
    "knee replace": "Endo",
    "tkr robot": "Endo",

    "endotracheal tubes": "Endo",
    "transducers": "Endo",
    "cartridges": "Endo",

    "disposable medical item": "Endo",
    "medical consumable": "Endo",
    "medical item": "Endo",

    "intra uterine": "Endo",
    "iud": "Endo",
    "iucd": "Endo",
    "hormonal intrauterine": "Endo",
    "anti contraceptive": "Endo",
    "contraceptive": "Endo",

    "skill lab": "Endo",
    "polyglactine": "Endo",
    "CHROMIC CATGUT": "Endo",
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
    out["emb_score"] = float(raw.get("emb_score") or raw.get("emb") or 0.0)
    out["token_score"] = float(raw.get("token_score") or raw.get("token") or 0.0)
    out["title_overlap"] = float(raw.get("title_overlap") or raw.get("title_tok") or 0.0)
    out["raw_score"] = float(raw.get("raw_score") or 0.0)
    out["relevancy"] = float(raw.get("relevancy") or raw.get("relevancy_score") or raw.get("relevancy_local") or 0.0)
    return out

# ----------------- query splitting logic -----------------
def split_multi_query(query: str) -> List[str]:
    """
    Split a complex query into individual product queries.
    Handles various separators: commas, semicolons, newlines, 'and', numbered lists
    
    Examples:
    - "product1, product2, product3"
    - "1. product1 2. product2"
    - "supply of - product1, product2"
    """
    # Normalize the query first
    query = normalize_text(query)
    
    # Remove common prefixes like "supply of -", "requirement of", etc.
    prefixes_to_remove = [
        r"^supply\s+of\s*[-:]*\s*",
        r"^requirement\s+of\s*[-:]*\s*",
        r"^procurement\s+of\s*[-:]*\s*",
        r"^purchase\s+of\s*[-:]*\s*",
        r"^quotation\s+for\s*[-:]*\s*",
    ]
    
    for prefix_pattern in prefixes_to_remove:
        query = re.sub(prefix_pattern, "", query, flags=re.IGNORECASE)
    
    # Split by common delimiters: comma, semicolon, newline, pipe
    # Also split by numbered lists like "1.", "2)", etc.
    parts = re.split(r'[,;|\n]|\d+[\.)]\s*', query)
    
    # Clean up each part
    queries = []
    for part in parts:
        part = part.strip()
        
        # Skip empty parts or very short parts (likely noise)
        if len(part) < 3:
            continue
            
        # Remove leading/trailing dashes, colons, etc.
        part = re.sub(r'^[-:•\s]+|[-:•\s]+$', '', part)
        
        # Skip if still too short after cleaning
        if len(part) < 5:
            continue
        
        # Remove trailing location/identifier patterns like "- gmc jagdalpur equipments"
        part = re.sub(r'\s*[-–]\s*[a-z\s]+equipments?\s*$', '', part, flags=re.IGNORECASE)
        part = re.sub(r'\s*[-–]\s*[a-z\s]+hospital\s*$', '', part, flags=re.IGNORECASE)
        
        queries.append(part.strip())
    
    # If no queries were extracted (maybe single query), return original
    if not queries:
        return [query.strip()]
    
    return queries

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
#                  SINGLE QUERY PREDICT FUNCTION
# ------------------------------------------------------------------
def predict_single(query: str, top_k: int = TOP_K) -> Dict[str, Any]:
    """Process a single query and return results"""
    query = normalize_text(query)
    detected_category = detect_category_from_query(query, INDEX)

    # -------------- route to analyser model if detected ----------------
    if detected_category and detected_category.lower() == "analyser" and HAS_ANALYSER_MODEL:
        try:
            print(f"  → Routing to analyser_relevancy.py")
            r = analyser_predict(query, top_k=top_k)
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
            print(f"  → Error running analyser_relevancy: {e}")

    # -------------- route to endo model if detected ----------------
    if detected_category and detected_category.lower() == "endo" and HAS_ENDO_MODEL:
        try:
            print(f"  → Routing to endo_relevancy.py")
            r = predict_endo(query, top_k=top_k)
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
            print(f"  → Error running endo_relevancy: {e}")

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
        "top_matches": top,
        "model_used": "global_index"
    }

# ------------------------------------------------------------------
#                  MULTI-QUERY PREDICT FUNCTION
# ------------------------------------------------------------------
def predict(query: str, top_k: int = TOP_K, return_individual: bool = True) -> Dict[str, Any]:
    """
    Main prediction function that handles both single and multi-query inputs.
    
    Args:
        query: Input query string (can contain multiple queries separated by delimiters)
        top_k: Number of top results to return per query
        return_individual: If True, returns detailed results for each query separately
    
    Returns:
        Dictionary containing:
        - is_multi_query: Whether multiple queries were detected
        - query_count: Number of individual queries found
        - results: List of results (one per query if multi-query)
        - summary: Overall summary statistics
    """
    # Split the query into individual queries
    individual_queries = split_multi_query(query)
    
    is_multi = len(individual_queries) > 1
    
    print(f"\n{'='*70}")
    print(f"Processing {'MULTI-QUERY' if is_multi else 'SINGLE QUERY'} input")
    print(f"Found {len(individual_queries)} individual {'queries' if is_multi else 'query'}")
    print(f"{'='*70}\n")
    
    all_results = []
    
    for idx, single_query in enumerate(individual_queries, 1):
        print(f"[Query {idx}/{len(individual_queries)}]: {single_query}")
        
        result = predict_single(single_query, top_k=top_k)
        result["query_number"] = idx
        all_results.append(result)
        
        # Print quick summary
        if result.get("best_match"):
            best = result["best_match"]
            print(f"  ✓ Best: {best.get('title', 'N/A')} (relevancy: {result['relevancy_score']:.3f})")
        else:
            print(f"  ✗ No match found")
        print()
    
    # Compute summary statistics
    relevant_count = sum(1 for r in all_results if r.get("relevant"))
    avg_relevancy = sum(r.get("relevancy_score", 0) for r in all_results) / len(all_results) if all_results else 0
    
    summary = {
        "total_queries": len(individual_queries),
        "relevant_matches": relevant_count,
        "irrelevant_matches": len(all_results) - relevant_count,
        "average_relevancy": float(avg_relevancy),
        "success_rate": float(relevant_count / len(all_results)) if all_results else 0.0
    }
    
    response = {
        "is_multi_query": is_multi,
        "original_query": query,
        "query_count": len(individual_queries),
        "individual_queries": individual_queries,
        "results": all_results,
        "summary": summary
    }
    
    return response

# ------------------------------------------------------------------
#                  BATCH PROCESSING FUNCTION
# ------------------------------------------------------------------
def predict_batch(queries: List[str], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Process multiple queries in batch mode.
    Each query can itself contain multiple sub-queries.
    
    Args:
        queries: List of query strings
        top_k: Number of top results per query
    
    Returns:
        List of prediction results
    """
    results = []
    for query in queries:
        result = predict(query, top_k=top_k)
        results.append(result)
    return results

# ------------------------------------------------------------------
#                  OUTPUT FORMATTING
# ------------------------------------------------------------------
def format_output(result: Dict[str, Any], verbose: bool = False) -> str:
    """Format the result for console output"""
    lines = []
    lines.append("\n" + "="*70)
    lines.append("MULTI-QUERY RELEVANCY SEARCH RESULTS")
    lines.append("="*70)
    
    if result.get("is_multi_query"):
        lines.append(f"\n📋 Original Query: {result['original_query'][:100]}...")
        lines.append(f"🔍 Detected {result['query_count']} individual queries\n")
        
        for idx, query_result in enumerate(result["results"], 1):
            lines.append(f"\n{'─'*70}")
            lines.append(f"Query {idx}: {query_result['query']}")
            lines.append(f"{'─'*70}")
            
            if query_result.get("detected_category"):
                lines.append(f"Category: {query_result['detected_category']}")
            
            best = query_result.get("best_match")
            if best and best.get("title"):
                lines.append(f"\n✓ BEST MATCH:")
                lines.append(f"  Product Code: {best.get('product_code', 'N/A')}")
                lines.append(f"  Title: {best['title']}")
                lines.append(f"  Category: {best.get('category', 'N/A')}")
                lines.append(f"  Type: {best.get('type', 'N/A')}")
                lines.append(f"  Relevancy: {query_result['relevancy_score']:.3f}")
                
                if verbose and best.get("specification"):
                    lines.append(f"  Specification: {best['specification'][:200]}...")
            else:
                lines.append(f"\n✗ NO MATCH FOUND")
            
            if verbose and query_result.get("top_matches"):
                lines.append(f"\n  Other top matches:")
                for i, match in enumerate(query_result["top_matches"][1:4], 2):
                    lines.append(f"    {i}. {match.get('title', 'N/A')} (rel: {match.get('relevancy', 0):.3f})")
        
        lines.append(f"\n{'='*70}")
        lines.append("SUMMARY")
        lines.append(f"{'='*70}")
        summary = result["summary"]
        lines.append(f"Total Queries: {summary['total_queries']}")
        lines.append(f"Relevant Matches: {summary['relevant_matches']}")
        lines.append(f"Success Rate: {summary['success_rate']*100:.1f}%")
        lines.append(f"Average Relevancy: {summary['average_relevancy']:.3f}")
        
    else:
        # Single query output
        query_result = result["results"][0]
        lines.append(f"\nQuery: {query_result['query']}")
        
        if query_result.get("detected_category"):
            lines.append(f"Category: {query_result['detected_category']}")
        
        best = query_result.get("best_match")
        if best and best.get("title"):
            lines.append(f"\n✓ BEST MATCH:")
            lines.append(f"  Product Code: {best.get('product_code', 'N/A')}")
            lines.append(f"  Title: {best['title']}")
            lines.append(f"  Category: {best.get('category', 'N/A')}")
            lines.append(f"  Relevancy: {query_result['relevancy_score']:.3f}")
        else:
            lines.append(f"\n✗ NO MATCH FOUND")
    
    lines.append("="*70 + "\n")
    return "\n".join(lines)

# -------------------- CLI --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Global relevancy search with multi-query support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Single query:
    python global_relevancy.py "5 part hematology analyser"
  
  Multi-query (comma-separated):
    python global_relevancy.py "5 part analyser, laparoscope, microscope"
  
  Complex multi-query:
    python global_relevancy.py "supply of - 5 part analyser, laparoscope, microscope - hospital"
  
  With options:
    python global_relevancy.py "analyser, microscope" --top 3 --verbose
    python global_relevancy.py "query" --json
        """
    )
    parser.add_argument("query", nargs="*", help="Query text (supports multiple queries)")
    parser.add_argument("--top", type=int, default=5, help="Top K results per query")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--batch", type=str, help="Path to file with queries (one per line)")
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch mode from file
        with open(args.batch, 'r', encoding='utf-8') as f:
            queries = [line.strip() for line in f if line.strip()]
        
        results = predict_batch(queries, top_k=args.top)
        
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            for result in results:
                print(format_output(result, verbose=args.verbose))
    else:
        # Single/multi query mode
        q = " ".join(args.query) if args.query else "5 Part Automated Hematology Analyser (V2) (Q2) , Operating Laparoscope with instrumentation , HD Endoscopy system , Diagnostic clinical Audiometer , Eclampsia Cot , Fully Automatic Rotary Microtome , Binocular Microscope , Monocular Version Compound Microscope, Polyglactine 910 Polyglycolic Acid"
        
        res = predict(q, top_k=args.top)
        
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(format_output(res, verbose=args.verbose))