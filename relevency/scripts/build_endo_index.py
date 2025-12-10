#!/usr/bin/env python3
import os
import json
import pandas as pd
import re
import unicodedata

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
XLSX_PATH = os.path.join(ROOT, "data", "endo.xlsx")
OUT_PATH = os.path.join(ROOT, "data", "embeddings", "endo_index.json")

def norm(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"\s+", " ", s.strip())
    return s

print("Reading:", XLSX_PATH)
df = pd.read_excel(XLSX_PATH)

items = []

for _, row in df.iterrows():
    type_ = norm(row.get("Type", ""))
    code = norm(row.get("SKU Code", ""))
    short = norm(row.get("Short Specification", ""))
    detailed = norm(row.get("Detailed Specification", ""))
    brand = norm(row.get("Brand", ""))
    sub_brand = norm(row.get("Sub Brand", ""))

    title = f"{short}".strip()

    merged = " ".join([
        title, detailed, type_, brand, sub_brand
    ])

    items.append({
        "product_code": code,
        "title": title,
        "type": type_,
        "brand": brand,
        "sub_brand": sub_brand,
        "short_spec": short,
        "detailed_spec": detailed,
        "merged_text": merged
    })

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(items, f, indent=2, ensure_ascii=False)

print("Saved:", OUT_PATH)
print("Total items:", len(items))
