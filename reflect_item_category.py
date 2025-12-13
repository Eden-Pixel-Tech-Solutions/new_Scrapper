#!/usr/bin/env python3
"""
Reflect Item Category from OCR JSON into gem_tenders.items
Runs every 30 minutes continuously
"""

import os
import json
import time
import mysql.connector
from datetime import datetime

DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tender_automation_with_ai",
    "autocommit": False
}

INTERVAL_SECONDS = 30 * 60  # 30 minutes


# --------------------------------------------------
# SAFE JSON LOADER
# --------------------------------------------------
def load_json_safe(json_path: str):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)

    except json.JSONDecodeError:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                raw = f.read().strip()

            if raw.startswith('"source_file"'):
                raw = "{\n" + raw
            if not raw.endswith("}"):
                raw += "\n}"

            return json.loads(raw)

        except Exception:
            return None

    except Exception:
        return None


# --------------------------------------------------
# TABLE-BASED EXTRACTION
# --------------------------------------------------
def extract_item_category(json_data: dict) -> str | None:
    pages = json_data.get("pages", [])

    for page in pages:
        for table in page.get("tables", []):
            for row in table:
                if not row or len(row) < 2:
                    continue

                key, value = row[0], row[1]
                if not key or not value:
                    continue

                norm_key = key.lower().replace(" ", "").replace("/", "")
                if norm_key == "itemcategory":
                    return value.strip()

    return None


# --------------------------------------------------
# SINGLE RUN
# --------------------------------------------------
def process_once():
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor(dictionary=True)

    cursor.execute("""
        SELECT bid_number, json_path
        FROM gem_tender_docs
        WHERE did_reflected_master = 0
    """)
    rows = cursor.fetchall()

    print(f"\n[{datetime.now()}] 🔍 Pending: {len(rows)}")

    for row in rows:
        bid_number = row["bid_number"]
        json_path = row["json_path"]

        if not json_path or not os.path.exists(json_path):
            print(f"⚠️ JSON missing: {bid_number}")
            continue

        json_data = load_json_safe(json_path)
        if not json_data:
            print(f"❌ Invalid JSON: {bid_number}")
            continue

        item_category = extract_item_category(json_data)
        if not item_category:
            print(f"⚠️ Item Category not found: {bid_number}")
            continue

        cursor.execute("""
            UPDATE gem_tenders
            SET items = %s
            WHERE bid_number = %s
        """, (item_category, bid_number))

        cursor.execute("""
            UPDATE gem_tender_docs
            SET did_reflected_master = 1
            WHERE bid_number = %s
        """, (bid_number,))

        print(f"✅ Updated: {bid_number}")

    conn.commit()
    cursor.close()
    conn.close()


# --------------------------------------------------
# CONTINUOUS LOOP
# --------------------------------------------------
def main():
    print("🚀 Item Category Reflector started (runs every 30 minutes)")
    try:
        while True:
            process_once()
            print(f"⏳ Sleeping for 30 minutes...\n")
            time.sleep(INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\n🛑 Stopped by user")


if __name__ == "__main__":
    main()
