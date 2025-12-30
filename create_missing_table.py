import os
import mysql.connector

# DB config should match what you use in run.py
# If you use environment variables on the server, this will pick them up.
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "tender_user"),
    "password": os.getenv("DB_PASSWORD", "StrongPassword@123"),
    "database": os.getenv("DB_NAME", "tender_automation_with_ai"),
    "autocommit": True,
    "charset": "utf8mb4",
    "use_unicode": True,
}

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS Main_Relevency (
    id INT AUTO_INCREMENT PRIMARY KEY,
    bid_number VARCHAR(100),
    query TEXT,
    detected_category VARCHAR(100),
    relevancy_score FLOAT,
    relevant TINYINT(1),
    best_match JSON,
    top_matches JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_bid_number (bid_number)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""

CREATE_GEM_TENDERS_SQL = """
CREATE TABLE IF NOT EXISTS gem_tenders (
    id INT AUTO_INCREMENT PRIMARY KEY,
    page_no INT,
    bid_number VARCHAR(100) UNIQUE,
    detail_url TEXT,
    items TEXT,
    quantity VARCHAR(255),
    department TEXT,
    start_date VARCHAR(100),
    end_date VARCHAR(100),
    relevance INT,
    relevance_score FLOAT,
    match_count INT,
    match_relevency VARCHAR(50),
    matches JSON,
    matches_status VARCHAR(50),
    relevency_result INT,
    main_relevency_score FLOAT,
    dept VARCHAR(255),
    ra_no VARCHAR(255),
    ra_url TEXT,
    Representation_json JSON,
    Corrigendum_json JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
"""


def create_table():
    print("--- Database Setup Tool ---")
    print(f"Target Database: {DB_CONFIG['database']}")
    print(f"Target Host: {DB_CONFIG['host']}")

    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        print("\nExecuting CREATE TABLE statement for Main_Relevency...")
        cursor.execute(CREATE_TABLE_SQL)
        print("✅ Table 'Main_Relevency' created successfully (or already exists).")

        print("\nExecuting CREATE TABLE statement for gem_tenders...")
        cursor.execute(CREATE_GEM_TENDERS_SQL)
        print("✅ Table 'gem_tenders' created successfully (or already exists).")

        cursor.close()
        conn.close()
    except mysql.connector.Error as err:
        print(f"\n❌ Database Error: {err}")
    except Exception as e:
        print(f"\n❌ Error: {e}")


if __name__ == "__main__":
    create_table()
