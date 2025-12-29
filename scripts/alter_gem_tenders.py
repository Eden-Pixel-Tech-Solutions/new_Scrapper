import os
import mysql.connector

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "tender_automation_with_ai"),
    "autocommit": True,
}

def alter_table():
    print("--- Altering gem_tenders Table ---")
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()

        # Add ra_no
        try:
            print("Adding column 'ra_no'...")
            cursor.execute("ALTER TABLE gem_tenders ADD COLUMN ra_no VARCHAR(100) DEFAULT NULL")
            print("✅ Column 'ra_no' added.")
        except mysql.connector.Error as err:
            if err.errno == 1060: # Duplicate column name
                print("⚠️ Column 'ra_no' already exists.")
            else:
                print(f"❌ Error adding 'ra_no': {err}")

        # Add ra_url
        try:
            print("Adding column 'ra_url'...")
            cursor.execute("ALTER TABLE gem_tenders ADD COLUMN ra_url TEXT DEFAULT NULL")
            print("✅ Column 'ra_url' added.")
        except mysql.connector.Error as err:
            if err.errno == 1060: # Duplicate column name
                print("⚠️ Column 'ra_url' already exists.")
            else:
                print(f"❌ Error adding 'ra_url': {err}")

        cursor.close()
        conn.close()
    except Exception as e:
        print(f"❌ Database Connection Error: {e}")

if __name__ == "__main__":
    alter_table()
