
import time
import logging
import mysql.connector
from tasks import process_tender

# Config
FETCH_LIMIT = 50
SLEEP_BETWEEN_BATCHES = 10

# DB Config (Same as tasks.py effectively, but running on host likely)
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "",
    "database": "tender_automation_with_ai"
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [DISPATCHER] %(message)s")
logger = logging.getLogger("dispatcher")

def db_connect():
    return mysql.connector.connect(**DB_CONFIG)

def fetch_pending_tenders(limit=FETCH_LIMIT):
    conn = None
    try:
        conn = db_connect()
        cur = conn.cursor(dictionary=True)
        cur.execute("""
            SELECT bid_number, detail_url
            FROM gem_tenders
            WHERE bid_number NOT IN (SELECT bid_number FROM gem_tender_docs)
            LIMIT %s
        """, (limit,))
        rows = cur.fetchall()
        cur.close()
        return rows or []
    except Exception as e:
        logger.error(f"DB Fetch Error: {e}")
        return []
    finally:
        if conn: conn.close()

def main_loop():
    logger.info("Starting Job Dispatcher...")
    while True:
        try:
            rows = fetch_pending_tenders()
            if not rows:
                logger.info("No pending tenders. Sleeping...")
                time.sleep(SLEEP_BETWEEN_BATCHES)
                continue
            
            logger.info(f"Main Loop: Found {len(rows)} pending tenders. Dispatching to Celery...")
            
            for row in rows:
                # Dispatch task to Celery
                process_tender.delay(row)
                logger.info(f"Queued bid {row.get('bid_number')}")
            
            # Simple throttle to not overwhelm if DB is huge
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("Dispatcher stopped by user.")
            break
        except Exception as e:
            logger.exception("Dispatcher main loop error")
            time.sleep(10)

if __name__ == "__main__":
    main_loop()
