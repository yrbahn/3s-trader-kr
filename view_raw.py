import sqlite3
import json
import sys
import os

def view_db(ticker):
    db_path = "state/trader_database.db"
    if not os.path.exists(db_path):
        print("Database not found.")
        return

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute("SELECT * FROM raw_data WHERE ticker = ?", (ticker,))
        row = cur.fetchone()
        
        if not row:
            print(f"Ticker {ticker} not found.")
            return

        print(f"\n=== [RDB Raw Data Monitor: {row['name']} ({ticker})] ===")
        print(f"Last Update: {row['timestamp']}")
        
        for key in ['technical', 'fundamental', 'dart', 'investor', 'news']:
            print(f"\n[{key.upper()}]")
            data = json.loads(row[key])
            print(json.dumps(data, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    if len(sys.argv) > 1:
        view_db(sys.argv[1])
    else:
        print("Usage: python3 view_raw.py <ticker.KQ>")
