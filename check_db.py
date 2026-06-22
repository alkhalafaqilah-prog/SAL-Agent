import sqlite3

conn = sqlite3.connect('app/beamdata_crm.db')
conn.row_factory = sqlite3.Row
cur = conn.cursor()

# Show all tables
cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [r[0] for r in cur.fetchall()]
print("Tables:", tables)
print()

# Show rows in each table
for table in tables:
    cur.execute(f"SELECT COUNT(*) FROM {table}")
    count = cur.fetchone()[0]
    print(f"=== {table.upper()} ({count} rows) ===")
    cur.execute(f"SELECT * FROM {table} LIMIT 5")
    rows = cur.fetchall()
    for r in rows:
        print(dict(r))
    print()

conn.close()