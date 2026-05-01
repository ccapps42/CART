import sqlite3

conn = sqlite3.connect("results.db")
conn.execute(
    "UPDATE configs SET status='pending', retry_count=0, started_at=NULL, completed_at=NULL "
    "WHERE status='failed'"
)
conn.commit()
rows = conn.execute(
    "SELECT config_id, d_model, n_loops, n_prelude, status, retry_count FROM configs WHERE retry_count=0 AND started_at IS NULL AND status='pending'"
).fetchall()
print(f"Reset {len(rows)} config(s):")
for r in rows:
    print(f"  d={r[1]} R={r[2]} P={r[3]}  id={r[0]}")
conn.close()
