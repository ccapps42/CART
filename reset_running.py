import sqlite3

conn = sqlite3.connect("results.db")
rows = conn.execute("SELECT config_id, d_model, n_loops, n_prelude FROM configs WHERE status='running'").fetchall()
conn.execute("UPDATE configs SET status='pending', started_at=NULL WHERE status='running'")
conn.commit()
print(f"Reset {len(rows)} running config(s) to pending:")
for r in rows:
    print(f"  d={r[1]} R={r[2]} P={r[3]}  id={r[0]}")
conn.close()
