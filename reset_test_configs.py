import sqlite3

conn = sqlite3.connect("results.db")
conn.execute(
    "UPDATE configs SET status='pending', started_at=NULL, completed_at=NULL "
    "WHERE config_id IN ('4fccb01a3c2e2a33','7e707fc287e277ed')"
)
conn.commit()
rows = conn.execute(
    "SELECT config_id, d_model, n_loops, n_prelude, status FROM configs "
    "WHERE config_id IN ('4fccb01a3c2e2a33','7e707fc287e277ed')"
).fetchall()
for r in rows:
    print(r)
conn.close()
