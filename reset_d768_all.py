import sqlite3

conn = sqlite3.connect("results.db")

# Reset all d=768 configs — complete, running, or pending — back to clean pending
rows = conn.execute("SELECT config_id, n_loops, n_prelude, status FROM configs WHERE d_model=768").fetchall()

for row in rows:
    cid = row[0]
    conn.execute("DELETE FROM results WHERE config_id=?", (cid,))
    conn.execute("DELETE FROM train_log WHERE config_id=?", (cid,))
    conn.execute(
        "UPDATE configs SET status='pending', started_at=NULL, completed_at=NULL, error_msg=NULL WHERE config_id=?",
        (cid,)
    )

conn.commit()
print(f"Reset {len(rows)} d=768 config(s) to pending:")
for row in rows:
    print(f"  R={row[1]} P={row[2]}  was: {row[3]}")
conn.close()
