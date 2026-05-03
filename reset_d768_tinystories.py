import sqlite3

conn = sqlite3.connect("results.db")

# Reset the R=2 P=2 run that completed on wrong (TinyStories-only) data
# Also reset anything still running
for status in ("complete", "running"):
    rows = conn.execute(
        "SELECT config_id, n_loops, n_prelude FROM configs WHERE d_model=768 AND status=?",
        (status,)
    ).fetchall()
    for row in rows:
        cid = row[0]
        conn.execute("DELETE FROM results WHERE config_id=?", (cid,))
        conn.execute("DELETE FROM train_log WHERE config_id=?", (cid,))
        conn.execute(
            "UPDATE configs SET status='pending', started_at=NULL, completed_at=NULL, error_msg=NULL WHERE config_id=?",
            (cid,)
        )
        print(f"  Reset R={row[1]} P={row[2]} (was {status})")

conn.commit()
conn.close()
print("Done.")
