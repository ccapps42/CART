"""Reset ALL configs to pending and clear all results/logs. Full sweep restart."""
import sqlite3

conn = sqlite3.connect("results.db")

ids = [r[0] for r in conn.execute("SELECT config_id FROM configs").fetchall()]
for cid in ids:
    conn.execute("DELETE FROM results WHERE config_id=?", (cid,))
    conn.execute("DELETE FROM train_log WHERE config_id=?", (cid,))
    conn.execute(
        "UPDATE configs SET status='pending', started_at=NULL, completed_at=NULL, "
        "error_msg=NULL, retry_count=0 WHERE config_id=?", (cid,)
    )
conn.commit()

rows = conn.execute(
    "SELECT d_model, COUNT(*) FROM configs GROUP BY d_model ORDER BY d_model"
).fetchall()
total = conn.execute("SELECT COUNT(*) FROM configs WHERE status='pending'").fetchone()[0]
n_results = conn.execute("SELECT COUNT(*) FROM results").fetchone()[0]
n_logs = conn.execute("SELECT COUNT(*) FROM train_log").fetchone()[0]

print(f"Reset {total} configs to pending:")
for r in rows:
    print(f"  d={r[0]}: {r[1]} configs")
print(f"Results rows remaining: {n_results}  (should be 0)")
print(f"Train_log rows remaining: {n_logs}  (should be 0)")
conn.close()
