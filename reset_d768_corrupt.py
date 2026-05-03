import sqlite3

conn = sqlite3.connect("results.db")

# All d=768 configs that ran with buggy gradient checkpointing
target_ids = [
    "1a059b35cfc44d59",  # R=2 P=2 complete
    "258f3051cf992c01",  # R=2 P=3 complete
    "a6bfd2964205ac5d",  # R=4 P=2 complete
    "25b6082d73b4d1ad",  # R=4 P=3 running (killed)
]

for cid in target_ids:
    conn.execute("DELETE FROM results WHERE config_id=?", (cid,))
    conn.execute("DELETE FROM train_log WHERE config_id=?", (cid,))
    conn.execute(
        "UPDATE configs SET status='pending', started_at=NULL, completed_at=NULL, error_msg=NULL "
        "WHERE config_id=?", (cid,)
    )

conn.commit()

rows = conn.execute(
    "SELECT config_id, d_model, n_loops, n_prelude, status FROM configs WHERE d_model=768 ORDER BY n_loops, n_prelude"
).fetchall()
print("d=768 config statuses after reset:")
for r in rows:
    print(f"  d={r[1]} R={r[2]} P={r[3]}  {r[4]}")

conn.close()
