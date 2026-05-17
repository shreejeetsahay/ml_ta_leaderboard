import sqlite3
conn = sqlite3.connect("file:leaderboard.db?mode=ro", uri=True, timeout=30)
cur = conn.cursor()
cur.execute("SELECT * FROM results where artifact = 'udz8gm_latest.pt'")
for row in cur.fetchall():
    print(row)
conn.close()

# import sqlite3

# NEW_STATUS = "Not displayed on leaderboard due to latest error"

# # Note: no mode=ro here, we need write access
# conn = sqlite3.connect("leaderboard.db", timeout=1.0)  # small timeout so we don't block long
# try:
#     cur = conn.cursor()
#     cur.execute(
#         """
#         UPDATE results
#         SET submitted_at = '2025-11-25 21:22:46.818398'
#         WHERE id=1086
#         """
#     )
#     conn.commit()
#     print("Rows updated:", cur.rowcount)
# finally:
#     conn.close()


