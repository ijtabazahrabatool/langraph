import sqlite3

DB_PATH = "chatbot.db"

#create threads table once
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(""" 
        CREATE TABLE IF NOT EXISTS threads(
        thread_id TEXT PRIMARY KEY,
        title TEXT NOT NULL
        )"""
                     )
        conn.commit()


# Save a new thread title permanently.
def save_thread(thread_id, title):
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            "INSERT OR IGNORE INTO threads (thread_id, title) VALUES (?, ?)",
            (str(thread_id), title)
        )
        conn.commit()

# Fetch all saved thread IDs and Titles for the sidebar UI.
def load_threads():
    with sqlite3.connect(DB_PATH) as conn: 
        cursor = conn.cursor()
        cursor.execute("SELECT thread_id, title FROM threads ORDER BY rowid DESC")
        rows = cursor.fetchall()
        return [{"id": row[0] , "title": row[1]} for row in rows]
