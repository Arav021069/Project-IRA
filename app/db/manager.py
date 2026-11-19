import sqlite3
from pathlib import Path
import threading

class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_folder = Path.home() / ".ira"
        self.db_folder.mkdir(parents=True, exist_ok=True)
        
        filename = db_path if db_path else "ira.db"
        self.db_path = str(self.db_folder / filename)
        
        # Persistent connection (thread-safe with check_same_thread=False)
        self._conn = None
        self._lock = threading.Lock()
        self._init_connection()

    def _init_connection(self):
        """Initialize the persistent connection with optimal pragmas."""
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._conn.execute("PRAGMA journal_mode = WAL")  # Better concurrency
        self._conn.execute("PRAGMA synchronous = NORMAL")  # Faster writes, safe with WAL
        self._conn.row_factory = sqlite3.Row

    def _get_cursor(self):
        """Thread-safe cursor access."""
        with self._lock:
            return self._conn.cursor()

    def init_db(self):
        cursor = self._get_cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            role TEXT,
            content TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions (id) ON DELETE CASCADE
        )
        """)
        self._conn.commit()

    def create_session(self, session_id: str, title: str):
        cursor = self._get_cursor()
        cursor.execute(
            "INSERT OR IGNORE INTO sessions (id, title) VALUES (?, ?)",
            (session_id, title)
        )
        self._conn.commit()

    def get_all_sessions(self):
        cursor = self._get_cursor()
        cursor.execute("SELECT id, title, timestamp FROM sessions ORDER BY timestamp DESC")
        return [dict(row) for row in cursor.fetchall()]

    def get_messages(self, session_id: str):
        cursor = self._get_cursor()
        cursor.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,)
        )
        return [dict(row) for row in cursor.fetchall()]

    def add_message(self, session_id: str, role: str, content: str):
        cursor = self._get_cursor()
        cursor.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content)
        )
        self._conn.commit()

    def delete_session(self, session_id: str):
        cursor = self._get_cursor()
        cursor.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        self._conn.commit()

    def session_exists(self, session_id: str) -> bool:
        cursor = self._get_cursor()
        cursor.execute("SELECT 1 FROM sessions WHERE id = ?", (session_id,))
        return cursor.fetchone() is not None

    def close(self):
        """Call on app shutdown."""
        if self._conn:
            self._conn.close()
            self._conn = None