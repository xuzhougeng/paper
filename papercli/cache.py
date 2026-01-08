"""SQLite-based caching for search results and LLM responses."""

import hashlib
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional


class Cache:
    """SQLite-based cache for papercli."""

    def __init__(self, path: str | Path | None = None, ttl_hours: int = 24 * 7):
        """
        Initialize cache.

        Args:
            path: Path to SQLite database file. Defaults to ~/.cache/papercli.sqlite
            ttl_hours: Cache TTL in hours. Default 7 days.
        """
        if path is None:
            path = Path.home() / ".cache" / "papercli.sqlite"
        self.path = Path(path).expanduser()
        self.ttl = timedelta(hours=ttl_hours)

        # Ensure parent directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self._init_db()

    def _init_db(self) -> None:
        """Initialize the database schema."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_cache_expires
                ON cache(expires_at)
            """)
            conn.commit()

    async def get(self, key: str) -> Optional[Any]:
        """
        Get a value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                """
                SELECT value, expires_at FROM cache
                WHERE key = ?
                """,
                (key,),
            )
            row = cursor.fetchone()

            if row is None:
                return None

            value, expires_at = row

            # Check expiration
            if expires_at:
                expires = datetime.fromisoformat(expires_at)
                if expires < datetime.now():
                    # Expired - delete and return None
                    conn.execute("DELETE FROM cache WHERE key = ?", (key,))
                    conn.commit()
                    return None

            return json.loads(value)

    async def set(self, key: str, value: Any, ttl: Optional[timedelta] = None) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Optional TTL override
        """
        ttl = ttl or self.ttl
        expires_at = datetime.now() + ttl

        with sqlite3.connect(self.path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO cache (key, value, created_at, expires_at)
                VALUES (?, ?, ?, ?)
                """,
                (key, json.dumps(value), datetime.now().isoformat(), expires_at.isoformat()),
            )
            conn.commit()

    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM cache WHERE key = ?", (key,))
            conn.commit()

    async def clear(self) -> None:
        """Clear all cache entries."""
        with sqlite3.connect(self.path) as conn:
            conn.execute("DELETE FROM cache")
            conn.commit()

    async def cleanup_expired(self) -> int:
        """
        Remove all expired entries.

        Returns:
            Number of entries removed
        """
        with sqlite3.connect(self.path) as conn:
            cursor = conn.execute(
                """
                DELETE FROM cache
                WHERE expires_at < ?
                """,
                (datetime.now().isoformat(),),
            )
            conn.commit()
            return cursor.rowcount

    @staticmethod
    def hash_key(*args: str) -> str:
        """Generate a cache key from multiple strings."""
        combined = ":".join(args)
        return hashlib.sha256(combined.encode()).hexdigest()[:32]

