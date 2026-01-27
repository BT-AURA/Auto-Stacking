"""Historical metrics caching using SQLite (optional)."""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


def _get_config_path() -> Path:
    """Get configuration file path in home directory."""
    home_dir = getattr(Path, 'home')()
    config_name = '.bitsconfig'
    return home_dir / config_name


class MetricsCache:
    """SQLite-based cache for historical subnet metrics."""

    def __init__(self, cache_path: Optional[Path] = None):
        """Initialize cache with optional custom path."""
        if cache_path is None:
            cache_path = Path(__file__).parent / "historical" / "metrics.db"
        else:
            cache_path = Path(cache_path)

        # Ensure directory exists
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.cache_path = cache_path
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS subnet_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp INTEGER NOT NULL,
                    netuid INTEGER NOT NULL,
                    block INTEGER NOT NULL,
                    metrics_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
            """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_netuid_timestamp 
                ON subnet_metrics(netuid, timestamp)
            """
            )
            conn.commit()

    def save_metrics(
        self, netuid: int, block: int, metrics: List[Dict]
    ) -> None:
        """Save metrics to cache."""
        try:
            timestamp = int(datetime.now().timestamp())
            metrics_json = json.dumps(metrics)

            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT INTO subnet_metrics (timestamp, netuid, block, metrics_json, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """,
                    (timestamp, netuid, block, metrics_json, datetime.now().isoformat()),
                )
                conn.commit()

            logger.debug(f"Cached metrics for subnet {netuid} at block {block}")
        except Exception as e:
            logger.error(f"Error saving metrics to cache: {e}")

    def get_latest_metrics(self, netuid: int) -> Optional[List[Dict]]:
        """Get latest cached metrics for a subnet."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT metrics_json FROM subnet_metrics
                    WHERE netuid = ?
                    ORDER BY timestamp DESC
                    LIMIT 1
                """,
                    (netuid,),
                )
                row = cursor.fetchone()
                if row:
                    return json.loads(row[0])
        except Exception as e:
            logger.error(f"Error reading metrics from cache: {e}")

        return None

    def get_metrics_history(
        self, netuid: int, limit: int = 100
    ) -> List[Dict]:
        """Get historical metrics for a subnet."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    """
                    SELECT timestamp, block, metrics_json FROM subnet_metrics
                    WHERE netuid = ?
                    ORDER BY timestamp DESC
                    LIMIT ?
                """,
                    (netuid, limit),
                )
                results = []
                for row in cursor.fetchall():
                    results.append(
                        {
                            "timestamp": row[0],
                            "block": row[1],
                            "metrics": json.loads(row[2]),
                        }
                    )
                return results
        except Exception as e:
            logger.error(f"Error reading metrics history: {e}")

        return []

    def clear_cache(self, netuid: Optional[int] = None) -> None:
        """Clear cache for a specific subnet or all subnets."""
        try:
            with sqlite3.connect(self.cache_path) as conn:
                if netuid is not None:
                    conn.execute(
                        "DELETE FROM subnet_metrics WHERE netuid = ?", (netuid,)
                    )
                else:
                    conn.execute("DELETE FROM subnet_metrics")
                conn.commit()
            logger.info(f"Cleared cache for subnet {netuid if netuid else 'all'}")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
