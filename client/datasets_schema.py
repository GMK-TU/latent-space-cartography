# datasets_schema.py
import re
import sqlite3

_DATASET_ID_RE = re.compile(r"^[a-zA-Z0-9_]{3,64}$")

def ensure_dataset_feature_tables(conn: sqlite3.Connection, dataset_id: str):
    if not _DATASET_ID_RE.match(dataset_id):
        raise ValueError("Invalid dataset_id")

    cur = conn.cursor()

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}_group` (
      `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT,
      `alias` varchar(255) DEFAULT NULL,
      `list` text,
      `creation_time` datetime DEFAULT CURRENT_TIMESTAMP,
      `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """)

    cur.execute(f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}_vector` (
      `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT,
      `description` varchar(255) DEFAULT NULL,
      `start` integer DEFAULT NULL,
      `end` integer DEFAULT NULL,
      `creation_time` datetime DEFAULT CURRENT_TIMESTAMP,
      `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """)

    conn.commit()
