import sqlite3
from datetime import datetime
import json

def _now():
    return datetime.utcnow().isoformat() + "Z"

class DatasetsDB:
    """
    Helper around sqlite3 for:
      - datasets registry (id, name, status, progress, message, created_at)
      - dataset_jobs (job_id, dataset_id, status, progress, message, stage, timestamps)
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self):
        cur = self.conn.cursor()
        cur.execute("""
        CREATE TABLE IF NOT EXISTS datasets (
          id TEXT PRIMARY KEY,
          name TEXT NOT NULL,
          type TEXT NOT NULL DEFAULT 'image',
          status TEXT NOT NULL DEFAULT 'empty',
          progress INTEGER NOT NULL DEFAULT 0,
          message TEXT DEFAULT '',
          error TEXT DEFAULT '',
          preview_images TEXT DEFAULT '',
          preview_meta TEXT DEFAULT '',
          matched_preview TEXT DEFAULT '',
          created_at TEXT NOT NULL
        );
        """)
        cur.execute("""
        CREATE TABLE IF NOT EXISTS dataset_jobs (
          job_id TEXT PRIMARY KEY,
          dataset_id TEXT NOT NULL,
          status TEXT NOT NULL DEFAULT 'running',
          progress INTEGER NOT NULL DEFAULT 0,
          message TEXT DEFAULT '',
          stage TEXT DEFAULT '',
          done INTEGER NOT NULL DEFAULT 0,
          created_at TEXT NOT NULL,
          updated_at TEXT NOT NULL,
          FOREIGN KEY(dataset_id) REFERENCES datasets(id)
        );
        """)
        self.conn.commit()

    def list_datasets(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM datasets ORDER BY created_at DESC")
        rows = [dict(r) for r in cur.fetchall()]
        return [self._deserialize_dataset(r) for r in rows]

    def get_dataset(self, dataset_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM datasets WHERE id = ?", (dataset_id,))
        r = cur.fetchone()
        return self._deserialize_dataset(dict(r)) if r else None

    def create_dataset(self, dataset_id: str, name: str, type: str = "image"):
        cur = self.conn.cursor()
        created_at = _now()
        cur.execute(
            """
            INSERT OR REPLACE INTO datasets
              (id, name, type, status, progress, message, created_at)
            VALUES (?, ?, ?, 'empty', 0, '', ?)
            """,
            (dataset_id, name, type, created_at),
        )
        self.conn.commit()
        return self.get_dataset(dataset_id)

    def update_dataset(
        self,
        dataset_id: str,
        status: str = None,
        progress: int = None,
        message: str = None,
        error: str = None,
        extra: dict = None,
    ):
        extra = extra or {}
        ds = self.get_dataset(dataset_id) or {}
        status = status if status is not None else ds.get("status", "empty")
        progress = int(progress) if progress is not None else int(ds.get("progress", 0) or 0)
        message = message if message is not None else ds.get("message", "")
        error = error if error is not None else ds.get("error", "")

        preview_images = extra.get("previewImages", ds.get("previewImages", []))
        preview_meta = extra.get("previewMeta", ds.get("previewMeta", []))
        matched_preview = extra.get("matchedPreview", ds.get("matchedPreview", []))

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO datasets
              (id, name, type, status, progress, message, error,
               preview_images, preview_meta, matched_preview, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset_id,
                ds.get("name", dataset_id),
                ds.get("type", "image"),
                status,
                progress,
                message,
                error,
                self._dump(preview_images),
                self._dump(preview_meta),
                self._dump(matched_preview),
                ds.get("created_at", _now()),
            ),
        )
        self.conn.commit()
        return self.get_dataset(dataset_id)

    def create_job(
        self,
        job_id: str,
        dataset_id: str,
        progress: int = 0,
        message: str = "",
        stage: str = "",
        status: str = "running",
    ):
        cur = self.conn.cursor()
        now = _now()
        cur.execute(
            """
            INSERT OR REPLACE INTO dataset_jobs
              (job_id, dataset_id, status, progress, message, stage, done, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (job_id, dataset_id, status, int(progress), message, stage, now, now),
        )
        self.conn.commit()
        return self.get_job(job_id)

    def update_job(
        self,
        job_id: str,
        status: str = None,
        progress: int = None,
        message: str = None,
        stage: str = None,
        done: bool = None,
    ):
        job = self.get_job(job_id) or {}
        status = status if status is not None else job.get("status", "running")
        progress = int(progress) if progress is not None else int(job.get("progress", 0) or 0)
        message = message if message is not None else job.get("message", "")
        stage = stage if stage is not None else job.get("stage", "")
        done_val = 1 if (done is True) else (int(job.get("done", 0) or 0) if done is None else (1 if done else 0))

        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO dataset_jobs
              (job_id, dataset_id, status, progress, message, stage, done, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id,
                job.get("dataset_id", ""),
                status,
                progress,
                message,
                stage,
                done_val,
                job.get("created_at", _now()),
                _now(),
            ),
        )
        self.conn.commit()
        return self.get_job(job_id)

    def get_job(self, job_id: str):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM dataset_jobs WHERE job_id = ?", (job_id,))
        r = cur.fetchone()
        if not r:
            return None
        d = dict(r)
        d["done"] = bool(d.get("done", 0))
        return d

    def list_jobs_for_dataset(self, dataset_id: str, limit: int = 20):
        cur = self.conn.cursor()
        cur.row_factory = sqlite3.Row
        cur.execute(
            """
            SELECT job_id, dataset_id, status, progress, message, stage, done, created_at, updated_at
            FROM dataset_jobs
            WHERE dataset_id = ?
            ORDER BY created_at DESC
            LIMIT ?
            """,
            (dataset_id, int(limit)),
        )
        return [dict(r) for r in cur.fetchall()]

    def get_latest_job_for_dataset(self, dataset_id: str):
        rows = self.list_jobs_for_dataset(dataset_id, limit=1)
        return rows[0] if rows else None

    def _dump(self, obj):
        try:
            return json.dumps(obj)
        except Exception:
            return "[]"

    def _loads(self, s, default):
        if not s:
            return default
        try:
            return json.loads(s)
        except Exception:
            return default

    def _deserialize_dataset(self, r):
        r = dict(r)
        r["progress"] = int(r.get("progress", 0) or 0)
        r["previewImages"] = self._loads(r.get("preview_images"), [])
        r["previewMeta"] = self._loads(r.get("preview_meta"), [])
        r["matchedPreview"] = self._loads(r.get("matched_preview"), [])
        r.pop("preview_images", None)
        r.pop("preview_meta", None)
        r.pop("matched_preview", None)
        return r
