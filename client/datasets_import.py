import os
import csv
import sqlite3

def _connect(db_path):
    conn = sqlite3.connect(db_path, check_same_thread=False)
    cursor = conn.cursor()
    return conn, cursor

def import_metadata_csv(db_path, dataset_id, csv_path, raw_dir=None, preview_n=5, matched_n=5):
    """
    Creates (if needed) per-dataset tables:
      - `<dataset_id>_group`
      - `<dataset_id>_meta`

    and imports the CSV into `<dataset_id>_meta`.

    Assumptions (same as existing codebase):
      - CSV has at least columns: i,name
      - any extra columns become VARCHAR(255) columns

    Returns:
      (preview_meta_rows, matched_preview_rows)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(csv_path)

    conn, cursor = _connect(db_path)

    q_group = f"""
    CREATE TABLE IF NOT EXISTS `{dataset_id}_group` (
        `id` integer NOT NULL PRIMARY KEY AUTOINCREMENT,
        `alias` varchar(255) DEFAULT NULL,
        `list` text,
        `creation_time` datetime DEFAULT CURRENT_TIMESTAMP,
        `timestamp` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP
    );
    """
    cursor.execute(q_group)

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        if "i" not in fieldnames or "name" not in fieldnames:
            raise ValueError("CSV must include at least columns: i,name")

        extra_cols = [c for c in fieldnames if c not in ("i", "name")]

        q_meta = f"CREATE TABLE IF NOT EXISTS `{dataset_id}_meta` (\n"
        q_meta += "  `i` int NOT NULL,\n"
        q_meta += "  `name` varchar(255) DEFAULT NULL,\n"
        for col in extra_cols:
            q_meta += f"  `{col}` varchar(255) DEFAULT NULL,\n"
        q_meta += "  PRIMARY KEY (`i`)\n);"
        cursor.execute(q_meta)

        cursor.execute(f"DELETE FROM `{dataset_id}_meta`;")

        to_db = []
        preview_meta = []
        for row in reader:
            i_val = int(row["i"])
            name = row.get("name", "")
            values = [i_val, name] + [row.get(c, "") for c in extra_cols]
            to_db.append(tuple(values))

            if len(preview_meta) < preview_n:
                preview_meta.append({c: row.get(c, "") for c in fieldnames})

        marks = ",".join(["?"] * (2 + len(extra_cols)))
        cols_sql = "i,name" + ("," + ",".join([f"`{c}`" for c in extra_cols]) if extra_cols else "")
        cursor.executemany(
            f"INSERT INTO `{dataset_id}_meta` ({cols_sql}) VALUES ({marks})",
            to_db
        )

    conn.commit()
    cursor.close()
    conn.close()

    matched_preview = _make_matched_preview(preview_meta, raw_dir, limit=matched_n)
    return preview_meta, matched_preview

def _make_matched_preview(preview_meta, raw_dir, limit=5):
    if not raw_dir or not os.path.isdir(raw_dir):
        return []

    exts = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"}

    images = []
    for dirpath, _, filenames in os.walk(raw_dir):
        for fn in filenames:
            if os.path.splitext(fn.lower())[1] in exts:
                images.append(os.path.join(dirpath, fn))

    by_base = {}
    by_stem = {}
    for full in images:
        base = os.path.basename(full).lower()
        stem = os.path.splitext(base)[0]
        by_base[base] = full
        by_stem[stem] = full

    # raw_dir is ./data/<id>/raw ; compute rel from ./data
    data_dir = os.path.abspath(os.path.join(raw_dir, "..", ".."))

    matched = []
    for row in preview_meta:
        if len(matched) >= limit:
            break
        name = (row.get("name") or "").strip().lower()
        if not name:
            continue

        base = name
        stem = os.path.splitext(base)[0]
        path = by_base.get(base) or by_stem.get(stem)

        if path:
            rel = os.path.relpath(path, data_dir).replace(os.sep, "/")
            matched.append({"imageUrl": f"/data/{rel}", "metaRow": row})
        else:
            matched.append({"metaRow": row})

    return matched
