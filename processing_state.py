from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS processed_pdfs (
    sha256            TEXT PRIMARY KEY,
    fingerprint_key   TEXT NOT NULL UNIQUE,
    filename          TEXT NOT NULL,
    size              INTEGER NOT NULL,
    source_path       TEXT NOT NULL,
    publication_type  TEXT NOT NULL,
    publication_date  TEXT NOT NULL,
    pdf_id             TEXT NOT NULL,
    output_path        TEXT NOT NULL,
    article_count      INTEGER NOT NULL,
    processed_at       REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_processed_pdf_filename
    ON processed_pdfs(filename, publication_type, publication_date);
"""


class ProcessingStateDB:
    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(str(self.db_path), timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def find_processed(
        self,
        sha256: str,
        publication_type: str,
        publication_date: str,
        filename: str,
    ) -> dict[str, Any] | None:
        key = build_fingerprint_key(publication_type, publication_date, filename)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM processed_pdfs WHERE sha256 = ? OR fingerprint_key = ?",
                (sha256, key),
            ).fetchone()
        return dict(row) if row else None

    def mark_processed(
        self,
        *,
        sha256: str,
        filename: str,
        size: int,
        source_path: str,
        publication_type: str,
        publication_date: str,
        pdf_id: str,
        output_path: str,
        article_count: int,
        processed_at: float | None = None,
    ) -> None:
        key = build_fingerprint_key(publication_type, publication_date, filename)
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO processed_pdfs (
                    sha256, fingerprint_key, filename, size, source_path,
                    publication_type, publication_date, pdf_id, output_path,
                    article_count, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(fingerprint_key) DO UPDATE SET
                    sha256 = excluded.sha256,
                    size = excluded.size,
                    source_path = excluded.source_path,
                    pdf_id = excluded.pdf_id,
                    output_path = excluded.output_path,
                    article_count = excluded.article_count,
                    processed_at = excluded.processed_at
                """,
                (
                    sha256,
                    key,
                    filename,
                    size,
                    source_path,
                    publication_type,
                    publication_date,
                    pdf_id,
                    output_path,
                    article_count,
                    processed_at if processed_at is not None else time.time(),
                ),
            )

    def list_all(self) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM processed_pdfs ORDER BY processed_at DESC"
            ).fetchall()
        return [dict(row) for row in rows]

    def count(self) -> int:
        with self._connect() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM processed_pdfs").fetchone()[0])

    def import_existing_outputs(self, output_root: Path) -> int:
        canonical_paths = read_index_output_paths(output_root / "database_index.js")
        if canonical_paths:
            with self._connect() as connection:
                placeholders = ",".join("?" for _ in canonical_paths)
                connection.execute(
                    f"DELETE FROM processed_pdfs WHERE output_path NOT IN ({placeholders})",
                    tuple(sorted(canonical_paths)),
                )
        imported = 0
        for database_path in sorted(output_root.glob("*/*/database.js")):
            relative_output = database_path.parent.relative_to(output_root).as_posix()
            if canonical_paths and relative_output not in canonical_paths:
                continue
            payload = read_pdf_database(database_path)
            if not payload:
                continue
            publication_type = str(payload.get("publication_type") or database_path.parent.parent.name)
            publication_date = str(payload.get("publication_date") or database_path.parent.name)
            filename = str(payload.get("original_filename") or "unknown.pdf")
            key = build_fingerprint_key(publication_type, publication_date, filename)
            legacy_sha = "legacy:" + hashlib.sha256(key.encode("utf-8")).hexdigest()
            if self.find_processed(legacy_sha, publication_type, publication_date, filename):
                continue
            self.mark_processed(
                sha256=legacy_sha,
                filename=filename,
                size=0,
                source_path="",
                publication_type=publication_type,
                publication_date=publication_date,
                pdf_id=str(payload.get("id") or key),
                output_path=database_path.parent.relative_to(output_root).as_posix(),
                article_count=int(payload.get("article_count") or len(payload.get("articles") or [])),
                processed_at=database_path.stat().st_mtime,
            )
            imported += 1
        return imported


def compute_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def build_fingerprint_key(
    publication_type: str,
    publication_date: str,
    filename: str,
) -> str:
    return "|".join(
        [publication_type.strip().upper(), publication_date.strip(), filename.strip().lower()]
    )


def read_pdf_database(database_path: Path) -> dict[str, Any] | None:
    try:
        text = database_path.read_text(encoding="utf-8")
        match = re.search(
            r"window\.paper_databases\[[^\]]+\]\s*=\s*([\s\S]*?);\s*$",
            text,
        )
        data = json.loads(match.group(1)) if match else None
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def read_index_output_paths(index_path: Path) -> set[str]:
    """Return issue directories that are intentionally published."""
    try:
        text = index_path.read_text(encoding="utf-8")
        match = re.search(r"window\.paper_db_index\s*=\s*([\s\S]*?);\s*$", text)
        items = json.loads(match.group(1)) if match else []
    except (OSError, json.JSONDecodeError):
        return set()
    paths: set[str] = set()
    for item in items if isinstance(items, list) else []:
        database_path = str(item.get("database_path") or "") if isinstance(item, dict) else ""
        if database_path:
            paths.add(str(Path(database_path).parent).replace("\\", "/"))
    return paths
