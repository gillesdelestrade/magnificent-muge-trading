"""Backup of the data that cannot be downloaded again: saved simulations and settings,
AI analyses (paid), news articles and configuration files.

Prices are left out (they are downloaded again for free) and so is the .env file (API keys).
"""
from __future__ import annotations

import sqlite3
import tempfile
import zipfile
from contextlib import closing
from datetime import datetime
from pathlib import Path

from .. import paths


def _sqlite_snapshot(source: Path, target: Path) -> None:
    """Consistent copy of a SQLite file, even if the application is writing to it."""
    with closing(sqlite3.connect(source)) as src, closing(sqlite3.connect(target)) as dst:
        src.backup(dst)


def create_backup(dest_dir: Path | None = None, now: datetime | None = None) -> Path:
    """Write a zip archive whose internal paths are relative to the application folder,
    so that unzipping it there restores everything."""
    dest_dir = dest_dir or paths.BACKUP_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = (now or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    target = dest_dir / f"muge-trading-sauvegarde-{stamp}.zip"
    root = paths.ROOT
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive, \
            tempfile.TemporaryDirectory() as tmp:
        for db_path in (paths.DB_PATH, paths.AI_CACHE_PATH):
            if db_path.exists():
                snapshot = Path(tmp) / db_path.name
                _sqlite_snapshot(db_path, snapshot)
                archive.write(snapshot, db_path.relative_to(root).as_posix())
        for folder in (paths.DATA_DIR / "news", paths.CONFIG_DIR):
            if folder.exists():
                for file in sorted(folder.rglob("*")):
                    if file.is_file():
                        archive.write(file, file.relative_to(root).as_posix())
    return target


def list_backups(dest_dir: Path | None = None) -> list[Path]:
    """Existing backups, most recent first."""
    dest_dir = dest_dir or paths.BACKUP_DIR
    if not dest_dir.exists():
        return []
    return sorted(dest_dir.glob("muge-trading-sauvegarde-*.zip"), reverse=True)
