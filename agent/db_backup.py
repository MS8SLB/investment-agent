"""
Database backup and recovery utilities.
Ensures portfolio database is never lost.
"""

import os
import shutil
from datetime import datetime
from pathlib import Path

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "portfolio.db")
BACKUP_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "backups")


def ensure_backup_dir() -> None:
    """Create backup directory if it doesn't exist."""
    os.makedirs(BACKUP_DIR, exist_ok=True)


def create_backup(label: str = "manual") -> str:
    """
    Create a timestamped backup of the portfolio database.

    Args:
        label: Optional label for the backup (e.g., "pre-code-review", "before-trades")

    Returns:
        Path to the backup file created.
    """
    ensure_backup_dir()

    if not os.path.exists(DB_PATH):
        return None

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(BACKUP_DIR, f"portfolio_{timestamp}_{label}.db")

    try:
        shutil.copy2(DB_PATH, backup_path)
        return backup_path
    except Exception as e:
        print(f"WARNING: Failed to create backup: {e}")
        return None


def list_backups() -> list[dict]:
    """List all available backups with their timestamps and sizes."""
    ensure_backup_dir()
    backups = []

    for fname in sorted(os.listdir(BACKUP_DIR), reverse=True):
        if fname.startswith("portfolio_") and fname.endswith(".db"):
            fpath = os.path.join(BACKUP_DIR, fname)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            mtime = os.path.getmtime(fpath)
            mtime_str = datetime.fromtimestamp(mtime).isoformat()

            backups.append({
                "filename": fname,
                "path": fpath,
                "size_mb": round(size_mb, 2),
                "created_at": mtime_str,
            })

    return backups


def restore_from_backup(backup_path: str) -> bool:
    """
    Restore the portfolio database from a backup.

    Args:
        backup_path: Path to the backup file

    Returns:
        True if successful, False otherwise.
    """
    if not os.path.exists(backup_path):
        print(f"ERROR: Backup file not found: {backup_path}")
        return False

    try:
        # Create a backup of the current DB before restoring (in case of mistakes)
        if os.path.exists(DB_PATH):
            create_backup("pre-restore")

        shutil.copy2(backup_path, DB_PATH)
        print(f"✓ Database restored from: {backup_path}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to restore backup: {e}")
        return False


def get_latest_backup() -> str:
    """Return the path to the most recent backup, or None if none exist."""
    backups = list_backups()
    return backups[0]["path"] if backups else None
