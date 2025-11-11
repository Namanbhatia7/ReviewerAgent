# app/services/storage/local_fs.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import hashlib, mimetypes, os, tempfile
from typing import Optional
from app.core.config import settings

_MAX_BYTES = 64 * 1024 * 1024  # 64 MB POC limit;

@dataclass
class StoredFile:
    path: str
    bytes: int
    sha256: str
    mime: str

def _safe_name(name: str) -> str:
    # prevent traversal, strip weird chars
    n = os.path.basename(name).strip().replace("\x00", "")
    return n or "unnamed.bin"

def save_bytes(data: bytes, subdir: str, filename: str, mime_hint: Optional[str] = None) -> StoredFile:
    if len(data) > _MAX_BYTES:
        raise ValueError(f"file too large (> {_MAX_BYTES} bytes)")

    base = Path(settings.artifact_root_path)
    target_dir = base / subdir
    target_dir.mkdir(parents=True, exist_ok=True)

    fname = _safe_name(filename)
    target = target_dir / fname

    # atomic write
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=str(target_dir)) as tmp:
        tmp.write(data)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_path = tmp.name
    os.replace(tmp_path, target)

    sha = hashlib.sha256(data).hexdigest()
    mime = mime_hint or (mimetypes.guess_type(target.name)[0] or "application/octet-stream")
    return StoredFile(path=str(target), bytes=len(data), sha256=sha, mime=mime)
