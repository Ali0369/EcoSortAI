"""Runtime settings (environment variables, all optional)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_bool(name: str, default: bool) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = field(default_factory=lambda: Path(os.environ.get("ECOSORT_ARTIFACTS_DIR", REPO_ROOT / "ml" / "artifacts")).resolve())
    ml_src: Path = field(default_factory=lambda: Path(os.environ.get("ECOSORT_ML_SRC", REPO_ROOT / "ml" / "src")).resolve())
    max_upload_mb: float = field(default_factory=lambda: float(os.environ.get("ECOSORT_MAX_UPLOAD_MB", "10")))
    max_pixels: int = 40_000_000          # decompression-bomb guard
    max_side_before_downscale: int = 2048  # very large photos are shrunk before inference
    allowed_formats: tuple = ("JPEG", "PNG", "WEBP")
    allowed_content_types: tuple = ("image/jpeg", "image/png", "image/webp")
    cors_origins: tuple = field(default_factory=lambda: tuple(
        o.strip() for o in os.environ.get("ECOSORT_CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if o.strip()))
    verify_hashes: bool = field(default_factory=lambda: _env_bool("ECOSORT_VERIFY_HASHES", True))
    verify_reference: bool = field(default_factory=lambda: _env_bool("ECOSORT_VERIFY_REFERENCE", True))

    @property
    def max_upload_bytes(self) -> int:
        return int(self.max_upload_mb * 1024 * 1024)
