"""
Centralised configuration for Flask + Celery.

Environment variables override the defaults to simplify container / cloud
deployments.
"""
from __future__ import annotations

import os
from pathlib import Path

_BASE_DIR = Path(__file__).resolve().parent.parent
_RESULTS_DIR = _BASE_DIR / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

class Config:  # pylint: disable=too-few-public-methods
    _secret = os.getenv("NCORR_SECRET_KEY")
    if not _secret:
        raise RuntimeError(
            "NCORR_SECRET_KEY environment variable must be set for security"
        )
    SECRET_KEY = _secret
    JSON_SORT_KEYS = False

    # Celery / Redis
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)

    # Where to stash NumPy / JSON artefacts produced by tasks
    RESULTS_BASE_DIR = Path(os.getenv("NCORR_RESULTS_DIR", _RESULTS_DIR)).resolve()

    # Max upload size (Flask) – 256 MiB default
    MAX_CONTENT_LENGTH = int(os.getenv("NCORR_MAX_UPLOAD_BYTES", 256 * 1024 * 1024))

# Export a ready instance so other modules can `from webapp.config import cfg`
cfg = Config()
