import os
from pathlib import Path


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    return int(raw)


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    return float(raw)


LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

ASR_BASE_URL = os.getenv("ASR_BASE_URL", "http://127.0.0.1:18000/v1").rstrip("/")
ASR_MODEL = os.getenv("ASR_MODEL", "qwen3-asr")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ko")
REQUEST_TIMEOUT_SECONDS = _get_float("REQUEST_TIMEOUT_SECONDS", 600.0)

CHUNK_SECONDS = _get_int("CHUNK_SECONDS", 120)
CHUNK_OVERLAP_SECONDS = _get_int("CHUNK_OVERLAP_SECONDS", 2)
MAX_CONCURRENT_CHUNKS = _get_int("MAX_CONCURRENT_CHUNKS", 4)

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/data"))
TMP_DIR = STORAGE_DIR / "tmp"

SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}
SUPPORTED_TIMESTAMP_GRANULARITIES = {"segment", "word"}
