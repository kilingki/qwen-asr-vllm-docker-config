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
ASR_MODEL_PATH = os.getenv("ASR_MODEL_PATH", "/models/Qwen3-ASR-1.7B")
QWEN_ASR_INTERNAL_HOST = os.getenv("QWEN_ASR_INTERNAL_HOST", "127.0.0.1")
QWEN_ASR_INTERNAL_PORT = os.getenv("QWEN_ASR_INTERNAL_PORT", "18000")
ASR_GPU_MEMORY_UTILIZATION = os.getenv("ASR_GPU_MEMORY_UTILIZATION", "0.72")
ASR_MAX_MODEL_LEN = os.getenv("ASR_MAX_MODEL_LEN", "10000")
ASR_GENERATION_CONFIG = os.getenv("ASR_GENERATION_CONFIG", "vllm")
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE", "ko")
REQUEST_TIMEOUT_SECONDS = _get_float("REQUEST_TIMEOUT_SECONDS", 600.0)
LOAD_TIMEOUT_SECONDS = _get_float("LOAD_TIMEOUT_SECONDS", 600.0)
WORKER_STOP_TIMEOUT_SECONDS = _get_float("WORKER_STOP_TIMEOUT_SECONDS", 30.0)

CHUNK_SECONDS = _get_int("CHUNK_SECONDS", 120)
CHUNK_OVERLAP_SECONDS = _get_int("CHUNK_OVERLAP_SECONDS", 2)
MAX_CONCURRENT_CHUNKS = _get_int("MAX_CONCURRENT_CHUNKS", 4)

STORAGE_DIR = Path(os.getenv("STORAGE_DIR", "/data"))
TMP_DIR = STORAGE_DIR / "tmp"

SUPPORTED_RESPONSE_FORMATS = {"json", "text", "srt", "vtt", "verbose_json"}
SUPPORTED_TIMESTAMP_GRANULARITIES = {"segment"}
