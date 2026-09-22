from pathlib import Path
from typing import Any

import httpx


HEALTH_PROBE_TIMEOUT_SECONDS = 2.0

_LANGUAGE_ALIASES = {
    "ko": "Korean",
    "kor": "Korean",
    "korean": "Korean",
    "ja": "Japanese",
    "jp": "Japanese",
    "japanese": "Japanese",
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "chinese": "Chinese",
    "en": "English",
    "english": "English",
    "yue": "Cantonese",
    "cantonese": "Cantonese",
    "fr": "French",
    "french": "French",
    "de": "German",
    "german": "German",
    "it": "Italian",
    "italian": "Italian",
    "pt": "Portuguese",
    "portuguese": "Portuguese",
    "ru": "Russian",
    "russian": "Russian",
    "es": "Spanish",
    "spanish": "Spanish",
}


def to_fa_language(language: str) -> str:
    normalized = language.strip().lower()
    if not normalized:
        return ""
    mapped = _LANGUAGE_ALIASES.get(normalized)
    if mapped:
        return mapped
    return normalized[0].upper() + normalized[1:]


class ForcedAlignerClient:
    def __init__(self, base_url: str, timeout_seconds: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=timeout_seconds)

    async def close(self) -> None:
        await self.client.aclose()

    async def reachable(self) -> bool:
        try:
            response = await self.client.get(
                f"{self.base_url}/health",
                timeout=HEALTH_PROBE_TIMEOUT_SECONDS,
            )
        except Exception:
            return False
        return response.is_success

    async def align(
        self,
        audio_path: Path,
        text: str,
        language: str,
    ) -> list[dict[str, Any]]:
        canonical_language = to_fa_language(language)
        audio_bytes = audio_path.read_bytes()
        files = {
            "file": (audio_path.name, audio_bytes, "audio/wav"),
        }
        form_data = {
            "text": text,
            "language": canonical_language,
        }
        try:
            response = await self.client.post(
                f"{self.base_url}/align",
                data=form_data,
                files=files,
            )
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Forced aligner request failed: {exc}") from exc

        if response.is_error:
            raise RuntimeError(
                f"Forced aligner returned status={response.status_code}: {response.text}"
            )

        try:
            payload = response.json()
        except ValueError as exc:
            raise RuntimeError("Forced aligner returned invalid JSON") from exc

        if not isinstance(payload, dict):
            raise RuntimeError("Forced aligner response is not an object")

        items = payload.get("items")
        if not isinstance(items, list):
            raise RuntimeError("Forced aligner response items is not a list")

        return _words_from_items(items)


def _words_from_items(items: list[Any]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        word = str(item.get("text") or "").strip()
        if not word:
            continue
        start = _as_float(item.get("start_time"))
        end = _as_float(item.get("end_time"))
        if start is None or end is None:
            continue
        words.append(
            {
                "word": word,
                "start": start,
                "end": end,
            }
        )
    return words


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
