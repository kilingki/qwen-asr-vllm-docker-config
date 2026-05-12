import asyncio
import re
from pathlib import Path
from typing import Any

from .asr_client import QwenAsrClient
from .audio import AudioChunk, create_chunks, probe_duration
from .config import (
    CHUNK_OVERLAP_SECONDS,
    CHUNK_SECONDS,
    MAX_CONCURRENT_CHUNKS,
)


PUNCTUATION = {".", "!", "?", ",", ";", ":", "。", "！", "？", "，", "、"}
HANGUL_RE = re.compile(r"[\uac00-\ud7af]")
NO_SPACE_BEFORE_HANGUL_TOKENS = {
    "은",
    "는",
    "이",
    "가",
    "을",
    "를",
    "에",
    "에서",
    "에게",
    "한테",
    "께",
    "으로",
    "로",
    "와",
    "과",
    "랑",
    "하고",
    "도",
    "만",
    "부터",
    "까지",
    "보다",
    "처럼",
    "이라",
    "라고",
    "이라고",
    "입니다",
    "입니다만",
    "이에요",
    "예요",
    "네요",
    "거든요",
    "죠",
    "요",
    "한",
}
ASR_ARTIFACT_REPLACEMENTS = (
    re.compile(
        r"(?:language\s*)?Korean\s*<?\s*asr[\s_-]*text\s*>?",
        re.IGNORECASE,
    ),
    re.compile(r"language\s*Koreanasrtext", re.IGNORECASE),
)
ASR_TRANSCRIPT_MARKER = re.compile(
    r"\blanguage\s+(?P<language>[A-Za-z][A-Za-z_-]*)\s*<\s*asr[\s_-]*text\s*>",
    re.IGNORECASE,
)
STANDALONE_LANGUAGE_ARTIFACT = re.compile(
    r"(?:(?<=^)|(?<=[\s\]\)])|(?<=[\uac00-\ud7af0-9]))language(?=$|[\s\uac00-\ud7af0-9])",
    re.IGNORECASE,
)
REPEATED_NO_TAIL = re.compile(r"(?:[\s,.;:!?]*(?:no)\b){20,}\s*$", re.IGNORECASE)
TEXT_TOKEN_RE = re.compile(r"[\uac00-\ud7afA-Za-z0-9]+")


class TranscriptionPipeline:
    def __init__(self, asr_client: QwenAsrClient) -> None:
        self.asr_client = asr_client

    async def transcribe(
        self,
        wav_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict[str, Any]:
        duration = probe_duration(wav_path)
        chunks = create_chunks(
            source_wav=wav_path,
            chunks_dir=wav_path.parent / "chunks",
            chunk_seconds=CHUNK_SECONDS,
            overlap_seconds=CHUNK_OVERLAP_SECONDS,
        )

        merged_language = language
        merged_text = ""
        merged_segments: list[dict[str, Any]] = []

        normalized_results = await self._transcribe_chunks(
            chunks=chunks,
            model=model,
            language=language,
            prompt=prompt,
            timestamp_granularities=timestamp_granularities,
        )

        for chunk, normalized in zip(chunks, normalized_results, strict=True):
            if not merged_language:
                merged_language = normalized.get("language") or language

            merged_text = _merge_text(merged_text, normalized["text"])
            dedupe_before = 0.0 if chunk.index == 0 else chunk.start + CHUNK_OVERLAP_SECONDS
            merged_segments.extend(
                _offset_and_clip_segments(
                    normalized["segments"],
                    offset=chunk.start,
                    dedupe_before=dedupe_before,
                )
            )

        if merged_segments:
            merged_text = ""
            for segment in merged_segments:
                segment_text = segment["text"].strip()
                if segment_text:
                    merged_text = _merge_text(merged_text, segment_text)

        return {
            "text": merged_text.strip(),
            "language": merged_language,
            "duration": duration,
            "segments": _reindex_segments(merged_segments),
        }

    async def _transcribe_chunks(
        self,
        chunks: list[AudioChunk],
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> list[dict[str, Any]]:
        semaphore = asyncio.Semaphore(max(1, MAX_CONCURRENT_CHUNKS))

        async def transcribe_one(chunk: AudioChunk) -> dict[str, Any]:
            async with semaphore:
                raw_result = await self.asr_client.transcribe(
                    audio_path=chunk.path,
                    model=model,
                    language=language,
                    prompt=prompt,
                    timestamp_granularities=timestamp_granularities,
                )
            normalized = self._normalize_backend_result(raw_result, chunk)
            return _clean_normalized_result(normalized)

        return await asyncio.gather(*(transcribe_one(chunk) for chunk in chunks))

    def _normalize_backend_result(
        self,
        payload: dict[str, Any],
        chunk: AudioChunk,
    ) -> dict[str, Any]:
        language = payload.get("language")
        text = _extract_text(payload)
        payload_words = _normalize_words(payload.get("words"))

        segments = payload.get("segments")
        if isinstance(segments, list) and segments:
            normalized_segments = []
            for item in segments:
                start = _as_float(item.get("start"), 0.0)
                end = _as_float(item.get("end"), chunk.duration)
                normalized_item = {
                    "id": item.get("id", 0),
                    "start": max(0.0, start),
                    "end": max(start, end),
                    "text": str(item.get("text", "")).strip() or text,
                    "words": [],
                }

                normalized_item["words"] = _normalize_words(item.get("words"))
                if not normalized_item["words"] and payload_words:
                    normalized_item["words"] = _words_for_segment(
                        payload_words,
                        start=normalized_item["start"],
                        end=normalized_item["end"],
                    )

                normalized_segments.append(normalized_item)

            return {
                "text": text,
                "language": language,
                "segments": normalized_segments,
            }

        if payload_words:
            return {
                "text": text,
                "language": language,
                "segments": _group_words_into_segments(payload_words),
            }

        return {
            "text": text,
            "language": language,
            "segments": [
                {
                    "id": 0,
                    "start": 0.0,
                    "end": chunk.duration,
                    "text": text,
                    "words": [],
                }
            ],
        }


def _extract_text(payload: dict[str, Any]) -> str:
    if isinstance(payload.get("text"), str):
        return payload["text"].strip()

    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        message = choices[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()

    return ""


def _normalize_words(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []

    words: list[dict[str, Any]] = []
    for raw_word in value:
        if not isinstance(raw_word, dict):
            continue
        words.append(
            {
                "start": _as_float(raw_word.get("start"), None),
                "end": _as_float(raw_word.get("end"), None),
                "word": str(raw_word.get("word") or raw_word.get("text") or ""),
            }
        )
    return words


def _words_for_segment(
    words: list[dict[str, Any]],
    start: float,
    end: float,
) -> list[dict[str, Any]]:
    segment_words: list[dict[str, Any]] = []
    for word in words:
        word_start = _as_float(word.get("start"), None)
        word_end = _as_float(word.get("end"), None)
        if word_start is None or word_end is None:
            continue
        if word_end <= start or word_start >= end:
            continue
        segment_words.append(word)
    return segment_words


def _clean_normalized_result(result: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(result)
    cleaned["text"] = _clean_asr_text(str(cleaned.get("text", "")))

    cleaned_segments: list[dict[str, Any]] = []
    for segment in cleaned.get("segments", []):
        item = dict(segment)
        item["text"] = _clean_asr_text(str(item.get("text", "")))
        raw_words = item.get("words", [])
        item["words"] = _clean_words(raw_words if isinstance(raw_words, list) else [])
        quality_text = item["text"] or " ".join(str(word.get("word", "")) for word in item["words"])
        if _is_asr_garbage_text(quality_text):
            continue
        if item["text"] or item["words"]:
            cleaned_segments.append(item)
    cleaned["segments"] = cleaned_segments
    return cleaned


def _clean_words(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cleaned_words: list[dict[str, Any]] = []
    for word in words:
        token = _clean_asr_text(str(word.get("word", "")))
        if not token:
            continue
        item = dict(word)
        item["word"] = token
        cleaned_words.append(item)
    return cleaned_words


def _clean_asr_text(text: str) -> str:
    cleaned = _clean_transcript_marker(text)
    for pattern in ASR_ARTIFACT_REPLACEMENTS:
        cleaned = pattern.sub(" ", cleaned)
    cleaned = STANDALONE_LANGUAGE_ARTIFACT.sub(" ", cleaned)
    cleaned = REPEATED_NO_TAIL.sub("", cleaned)
    cleaned = _strip_repetitive_non_korean_tail(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def _clean_transcript_marker(text: str) -> str:
    marker = ASR_TRANSCRIPT_MARKER.search(text)
    if not marker:
        return text

    prefix = text[: marker.start()].strip()
    suffix = text[marker.end() :].strip()
    language = marker.group("language").lower()
    is_korean_marker = language in {"ko", "kor", "korean"}

    if prefix and suffix and is_korean_marker:
        return f"{prefix} {suffix}"
    if prefix and not is_korean_marker:
        return prefix
    return suffix


def _strip_repetitive_non_korean_tail(text: str) -> str:
    token_matches = list(TEXT_TOKEN_RE.finditer(text))
    if len(token_matches) < 24:
        return text

    max_tail_tokens = min(240, len(token_matches))
    for tail_size in range(max_tail_tokens, 23, -1):
        tail_matches = token_matches[-tail_size:]
        tail_text = text[tail_matches[0].start() :]
        if _is_asr_garbage_text(tail_text):
            return text[: tail_matches[0].start()].rstrip(" ,.;:!?")

    return text


def _is_asr_garbage_text(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return False

    tokens = [match.group(0).lower() for match in TEXT_TOKEN_RE.finditer(normalized)]
    if len(tokens) < 20:
        return False

    hangul_chars = len(HANGUL_RE.findall(normalized))
    text_chars = sum(1 for char in normalized if char.isalnum())
    hangul_ratio = hangul_chars / text_chars if text_chars else 0.0
    unique_ratio = len(set(tokens)) / len(tokens)

    if hangul_ratio >= 0.05:
        return False
    if unique_ratio <= 0.15:
        return True

    most_common_count = max(tokens.count(token) for token in set(tokens))
    return most_common_count >= 20 and most_common_count / len(tokens) >= 0.7


def _offset_and_clip_segments(
    segments: list[dict[str, Any]],
    offset: float,
    dedupe_before: float,
) -> list[dict[str, Any]]:
    adjusted_segments: list[dict[str, Any]] = []
    for segment in segments:
        start = _as_float(segment.get("start"), 0.0) + offset
        end = _as_float(segment.get("end"), start) + offset
        if end <= dedupe_before:
            continue
        if start < dedupe_before:
            start = dedupe_before

        adjusted = {
            "id": segment.get("id", 0),
            "start": start,
            "end": max(start, end),
            "text": str(segment.get("text", "")).strip(),
            "words": [],
        }

        for raw_word in segment.get("words", []):
            word_start = _as_float(raw_word.get("start"), None)
            word_end = _as_float(raw_word.get("end"), None)
            if word_start is None or word_end is None:
                continue

            word_start += offset
            word_end += offset
            if word_end <= dedupe_before:
                continue
            if word_start < dedupe_before:
                word_start = dedupe_before

            adjusted["words"].append(
                {
                    "start": word_start,
                    "end": max(word_start, word_end),
                    "word": str(raw_word.get("word", "")).strip(),
                }
            )

        adjusted_segments.append(adjusted)

    return adjusted_segments


def _group_words_into_segments(words: list[dict[str, Any]]) -> list[dict[str, Any]]:
    segments: list[dict[str, Any]] = []
    current_words: list[dict[str, Any]] = []

    for index, word in enumerate(words):
        current_words.append(
            {
                "start": _as_float(word.get("start"), 0.0),
                "end": _as_float(word.get("end"), 0.0),
                "word": str(word.get("text") or word.get("word") or "").strip(),
            }
        )

        current_word = current_words[-1]["word"]
        next_start = None
        if index + 1 < len(words):
            next_start = _as_float(words[index + 1].get("start"), None)

        should_split = False
        if current_word and current_word[-1] in PUNCTUATION:
            should_split = True
        if next_start is not None:
            gap = next_start - current_words[-1]["end"]
            if gap > 0.8:
                should_split = True
        if len(current_words) >= 32:
            should_split = True

        if should_split:
            segments.append(_build_segment(current_words))
            current_words = []

    if current_words:
        segments.append(_build_segment(current_words))

    return _reindex_segments(segments)


def _build_segment(words: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "id": 0,
        "start": words[0]["start"],
        "end": words[-1]["end"],
        "text": _join_words([word["word"] for word in words]),
        "words": words,
    }


def _reindex_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    reindexed: list[dict[str, Any]] = []
    for index, segment in enumerate(segments):
        item = dict(segment)
        item["id"] = index
        reindexed.append(item)
    return reindexed


def _merge_text(existing: str, incoming: str) -> str:
    existing = existing.strip()
    incoming = incoming.strip()
    if not existing:
        return incoming
    if not incoming:
        return existing

    if " " not in existing or " " not in incoming:
        return _merge_char_overlap(existing, incoming)

    existing_words = existing.split()
    incoming_words = incoming.split()
    max_window = min(30, len(existing_words), len(incoming_words))

    for size in range(max_window, 0, -1):
        if _normalize_tokens(existing_words[-size:]) == _normalize_tokens(incoming_words[:size]):
            merged = existing_words + incoming_words[size:]
            return " ".join(merged).strip()

    return f"{existing} {incoming}".strip()


def _merge_char_overlap(existing: str, incoming: str) -> str:
    max_window = min(48, len(existing), len(incoming))
    for size in range(max_window, 0, -1):
        if _normalize_string(existing[-size:]) == _normalize_string(incoming[:size]):
            return f"{existing}{incoming[size:]}".strip()
    return f"{existing} {incoming}".strip()


def _join_words(words: list[str]) -> str:
    if not words:
        return ""

    parts: list[str] = []
    for word in words:
        if not word:
            continue
        if not parts:
            parts.append(word)
            continue

        previous = parts[-1]
        if _needs_space(previous, word):
            parts.append(f" {word}")
        else:
            parts.append(word)

    text = "".join(parts)
    text = re.sub(r"\s+([,.;:!?])", r"\1", text)
    return text.strip()


def _needs_space(previous: str, current: str) -> bool:
    if current[0] in PUNCTUATION:
        return False
    if previous[-1] in {"(", "[", "{", "/", "-"}:
        return False
    if _is_hangul(previous[-1]) and _is_hangul(current[0]):
        return not _attaches_to_previous_hangul(current)
    if re.match(r"[\u4e00-\u9fff\u3040-\u30ff]", previous[-1]):
        return False
    if re.match(r"[\u4e00-\u9fff\u3040-\u30ff]", current[0]):
        return False
    return True


def _is_hangul(value: str) -> bool:
    return bool(HANGUL_RE.fullmatch(value))


def _attaches_to_previous_hangul(token: str) -> bool:
    if token in NO_SPACE_BEFORE_HANGUL_TOKENS:
        return True
    if len(token) == 1 and token in {"들", "적", "성", "식", "짜리"}:
        return True
    return False


def _normalize_tokens(tokens: list[str]) -> list[str]:
    return [_normalize_string(token) for token in tokens]


def _normalize_string(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _as_float(value: Any, default: float | None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default
