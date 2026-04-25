from typing import Any


def to_json_response(result: dict[str, Any]) -> dict[str, Any]:
    return {"text": result["text"]}


def to_verbose_json_response(result: dict[str, Any]) -> dict[str, Any]:
    response = {
        "task": "transcribe",
        "language": result.get("language"),
        "duration": result.get("duration"),
        "text": result["text"],
        "segments": result.get("segments", []),
    }
    words = _flatten_words(result.get("segments", []))
    if words:
        response["words"] = words
    return response


def to_text_response(result: dict[str, Any]) -> str:
    return result["text"]


def to_srt(result: dict[str, Any]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(result.get("segments", []), start=1):
        start = _format_timestamp(seg["start"], srt=True)
        end = _format_timestamp(seg["end"], srt=True)
        text = seg["text"].strip()
        lines.append(f"{i}\n{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def to_vtt(result: dict[str, Any]) -> str:
    lines: list[str] = ["WEBVTT\n"]
    for seg in result.get("segments", []):
        start = _format_timestamp(seg["start"], srt=False)
        end = _format_timestamp(seg["end"], srt=False)
        text = seg["text"].strip()
        lines.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(lines)


def _flatten_words(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    words: list[dict[str, Any]] = []
    for segment in segments:
        for word in segment.get("words", []):
            words.append(word)
    return words


def _format_timestamp(seconds: float, srt: bool = True) -> str:
    if seconds < 0:
        seconds = 0.0

    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int(round((seconds - int(seconds)) * 1000))

    if millis == 1000:
        millis = 0
        secs += 1
    if secs == 60:
        secs = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1

    sep = "," if srt else "."
    return f"{hours:02}:{minutes:02}:{secs:02}{sep}{millis:03}"
