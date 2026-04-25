#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib import error, request


def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dotenv(path: Path | None = None) -> None:
    env_path = path or (project_root() / ".env")
    if not env_path.is_file():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, _, value = line.partition("=")
        key = key.strip()
        if not key or key in os.environ:
            continue

        val = value.strip()
        if len(val) >= 2 and val[0] == val[-1] and val[0] in ("'", '"'):
            val = val[1:-1]
        os.environ[key] = val


def read_env(name: str, default: str) -> str:
    value = os.environ.get(name, default).strip()
    return value if value else default


def parse_timestamp_granularities(raw: str) -> list[str]:
    values = [item.strip() for item in raw.split(",")]
    allowed = {"segment", "word"}
    result: list[str] = []
    for value in values:
        if not value:
            continue
        if value not in allowed:
            raise RuntimeError(
                f"Unsupported timestamp granularity: {value}. "
                "Supported values: segment, word"
            )
        if value not in result:
            result.append(value)
    return result or ["segment"]


load_dotenv()

# Set the test target directly here when you want to run the script.
YOUTUBE_URL = "https://www.youtube.com/watch?v=ZYbKUUrbatI"

BASE_URL = read_env("STT_BASE_URL", "http://localhost:8080").rstrip("/")
MODEL = read_env("STT_MODEL", "qwen3-asr")
LANGUAGE = read_env("DEFAULT_LANGUAGE", "ko")
RESPONSE_FORMAT = read_env("STT_RESPONSE_FORMAT", "verbose_json")
TIMESTAMP_GRANULARITIES = parse_timestamp_granularities(
    read_env("STT_TIMESTAMP_GRANULARITIES", "segment,word")
)
RETRIES = int(read_env("STT_HEALTH_RETRIES", "30"))
BACKOFF_SECONDS = float(read_env("STT_HEALTH_BACKOFF_SEC", "2"))
REQUEST_TIMEOUT_SECONDS = int(read_env("STT_REQUEST_TIMEOUT_SECONDS", "7200"))
OUTPUT_DIR = Path(
    read_env("STT_OUTPUT_DIR", str(project_root() / "scripts" / "outputs"))
)
ASR_ARTIFACT_REPLACEMENTS = (
    re.compile(r"language\s*(?:Korean\s*asr\s*text|Koreanasrtext)", re.IGNORECASE),
    re.compile(r"Korean\s*asr\s*text", re.IGNORECASE),
    re.compile(r"Koreanasrtext", re.IGNORECASE),
)
STANDALONE_LANGUAGE_ARTIFACT = re.compile(
    r"(?:(?<=^)|(?<=[\s\]\)])|(?<=[\uac00-\ud7af0-9]))language(?=$|[\s\uac00-\ud7af0-9])",
    re.IGNORECASE,
)


def http_json(
    method: str,
    url: str,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    req = request.Request(url=url, method=method, headers=headers or {})
    try:
        with request.urlopen(req, timeout=30) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body) if body else {}
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Unexpected JSON payload type from {url}: {type(payload).__name__}"
                )
            return resp.status, payload
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        print(f"[HTTP ERROR] status={exc.code} url={url}")
        print(body)
        raise
    except error.URLError as exc:
        print(f"[NETWORK ERROR] url={url} reason={exc.reason}")
        raise


def wait_for_health() -> None:
    health_url = f"{BASE_URL}/health"
    last_error: Exception | None = None

    for attempt in range(1, RETRIES + 1):
        try:
            status, payload = http_json("GET", health_url)
            backend_reachable = bool(payload.get("backend_reachable"))
            if status == 200 and payload.get("status") == "ok" and backend_reachable:
                print(f"[INFO] health ready: {payload}")
                return
            print(f"[WARN] Unexpected /health payload: {payload}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WAIT] attempt={attempt}/{RETRIES} failed, retrying...")

        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS)

    print("[FAIL] /health did not become ready in time.")
    if last_error is not None:
        raise last_error
    raise RuntimeError("STT health endpoint unavailable")


def ensure_command(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Required command not found in PATH: {name}")


def download_audio(youtube_url: str, target_dir: Path) -> Path:
    if not youtube_url:
        raise RuntimeError(
            "YouTube URL is empty. Set YOUTUBE_URL at the top of this script."
        )

    ensure_command("yt-dlp")

    outtmpl = str(target_dir / "%(title).200s.%(ext)s")
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "-f",
        "bestaudio/best",
        "--print",
        "after_move:filepath",
        "-o",
        outtmpl,
        youtube_url,
    ]
    print("[INFO] Downloading full audio with yt-dlp...")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        if proc.stdout.strip():
            print(proc.stdout)
        if proc.stderr.strip():
            print(proc.stderr)
        raise RuntimeError(f"yt-dlp failed with exit code {proc.returncode}")

    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("yt-dlp did not report the downloaded file path.")

    audio_path = Path(lines[-1])
    if not audio_path.is_file():
        raise RuntimeError(f"Downloaded file not found: {audio_path}")

    safe_path = target_dir / f"audio{audio_path.suffix}"
    audio_path.rename(safe_path)
    print(f"[INFO] downloaded_audio={safe_path}")
    return safe_path


def request_transcription(audio_path: Path) -> dict[str, Any] | str:
    ensure_command("curl")
    url = f"{BASE_URL}/v1/audio/transcriptions"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".out", delete=False) as resp_tmp:
        response_path = Path(resp_tmp.name)

    form_args = [
        "-F",
        f"file=@{audio_path}",
        "-F",
        f"model={MODEL}",
        "-F",
        f"language={LANGUAGE}",
        "-F",
        f"response_format={RESPONSE_FORMAT}",
    ]
    for granularity in TIMESTAMP_GRANULARITIES:
        form_args.extend(["-F", f"timestamp_granularities[]={granularity}"])

    try:
        t0 = time.perf_counter()
        proc = subprocess.run(
            [
                "curl",
                "-sS",
                "-o",
                str(response_path),
                "-w",
                "%{http_code}",
                "-X",
                "POST",
                *form_args,
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        elapsed = time.perf_counter() - t0
        print(f"[INFO] transcription_elapsed_sec={elapsed:.3f}")
        if proc.returncode != 0:
            raise RuntimeError(f"curl failed with exit code {proc.returncode}: {proc.stderr.strip()}")

        status_code = proc.stdout.strip()
        raw = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        if status_code != "200":
            raise RuntimeError(f"STT request failed with status={status_code}: {raw}")

        if RESPONSE_FORMAT in {"json", "verbose_json"}:
            payload = json.loads(raw) if raw else {}
            if not isinstance(payload, dict):
                raise RuntimeError(
                    f"Unexpected transcription response type: {type(payload).__name__}"
                )
            return payload
        return raw
    finally:
        if response_path.exists():
            response_path.unlink()


def format_timestamp(seconds: float | int | None) -> str:
    if seconds is None:
        return "??:??:??.???"

    value = max(float(seconds), 0.0)
    hours = int(value // 3600)
    minutes = int((value % 3600) // 60)
    secs = int(value % 60)
    millis = int(round((value - int(value)) * 1000))

    if millis == 1000:
        millis = 0
        secs += 1
    if secs == 60:
        secs = 0
        minutes += 1
    if minutes == 60:
        minutes = 0
        hours += 1

    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def clean_asr_text(text: str) -> str:
    cleaned = text
    for pattern in ASR_ARTIFACT_REPLACEMENTS:
        cleaned = pattern.sub("", cleaned)
    cleaned = STANDALONE_LANGUAGE_ARTIFACT.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    return cleaned.strip()


def build_clean_transcript(payload: dict[str, Any]) -> str:
    segments = payload.get("segments", [])
    if not isinstance(segments, list) or not segments:
        return clean_asr_text(str(payload.get("text", "")))

    paragraphs: list[str] = []
    current: list[str] = []
    previous_end: float | None = None

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        text = clean_asr_text(str(segment.get("text", "")))
        if not text:
            continue

        start = as_float(segment.get("start"))
        end = as_float(segment.get("end"))
        if (
            current
            and previous_end is not None
            and start is not None
            and start - previous_end >= 2.5
        ):
            paragraphs.append(" ".join(current).strip())
            current = []

        current.append(text)
        if end is not None:
            previous_end = end

    if current:
        paragraphs.append(" ".join(current).strip())

    return "\n\n".join(paragraph for paragraph in paragraphs if paragraph)


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def print_verbose_payload(payload: dict[str, Any]) -> None:
    text = payload.get("text", "")
    segments = payload.get("segments", [])
    words = payload.get("words", [])

    print("\n=== FULL TEXT ===")
    print(text.strip() if isinstance(text, str) else text)

    print("\n=== TIMESTAMPED SEGMENTS ===")
    if isinstance(segments, list) and segments:
        for seg in segments:
            if not isinstance(seg, dict):
                continue
            start = format_timestamp(seg.get("start"))
            end = format_timestamp(seg.get("end"))
            seg_text = str(seg.get("text", "")).strip()
            print(f"[{start} - {end}] {seg_text}")
    else:
        print("[WARN] No segments returned.")

    if isinstance(words, list) and words:
        print("\n=== WORD TIMESTAMPS ===")
        for word in words:
            if not isinstance(word, dict):
                continue
            start = format_timestamp(word.get("start"))
            end = format_timestamp(word.get("end"))
            token = str(word.get("word", "")).strip()
            print(f"[{start} - {end}] {token}")


def print_result(result: dict[str, Any] | str) -> None:
    if isinstance(result, dict):
        if RESPONSE_FORMAT == "verbose_json":
            print_verbose_payload(result)
            return
        print("\n=== JSON RESPONSE ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"\n=== {RESPONSE_FORMAT.upper()} RESPONSE ===")
    print(result.strip())


def save_outputs(audio_path: Path, result: dict[str, Any] | str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    base_name = audio_path.stem
    if isinstance(result, dict):
        json_path = OUTPUT_DIR / f"{base_name}.{RESPONSE_FORMAT}.json"
        json_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[INFO] saved_json={json_path}")

        segments = result.get("segments", [])
        if isinstance(segments, list) and segments:
            txt_path = OUTPUT_DIR / f"{base_name}.segments.txt"
            lines: list[str] = []
            for seg in segments:
                if not isinstance(seg, dict):
                    continue
                start = format_timestamp(seg.get("start"))
                end = format_timestamp(seg.get("end"))
                seg_text = str(seg.get("text", "")).strip()
                lines.append(f"[{start} - {end}] {seg_text}")
            txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            print(f"[INFO] saved_segments={txt_path}")

        words = result.get("words", [])
        if isinstance(words, list) and words:
            words_path = OUTPUT_DIR / f"{base_name}.words.txt"
            word_lines: list[str] = []
            for word in words:
                if not isinstance(word, dict):
                    continue
                start = format_timestamp(word.get("start"))
                end = format_timestamp(word.get("end"))
                token = str(word.get("word", "")).strip()
                word_lines.append(f"[{start} - {end}] {token}")
            words_path.write_text("\n".join(word_lines) + "\n", encoding="utf-8")
            print(f"[INFO] saved_words={words_path}")

        clean_text = build_clean_transcript(result)
        if clean_text:
            clean_path = OUTPUT_DIR / f"{base_name}.clean.txt"
            clean_path.write_text(clean_text + "\n", encoding="utf-8")
            print(f"[INFO] saved_clean_transcript={clean_path}")
        return

    suffix = {
        "text": "txt",
        "srt": "srt",
        "vtt": "vtt",
    }.get(RESPONSE_FORMAT, "txt")
    output_path = OUTPUT_DIR / f"{base_name}.{suffix}"
    output_path.write_text(result, encoding="utf-8")
    print(f"[INFO] saved_output={output_path}")
    if RESPONSE_FORMAT == "text":
        clean_text = clean_asr_text(result)
        if clean_text:
            clean_path = OUTPUT_DIR / f"{base_name}.clean.txt"
            clean_path.write_text(clean_text + "\n", encoding="utf-8")
            print(f"[INFO] saved_clean_transcript={clean_path}")


def main() -> int:
    try:
        print(f"[INFO] base_url={BASE_URL}")
        print(f"[INFO] model={MODEL}")
        print(f"[INFO] language={LANGUAGE}")
        print(f"[INFO] response_format={RESPONSE_FORMAT}")
        print(f"[INFO] timestamp_granularities={TIMESTAMP_GRANULARITIES}")
        wait_for_health()

        with tempfile.TemporaryDirectory(prefix="yt-qwen-asr-") as tmp_dir:
            work_dir = Path(tmp_dir)
            audio_path = download_audio(YOUTUBE_URL, work_dir)
            result = request_transcription(audio_path)
            #print_result(result)
            save_outputs(audio_path, result)

        print("[SUCCESS] YouTube -> Qwen ASR API transcription completed.")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
