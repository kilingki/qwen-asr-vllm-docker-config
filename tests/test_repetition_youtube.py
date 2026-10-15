#!/usr/bin/env python3
"""Transcribe known runaway videos and check chunk text for FA.

Downloads each of Fw-8hZhnH3Q and H2WN130OzLg, sends the 16 kHz PCM WAV with
include_chunks=true, and writes tests/outputs/<video_id>/response.json.
The saved chunks[].text must not still contain a phrase, including one whose
only changes are numbers, repeated five or more times in a row. This talks to
a running ASR server. The mock tests in test_transcription.py do not.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path
from typing import Any
from urllib import error, request

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "asr-api"))

from app.pipeline import _find_repetition_loops


PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
VIDEO_IDS = ("Fw-8hZhnH3Q", "H2WN130OzLg")


def project_root() -> Path:
    return _REPO_ROOT


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


load_dotenv()

BASE_URL = read_env("STT_BASE_URL", "http://localhost:8080").rstrip("/")
MODEL = read_env("STT_MODEL", "qwen3-asr")
LANGUAGE = read_env("DEFAULT_LANGUAGE", "ko")
RETRIES = int(read_env("STT_HEALTH_RETRIES", "30"))
BACKOFF_SECONDS = float(read_env("STT_HEALTH_BACKOFF_SEC", "2"))
REQUEST_TIMEOUT_SECONDS = int(read_env("STT_REQUEST_TIMEOUT_SECONDS", "7200"))


def youtube_url(video_id: str) -> str:
    return f"https://www.youtube.com/watch?v={video_id}"


def output_dir_for(video_id: str) -> Path:
    return project_root() / "tests" / "outputs" / video_id


def http_json(method: str, url: str) -> tuple[int, dict[str, Any]]:
    req = request.Request(url=url, method=method)
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
            if status == 200 and payload.get("status") == "ok":
                print(f"[INFO] facade ready: {payload}")
                _load_model()
                return
            print(f"[WARN] Unexpected /health payload: {payload}")
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[WAIT] attempt={attempt}/{RETRIES} failed, retrying...")

        if attempt < RETRIES:
            time.sleep(BACKOFF_SECONDS)

    print(
        "[FAIL] Live ASR server is not ready. "
        "This check is separate from the mock tests in test_transcription.py."
    )
    if last_error is not None:
        raise last_error
    raise RuntimeError("STT health endpoint unavailable")


def _load_model() -> None:
    req = request.Request(
        url=f"{BASE_URL}/control/load",
        data=b"{}",
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as resp:
            body = resp.read().decode("utf-8")
            payload = json.loads(body) if body else {}
            if resp.status != 200 or payload.get("state") != "ready":
                raise RuntimeError(f"model load did not become ready: {payload}")
            print(f"[INFO] model loaded: {payload}")
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"model load failed status={exc.code}: {body}") from exc


def ensure_command(name: str) -> None:
    if shutil.which(name):
        return
    raise RuntimeError(f"Required command not found in PATH: {name}")


def download_audio(youtube_url: str, target_dir: Path) -> Path:
    ensure_command("yt-dlp")
    outtmpl = str(target_dir / "%(title).200s.%(ext)s")
    proc = subprocess.run(
        [
            "yt-dlp",
            "--no-playlist",
            "-f",
            "bestaudio/best",
            "--print",
            "after_move:filepath",
            "-o",
            outtmpl,
            youtube_url,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
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


def normalize_pcm_wav(src: Path, dst: Path) -> None:
    ensure_command("ffmpeg")
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-acodec",
            "pcm_s16le",
            "-vn",
            str(dst),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        detail = proc.stderr.strip() or proc.stdout.strip()
        raise RuntimeError(f"ffmpeg failed with exit code {proc.returncode}: {detail}")

    with wave.open(str(dst), "rb") as wav:
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        sample_rate = wav.getframerate()
        num_samples = wav.getnframes()
        comptype = wav.getcomptype()
    if (
        channels != PCM_CHANNELS
        or sample_width != PCM_SAMPLE_WIDTH
        or sample_rate != PCM_SAMPLE_RATE
        or comptype != "NONE"
        or num_samples <= 0
    ):
        raise RuntimeError(
            "Normalized WAV must be 16 kHz, mono, signed 16-bit PCM. "
            f"Got channels={channels}, sample_width={sample_width}, "
            f"sample_rate={sample_rate}, comptype={comptype}, "
            f"num_samples={num_samples}."
        )
    print(f"[INFO] normalized_wav={dst} num_samples={num_samples}")


def request_transcription(audio_path: Path) -> dict[str, Any]:
    ensure_command("curl")
    url = f"{BASE_URL}/v1/audio/transcriptions"
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".out", delete=False) as resp_tmp:
        response_path = Path(resp_tmp.name)

    try:
        started = time.perf_counter()
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
                "-F",
                f"file=@{audio_path}",
                "-F",
                f"model={MODEL}",
                "-F",
                f"language={LANGUAGE}",
                "-F",
                "response_format=verbose_json",
                "-F",
                "include_chunks=true",
                url,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        print(f"[INFO] transcription_elapsed_sec={time.perf_counter() - started:.3f}")
        if proc.returncode != 0:
            raise RuntimeError(
                f"curl failed with exit code {proc.returncode}: {proc.stderr.strip()}"
            )

        status_code = proc.stdout.strip()
        raw = response_path.read_text(encoding="utf-8") if response_path.exists() else ""
        if status_code != "200":
            raise RuntimeError(f"STT request failed with status={status_code}: {raw}")

        payload = json.loads(raw) if raw else {}
        if not isinstance(payload, dict):
            raise RuntimeError(
                f"Unexpected transcription response type: {type(payload).__name__}"
            )
        return payload
    finally:
        if response_path.exists():
            response_path.unlink()


def check_saved_response(path: Path) -> int:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("Saved response is not a JSON object.")

    audio = payload.get("audio")
    chunks = payload.get("chunks")
    if not isinstance(audio, dict) or not isinstance(chunks, list):
        raise RuntimeError(
            "Response has no audio/chunks. "
            "The request must use response_format=verbose_json and include_chunks=true."
        )

    nonempty = 0
    failures: list[str] = []
    for position, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise RuntimeError(f"chunks[{position}] is not an object.")
        text = chunk.get("text")
        index = chunk.get("index", position)
        if not isinstance(text, str):
            raise RuntimeError(f"chunks[{index}].text is not a string.")
        if text.strip():
            nonempty += 1
        for loop in _find_repetition_loops(text):
            failures.append(
                f"chunks[{index}] phrase={loop['phrase']!r} "
                f"repeat_count={loop['repeat_count']}"
            )

    if nonempty == 0:
        raise RuntimeError("Every chunks[].text is empty.")
    if failures:
        print("[FAIL] Repeated phrases remain in chunks[].text:")
        for failure in failures:
            print(failure)
        raise RuntimeError(f"{len(failures)} repeated phrase(s) remain.")

    print(f"[INFO] chunks={len(chunks)} nonempty={nonempty}")
    print(f"[INFO] saved_response={path}")
    return len(chunks)


def transcribe_video(video_id: str) -> None:
    url = youtube_url(video_id)
    output_dir = output_dir_for(video_id)
    print(f"[INFO] youtube_url={url}")
    print(f"[INFO] output_dir={output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    source_wav = output_dir / "source.wav"
    with tempfile.TemporaryDirectory(prefix=f"yt-repetition-{video_id}-") as tmp_dir:
        downloaded = download_audio(url, Path(tmp_dir))
        normalize_pcm_wav(downloaded, source_wav)

    payload = request_transcription(source_wav)
    response_path = output_dir / "response.json"
    response_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    check_saved_response(response_path)
    print(f"[SUCCESS] {video_id} chunk text has no 5+ consecutive repeats.")


def main() -> int:
    print(f"[INFO] base_url={BASE_URL}")
    print(f"[INFO] model={MODEL}")
    print(f"[INFO] language={LANGUAGE}")
    print("[INFO] response_format=verbose_json include_chunks=true")
    print(f"[INFO] video_ids={','.join(VIDEO_IDS)}")
    try:
        wait_for_health()
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] {type(exc).__name__}: {exc}")
        return 1

    failures: list[str] = []
    for video_id in VIDEO_IDS:
        try:
            transcribe_video(video_id)
        except Exception as exc:  # noqa: BLE001
            failures.append(video_id)
            print(f"[FAIL] {video_id} {type(exc).__name__}: {exc}")

    if failures:
        print(f"[FAIL] videos={','.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
