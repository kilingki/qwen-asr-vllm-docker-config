#!/usr/bin/env python3
"""Measure load, transcription, and unload against the running ASR container.

The model lives in a child process, so this records host GPU memory and
worker process exit instead of torch.cuda.memory_allocated().
"""

import json
import subprocess
import tempfile
import threading
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path


BASE = "http://127.0.0.1:8080"
STATUS_LIMIT_SEC = 2.0
ROUNDS = 3
RESIDUAL_LIMIT_MIB = 32
OUTPUT = Path(__file__).resolve().parent / "outputs" / "gpu_lifecycle.md"
CONTAINER = "asr-api"


def gpu_mib() -> int:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    return int(raw.strip().splitlines()[0])


def worker_cmds() -> list[str]:
    raw = subprocess.check_output(["docker", "exec", CONTAINER, "ps", "-eo", "cmd"], text=True)
    wanted = ("qwen-asr-serve", "VLLM::EngineCore")
    return [line for line in raw.splitlines() if any(name in line for name in wanted)]


def request(method: str, path: str, body: bytes | None = None, timeout: float = 600) -> tuple[int, dict]:
    data = body if body not in (None, b"") else None
    req = urllib.request.Request(BASE + path, data=data, method=method)
    if data is not None:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            payload = response.read()
            return response.status, json.loads(payload) if payload else {}
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, json.loads(raw) if raw else {}


def write_wav(path: Path) -> None:
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16000)
        handle.writeframes(b"\x00\x00" * 16000)


def transcribe(path: Path) -> tuple[int, dict]:
    boundary = "----asr-gpu-check"
    body = "\r\n".join(
        [
            f"--{boundary}",
            'Content-Disposition: form-data; name="file"; filename="silence.wav"',
            "Content-Type: audio/wav",
            "",
            path.read_bytes().decode("latin1"),
            f"--{boundary}",
            'Content-Disposition: form-data; name="model"',
            "",
            "qwen3-asr",
            f"--{boundary}",
            'Content-Disposition: form-data; name="response_format"',
            "",
            "json",
            f"--{boundary}--",
            "",
        ]
    ).encode("latin1")
    req = urllib.request.Request(BASE + "/v1/audio/transcriptions", data=body, method="POST")
    req.add_header("content-type", f"multipart/form-data; boundary={boundary}")
    try:
        with urllib.request.urlopen(req, timeout=600) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        raw = exc.read().decode()
        return exc.code, json.loads(raw) if raw else {}


def watch_status(stop: threading.Event, samples: list[float]) -> None:
    while not stop.is_set():
        started = time.perf_counter()
        try:
            request("GET", "/control/status", timeout=STATUS_LIMIT_SEC)
        except Exception:
            pass
        samples.append(time.perf_counter() - started)
        time.sleep(0.05)


def watch_memory(stop: threading.Event, peaks: list[int]) -> None:
    peak = gpu_mib()
    while not stop.is_set():
        peak = max(peak, gpu_mib())
        time.sleep(0.05)
    peaks.append(peak)


def write_report(rounds: list[dict], summary: dict | None, error: str | None) -> None:
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# GPU lifecycle 테스트 결과",
        "",
        "컨테이너의 자식 워커에 대해 load, 전사, unload를 세 번 반복한 기록이다.",
        "메모리는 호스트 nvidia-smi 사용량이다.",
        "",
    ]
    if error:
        lines.extend([f"실패: {error}", ""])
    for item in rounds:
        lines.extend(
            [
                f"## round {item['round']} {item['step']}",
                "",
                "```json",
                json.dumps(item, ensure_ascii=False, indent=2),
                "```",
                "",
            ]
        )
    if summary is not None:
        lines.extend(["## summary", "", "```json", json.dumps(summary, ensure_ascii=False, indent=2), "```", ""])
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def fail(rounds: list[dict], message: str) -> None:
    write_report(rounds, None, message)
    raise SystemExit(message)


def main() -> None:
    try:
        status, initial = request("GET", "/control/status", timeout=5)
    except Exception as exc:
        fail([], f"control status is not reachable: {exc}")
    if status != 200 or initial.get("state") != "unloaded" or initial.get("residency") != "not_resident":
        fail([{"round": 0, "step": "startup", "http_status": status, "body": initial}], "startup is not unloaded")

    wav_path = Path(tempfile.gettempdir()) / "asr-gpu-check.wav"
    write_wav(wav_path)
    rounds: list[dict] = []
    load_peaks: list[int] = []
    infer_peaks: list[int] = []
    residuals: list[int] = []

    for round_index in range(ROUNDS):
        before = gpu_mib()
        stop = threading.Event()
        latencies: list[float] = []
        peaks: list[int] = []
        threading.Thread(target=watch_status, args=(stop, latencies), daemon=True).start()
        threading.Thread(target=watch_memory, args=(stop, peaks), daemon=True).start()
        started = time.perf_counter()
        load_status, load_body = request("POST", "/control/load", b"{}")
        stop.set()
        time.sleep(0.2)
        load_sec = time.perf_counter() - started
        worst = max(latencies) if latencies else STATUS_LIMIT_SEC
        load_peak = max(peaks) if peaks else gpu_mib()
        rounds.append(
            {
                "round": round_index,
                "step": "load",
                "http_status": load_status,
                "body": load_body,
                "seconds": round(load_sec, 3),
                "status_max_sec": round(worst, 3),
                "mib_before": before,
                "mib_after": gpu_mib(),
                "load_peak_mib": load_peak,
            }
        )
        if load_status != 200 or load_body.get("state") != "ready" or worst >= STATUS_LIMIT_SEC:
            fail(rounds, "load did not keep status responsive")
        load_peaks.append(load_peak)

        stop = threading.Event()
        infer_latencies: list[float] = []
        infer_peak_box: list[int] = []
        threading.Thread(target=watch_status, args=(stop, infer_latencies), daemon=True).start()
        threading.Thread(target=watch_memory, args=(stop, infer_peak_box), daemon=True).start()
        tx_status, tx_body = transcribe(wav_path)
        stop.set()
        time.sleep(0.2)
        infer_worst = max(infer_latencies) if infer_latencies else STATUS_LIMIT_SEC
        infer_peak = max(infer_peak_box) if infer_peak_box else gpu_mib()
        rounds.append(
            {
                "round": round_index,
                "step": "transcribe",
                "http_status": tx_status,
                "body": tx_body,
                "status_max_sec": round(infer_worst, 3),
                "infer_peak_mib": infer_peak,
            }
        )
        if tx_status != 200 or infer_worst >= STATUS_LIMIT_SEC:
            fail(rounds, "inference did not keep status responsive")
        infer_peaks.append(infer_peak)

        unload_status, unload_body = request("POST", "/control/unload")
        time.sleep(2)
        workers = worker_cmds()
        residual = gpu_mib()
        residuals.append(residual)
        rounds.append(
            {
                "round": round_index,
                "step": "unload",
                "http_status": unload_status,
                "body": unload_body,
                "worker_cmds": workers,
                "mib_after": residual,
            }
        )
        if (
            unload_status != 200
            or unload_body.get("state") != "unloaded"
            or unload_body.get("residency") != "not_resident"
            or workers
        ):
            fail(rounds, "unload did not release the worker")

    grew = residuals[-1] > residuals[0] + RESIDUAL_LIMIT_MIB
    summary = {
        "load_peak_mib": max(load_peaks),
        "infer_peak_mib": max(infer_peaks),
        "residuals_mib": residuals,
        "residual_grew": grew,
    }
    write_report(rounds, summary, "GPU memory grew across unload rounds" if grew else None)
    if grew:
        raise SystemExit("GPU memory grew across unload rounds")


if __name__ == "__main__":
    main()
