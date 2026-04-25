import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile


@dataclass(slots=True)
class AudioChunk:
    index: int
    path: Path
    start: float
    end: float

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


async def save_upload(upload_file: UploadFile, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as target:
        while True:
            chunk = await upload_file.read(1024 * 1024)
            if not chunk:
                break
            target.write(chunk)


def convert_to_wav(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-vn",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return float(result.stdout.strip())


def create_chunks(
    source_wav: Path,
    chunks_dir: Path,
    chunk_seconds: int,
    overlap_seconds: int,
) -> list[AudioChunk]:
    duration = probe_duration(source_wav)
    if duration <= 0:
        raise ValueError("Audio duration must be greater than zero.")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    if duration <= chunk_seconds:
        single_path = chunks_dir / "chunk-0000.wav"
        shutil.copyfile(source_wav, single_path)
        return [AudioChunk(index=0, path=single_path, start=0.0, end=duration)]

    step = max(1, chunk_seconds - overlap_seconds)
    chunks: list[AudioChunk] = []
    start = 0.0
    index = 0

    while start < duration:
        end = min(duration, start + chunk_seconds)
        chunk_path = chunks_dir / f"chunk-{index:04d}.wav"
        _extract_chunk(source_wav, chunk_path, start=start, end=end)
        chunks.append(AudioChunk(index=index, path=chunk_path, start=start, end=end))
        if end >= duration:
            break
        start += step
        index += 1

    return chunks


def _extract_chunk(src: Path, dst: Path, start: float, end: float) -> None:
    duration = max(0.1, end - start)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(src),
            "-t",
            f"{duration:.3f}",
            "-acodec",
            "pcm_s16le",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(dst),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
