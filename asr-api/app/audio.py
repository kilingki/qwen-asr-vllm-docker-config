import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

PCM_SAMPLE_RATE = 16000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
PCM_FORMAT_DETAIL = "16 kHz, mono, signed 16-bit PCM WAV"


@dataclass(slots=True)
class AudioChunk:
    index: int
    path: Path
    start: float
    end: float
    start_sample: int | None = None
    end_sample: int | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)


@dataclass(slots=True)
class PcmWav:
    sample_rate: int
    channels: int
    num_samples: int


class PcmFormatError(ValueError):
    """Raised when include_chunks audio is not the required PCM WAV."""


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


def inspect_pcm_wav(path: Path) -> PcmWav:
    try:
        with wave.open(str(path), "rb") as wav:
            channels = wav.getnchannels()
            sample_width = wav.getsampwidth()
            sample_rate = wav.getframerate()
            num_samples = wav.getnframes()
            comptype = wav.getcomptype()
    except (wave.Error, EOFError) as exc:
        raise PcmFormatError(
            "include_chunks=true requires "
            f"{PCM_FORMAT_DETAIL}. The uploaded file is not a PCM WAV."
        ) from exc

    if (
        channels != PCM_CHANNELS
        or sample_width != PCM_SAMPLE_WIDTH
        or sample_rate != PCM_SAMPLE_RATE
        or comptype != "NONE"
        or num_samples <= 0
    ):
        raise PcmFormatError(
            "include_chunks=true requires "
            f"{PCM_FORMAT_DETAIL}. Got channels={channels}, "
            f"sample_width={sample_width}, sample_rate={sample_rate}, "
            f"comptype={comptype}, num_samples={num_samples}."
        )
    return PcmWav(
        sample_rate=sample_rate,
        channels=channels,
        num_samples=num_samples,
    )


def create_pcm_chunks(
    source_wav: Path,
    chunks_dir: Path,
    chunk_seconds: int,
    overlap_seconds: int,
    sample_rate: int,
    num_samples: int,
) -> list[AudioChunk]:
    if num_samples <= 0:
        raise ValueError("Audio duration must be greater than zero.")

    chunks_dir.mkdir(parents=True, exist_ok=True)
    chunk_samples = chunk_seconds * sample_rate
    overlap_samples = overlap_seconds * sample_rate
    if num_samples <= chunk_samples:
        single_path = chunks_dir / "chunk-0000.wav"
        shutil.copyfile(source_wav, single_path)
        return [
            AudioChunk(
                index=0,
                path=single_path,
                start=0.0,
                end=num_samples / sample_rate,
                start_sample=0,
                end_sample=num_samples,
            )
        ]

    step = max(1, chunk_samples - overlap_samples)
    chunks: list[AudioChunk] = []
    start = 0
    index = 0
    while start < num_samples:
        end = min(num_samples, start + chunk_samples)
        chunk_path = chunks_dir / f"chunk-{index:04d}.wav"
        _write_pcm_slice(
            source_wav,
            chunk_path,
            start_sample=start,
            end_sample=end,
            sample_rate=sample_rate,
        )
        chunks.append(
            AudioChunk(
                index=index,
                path=chunk_path,
                start=start / sample_rate,
                end=end / sample_rate,
                start_sample=start,
                end_sample=end,
            )
        )
        if end >= num_samples:
            break
        start += step
        index += 1
    return chunks


def _write_pcm_slice(
    src: Path,
    dst: Path,
    start_sample: int,
    end_sample: int,
    sample_rate: int,
) -> None:
    frame_count = end_sample - start_sample
    with wave.open(str(src), "rb") as reader:
        channels = reader.getnchannels()
        sample_width = reader.getsampwidth()
        reader.setpos(start_sample)
        frames = reader.readframes(frame_count)
    expected_bytes = frame_count * sample_width * channels
    if len(frames) != expected_bytes:
        raise ValueError(
            f"PCM slice [{start_sample}, {end_sample}) returned "
            f"{len(frames)} bytes, expected {expected_bytes}."
        )
    with wave.open(str(dst), "wb") as writer:
        writer.setnchannels(channels)
        writer.setsampwidth(sample_width)
        writer.setframerate(sample_rate)
        writer.writeframes(frames)
