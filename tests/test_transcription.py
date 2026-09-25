import array
import asyncio
import os
import sys
import tempfile
import threading
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "asr-api"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

_TEST_ROOT = tempfile.mkdtemp(prefix="asr-facade-")
os.environ["STORAGE_DIR"] = _TEST_ROOT

from api_report import install

install()
os.environ["DEFAULT_LANGUAGE"] = "ko"
os.environ["CHUNK_SECONDS"] = "120"
os.environ["CHUNK_OVERLAP_SECONDS"] = "2"
os.environ["MAX_CONCURRENT_CHUNKS"] = "2"

from fastapi.testclient import TestClient

from app import main as api
from app.audio import AudioChunk
from app.lifecycle import Lifecycle
from app.pipeline import TranscriptionPipeline


def write_pcm(
    path: Path,
    num_samples: int,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    frame_count = num_samples * channels
    if sample_width == 2:
        payload = array.array("h", ((index * 17) % 32767 for index in range(frame_count)))
        raw = payload.tobytes()
    else:
        raw = bytes(index % 251 for index in range(frame_count * sample_width))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(sample_width)
        wav.setframerate(sample_rate)
        wav.writeframes(raw)
    return read_frames(path)


def read_frames(path: Path) -> bytes:
    with wave.open(str(path), "rb") as wav:
        return wav.readframes(wav.getnframes())


class FakeAsr:
    def __init__(self, texts: dict[str, object] | None = None) -> None:
        self.texts = texts or {}
        self.languages: list[str | None] = []
        self.calls = 0
        self.frames: dict[str, bytes] = {}

    async def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict:
        del model, prompt, timestamp_granularities
        self.calls += 1
        self.languages.append(language)
        if audio_path.is_file():
            self.frames[audio_path.name] = read_frames(audio_path)
        payload = self.texts.get(audio_path.name, {"text": "문장", "language": "ko"})
        if isinstance(payload, str):
            return {"text": payload, "language": "ko"}
        return payload

    async def list_models(self) -> dict:
        return {"data": []}

    async def close(self) -> None:
        return None


class OverlapAsr(FakeAsr):
    def __init__(self) -> None:
        super().__init__()
        self.active = 0
        self.max_active = 0
        self._init_lock = threading.Lock()
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        with self._init_lock:
            if self._ready:
                return
            self.lock = asyncio.Lock()
            self.release = asyncio.Event()
            self._ready = True

    async def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict:
        self._ensure()
        async with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= 2:
                self.release.set()
        try:
            await asyncio.wait_for(self.release.wait(), timeout=2)
        finally:
            async with self.lock:
                self.active -= 1
        return await super().transcribe(
            audio_path,
            model,
            language,
            prompt,
            timestamp_granularities,
        )


class DelayedAsr(FakeAsr):
    async def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict:
        index = int(audio_path.stem.split("-")[1])
        if index == 0:
            await asyncio.sleep(0.05)
        return await super().transcribe(
            audio_path,
            model,
            language,
            prompt,
            timestamp_granularities,
        )


def _form(file_bytes: bytes, filename: str, *fields: tuple[str, str]) -> list[tuple[str, tuple]]:
    form: list[tuple[str, tuple]] = [
        ("file", (filename, file_bytes, "audio/wav")),
    ]
    for name, value in fields:
        form.append((name, (None, value)))
    return form


def _chunk(index: int, name: str, start: float, end: float) -> AudioChunk:
    return AudioChunk(index=index, path=Path(name), start=start, end=end)


class _PatchedAudio:
    def __init__(self, chunks: list[AudioChunk], duration: float) -> None:
        self.chunks = chunks
        self.duration = duration
        self.converted = False
        self._patches: list = []

    def __enter__(self) -> "_PatchedAudio":
        def probe_duration(path: Path) -> float:
            del path
            return self.duration

        def create_chunks(source_wav, chunks_dir, chunk_seconds, overlap_seconds):
            del source_wav, chunks_dir, chunk_seconds, overlap_seconds
            return self.chunks

        def convert_to_wav(src: Path, dst: Path) -> None:
            del src
            self.converted = True
            dst.write_bytes(b"RIFF")

        self._patches = [
            patch("app.pipeline.probe_duration", probe_duration),
            patch("app.pipeline.create_chunks", create_chunks),
            patch("app.main.convert_to_wav", convert_to_wav),
        ]
        for item in self._patches:
            item.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for item in reversed(self._patches):
            item.stop()


class _ImmediateWorker:
    def __init__(self) -> None:
        self.alive = False

    async def start(self) -> None:
        self.alive = True

    async def stop(self) -> None:
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


def _load_model(client: TestClient) -> None:
    response = client.post("/control/load")
    if response.status_code != 200:
        raise AssertionError(response.text)


class TranscriptionApiTest(unittest.TestCase):
    def setUp(self) -> None:
        api.lifecycle = Lifecycle(_ImmediateWorker())

    def test_health_reports_backend_only(self) -> None:
        with TestClient(api.app) as client:
            body = client.get("/health").json()
        self.assertEqual(body, {"status": "ok", "backend_reachable": False})

        with TestClient(api.app) as client:
            _load_model(client)
            body = client.get("/health").json()
        self.assertEqual(body, {"status": "ok", "backend_reachable": True})

    def test_segment_request_keeps_asr_only_verbose_json(self) -> None:
        chunks = [_chunk(0, "chunk-0000.wav", 0.0, 120.0)]
        asr = FakeAsr({"chunk-0000.wav": "language Korean <asr_text> 첫 문장"})
        with _PatchedAudio(chunks, duration=120.0):
            with TestClient(api.app) as client:
                _load_model(client)
                api.pipeline = TranscriptionPipeline(asr)
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_form(
                        b"audio",
                        "sample.wav",
                        ("response_format", "verbose_json"),
                        ("language", "ko"),
                        ("timestamp_granularities[]", "segment"),
                    ),
                )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertNotIn("words", body)
        self.assertNotIn("audio", body)
        self.assertNotIn("chunks", body)
        self.assertEqual(body["text"], "첫 문장")
        self.assertEqual(body["segments"][0]["words"], [])
        self.assertEqual(body["segments"][0]["start"], 0.0)
        self.assertEqual(body["segments"][0]["end"], 120.0)
        self.assertEqual(asr.languages, ["ko"])

    def test_formats_without_include_chunks_match_asr_only(self) -> None:
        chunks = [_chunk(0, "chunk-0000.wav", 0.0, 120.0)]
        expected = {
            "json": {"text": "첫 문장"},
            "text": "첫 문장",
            "srt": "00:00:00,000 --> 00:02:00,000\n첫 문장",
            "vtt": "WEBVTT\n\n00:00:00.000 --> 00:02:00.000\n첫 문장",
        }
        for response_format, expected_body in expected.items():
            with self.subTest(response_format=response_format):
                asr = FakeAsr({"chunk-0000.wav": "language Korean <asr_text> 첫 문장"})
                with _PatchedAudio(chunks, duration=120.0):
                    with TestClient(api.app) as client:
                        _load_model(client)
                        api.pipeline = TranscriptionPipeline(asr)
                        response = client.post(
                            "/v1/audio/transcriptions",
                            files=_form(
                                b"not-pcm",
                                "sample.bin",
                                ("response_format", response_format),
                            ),
                        )
                self.assertEqual(response.status_code, 200, response.text)
                if response_format == "json":
                    self.assertEqual(response.json(), expected_body)
                else:
                    self.assertIn(expected_body, response.text)

    def test_include_chunks_false_still_converts_audio(self) -> None:
        chunks = [_chunk(0, "chunk-0000.wav", 0.0, 1.0)]
        asr = FakeAsr({"chunk-0000.wav": "문장"})
        wav_path = Path(_TEST_ROOT) / "already.wav"
        write_pcm(wav_path, 160)
        audio = _PatchedAudio(chunks, duration=1.0)
        with audio:
            with TestClient(api.app) as client:
                _load_model(client)
                api.pipeline = TranscriptionPipeline(asr)
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_form(
                        wav_path.read_bytes(),
                        "already.wav",
                        ("response_format", "verbose_json"),
                        ("include_chunks", "false"),
                    ),
                )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(audio.converted)
        body = response.json()
        self.assertNotIn("audio", body)
        self.assertNotIn("chunks", body)
        self.assertEqual(body["text"], "문장")

    def test_long_chunk_is_not_rejected(self) -> None:
        chunks = [_chunk(0, "chunk-0000.wav", 0.0, 181.0)]
        asr = FakeAsr({"chunk-0000.wav": "긴 문장"})
        with _PatchedAudio(chunks, duration=181.0):
            with TestClient(api.app) as client:
                _load_model(client)
                api.pipeline = TranscriptionPipeline(asr)
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_form(b"audio", "sample.wav", ("response_format", "json")),
                )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(response.json(), {"text": "긴 문장"})

    def test_word_requests_are_400_before_conversion(self) -> None:
        cases = [
            (("timestamp_granularities[]", "word"),),
            (("timestamp_granularities", "word"),),
            (
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
            ),
        ]
        for fields in cases:
            with self.subTest(fields=fields):
                asr = FakeAsr()
                converted = {"called": False}

                def convert_to_wav(src: Path, dst: Path) -> None:
                    del src, dst
                    converted["called"] = True

                with patch("app.main.convert_to_wav", convert_to_wav):
                    with TestClient(api.app) as client:
                        api.pipeline = TranscriptionPipeline(asr)
                        response = client.post(
                            "/v1/audio/transcriptions",
                            files=_form(
                                b"audio",
                                "sample.wav",
                                ("response_format", "verbose_json"),
                                *fields,
                            ),
                        )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertEqual(
                    response.json()["detail"],
                    "This server does not provide word timestamps.",
                )
                self.assertFalse(converted["called"])
                self.assertEqual(asr.calls, 0)

    def test_include_chunks_rejects_non_verbose_formats(self) -> None:
        for response_format in ("json", "text", "srt", "vtt"):
            with self.subTest(response_format=response_format):
                asr = FakeAsr()
                with patch("app.main.convert_to_wav", _fail_convert):
                    with TestClient(api.app) as client:
                        api.pipeline = TranscriptionPipeline(asr)
                        response = client.post(
                            "/v1/audio/transcriptions",
                            files=_form(
                                b"audio",
                                "sample.wav",
                                ("response_format", response_format),
                                ("include_chunks", "true"),
                            ),
                        )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("verbose_json", response.json()["detail"])
                self.assertEqual(asr.calls, 0)

    def test_invalid_include_chunks_is_400(self) -> None:
        asr = FakeAsr()
        with patch("app.main.convert_to_wav", _fail_convert):
            with TestClient(api.app) as client:
                _load_model(client)
                api.pipeline = TranscriptionPipeline(asr)
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_form(
                        b"audio",
                        "sample.wav",
                        ("include_chunks", "maybe"),
                    ),
                )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("include_chunks", response.json()["detail"])
        self.assertEqual(asr.calls, 0)

    def test_bad_pcm_is_400_before_inference(self) -> None:
        samples = Path(_TEST_ROOT)
        write_pcm(samples / "stereo.wav", 32, channels=2)
        write_pcm(samples / "rate.wav", 32, sample_rate=44100)
        write_pcm(samples / "width.wav", 32, sample_width=1)
        cases = {
            "not-wav": (b"not-a-wav-file", "sample.wav"),
            "stereo": ((samples / "stereo.wav").read_bytes(), "stereo.wav"),
            "rate": ((samples / "rate.wav").read_bytes(), "rate.wav"),
            "width": ((samples / "width.wav").read_bytes(), "width.wav"),
        }
        for name, (payload, filename) in cases.items():
            with self.subTest(name=name):
                asr = FakeAsr()
                with patch("app.main.convert_to_wav", _fail_convert):
                    with TestClient(api.app) as client:
                        api.pipeline = TranscriptionPipeline(asr)
                        response = client.post(
                            "/v1/audio/transcriptions",
                            files=_form(
                                payload,
                                filename,
                                ("response_format", "verbose_json"),
                                ("include_chunks", "true"),
                            ),
                        )
                self.assertEqual(response.status_code, 400, response.text)
                self.assertIn("PCM", response.json()["detail"])
                self.assertEqual(asr.calls, 0)

    def test_pcm_ranges_match_audio_sent_to_asr(self) -> None:
        cases = [
            ("short", 17, None, None, [(0, 17)]),
            ("exact", 16000, 1, 0, [(0, 16000)]),
            ("tail", 16100, 1, 0, [(0, 16000), (16000, 16100)]),
            (
                "default-settings",
                120 * 16000 + 100,
                None,
                None,
                [(0, 120 * 16000), (118 * 16000, 120 * 16000 + 100)],
            ),
            (
                "custom",
                2 * 16000 + 50,
                2,
                1,
                [(0, 2 * 16000), (16000, 2 * 16000 + 50)],
            ),
        ]
        for name, num_samples, chunk_seconds, overlap_seconds, bounds in cases:
            with self.subTest(name=name):
                response, asr, original = self._post_pcm(
                    num_samples,
                    chunk_seconds=chunk_seconds,
                    overlap_seconds=overlap_seconds,
                )
                self.assertEqual(response.status_code, 200, response.text)
                body = response.json()
                self.assertEqual(body["audio"]["num_samples"], num_samples)
                self.assertEqual(body["audio"]["sample_rate"], 16000)
                self.assertEqual(body["audio"]["channels"], 1)
                self.assertEqual(body["duration"], num_samples / 16000)
                self.assertEqual(len(body["chunks"]), len(bounds))
                for chunk, (start, end) in zip(body["chunks"], bounds, strict=True):
                    self.assertEqual(chunk["start_sample"], start)
                    self.assertEqual(chunk["end_sample"], end)
                    sent = asr.frames[f"chunk-{chunk['index']:04d}.wav"]
                    self.assertEqual(sent, original[start * 2 : end * 2])
                    self.assertEqual(len(sent) // 2, end - start)
                self.assertNotIn("path", body)
                self.assertNotIn("words", body)

    def test_chunks_keep_premerge_text_and_index_order(self) -> None:
        asr = DelayedAsr(
            {
                "chunk-0000.wav": "알파 겹침",
                "chunk-0001.wav": "",
                "chunk-0002.wav": "겹침 베타",
            }
        )
        response, _, _ = self._post_pcm(
            48000,
            texts=asr.texts,
            asr=asr,
            chunk_seconds=1,
            overlap_seconds=0,
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["text"], "알파 겹침 베타")
        self.assertEqual([chunk["index"] for chunk in body["chunks"]], [0, 1, 2])
        self.assertEqual(
            [chunk["text"] for chunk in body["chunks"]],
            ["알파 겹침", "", "겹침 베타"],
        )
        self.assertEqual(
            [chunk["start_sample"] for chunk in body["chunks"]],
            [0, 16000, 32000],
        )

    def test_chunk_text_drops_filtered_segments(self) -> None:
        garbage = "no " * 25
        response, _, _ = self._post_pcm(
            160,
            texts={
                "chunk-0000.wav": {
                    "text": "좋은 문장 그리고 버린 말",
                    "language": "ko",
                    "segments": [
                        {"id": 0, "start": 0.0, "end": 1.0, "text": "좋은 문장"},
                        {"id": 1, "start": 1.0, "end": 2.0, "text": garbage},
                    ],
                }
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["chunks"][0]["text"], "좋은 문장")
        self.assertNotIn("no", body["chunks"][0]["text"])

    def test_unknown_chunk_language_is_null(self) -> None:
        with patch("app.main.DEFAULT_LANGUAGE", ""):
            response, _, _ = self._post_pcm(
                160,
                texts={"chunk-0000.wav": {"text": "안녕"}},
                fields=(("language", ""),),
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertIsNone(body["language"])
        self.assertIsNone(body["chunks"][0]["language"])

    def test_parallel_chunk_requests_overlap_within_the_limit(self) -> None:
        asr = OverlapAsr()
        response, used, _ = self._post_pcm(
            48000,
            asr=asr,
            chunk_seconds=1,
            overlap_seconds=0,
            max_concurrent=2,
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(len(response.json()["chunks"]), 3)
        self.assertGreaterEqual(used.max_active, 2)
        self.assertLessEqual(used.max_active, 2)
        self.assertEqual(used.calls, 3)

    def test_runtime_sources_do_not_reference_fa(self) -> None:
        root = _REPO_ROOT
        paths = [root / "asr-api" / "app", root / "docker-compose.yml", root / ".env.example"]
        paths.extend((root / "scripts").glob("*.py"))
        paths.append(root / "README.md")
        forbidden = (
            "FA_BASE_URL",
            "FA_MAX_AUDIO_SECONDS",
            "ForcedAlignerClient",
            "fa_client",
            "fa_configured",
            "fa_reachable",
            "host.docker.internal",
        )
        offenders: list[str] = []
        allowed_suffixes = {".py", ".yml", ".yaml", ".md", ".example", ""}
        for path in paths:
            files = path.rglob("*") if path.is_dir() else [path]
            for file_path in files:
                if not file_path.is_file() or file_path.suffix not in allowed_suffixes:
                    continue
                text = file_path.read_text(encoding="utf-8")
                for token in forbidden:
                    if token in text:
                        offenders.append(f"{file_path}:{token}")
        self.assertEqual(offenders, [])
        readme = (root / "README.md").read_text(encoding="utf-8")
        self.assertIn("to FA", readme)

    def _post_pcm(
        self,
        num_samples: int,
        *,
        texts: dict[str, object] | None = None,
        asr: FakeAsr | None = None,
        chunk_seconds: int | None = None,
        overlap_seconds: int | None = None,
        max_concurrent: int | None = None,
        fields: tuple[tuple[str, str], ...] = (),
    ):
        wav_path = Path(_TEST_ROOT) / f"pcm-{num_samples}-{chunk_seconds}-{overlap_seconds}.wav"
        original = write_pcm(wav_path, num_samples)
        used = asr or FakeAsr(texts)
        patches = []
        if chunk_seconds is not None:
            patches.append(patch("app.pipeline.CHUNK_SECONDS", chunk_seconds))
        if overlap_seconds is not None:
            patches.append(patch("app.pipeline.CHUNK_OVERLAP_SECONDS", overlap_seconds))
        if max_concurrent is not None:
            patches.append(patch("app.pipeline.MAX_CONCURRENT_CHUNKS", max_concurrent))
        for item in patches:
            item.start()
        try:
            with TestClient(api.app) as client:
                _load_model(client)
                api.pipeline = TranscriptionPipeline(used)
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_form(
                        wav_path.read_bytes(),
                        wav_path.name,
                        ("response_format", "verbose_json"),
                        ("include_chunks", "true"),
                        *fields,
                    ),
                )
        finally:
            for item in reversed(patches):
                item.stop()
        return response, used, original


def _fail_convert(src: Path, dst: Path) -> None:
    del src, dst
    raise AssertionError("convert_to_wav should not run")


if __name__ == "__main__":
    unittest.main()
