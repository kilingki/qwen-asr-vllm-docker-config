import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

_TEST_ROOT = tempfile.mkdtemp(prefix="asr-facade-")
os.environ["STORAGE_DIR"] = _TEST_ROOT
os.environ["FA_BASE_URL"] = ""
os.environ["DEFAULT_LANGUAGE"] = "ko"
os.environ["CHUNK_SECONDS"] = "120"
os.environ["CHUNK_OVERLAP_SECONDS"] = "2"

import httpx
from fastapi.testclient import TestClient

from app import main as api
from app.audio import AudioChunk
from app.fa_client import (
    HEALTH_PROBE_TIMEOUT_SECONDS,
    ForcedAlignerClient,
    to_fa_language,
)
from app.pipeline import TranscriptionPipeline, _assign_aligned_words


class FakeAsr:
    def __init__(self, texts: dict[str, str]) -> None:
        self.texts = texts
        self.languages: list[str | None] = []

    async def transcribe(
        self,
        audio_path: Path,
        model: str,
        language: str | None,
        prompt: str | None,
        timestamp_granularities: list[str],
    ) -> dict:
        del model, prompt, timestamp_granularities
        self.languages.append(language)
        return {"text": self.texts[audio_path.name], "language": "ko"}

    async def list_models(self) -> dict:
        return {"data": []}

    async def close(self) -> None:
        return None


class FakeFa:
    def __init__(
        self,
        words_for_text: dict[str, list[dict]],
        fail_texts: set[str] | None = None,
    ) -> None:
        self.words_for_text = words_for_text
        self.fail_texts = fail_texts or set()
        self.calls: list[dict] = []

    async def align(self, audio_path: Path, text: str, language: str) -> list[dict]:
        self.calls.append(
            {"text": text, "language": language, "name": audio_path.name}
        )
        if text in self.fail_texts:
            raise RuntimeError("align failed")
        return list(self.words_for_text[text])

    async def reachable(self) -> bool:
        return False

    async def close(self) -> None:
        return None


def _chunk(index: int, name: str, start: float, end: float) -> AudioChunk:
    return AudioChunk(index=index, path=Path(name), start=start, end=end)


def _transcription_form(*fields: tuple[str, str]) -> list[tuple[str, tuple]]:
    form: list[tuple[str, tuple]] = [
        ("file", ("sample.wav", b"audio", "audio/wav")),
    ]
    for name, value in fields:
        form.append((name, (None, value)))
    return form


class FacadeApiTest(unittest.TestCase):
    def tearDown(self) -> None:
        api.FA_BASE_URL = ""

    def test_health_without_fa_stays_asr_ok(self) -> None:
        with TestClient(api.app) as client:
            api.asr_client = FakeAsr({})
            body = client.get("/health").json()

        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["backend_reachable"])
        self.assertFalse(body["fa_configured"])
        self.assertFalse(body["fa_reachable"])

    def test_health_reports_unreachable_fa_without_changing_asr_status(self) -> None:
        with TestClient(api.app) as client:
            api.FA_BASE_URL = "http://host.docker.internal:8090"
            api.asr_client = FakeAsr({})
            api.fa_client = FakeFa({})
            body = client.get("/health").json()

        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["backend_reachable"])
        self.assertTrue(body["fa_configured"])
        self.assertFalse(body["fa_reachable"])

    def test_segment_request_does_not_call_fa(self) -> None:
        chunks = [_chunk(0, "chunk-0.wav", 0.0, 120.0)]
        asr = FakeAsr({"chunk-0.wav": "language Korean <asr_text> 첫 문장"})
        fa = FakeFa({})
        with self._audio(chunks, duration=120.0):
            with TestClient(api.app) as client:
                api.pipeline = TranscriptionPipeline(asr, fa)
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_transcription_form(
                        ("response_format", "verbose_json"),
                        ("language", "ko"),
                        ("timestamp_granularities[]", "segment"),
                    ),
                )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertNotIn("words", body)
        self.assertEqual(body["text"], "첫 문장")
        self.assertEqual(body["segments"][0]["words"], [])
        self.assertEqual(body["segments"][0]["start"], 0.0)
        self.assertEqual(body["segments"][0]["end"], 120.0)
        self.assertEqual(fa.calls, [])
        self.assertEqual(asr.languages, ["ko"])

    def test_word_without_fa_url_is_503_before_conversion(self) -> None:
        converted = {"called": False}

        def convert_to_wav(src: Path, dst: Path) -> None:
            del src, dst
            converted["called"] = True

        with patch("app.main.convert_to_wav", convert_to_wav):
            with TestClient(api.app) as client:
                api.fa_client = None
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_transcription_form(
                        ("response_format", "verbose_json"),
                        ("timestamp_granularities[]", "word"),
                    ),
                )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.json()["detail"], "Forced aligner is not configured")
        self.assertFalse(converted["called"])

    def test_word_request_aligns_each_chunk_on_the_original_timeline(self) -> None:
        chunks = [
            _chunk(0, "chunk-0.wav", 0.0, 120.0),
            _chunk(1, "chunk-1.wav", 118.0, 200.0),
        ]
        asr = FakeAsr(
            {
                "chunk-0.wav": "language Korean <asr_text> 첫 문장",
                "chunk-1.wav": "language Korean <asr_text> 둘째 문장",
            }
        )
        fa = FakeFa(
            {
                "첫 문장": [
                    {"word": "language", "start": 1.0, "end": 1.2},
                    {"word": "첫", "start": 1.2, "end": 1.6},
                ],
                "둘째 문장": [
                    {"word": "겹침", "start": 0.1, "end": 0.5},
                    {"word": "걸침", "start": 1.5, "end": 3.0},
                    {"word": "둘째", "start": 4.0, "end": 5.0},
                ],
            }
        )
        with self._audio(chunks, duration=200.0):
            with TestClient(api.app) as client:
                api.pipeline = TranscriptionPipeline(asr, fa)
                api.fa_client = fa
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_transcription_form(
                        ("response_format", "verbose_json"),
                        ("language", "ko"),
                        ("timestamp_granularities[]", "segment"),
                        ("timestamp_granularities[]", "word"),
                    ),
                )

        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(len(fa.calls), 2)
        self.assertEqual([call["language"] for call in fa.calls], ["Korean", "Korean"])
        sent_text = " ".join(call["text"] for call in fa.calls)
        self.assertNotIn("language", sent_text)
        self.assertNotIn("asr_text", sent_text)
        self.assertEqual(body["text"], "첫 문장 둘째 문장")
        self.assertEqual(body["segments"][0]["text"], "첫 문장")
        self.assertEqual(body["segments"][1]["text"], "둘째 문장")
        words = body["words"]
        self.assertEqual([word["word"] for word in words], ["language", "첫", "걸침", "둘째"])
        self.assertTrue(all(word["start"] >= 118.0 for word in words[2:]))
        crossing = next(word for word in words if word["word"] == "걸침")
        self.assertEqual(crossing["start"], 120.0)
        self.assertEqual(crossing["end"], 121.0)
        self.assertEqual(asr.languages, ["ko", "ko"])

    def test_one_fa_failure_fails_the_whole_request(self) -> None:
        chunks = [
            _chunk(0, "chunk-0.wav", 0.0, 10.0),
            _chunk(1, "chunk-1.wav", 8.0, 16.0),
        ]
        asr = FakeAsr(
            {
                "chunk-0.wav": "첫 문장",
                "chunk-1.wav": "둘째 문장",
            }
        )
        fa = FakeFa(
            {"첫 문장": [{"word": "첫", "start": 0.2, "end": 0.4}]},
            fail_texts={"둘째 문장"},
        )
        with self._audio(chunks, duration=16.0):
            with TestClient(api.app) as client:
                api.pipeline = TranscriptionPipeline(asr, fa)
                api.fa_client = fa
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_transcription_form(
                        ("response_format", "verbose_json"),
                        ("timestamp_granularities[]", "word"),
                    ),
                )

        self.assertEqual(response.status_code, 500)
        self.assertIn("align failed", response.json()["detail"])

    def test_chunk_longer_than_fa_limit_is_400(self) -> None:
        chunks = [_chunk(0, "chunk-0.wav", 0.0, 181.0)]
        asr = FakeAsr({"chunk-0.wav": "긴 문장"})
        fa = FakeFa({"긴 문장": [{"word": "긴", "start": 0.1, "end": 0.2}]})
        with self._audio(chunks, duration=181.0):
            with TestClient(api.app) as client:
                api.pipeline = TranscriptionPipeline(asr, fa)
                api.fa_client = fa
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_transcription_form(("timestamp_granularities[]", "word")),
                )

        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("CHUNK_SECONDS", response.json()["detail"])
        self.assertEqual(fa.calls, [])

    def test_empty_cleaned_text_skips_align(self) -> None:
        chunks = [_chunk(0, "chunk-0.wav", 0.0, 5.0)]
        asr = FakeAsr({"chunk-0.wav": "language Korean <asr_text>"})
        fa = FakeFa({})
        with self._audio(chunks, duration=5.0):
            with TestClient(api.app) as client:
                api.pipeline = TranscriptionPipeline(asr, fa)
                api.fa_client = fa
                response = client.post(
                    "/v1/audio/transcriptions",
                    files=_transcription_form(
                        ("response_format", "verbose_json"),
                        ("timestamp_granularities[]", "word"),
                    ),
                )

        self.assertEqual(response.status_code, 200, response.text)
        self.assertNotIn("words", response.json())
        self.assertEqual(fa.calls, [])

    def test_words_are_split_across_multiple_segments_without_rebuilding_text(self) -> None:
        segments = [
            {"id": 0, "start": 0.0, "end": 1.0, "text": "가 나.", "words": []},
            {"id": 1, "start": 2.0, "end": 4.0, "text": "다.", "words": []},
        ]
        _assign_aligned_words(
            segments,
            [
                {"word": "가", "start": 0.2, "end": 0.4},
                {"word": "나", "start": 0.5, "end": 0.8},
                {"word": "다", "start": 2.2, "end": 2.6},
            ],
        )
        self.assertEqual([word["word"] for word in segments[0]["words"]], ["가", "나"])
        self.assertEqual(segments[0]["text"], "가 나.")
        self.assertEqual(segments[0]["start"], 0.2)
        self.assertEqual(segments[0]["end"], 0.8)
        self.assertEqual([word["word"] for word in segments[1]["words"]], ["다"])
        self.assertEqual(segments[1]["text"], "다.")

    def _audio(self, chunks: list[AudioChunk], duration: float):
        return _patched_audio(chunks, duration)


class _PatchedAudio:
    def __init__(self, chunks: list[AudioChunk], duration: float) -> None:
        self.chunks = chunks
        self.duration = duration
        self._patches = []

    def __enter__(self):
        def probe_duration(path: Path) -> float:
            del path
            return self.duration

        def create_chunks(source_wav, chunks_dir, chunk_seconds, overlap_seconds):
            del source_wav, chunks_dir, chunk_seconds, overlap_seconds
            return self.chunks

        def convert_to_wav(src: Path, dst: Path) -> None:
            del src
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


def _patched_audio(chunks: list[AudioChunk], duration: float) -> _PatchedAudio:
    return _PatchedAudio(chunks, duration)


class FaClientTest(unittest.TestCase):
    def test_language_aliases(self) -> None:
        self.assertEqual(to_fa_language("ko"), "Korean")
        self.assertEqual(to_fa_language("JP"), "Japanese")
        self.assertEqual(to_fa_language("zh-CN"), "Chinese")
        self.assertEqual(to_fa_language("thai"), "Thai")
        self.assertEqual(to_fa_language("Korean"), "Korean")

    def test_align_maps_language_and_drops_invalid_items(self) -> None:
        seen: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            seen["url"] = str(request.url)
            seen["body"] = request.content
            return httpx.Response(
                200,
                json={
                    "items": [
                        {"text": "안녕", "start_time": 0.12, "end_time": 0.48},
                        {"text": "", "start_time": 0.5, "end_time": 0.6},
                        {"text": "버림", "start_time": "x", "end_time": 0.7},
                        "not-a-dict",
                    ]
                },
            )

        words = asyncio.run(self._align(handler, "ko"))
        self.assertEqual(seen["url"], "http://fa.example/align")
        body = seen["body"]
        assert isinstance(body, bytes)
        self.assertIn(b"Korean", body)
        self.assertIn("안녕하세요".encode(), body)
        self.assertEqual(words, [{"word": "안녕", "start": 0.12, "end": 0.48}])

    def test_align_raises_on_http_error_and_bad_payload(self) -> None:
        with self.assertRaises(RuntimeError):
            asyncio.run(self._align(lambda request: httpx.Response(500, text="no"), "Korean"))
        with self.assertRaises(RuntimeError):
            asyncio.run(
                self._align(lambda request: httpx.Response(200, json=["nope"]), "Korean")
            )
        with self.assertRaises(RuntimeError):
            asyncio.run(
                self._align(
                    lambda request: httpx.Response(200, json={"items": {}}),
                    "Korean",
                )
            )

    def test_reachable_uses_a_short_timeout(self) -> None:
        class RecordingClient:
            def __init__(self) -> None:
                self.timeout = None

            async def get(self, url: str, timeout: float | None = None) -> httpx.Response:
                del url
                self.timeout = timeout
                raise httpx.ConnectError("down")

            async def aclose(self) -> None:
                return None

        async def check() -> tuple[bool, float | None]:
            client = ForcedAlignerClient("http://fa.example", timeout_seconds=600)
            recording = RecordingClient()
            await client.close()
            client.client = recording
            reachable = await client.reachable()
            return reachable, recording.timeout

        reachable, timeout = asyncio.run(check())
        self.assertFalse(reachable)
        self.assertEqual(timeout, HEALTH_PROBE_TIMEOUT_SECONDS)

    async def _align_async(self, handler, language: str) -> list[dict]:
        with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
            handle.write(b"RIFF")
            handle.flush()
            client = ForcedAlignerClient("http://fa.example/", timeout_seconds=5)
            await client.client.aclose()
            client.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            try:
                return await client.align(Path(handle.name), "안녕하세요", language)
            finally:
                await client.close()

    def _align(self, handler, language: str) -> list[dict]:
        return self._align_async(handler, language)


if __name__ == "__main__":
    unittest.main()
