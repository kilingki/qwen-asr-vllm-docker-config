import asyncio
import os
import sys
import tempfile
import threading
import unittest
import unittest.mock
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "asr-api"))
sys.path.insert(0, str(_REPO_ROOT / "tests"))

os.environ.setdefault("STORAGE_DIR", tempfile.mkdtemp(prefix="asr-control-"))
os.environ.setdefault("DEFAULT_LANGUAGE", "ko")
os.environ.setdefault("CHUNK_SECONDS", "120")
os.environ.setdefault("CHUNK_OVERLAP_SECONDS", "2")
os.environ.setdefault("MAX_CONCURRENT_CHUNKS", "2")

from api_report import install

install()

from fastapi.testclient import TestClient

from app import main as api
from app.lifecycle import ControlError, Lifecycle, WorkerFailure


class GateWorker:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release_start = asyncio.Event()
        self.stop_entered = asyncio.Event()
        self.release_stop = asyncio.Event()
        self.alive = False
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        self.started.set()
        await self.release_start.wait()
        self.alive = True

    async def stop(self) -> None:
        self.stops += 1
        self.stop_entered.set()
        await self.release_stop.wait()
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class ScriptedWorker:
    def __init__(
        self,
        *,
        start_failure: str | None = None,
        stop_failure: str | None = None,
    ) -> None:
        self.start_failure = start_failure
        self.stop_failure = stop_failure
        self.alive = False
        self.starts = 0
        self.stops = 0

    async def start(self) -> None:
        self.starts += 1
        if self.start_failure is not None:
            self.alive = self.start_failure != "not_resident"
            raise WorkerFailure("load failed", self.start_failure)
        self.alive = True

    async def stop(self) -> None:
        self.stops += 1
        if self.stop_failure is not None:
            self.alive = self.stop_failure == "resident"
            raise WorkerFailure("failed to release model", self.stop_failure)
        self.alive = False

    def is_alive(self) -> bool:
        return self.alive


class BlockingPipeline:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.both = threading.Event()
        self.active_during: list[int] = []

    async def transcribe(self, **_kwargs: object) -> dict:
        self.calls += 1
        if self.calls >= 2:
            self.both.set()
        status = await api.lifecycle.status()
        self.active_during.append(status["active_requests"])
        self.entered.set()
        await asyncio.to_thread(self.release.wait)
        status = await api.lifecycle.status()
        self.active_during.append(status["active_requests"])
        return {
            "text": "문장",
            "language": "ko",
            "duration": 1.0,
            "segments": [{"id": 0, "start": 0.0, "end": 1.0, "text": "문장", "words": []}],
        }


def _run(coro):
    return asyncio.run(coro)


class ControlLifecycleTest(unittest.TestCase):
    def test_initial_status_and_inference_unavailable(self) -> None:
        api.lifecycle = Lifecycle(ScriptedWorker())
        with TestClient(api.app) as client:
            status = client.get("/control/status")
            self.assertEqual(status.status_code, 200)
            self.assertEqual(
                status.json(),
                {
                    "state": "unloaded",
                    "residency": "not_resident",
                    "active_requests": 0,
                    "last_error": None,
                },
            )
            rejected = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"not-wav", "audio/wav")},
                data={"response_format": "json"},
            )
            self.assertEqual(rejected.status_code, 503, rejected.text)
            after = client.get("/control/status").json()
            self.assertEqual(after["active_requests"], 0)
            bad = client.post("/control/load", json={"model": "qwen3-asr"})
            self.assertEqual(bad.status_code, 400)
            self.assertEqual(bad.json()["error"]["code"], "BAD_REQUEST")

    def test_load_inference_busy_and_unload(self) -> None:
        api.lifecycle = Lifecycle(ScriptedWorker())
        pipeline = BlockingPipeline()

        def convert_to_wav(src: Path, dst: Path) -> None:
            del src
            dst.write_bytes(b"RIFF")

        with TestClient(api.app) as client:
            api.pipeline = pipeline
            loaded = client.post("/control/load", content=b"{}")
            self.assertEqual(loaded.status_code, 200, loaded.text)
            self.assertEqual(loaded.json()["state"], "ready")
            self.assertEqual(loaded.json()["residency"], "resident")

            holder: dict[str, object] = {}

            def _post() -> None:
                try:
                    with unittest.mock.patch("app.main.convert_to_wav", convert_to_wav):
                        holder["response"] = client.post(
                            "/v1/audio/transcriptions",
                            files={"file": ("sample.bin", b"audio", "audio/wav")},
                            data={"response_format": "json"},
                        )
                except Exception as exc:  # noqa: BLE001
                    holder["error"] = exc

            thread = threading.Thread(target=_post)
            thread.start()
            entered = pipeline.entered.wait(timeout=5)
            if not entered:
                thread.join(timeout=1)
                response = holder.get("response")
                detail = getattr(response, "text", None)
                self.fail(f"transcription did not start: {holder} body={detail}")
            mid = client.get("/control/status")
            self.assertEqual(mid.status_code, 200)
            self.assertEqual(mid.json()["state"], "ready")
            self.assertEqual(mid.json()["residency"], "resident")
            self.assertGreater(mid.json()["active_requests"], 0)
            busy = client.post("/control/unload")
            self.assertEqual(busy.status_code, 409)
            self.assertEqual(busy.json()["error"]["code"], "BUSY")
            pipeline.release.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive())
            response = holder["response"]
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(pipeline.active_during, [1, 1])
            done = client.get("/control/status").json()
            self.assertEqual(done["active_requests"], 0)
            unloaded = client.post("/control/unload")
            self.assertEqual(unloaded.status_code, 200, unloaded.text)
            self.assertEqual(unloaded.json()["state"], "unloaded")
            self.assertEqual(unloaded.json()["residency"], "not_resident")
            self.assertEqual(unloaded.json()["active_requests"], 0)
            again = client.post("/control/unload")
            self.assertEqual(again.status_code, 200, again.text)
            self.assertEqual(again.json()["state"], "unloaded")

    def test_ready_load_is_noop(self) -> None:
        worker = ScriptedWorker()
        api.lifecycle = Lifecycle(worker)
        with TestClient(api.app) as client:
            first = client.post("/control/load")
            second = client.post("/control/load")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(second.json()["state"], "ready")
            self.assertEqual(worker.starts, 1)

    def test_shared_load_and_conflict(self) -> None:
        async def scenario() -> None:
            import httpx
            from api_report import remember

            worker = GateWorker()
            api.lifecycle = Lifecycle(worker)
            name = "test_shared_load_and_conflict"
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                first_task = asyncio.create_task(client.post("/control/load"))
                await worker.started.wait()
                status = await client.get("/control/status")
                conflict = await client.post("/control/unload")
                second_task = asyncio.create_task(client.post("/control/load"))
                worker.release_start.set()
                first, second = await asyncio.gather(first_task, second_task)
            for response, method, path in (
                (status, "GET", "/control/status"),
                (conflict, "POST", "/control/unload"),
                (first, "POST", "/control/load"),
                (second, "POST", "/control/load"),
            ):
                remember(name, method, path, response)
            self.assertEqual(status.status_code, 200)
            self.assertEqual(status.json()["state"], "loading")
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(first.json(), second.json())
            self.assertEqual(first.json()["state"], "ready")
            self.assertEqual(worker.starts, 1)

        _run(scenario())

    def test_shared_unload_and_conflict(self) -> None:
        async def scenario() -> None:
            import httpx
            from api_report import remember

            worker = GateWorker()
            api.lifecycle = Lifecycle(worker)
            worker.release_start.set()
            pipeline = BlockingPipeline()
            api.pipeline = pipeline
            name = "test_shared_unload_and_conflict"
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                loaded = await client.post("/control/load")
                self.assertEqual(loaded.status_code, 200)
                first_task = asyncio.create_task(client.post("/control/unload"))
                await worker.stop_entered.wait()
                status = await client.get("/control/status")
                conflict = await client.post("/control/load")
                denied = await client.post(
                    "/v1/audio/transcriptions",
                    files={"file": ("sample.bin", b"audio", "audio/wav")},
                    data={"response_format": "json"},
                )
                second_task = asyncio.create_task(client.post("/control/unload"))
                worker.release_stop.set()
                first, second = await asyncio.gather(first_task, second_task)
            for response, method, path in (
                (loaded, "POST", "/control/load"),
                (status, "GET", "/control/status"),
                (conflict, "POST", "/control/load"),
                (denied, "POST", "/v1/audio/transcriptions"),
                (first, "POST", "/control/unload"),
                (second, "POST", "/control/unload"),
            ):
                remember(name, method, path, response)
            self.assertEqual(status.json()["state"], "unloading")
            self.assertEqual(conflict.status_code, 409)
            self.assertEqual(conflict.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            self.assertEqual(denied.status_code, 503)
            self.assertEqual(pipeline.calls, 0)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json(), second.json())
            self.assertEqual(first.json()["state"], "unloaded")
            self.assertEqual(first.json()["residency"], "not_resident")
            self.assertEqual(worker.stops, 1)

        _run(scenario())

    def test_disconnect_does_not_cancel_operation(self) -> None:
        async def scenario() -> None:
            import httpx

            from api_report import remember

            worker = GateWorker()
            api.lifecycle = Lifecycle(worker)
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                load_task = asyncio.create_task(client.post("/control/load"))
                await worker.started.wait()
                load_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await load_task
                loading = await client.get("/control/status")
                remember("test_disconnect_does_not_cancel_operation", "GET", "/control/status", loading)
                self.assertEqual(loading.status_code, 200)
                self.assertEqual(loading.json()["state"], "loading")
                worker.release_start.set()
                for _ in range(50):
                    ready = await client.get("/control/status")
                    remember(
                        "test_disconnect_does_not_cancel_operation",
                        "GET",
                        "/control/status",
                        ready,
                    )
                    if ready.json()["state"] == "ready":
                        break
                    await asyncio.sleep(0.01)
                self.assertEqual(ready.json()["state"], "ready")
                self.assertEqual(worker.starts, 1)

                unload_task = asyncio.create_task(client.post("/control/unload"))
                await worker.stop_entered.wait()
                unload_task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await unload_task
                unloading = await client.get("/control/status")
                remember(
                    "test_disconnect_does_not_cancel_operation",
                    "GET",
                    "/control/status",
                    unloading,
                )
                self.assertEqual(unloading.json()["state"], "unloading")
                worker.release_stop.set()
                for _ in range(50):
                    done = await client.get("/control/status")
                    remember(
                        "test_disconnect_does_not_cancel_operation",
                        "GET",
                        "/control/status",
                        done,
                    )
                    if done.json()["state"] == "unloaded":
                        break
                    await asyncio.sleep(0.01)
            self.assertEqual(done.json()["state"], "unloaded")
            self.assertEqual(done.json()["residency"], "not_resident")
            self.assertEqual(worker.stops, 1)

        _run(scenario())

    def test_rejected_input_does_not_count(self) -> None:
        api.lifecycle = Lifecycle(ScriptedWorker())
        with TestClient(api.app) as client:
            loaded = client.post("/control/load")
            self.assertEqual(loaded.status_code, 200, loaded.text)
            rejected = client.post(
                "/v1/audio/transcriptions",
                files={"file": ("sample.wav", b"audio", "audio/wav")},
                data={
                    "response_format": "verbose_json",
                    "timestamp_granularities[]": "word",
                },
            )
            status = client.get("/control/status")
        self.assertEqual(rejected.status_code, 400, rejected.text)
        self.assertEqual(status.status_code, 200)
        self.assertEqual(status.json()["active_requests"], 0)
        self.assertEqual(status.json()["state"], "ready")

    def test_two_requests_count_separately(self) -> None:
        async def scenario() -> None:
            import httpx
            from api_report import remember

            api.lifecycle = Lifecycle(ScriptedWorker())
            pipeline = BlockingPipeline()
            api.pipeline = pipeline
            name = "test_two_requests_count_separately"
            transport = httpx.ASGITransport(app=api.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
                loaded = await client.post("/control/load")
                self.assertEqual(loaded.status_code, 200)

                def convert_to_wav(src: Path, dst: Path) -> None:
                    del src
                    dst.write_bytes(b"RIFF")

                with unittest.mock.patch("app.main.convert_to_wav", convert_to_wav):
                    first_task = asyncio.create_task(
                        client.post(
                            "/v1/audio/transcriptions",
                            files={"file": ("sample.bin", b"audio", "audio/wav")},
                            data={"response_format": "json"},
                        )
                    )
                    second_task = asyncio.create_task(
                        client.post(
                            "/v1/audio/transcriptions",
                            files={"file": ("sample.bin", b"audio", "audio/wav")},
                            data={"response_format": "json"},
                        )
                    )
                    await asyncio.wait_for(asyncio.to_thread(pipeline.both.wait), timeout=5)
                    status = await client.get("/control/status")
                    pipeline.release.set()
                    first, second = await asyncio.gather(first_task, second_task)
                done = await client.get("/control/status")
            for response, method, path in (
                (loaded, "POST", "/control/load"),
                (status, "GET", "/control/status"),
                (first, "POST", "/v1/audio/transcriptions"),
                (second, "POST", "/v1/audio/transcriptions"),
                (done, "GET", "/control/status"),
            ):
                remember(name, method, path, response)
            self.assertEqual(status.json()["active_requests"], 2)
            self.assertEqual(status.json()["state"], "ready")
            self.assertEqual(first.status_code, 200)
            self.assertEqual(second.status_code, 200)
            self.assertEqual(done.json()["active_requests"], 0)

        _run(scenario())

    def test_load_failure_not_resident_can_retry(self) -> None:
        worker = ScriptedWorker(start_failure="not_resident")
        api.lifecycle = Lifecycle(worker)
        with TestClient(api.app) as client:
            failed = client.post("/control/load")
            self.assertEqual(failed.status_code, 500)
            self.assertEqual(failed.json()["error"]["code"], "LOAD_FAILED")
            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["residency"], "not_resident")
            self.assertEqual(status["last_error"]["code"], "LOAD_FAILED")
            worker.start_failure = None
            retried = client.post("/control/load")
            self.assertEqual(retried.status_code, 200, retried.text)
            self.assertEqual(retried.json()["state"], "ready")
            self.assertIsNone(retried.json()["last_error"])

    def test_load_failure_unknown_blocks_load_until_recovery(self) -> None:
        worker = ScriptedWorker(start_failure="unknown")
        api.lifecycle = Lifecycle(worker)
        with TestClient(api.app) as client:
            failed = client.post("/control/load")
            self.assertEqual(failed.status_code, 500)
            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["residency"], "unknown")
            blocked = client.post("/control/load")
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            recovered = client.post("/control/unload")
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json()["state"], "unloaded")
            self.assertEqual(recovered.json()["residency"], "not_resident")
            self.assertIsNone(recovered.json()["last_error"])

    def test_unload_failure_keeps_residency_and_recovery_clears_error(self) -> None:
        worker = ScriptedWorker(stop_failure="resident")
        api.lifecycle = Lifecycle(worker)
        with TestClient(api.app) as client:
            self.assertEqual(client.post("/control/load").status_code, 200)
            failed = client.post("/control/unload")
            self.assertEqual(failed.status_code, 500)
            self.assertEqual(failed.json()["error"]["code"], "UNLOAD_FAILED")
            status = client.get("/control/status").json()
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["residency"], "resident")
            blocked = client.post("/control/load")
            self.assertEqual(blocked.status_code, 409)
            self.assertEqual(blocked.json()["error"]["code"], "LIFECYCLE_CONFLICT")
            worker.stop_failure = None
            recovered = client.post("/control/unload")
            self.assertEqual(recovered.status_code, 200, recovered.text)
            self.assertEqual(recovered.json()["state"], "unloaded")
            self.assertEqual(recovered.json()["residency"], "not_resident")
            self.assertIsNone(recovered.json()["last_error"])

    def test_status_failure_is_http_500(self) -> None:
        api.lifecycle = Lifecycle(ScriptedWorker())
        api.lifecycle.fail_status()
        with TestClient(api.app) as client:
            status = client.get("/control/status")
            self.assertEqual(status.status_code, 500)
            self.assertEqual(status.json()["error"]["code"], "STATUS_FAILED")
            self.assertNotIn("state", status.json())

    def test_admit_and_unload_are_ordered(self) -> None:
        async def scenario() -> None:
            worker = GateWorker()
            life = Lifecycle(worker)
            worker.release_start.set()
            await life.load()
            self.assertTrue(await life.try_admit())
            busy = await self._expect_control(life.unload())
            self.assertEqual(busy.code, "BUSY")
            self.assertEqual(worker.stops, 0)
            await life.release()

            unload_task = asyncio.create_task(life.unload())
            await worker.stop_entered.wait()
            self.assertFalse(await life.try_admit())
            self.assertEqual((await life.status())["state"], "unloading")
            worker.release_stop.set()
            body = await unload_task
            self.assertEqual(body["state"], "unloaded")

        _run(scenario())

    def test_models_requires_worker(self) -> None:
        api.lifecycle = Lifecycle(ScriptedWorker())
        with TestClient(api.app) as client:
            missing = client.get("/v1/models")
            self.assertEqual(missing.status_code, 503)
            client.post("/control/load")

            class Models:
                async def list_models(self) -> dict:
                    return {"data": [{"id": "qwen3-asr"}]}

                async def close(self) -> None:
                    return None

            api.asr_client = Models()
            listed = client.get("/v1/models")
            self.assertEqual(listed.status_code, 200)
            self.assertEqual(listed.json()["data"][0]["id"], "qwen3-asr")

    async def _expect_control(self, awaitable) -> ControlError:
        try:
            await awaitable
        except ControlError as exc:
            return exc
        raise AssertionError("expected ControlError")


class PrepareScriptTest(unittest.TestCase):
    def test_prepare_script_exists_and_is_executable_text(self) -> None:
        path = _REPO_ROOT / "prepare-inferswap"
        text = path.read_text(encoding="utf-8")
        self.assertIn("docker compose up -d", text)
        self.assertIn("not_resident", text)
        self.assertIn("leaving the running container unchanged", text)


if __name__ == "__main__":
    unittest.main()
