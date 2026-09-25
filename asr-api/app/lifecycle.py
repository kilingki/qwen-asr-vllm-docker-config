import asyncio
import os
import signal
from typing import Any, Protocol

import httpx

from .config import (
    ASR_BASE_URL,
    ASR_GENERATION_CONFIG,
    ASR_GPU_MEMORY_UTILIZATION,
    ASR_MAX_MODEL_LEN,
    ASR_MODEL,
    ASR_MODEL_PATH,
    LOAD_TIMEOUT_SECONDS,
    QWEN_ASR_INTERNAL_HOST,
    QWEN_ASR_INTERNAL_PORT,
    WORKER_STOP_TIMEOUT_SECONDS,
)


class ControlError(Exception):
    def __init__(self, status_code: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message


class WorkerFailure(Exception):
    def __init__(self, message: str, residency: str) -> None:
        super().__init__(message)
        self.residency = residency
        self.message = message


class ModelWorker(Protocol):
    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    def is_alive(self) -> bool: ...


def _group_alive(pid: int) -> bool | None:
    try:
        os.killpg(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None
    return True


class QwenServeWorker:
    def __init__(self) -> None:
        self._proc: asyncio.subprocess.Process | None = None
        self._pid: int | None = None

    def is_alive(self) -> bool:
        if self._proc is not None and self._proc.returncode is None:
            return True
        if self._pid is None:
            return False
        alive = _group_alive(self._pid)
        return alive is True

    async def start(self) -> None:
        if self.is_alive():
            await self._wait_until_models()
            return
        self._proc = await asyncio.create_subprocess_exec(
            "qwen-asr-serve",
            ASR_MODEL_PATH,
            "--host",
            QWEN_ASR_INTERNAL_HOST,
            "--port",
            str(QWEN_ASR_INTERNAL_PORT),
            "--served-model-name",
            ASR_MODEL,
            "--gpu-memory-utilization",
            ASR_GPU_MEMORY_UTILIZATION,
            "--max-model-len",
            str(ASR_MAX_MODEL_LEN),
            "--generation-config",
            ASR_GENERATION_CONFIG,
            start_new_session=True,
        )
        self._pid = self._proc.pid
        try:
            await self._wait_until_models()
        except WorkerFailure:
            raise
        except Exception as exc:
            residency = "resident" if self.is_alive() else "not_resident"
            raise WorkerFailure(str(exc), residency) from exc

    async def stop(self) -> None:
        proc = self._proc
        pid = self._pid
        if proc is None and pid is None:
            return
        if not self.is_alive():
            self._proc = None
            self._pid = None
            return
        self._signal_group(pid, proc, signal.SIGTERM)
        if await self._wait_dead(WORKER_STOP_TIMEOUT_SECONDS):
            self._proc = None
            self._pid = None
            return
        self._signal_group(pid, proc, signal.SIGKILL)
        if await self._wait_dead(WORKER_STOP_TIMEOUT_SECONDS):
            self._proc = None
            self._pid = None
            return
        residency = "resident" if self.is_alive() else "unknown"
        raise WorkerFailure("failed to release model", residency)

    def _signal_group(
        self,
        pid: int | None,
        proc: asyncio.subprocess.Process | None,
        sig: int,
    ) -> None:
        if pid is not None:
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                return
            except OSError as exc:
                raise WorkerFailure(
                    f"failed to signal worker: {exc}",
                    "unknown",
                ) from exc
            return
        if proc is None:
            return
        if sig == signal.SIGKILL:
            proc.kill()
        else:
            proc.send_signal(sig)

    async def _wait_dead(self, timeout: float) -> bool:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            proc = self._proc
            if proc is not None and proc.returncode is None:
                remaining = deadline - asyncio.get_running_loop().time()
                if remaining <= 0:
                    return not self.is_alive()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=min(0.2, remaining))
                except (TimeoutError, asyncio.TimeoutError):
                    pass
            if not self.is_alive():
                return True
            if asyncio.get_running_loop().time() >= deadline:
                return False
            await asyncio.sleep(0.2)

    async def _wait_until_models(self) -> None:
        deadline = asyncio.get_running_loop().time() + LOAD_TIMEOUT_SECONDS
        url = f"{ASR_BASE_URL}/models"
        async with httpx.AsyncClient(timeout=5.0) as client:
            while True:
                if self._proc is not None and self._proc.returncode is not None:
                    raise WorkerFailure(
                        f"qwen-asr-serve exited with status {self._proc.returncode}",
                        "not_resident",
                    )
                try:
                    response = await client.get(url)
                    if response.status_code == 200:
                        return
                except httpx.HTTPError:
                    pass
                if asyncio.get_running_loop().time() >= deadline:
                    residency = "unknown" if self.is_alive() else "not_resident"
                    raise WorkerFailure("model did not become ready", residency)
                await asyncio.sleep(0.5)


class Lifecycle:
    def __init__(self, worker: ModelWorker | None = None) -> None:
        self._worker = worker if worker is not None else QwenServeWorker()
        self._lock = asyncio.Lock()
        self._state = "unloaded"
        self._residency = "not_resident"
        self._active = 0
        self._last_error: dict[str, str] | None = None
        self._operation: asyncio.Task[dict[str, Any]] | None = None
        self._status_fault: str | None = None

    def install_worker(self, worker: ModelWorker) -> None:
        self._worker = worker

    def worker_alive(self) -> bool:
        return self._worker.is_alive()

    def fail_status(self, message: str = "status is unavailable") -> None:
        self._status_fault = message

    async def status(self) -> dict[str, Any]:
        async with self._lock:
            if self._status_fault is not None:
                raise ControlError(500, "STATUS_FAILED", self._status_fault)
            return self._snapshot()

    async def load(self) -> dict[str, Any]:
        async with self._lock:
            if self._state == "ready":
                return self._snapshot()
            if self._state == "loading" and self._operation is not None:
                task = self._operation
            elif self._state == "unloading":
                raise ControlError(
                    409,
                    "LIFECYCLE_CONFLICT",
                    "unload is in progress",
                )
            elif self._state == "failed" and self._residency != "not_resident":
                raise ControlError(
                    409,
                    "LIFECYCLE_CONFLICT",
                    "load is not allowed while residency remains",
                )
            else:
                self._state = "loading"
                task = asyncio.create_task(self._run_load())
                self._operation = task
        return await self._await_operation(task)

    async def unload(self) -> dict[str, Any]:
        async with self._lock:
            if self._state == "unloading" and self._operation is not None:
                task = self._operation
            elif self._state == "loading":
                raise ControlError(
                    409,
                    "LIFECYCLE_CONFLICT",
                    "load is in progress",
                )
            elif self._active > 0:
                raise ControlError(
                    409,
                    "BUSY",
                    "runtime has active inference requests",
                )
            elif self._state == "unloaded" and self._residency == "not_resident":
                return self._snapshot()
            elif self._state == "failed" and self._residency == "not_resident":
                self._state = "unloaded"
                self._residency = "not_resident"
                self._last_error = None
                return self._snapshot()
            else:
                self._state = "unloading"
                task = asyncio.create_task(self._run_unload())
                self._operation = task
        return await self._await_operation(task)

    async def try_admit(self) -> bool:
        async with self._lock:
            if self._state != "ready":
                return False
            self._active += 1
            return True

    async def release(self) -> None:
        async with self._lock:
            if self._active > 0:
                self._active -= 1

    async def shutdown(self) -> None:
        if self.worker_alive():
            try:
                await self._worker.stop()
            except WorkerFailure:
                pass

    async def _await_operation(self, task: asyncio.Task[dict[str, Any]]) -> dict[str, Any]:
        try:
            return await asyncio.shield(task)
        except WorkerFailure as exc:
            code = getattr(exc, "control_code", "LOAD_FAILED")
            raise ControlError(500, code, exc.message) from exc

    async def _run_load(self) -> dict[str, Any]:
        task = asyncio.current_task()
        if task is not None:
            task.set_name("load")
        try:
            await self._worker.start()
        except WorkerFailure as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = exc.residency
                self._last_error = {"code": "LOAD_FAILED", "message": exc.message}
                self._operation = None
            exc.control_code = "LOAD_FAILED"  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = "unknown"
                self._last_error = {"code": "LOAD_FAILED", "message": str(exc)}
                self._operation = None
            failure = WorkerFailure(str(exc), "unknown")
            failure.control_code = "LOAD_FAILED"  # type: ignore[attr-defined]
            raise failure from exc
        async with self._lock:
            self._state = "ready"
            self._residency = "resident"
            self._last_error = None
            self._operation = None
            return self._snapshot()

    async def _run_unload(self) -> dict[str, Any]:
        task = asyncio.current_task()
        if task is not None:
            task.set_name("unload")
        try:
            await self._worker.stop()
        except WorkerFailure as exc:
            async with self._lock:
                self._state = "failed"
                self._residency = exc.residency
                self._last_error = {"code": "UNLOAD_FAILED", "message": exc.message}
                self._operation = None
            exc.control_code = "UNLOAD_FAILED"  # type: ignore[attr-defined]
            raise
        except Exception as exc:
            async with self._lock:
                alive = False
                try:
                    alive = self._worker.is_alive()
                except Exception:
                    residency = "unknown"
                else:
                    residency = "resident" if alive else "unknown"
                self._state = "failed"
                self._residency = residency
                self._last_error = {"code": "UNLOAD_FAILED", "message": str(exc)}
                self._operation = None
            failure = WorkerFailure(str(exc), residency)
            failure.control_code = "UNLOAD_FAILED"  # type: ignore[attr-defined]
            raise failure from exc
        async with self._lock:
            self._state = "unloaded"
            self._residency = "not_resident"
            self._active = 0
            self._last_error = None
            self._operation = None
            return self._snapshot()

    def _snapshot(self) -> dict[str, Any]:
        last_error = None if self._last_error is None else dict(self._last_error)
        return {
            "state": self._state,
            "residency": self._residency,
            "active_requests": self._active,
            "last_error": last_error,
        }
