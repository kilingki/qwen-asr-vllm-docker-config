import asyncio
import shutil
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .asr_client import QwenAsrClient
from .audio import PcmFormatError, convert_to_wav, inspect_pcm_wav, save_upload
from .config import (
    ASR_BASE_URL,
    ASR_MODEL,
    DEFAULT_LANGUAGE,
    REQUEST_TIMEOUT_SECONDS,
    SUPPORTED_RESPONSE_FORMATS,
    SUPPORTED_TIMESTAMP_GRANULARITIES,
    TMP_DIR,
)
from .lifecycle import ControlError, Lifecycle
from .formatter import (
    to_json_response,
    to_srt,
    to_text_response,
    to_verbose_json_response,
    to_vtt,
)
from .pipeline import TranscriptionPipeline

app = FastAPI(
    title="Qwen3-ASR OpenAI-Compatible Facade",
    version="0.1.0",
)

asr_client: QwenAsrClient | None = None
pipeline: TranscriptionPipeline | None = None
lifecycle = Lifecycle()

WORD_TIMESTAMPS_DETAIL = "This server does not provide word timestamps."


@app.on_event("startup")
async def on_startup() -> None:
    global asr_client, pipeline
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    asr_client = QwenAsrClient(
        base_url=ASR_BASE_URL,
        timeout_seconds=REQUEST_TIMEOUT_SECONDS,
    )
    pipeline = TranscriptionPipeline(asr_client=asr_client)


@app.on_event("shutdown")
async def on_shutdown() -> None:
    await lifecycle.shutdown()
    if asr_client is not None:
        await asr_client.close()


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend_reachable": lifecycle.worker_alive(),
    }


@app.get("/control/status")
async def control_status() -> JSONResponse:
    try:
        body = await lifecycle.status()
    except ControlError as exc:
        return _control_error(exc)
    return JSONResponse(body)


@app.post("/control/load")
async def control_load(request: Request) -> JSONResponse:
    try:
        await _require_empty_body(request)
        body = await lifecycle.load()
    except ControlError as exc:
        return _control_error(exc)
    return JSONResponse(body)


@app.post("/control/unload")
async def control_unload(request: Request) -> JSONResponse:
    try:
        await _require_empty_body(request)
        body = await lifecycle.unload()
    except ControlError as exc:
        return _control_error(exc)
    return JSONResponse(body)


@app.get("/v1/models")
async def list_models() -> JSONResponse:
    if not lifecycle.worker_alive():
        raise HTTPException(status_code=503, detail="model is not ready")
    assert asr_client is not None
    payload = await asr_client.list_models()
    return JSONResponse(payload)


@app.post("/v1/audio/transcriptions", response_model=None)
async def create_transcription(
    request: Request,
    file: UploadFile = File(...),
    model: str = Form(default=ASR_MODEL),
    language: str | None = Form(default=None),
    prompt: str | None = Form(default=None),
    temperature: float = Form(default=0.0),
    response_format: str = Form(default="json"),
) -> JSONResponse | PlainTextResponse:
    del temperature
    try:
        if response_format not in SUPPORTED_RESPONSE_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported response_format: {response_format}",
            )

        form = await request.form()
        timestamp_granularities = form.getlist("timestamp_granularities[]")
        if not timestamp_granularities:
            timestamp_granularities = form.getlist("timestamp_granularities")
        if not timestamp_granularities:
            timestamp_granularities = ["segment"]

        if "word" in timestamp_granularities:
            raise HTTPException(status_code=400, detail=WORD_TIMESTAMPS_DETAIL)

        unsupported_granularities = sorted(
            set(timestamp_granularities) - SUPPORTED_TIMESTAMP_GRANULARITIES
        )
        if unsupported_granularities:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Unsupported timestamp_granularities: "
                    + ", ".join(unsupported_granularities)
                ),
            )

        include_chunks = _form_flag(form.get("include_chunks"))
        if include_chunks and response_format != "verbose_json":
            raise HTTPException(
                status_code=400,
                detail="include_chunks=true requires response_format=verbose_json",
            )

        assert pipeline is not None

        filename = Path(file.filename or "upload.bin").name
        suffix = Path(filename).suffix or ".bin"
        effective_language = language or DEFAULT_LANGUAGE or None

        tmp_root = Path(tempfile.mkdtemp(dir=TMP_DIR))
        work: asyncio.Task[Any] | None = None
        try:
            upload_path = tmp_root / f"upload{suffix}"
            await save_upload(file, upload_path)

            pcm = None
            if include_chunks:
                try:
                    pcm = inspect_pcm_wav(upload_path)
                except PcmFormatError as exc:
                    raise HTTPException(status_code=400, detail=str(exc)) from exc
                wav_path = upload_path
            else:
                wav_path = tmp_root / "input.wav"

            if not await lifecycle.try_admit():
                raise HTTPException(status_code=503, detail="model is not ready")

            async def _run_transcription() -> Any:
                try:
                    if pcm is None:
                        await asyncio.to_thread(convert_to_wav, upload_path, wav_path)
                    return await pipeline.transcribe(
                        wav_path=wav_path,
                        model=model,
                        language=effective_language,
                        prompt=prompt,
                        timestamp_granularities=timestamp_granularities,
                        pcm=pcm,
                    )
                finally:
                    await lifecycle.release()
                    shutil.rmtree(tmp_root, ignore_errors=True)

            work = asyncio.create_task(_run_transcription())
            result = await asyncio.shield(work)
        finally:
            if work is None:
                shutil.rmtree(tmp_root, ignore_errors=True)

        if response_format == "json":
            return JSONResponse(to_json_response(result))
        if response_format == "verbose_json":
            return JSONResponse(to_verbose_json_response(result))
        if response_format == "text":
            return PlainTextResponse(to_text_response(result), media_type="text/plain")
        if response_format == "srt":
            return PlainTextResponse(to_srt(result), media_type="text/plain")
        if response_format == "vtt":
            return PlainTextResponse(to_vtt(result), media_type="text/vtt")

        raise HTTPException(status_code=400, detail="Invalid response_format")
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        await file.close()


async def _require_empty_body(request: Request) -> None:
    raw = await request.body()
    if raw.strip() in {b"", b"{}"}:
        return
    raise ControlError(400, "BAD_REQUEST", "request body must be empty or {}")


def _control_error(exc: ControlError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


def _form_flag(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, str):
        raise HTTPException(status_code=400, detail="Invalid include_chunks value")
    text = value.strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"", "0", "false", "no", "off"}:
        return False
    raise HTTPException(
        status_code=400,
        detail=f"Invalid include_chunks value: {value}",
    )
