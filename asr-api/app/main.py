import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse

from .asr_client import QwenAsrClient
from .audio import convert_to_wav, save_upload
from .config import (
    ASR_BASE_URL,
    ASR_MODEL,
    DEFAULT_LANGUAGE,
    REQUEST_TIMEOUT_SECONDS,
    SUPPORTED_RESPONSE_FORMATS,
    SUPPORTED_TIMESTAMP_GRANULARITIES,
    TMP_DIR,
)
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
    if asr_client is not None:
        await asr_client.close()


@app.get("/health")
async def health() -> dict[str, Any]:
    assert asr_client is not None
    backend_reachable = False
    try:
        await asr_client.list_models()
        backend_reachable = True
    except Exception:
        backend_reachable = False

    return {
        "status": "ok",
        "backend_reachable": backend_reachable,
    }


@app.get("/v1/models")
async def list_models() -> JSONResponse:
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

    assert pipeline is not None

    filename = Path(file.filename or "upload.bin").name
    suffix = Path(filename).suffix or ".bin"
    effective_language = language or DEFAULT_LANGUAGE or None

    try:
        with tempfile.TemporaryDirectory(dir=TMP_DIR) as tmp_dir:
            tmp_root = Path(tmp_dir)
            upload_path = tmp_root / f"upload{suffix}"
            wav_path = tmp_root / "input.wav"

            await save_upload(file, upload_path)
            convert_to_wav(upload_path, wav_path)

            result = await pipeline.transcribe(
                wav_path=wav_path,
                model=model,
                language=effective_language,
                prompt=prompt,
                timestamp_granularities=timestamp_granularities,
            )

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
