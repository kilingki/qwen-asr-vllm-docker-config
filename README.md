# qwen-asr-vllm-docker-config

This project provides a practical single-container Docker deployment for Qwen3-ASR with an OpenAI-compatible facade API.

The runtime uses one Docker Compose service:

- `asr-api`: FastAPI facade for upload handling, audio preprocessing, chunking, result merging, and the internal `qwen-asr-serve` backend

## Architecture

```text
[client]
   |
   v
[asr-api :8080]
   |
   +--> [qwen-asr-serve 127.0.0.1:18000] Qwen3-ASR-1.7B
```

Why this layout:

- run only one container while keeping the Qwen backend private to that container
- expose only one stable public API
- normalize responses into an OpenAI-style transcription API
- handle long-audio chunking in the facade

## Project Structure

- `docker-compose.yml`: runtime definition for the single ASR API container
- `.env.example`: environment variables for ports, model paths, and chunking
- `asr-api/`: FastAPI facade image and app code
- `scripts/test_qwen_asr_youtube.py`: test script that downloads YouTube audio and sends it to the facade API
- `data/`: runtime scratch storage for uploaded and converted audio

## Requirements

- NVIDIA GPU
- NVIDIA Container Toolkit
- Docker / Docker Compose
- Hugging Face access for model download

## Model Download

Prepare the local model directory before starting the stack:

```bash
mkdir -p ../models/stt/hf

hf download Qwen/Qwen3-ASR-1.7B \
  --local-dir ../models/stt/hf/Qwen3-ASR-1.7B
```

If you already downloaded the models to the following paths, no extra download step is needed:

- `../models/stt/hf/Qwen3-ASR-1.7B`

## Quick Start

1. Copy the example environment file.

```bash
cp .env.example .env
```

2. Review the model directory and GPU memory setting in `.env`.

```env
MODEL_HOST_DIR=../models/stt/hf
ASR_MODEL_PATH=/models/Qwen3-ASR-1.7B
ASR_GPU_MEMORY_UTILIZATION=0.72
```

The GPU device is fixed to `0` in `docker-compose.yml` (`NVIDIA_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES`).

3. Build and start the ASR container.

```bash
docker compose up --build -d
```

4. Check the public facade API.

```bash
curl http://localhost:8080/health
```

Expected response:

```json
{
  "status": "ok",
  "backend_reachable": true
}
```

The compose healthcheck requires `backend_reachable`.

## API Example

Basic transcription:

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr"
```

Verbose response with segment timestamps:

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment"
```

Chunk ranges with `verbose_json`. `include_chunks` is an extension of this API, not an OpenAI field. The upload must already be 16 kHz, mono, signed 16-bit PCM WAV. The server checks the WAV header and does not resample that request.

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@sample.pcm.wav" \
  -F "model=qwen3-asr" \
  -F "response_format=verbose_json" \
  -F "include_chunks=true"
```

## API Behavior

The facade API performs the following steps:

1. store the uploaded audio to a temporary workspace
2. on the default path, convert it into `16kHz mono wav` via `ffmpeg`
3. split long audio into overlapping chunks
4. call the internal Qwen server for chunks, with bounded concurrency
5. merge text and segment offsets back into a single timeline
6. return an OpenAI-style response

`GET /v1/models` returns the model list from the internal Qwen server.

`language` defaults to `DEFAULT_LANGUAGE` (`ko` unless changed). `prompt` is forwarded to the internal Qwen server. `temperature` is accepted and ignored.

`timestamp_granularities` accepts `segment` or may be omitted. Both use the ASR-only path. `verbose_json` segments are chunk-based approximate times, not precise speech or word times. Each segment includes `"words": []`. The top-level body has no `words` field. A request that includes `word`, including `segment,word`, returns HTTP 400 before conversion or inference. This server does not provide word timestamps.

`include_chunks` defaults to false when omitted. False values are an empty string, `0`, `false`, `no`, and `off`. Those requests keep the ffmpeg conversion path and the existing `json`, `text`, `verbose_json`, `srt`, and `vtt` bodies. `json` stays `{text}` and does not gain `audio` or `chunks`. True values are `1`, `true`, `yes`, and `on`. Any other value returns HTTP 400.

`include_chunks=true` is allowed only with `response_format=verbose_json`. Any other format returns HTTP 400. The file must be 16 kHz, mono, signed 16-bit PCM WAV. The server reads the WAV header, not the extension or MIME type. A different format returns HTTP 400 and is not resampled or downmixed. Matching input is sliced by frame count with `CHUNK_SECONDS` and `CHUNK_OVERLAP_SECONDS`. Those slices are the audio sent to ASR. The response adds:

```json
{
  "audio": {"sample_rate": 16000, "channels": 1, "num_samples": 1920000},
  "chunks": [
    {"index": 0, "start_sample": 0, "end_sample": 1920000, "text": "첫 청크 전사문", "language": "ko"}
  ]
}
```

That example is one 120-second chunk, which fits the default `CHUNK_SECONDS`. Longer audio returns one object per slice. `start_sample` and `end_sample` are integer indexes into that PCM, half-open as `[start_sample, end_sample)`. `chunks` are ordered by index. A silent or fully filtered chunk stays in the list with `text` set to `""`. `chunks[].text` is the cleaned text of that chunk before cross-chunk merge and overlap removal. `chunks[].language` is null when neither the backend nor the request supplies a language. `text` and `segments` are still the merged ASR result. The response does not include paths, internal URLs, or base64 chunk audio.

Cleanup can drop repeated non-speech tails and other ASR garbage. That can also remove real speech, so an empty chunk text does not prove the audio was silent.

## Environment Variables

The main settings are documented in `.env.example`.

- `ASR_API_PORT`: host port for the public facade API
- `QWEN_ASR_INTERNAL_HOST`: container-local host for the internal Qwen server
- `QWEN_ASR_INTERNAL_PORT`: container-local port for the internal Qwen server
- `MODEL_HOST_DIR`: local directory mounted into the ASR container
- `ASR_MODEL_PATH`: container path to `Qwen3-ASR-1.7B`
- `ASR_BASE_URL`: internal base URL used by the facade
- `ASR_MODEL`: served model name exposed by the Qwen server
- `ASR_GPU_MEMORY_UTILIZATION`: GPU memory fraction passed to `qwen-asr-serve`
- `ASR_MAX_MODEL_LEN`: `--max-model-len` passed to `qwen-asr-serve`
- `ASR_GENERATION_CONFIG`: `--generation-config` passed to `qwen-asr-serve`
- `DEFAULT_LANGUAGE`: default transcription language used by the API and test script
- `CHUNK_SECONDS`: chunk size for long audio
- `CHUNK_OVERLAP_SECONDS`: overlap between adjacent chunks
- `MAX_CONCURRENT_CHUNKS`: maximum number of audio chunks transcribed at the same time
- `REQUEST_TIMEOUT_SECONDS`: facade timeout for each internal ASR request
- `LOG_LEVEL`: uvicorn log level

Compose and `.env.example` set `MAX_CONCURRENT_CHUNKS=2`. If the variable is unset and the app is started outside Compose, the code fallback is `4`.

For throughput tuning, start with `MAX_CONCURRENT_CHUNKS=2`. If vLLM logs still show
low GPU memory use and only one running request, increase it gradually. If latency
or memory pressure gets worse, reduce it back to `1`.

## Test Script

The repository includes a test script that downloads audio from YouTube and sends the full file to the public facade API. The host needs `yt-dlp` and `curl`.

```bash
python3 scripts/test_qwen_asr_youtube.py
```

Set the YouTube URL directly at the top of `scripts/test_qwen_asr_youtube.py`, and adjust these values in `.env` if needed:

- `STT_BASE_URL`
- `STT_MODEL`
- `STT_RESPONSE_FORMAT`
- `STT_TIMESTAMP_GRANULARITIES`
- `STT_HEALTH_RETRIES`
- `STT_HEALTH_BACKOFF_SEC`
- `STT_REQUEST_TIMEOUT_SECONDS`
- `STT_OUTPUT_DIR`

By default the script requests `verbose_json` with segment timestamps. Generated outputs are saved to `scripts/outputs/` by default, and that directory is excluded from git tracking.

Generated output files:

- `<audio>.verbose_json.json`: full JSON response from the facade API, including `text`, `language`, `duration`, and timestamped `segments`.
- `<audio>.segments.txt`: one line per segment, formatted as `[start - end] text` for quick timestamp review.
- `<audio>.clean.txt`: cleaned transcript text without timestamps, intended for reading or downstream text processing.
- `<audio>.txt`, `<audio>.srt`, or `<audio>.vtt`: plain text or subtitle output when `STT_RESPONSE_FORMAT` is set to `text`, `srt`, or `vtt`.

## External alignment

This container does not call a forced aligner. An external service can send the same audio to FA after ASR:

1. Normalize the source to 16 kHz, mono, signed 16-bit PCM WAV once, and keep that file.
2. Send that WAV to ASR with `include_chunks=true` and `response_format=verbose_json`.
3. Cut the returned sample ranges from the same WAV.
4. Send each range and its `text` to FA. Skip chunks whose `text` is empty.
5. Add `start_sample / sample_rate` to the FA times.
6. Remove overlap duplicates after FA returns. Do not dedupe chunk text before that.

The ASR server does not check the FA input limit. Match `CHUNK_SECONDS` to what FA accepts in the deployment configuration. Segment timestamps in the ASR response remain approximate chunk times.

## Notes

- Only `ASR_API_PORT` is published to the host. The internal Qwen server defaults to `127.0.0.1:18000` inside the container, so it does not conflict with other host services on port `8000`.
- GPU selection is device `0` only, set in `docker-compose.yml`.
