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

2. Review the model directory and GPU settings in `.env`.

```env
MODEL_HOST_DIR=../models/stt/hf
ASR_MODEL_PATH=/models/Qwen3-ASR-1.7B
```

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

Verbose response with word timestamps:

```bash
curl -X POST "http://localhost:8080/v1/audio/transcriptions" \
  -F "file=@sample.wav" \
  -F "model=qwen3-asr" \
  -F "response_format=verbose_json" \
  -F "timestamp_granularities[]=segment" \
  -F "timestamp_granularities[]=word"
```

## API Behavior

The facade API performs the following steps:

1. store the uploaded audio to a temporary workspace
2. convert it into `16kHz mono wav` via `ffmpeg`
3. split long audio into overlapping chunks
4. call the internal Qwen server for each chunk
5. merge text and segment offsets back into a single timeline
6. return an OpenAI-style response

## Environment Variables

The main settings are documented in `.env.example`.

- `ASR_API_PORT`: host port for the public facade API
- `QWEN_ASR_INTERNAL_HOST`: container-local host for the internal Qwen server
- `QWEN_ASR_INTERNAL_PORT`: container-local port for the internal Qwen server
- `MODEL_HOST_DIR`: local directory mounted into the ASR container
- `ASR_MODEL_PATH`: container path to `Qwen3-ASR-1.7B`
- `ASR_BASE_URL`: internal base URL used by the facade
- `ASR_MODEL`: served model name exposed by the Qwen server
- `DEFAULT_LANGUAGE`: default transcription language used by the API and test script
- `CHUNK_SECONDS`: chunk size for long audio
- `CHUNK_OVERLAP_SECONDS`: overlap between adjacent chunks

## Test Script

The repository includes a test script that downloads audio from YouTube and sends the full file to the public facade API.

```bash
python3 scripts/test_qwen_asr_youtube.py
```

Set the YouTube URL directly at the top of `scripts/test_qwen_asr_youtube.py`, and adjust these values in `.env` if needed:

- `STT_BASE_URL`
- `STT_MODEL`
- `STT_RESPONSE_FORMAT`
- `STT_TIMESTAMP_GRANULARITIES`
- `STT_OUTPUT_DIR`

By default the script requests `verbose_json` with segment timestamps, and can also request word timestamps through `STT_TIMESTAMP_GRANULARITIES`. Generated outputs are saved to `scripts/outputs/` by default, and that directory is excluded from git tracking.

Generated output files:

- `<audio>.verbose_json.json`: full JSON response from the facade API, including `text`, `language`, `duration`, and timestamped `segments`.
- `<audio>.segments.txt`: one line per segment, formatted as `[start - end] text` for quick timestamp review.
- `<audio>.words.txt`: one line per word timestamp when `word` granularity is requested.
- `<audio>.clean.txt`: cleaned transcript text without timestamps, intended for reading or downstream text processing.
- `<audio>.txt`, `<audio>.srt`, or `<audio>.vtt`: plain text or subtitle output when `STT_RESPONSE_FORMAT` is set to `text`, `srt`, or `vtt`.

## Notes

- Only `ASR_API_PORT` is published to the host. The internal Qwen server defaults to `127.0.0.1:18000` inside the container, so it does not conflict with other host services on port `8000`.
