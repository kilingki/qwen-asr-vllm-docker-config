#!/usr/bin/env bash
set -euo pipefail

qwen_pid=""
api_pid=""

shutdown() {
  if [[ -n "${api_pid}" ]] && kill -0 "${api_pid}" 2>/dev/null; then
    kill "${api_pid}" 2>/dev/null || true
  fi
  if [[ -n "${qwen_pid}" ]] && kill -0 "${qwen_pid}" 2>/dev/null; then
    kill "${qwen_pid}" 2>/dev/null || true
  fi
}

trap shutdown EXIT INT TERM

cuda_lib="$(ldconfig -p | awk '/libcuda\.so\.1/{print $NF; exit}')"
if [[ -n "${cuda_lib}" ]]; then
  mkdir -p /usr/local/cuda/lib64
  ln -sf "${cuda_lib}" /usr/local/cuda/lib64/libcuda.so
  export LIBRARY_PATH="/usr/local/cuda/lib64:${LIBRARY_PATH:-}"
  export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

qwen-asr-serve "${ASR_MODEL_PATH:-/models/Qwen3-ASR-1.7B}" \
  --host "${QWEN_ASR_INTERNAL_HOST:-127.0.0.1}" \
  --port "${QWEN_ASR_INTERNAL_PORT:-18000}" \
  --served-model-name "${ASR_MODEL:-qwen3-asr}" \
  --gpu-memory-utilization "${ASR_GPU_MEMORY_UTILIZATION:-0.72}" \
  --max-model-len "${ASR_MAX_MODEL_LEN:-10000}" \
  --generation-config "${ASR_GENERATION_CONFIG:-vllm}" &
qwen_pid="$!"

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level "${LOG_LEVEL:-info}" &
api_pid="$!"

wait -n "${qwen_pid}" "${api_pid}"
