#!/usr/bin/env bash
set -euo pipefail

api_pid=""

shutdown() {
  if [[ -n "${api_pid}" ]] && kill -0 "${api_pid}" 2>/dev/null; then
    kill "${api_pid}" 2>/dev/null || true
    wait "${api_pid}" 2>/dev/null || true
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

uvicorn app.main:app \
  --host 0.0.0.0 \
  --port 8080 \
  --log-level "${LOG_LEVEL:-info}" &
api_pid="$!"

wait "${api_pid}"
