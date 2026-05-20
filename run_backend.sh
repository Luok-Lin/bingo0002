#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
HOST="${BACKEND_HOST:-127.0.0.1}"
PORT="${BACKEND_PORT:-8000}"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${ROOT_DIR}/.backend.pid"
LOG_FILE="${LOG_DIR}/backend.log"

mkdir -p "${LOG_DIR}"

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[+] 后端已在运行: http://${HOST}:${PORT}"
  curl -s "http://${HOST}:${PORT}/api/health" || true
  echo ""
  exit 0
fi

echo "[*] 启动后端: http://${HOST}:${PORT}"
nohup "${PYTHON_BIN}" -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"

for _ in $(seq 1 30); do
  if curl -s "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
    echo "[+] 后端已就绪。日志: ${LOG_FILE}"
    echo "[*] 请打开: http://${HOST}:${PORT}"
    exit 0
  fi
  sleep 0.5
done

echo "[!] 后端启动超时，请查看日志: ${LOG_FILE}"
exit 1
