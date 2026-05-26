#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "${PYTHON_BIN}" ] && [ -x "${ROOT_DIR}/.venv/bin/python" ]; then
  PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
if [ "${PUBLIC_WEB:-0}" = "1" ] && [ -z "${BACKEND_HOST:-}" ]; then
  HOST="0.0.0.0"
else
  HOST="${BACKEND_HOST:-127.0.0.1}"
fi
PORT="${BACKEND_PORT:-8000}"
HEALTH_HOST="${HOST}"
if [ "${HOST}" = "0.0.0.0" ]; then
  HEALTH_HOST="127.0.0.1"
fi
PUBLIC_URL="${PUBLIC_URL:-}"
if [ -n "${PUBLIC_URL}" ]; then
  ACCESS_URL="${PUBLIC_URL%/}"
elif [ "${HOST}" = "0.0.0.0" ]; then
  ACCESS_URL="http://<服务器公网IP或域名>:${PORT}"
else
  ACCESS_URL="http://${HOST}:${PORT}"
fi
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${ROOT_DIR}/.backend.pid"
LOG_FILE="${LOG_DIR}/backend.log"

mkdir -p "${LOG_DIR}"

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "[+] 后端已在运行: ${ACCESS_URL}"
  curl -s "http://${HEALTH_HOST}:${PORT}/api/health" || true
  echo ""
  exit 0
fi

echo "[*] 启动后端: http://${HOST}:${PORT}"
if [ "${HOST}" = "0.0.0.0" ]; then
  echo "[!] 公网/外部访问模式：后端将监听所有网卡。请确认已设置强管理员密码，并在防火墙/反向代理中只开放必要端口。"
fi
nohup "${PYTHON_BIN}" -m uvicorn backend.app:app --host "${HOST}" --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
echo $! >"${PID_FILE}"

for _ in $(seq 1 30); do
  if curl -s "http://${HEALTH_HOST}:${PORT}/api/health" >/dev/null 2>&1; then
    echo "[+] 后端已就绪。日志: ${LOG_FILE}"
    echo "[*] 请打开: ${ACCESS_URL}"
    if [ "${HOST}" = "0.0.0.0" ] && [ -z "${PUBLIC_URL}" ]; then
      echo "[*] 如需非局域网访问，请把 PUBLIC_URL 设置为你的公网域名或穿透地址。"
    fi
    exit 0
  fi
  sleep 0.5
done

echo "[!] 后端启动超时，请查看日志: ${LOG_FILE}"
exit 1
