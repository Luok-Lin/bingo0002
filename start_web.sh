#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BACKEND_HOST="127.0.0.1"
BACKEND_PORT="8000"
FRONTEND_PORT="5173"
PYTHON_BIN="${PYTHON_BIN:-python3}"

backend_pid=""

is_port_listening() {
  local port="$1"
  lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
}

cleanup() {
  if [ -n "${backend_pid}" ] && kill -0 "${backend_pid}" >/dev/null 2>&1; then
    echo ""
    echo "[*] 正在停止后端服务 (PID: ${backend_pid}) ..."
    kill "${backend_pid}" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT INT TERM

echo "==============================================="
echo " TradingAgents Web 一键启动脚本"
echo "==============================================="
echo "[*] 项目目录: ${ROOT_DIR}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[!] 未找到 ${PYTHON_BIN}，请安装 Python 3 后重试。"
  exit 1
fi

if is_port_listening "${BACKEND_PORT}"; then
  echo "[!] 端口 ${BACKEND_PORT} 已被占用，请先释放后再启动。"
  echo "    参考命令: lsof -nP -iTCP:${BACKEND_PORT} -sTCP:LISTEN"
  exit 1
fi

RELOAD_FLAG=""
if [ "${WEB_RELOAD:-0}" = "1" ]; then
  RELOAD_FLAG="--reload"
  echo "[!] 已启用 --reload（开发模式）。长任务生成建议时请勿重启后端。"
fi

echo "[*] 启动后端 API（含前端页面）: http://${BACKEND_HOST}:${BACKEND_PORT}"
"${PYTHON_BIN}" -m uvicorn backend.app:app --host "${BACKEND_HOST}" --port "${BACKEND_PORT}" ${RELOAD_FLAG} &
backend_pid=$!

echo "[*] 等待后端启动..."
for _ in $(seq 1 20); do
  if curl -s "http://${BACKEND_HOST}:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -s "http://${BACKEND_HOST}:${BACKEND_PORT}/api/health" >/dev/null 2>&1; then
  echo "[!] 后端健康检查失败，启动终止。"
  exit 1
fi

echo "[+] 后端已就绪。"
echo "[*] 推荐直接打开: http://${BACKEND_HOST}:${BACKEND_PORT}"
echo "[*] 也可使用独立前端: http://127.0.0.1:${FRONTEND_PORT}（可选）"
echo "[*] 按 Ctrl+C 停止后端。"

if is_port_listening "${FRONTEND_PORT}"; then
  echo "[!] 端口 ${FRONTEND_PORT} 已被占用，跳过独立前端服务。"
  wait "${backend_pid}"
  exit 0
fi

cd "${ROOT_DIR}/frontend"
"${PYTHON_BIN}" -m http.server "${FRONTEND_PORT}" --bind "${BACKEND_HOST}"
