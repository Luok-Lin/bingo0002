#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

BACKEND_PORT="${BACKEND_PORT:-8000}"
BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:${BACKEND_PORT}}"
NGROK_BIN="${NGROK_BIN:-ngrok}"
NGROK_API_PORT="${NGROK_API_PORT:-4040}"
NGROK_API_URL="http://127.0.0.1:${NGROK_API_PORT}/api/tunnels"
LOG_DIR="${ROOT_DIR}/logs"
PID_FILE="${ROOT_DIR}/.ngrok.pid"
LOG_FILE="${LOG_DIR}/ngrok.log"
LAUNCHD_LABEL="com.tradingagents.ngrok"
LAUNCHD_PLIST="${LOG_DIR}/${LAUNCHD_LABEL}.plist"
SCREEN_SESSION="${NGROK_SCREEN_SESSION:-tradingagents-ngrok}"

mkdir -p "${LOG_DIR}"

read_public_url() {
  curl -sS "${NGROK_API_URL}" 2>/dev/null | python3 -c '
import json
import sys

try:
    payload = json.load(sys.stdin)
except Exception:
    sys.exit(1)

tunnels = payload.get("tunnels", [])
https_urls = [
    item.get("public_url", "")
    for item in tunnels
    if str(item.get("public_url", "")).startswith("https://")
]
http_urls = [
    item.get("public_url", "")
    for item in tunnels
    if str(item.get("public_url", "")).startswith("http://")
]
url = (https_urls or http_urls or [""])[0]
if not url:
    sys.exit(1)
print(url)
'
}

if ! command -v "${NGROK_BIN}" >/dev/null 2>&1; then
  echo "[!] 未找到 ngrok。请先安装 ngrok，或设置 NGROK_BIN=/path/to/ngrok。"
  exit 1
fi
NGROK_BIN="$(command -v "${NGROK_BIN}")"

if ! curl -sS "${BACKEND_URL}/api/health" >/dev/null 2>&1; then
  echo "[!] 后端没有运行或无法访问: ${BACKEND_URL}"
  echo "    请先执行: bash start_web.sh"
  exit 1
fi

if public_url="$(read_public_url)"; then
  echo "[+] ngrok 已在运行。"
  echo "[*] 公网访问地址: ${public_url}"
  echo "[*] 本地后端地址: ${BACKEND_URL}"
  exit 0
fi

echo "[*] 启动 ngrok 隧道: ${BACKEND_URL}"
if command -v screen >/dev/null 2>&1; then
  screen -S "${SCREEN_SESSION}" -X quit >/dev/null 2>&1 || true
  screen -dmS "${SCREEN_SESSION}" "${NGROK_BIN}" http "${BACKEND_URL}" --log=stdout
  echo "screen:${SCREEN_SESSION}" >"${PID_FILE}"
elif [ "$(uname -s)" = "Darwin" ] && command -v launchctl >/dev/null 2>&1; then
  python3 - "${LAUNCHD_PLIST}" "${LAUNCHD_LABEL}" "${NGROK_BIN}" "${BACKEND_URL}" "${LOG_FILE}" "${ROOT_DIR}" "${HOME}" "${PATH}" <<'PY'
import plistlib
import sys

plist_path, label, ngrok_bin, backend_url, log_file, root_dir, home_dir, path = sys.argv[1:]
payload = {
    "Label": label,
    "ProgramArguments": [ngrok_bin, "http", backend_url, "--log=stdout"],
    "RunAtLoad": True,
    "WorkingDirectory": root_dir,
    "StandardOutPath": log_file,
    "StandardErrorPath": log_file,
    "EnvironmentVariables": {
        "HOME": home_dir,
        "PATH": path,
    },
}
with open(plist_path, "wb") as f:
    plistlib.dump(payload, f)
PY
  launchctl bootout "gui/$(id -u)" "${LAUNCHD_PLIST}" >/dev/null 2>&1 || true
  launchctl bootstrap "gui/$(id -u)" "${LAUNCHD_PLIST}"
  echo "${LAUNCHD_LABEL}" >"${PID_FILE}"
else
  nohup "${NGROK_BIN}" http "${BACKEND_URL}" --log=stdout >"${LOG_FILE}" 2>&1 </dev/null &
  echo $! >"${PID_FILE}"
fi

for _ in $(seq 1 30); do
  if public_url="$(read_public_url)"; then
    echo "[+] ngrok 已就绪。"
    echo "[*] 公网访问地址: ${public_url}"
    echo "[*] 本地后端地址: ${BACKEND_URL}"
    echo "[*] 日志文件: ${LOG_FILE}"
    echo "[*] 停止公网访问: bash stop_public_ngrok.sh"
    exit 0
  fi
  sleep 1
done

echo "[!] ngrok 启动超时，请查看日志: ${LOG_FILE}"
tail -n 30 "${LOG_FILE}" 2>/dev/null || true
exit 1
