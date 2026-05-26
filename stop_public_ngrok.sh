#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${ROOT_DIR}"

LABEL="com.tradingagents.ngrok"
PLIST_FILE="${ROOT_DIR}/logs/${LABEL}.plist"
PID_FILE="${ROOT_DIR}/.ngrok.pid"
SCREEN_SESSION="${NGROK_SCREEN_SESSION:-tradingagents-ngrok}"

stopped=0

if [ "$(uname -s)" = "Darwin" ] && command -v launchctl >/dev/null 2>&1 && [ -f "${PLIST_FILE}" ]; then
  if launchctl bootout "gui/$(id -u)" "${PLIST_FILE}" >/dev/null 2>&1; then
    stopped=1
  fi
fi

if [ -f "${PID_FILE}" ]; then
  pid_or_label="$(cat "${PID_FILE}")"
  if [[ "${pid_or_label}" == screen:* ]]; then
    screen_name="${pid_or_label#screen:}"
    screen -S "${screen_name}" -X quit >/dev/null 2>&1 || true
    stopped=1
  fi
  if [[ "${pid_or_label}" =~ ^[0-9]+$ ]] && kill -0 "${pid_or_label}" >/dev/null 2>&1; then
    kill "${pid_or_label}" >/dev/null 2>&1 || true
    stopped=1
  fi
fi

screen -S "${SCREEN_SESSION}" -X quit >/dev/null 2>&1 || true
pkill -f "ngrok http http://127.0.0.1:8000" >/dev/null 2>&1 || true
rm -f "${PID_FILE}"

if [ "${stopped}" = "1" ]; then
  echo "[+] ngrok 公网访问已停止。"
else
  echo "[*] 未发现正在托管的 ngrok 公网访问。"
fi
