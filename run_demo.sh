#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SV_PORT="${SV_PORT:-5050}"

open_url() {
  if [[ "$(uname -s)" == "Darwin" ]]; then open "$1"
  elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$1" >/dev/null 2>&1 &
  else echo "Open this URL in a browser: $1"; fi
}

python server.py --port "$SV_PORT" &
SERVER_PID=$!
cleanup() { kill "$SERVER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM
for _ in $(seq 1 40); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:${SV_PORT}/', timeout=1).close()" >/dev/null 2>&1; then break; fi
  sleep 0.5
done
echo "RECORDED CLIP ANALYSED OFFLINE - this is not a live camera feed."
open_url "http://127.0.0.1:${SV_PORT}/police?offline=1"
open_url "http://127.0.0.1:${SV_PORT}/hospital?offline=1"
python make_demo.py --build-only
open_url "$(pwd)/verification/offline_demo/normal_quiet.mp4"
sleep 2
python make_demo.py --post-only
echo "RECORDED CLIP ANALYSED OFFLINE - review the clip beside the dashboards."
wait "$SERVER_PID"
