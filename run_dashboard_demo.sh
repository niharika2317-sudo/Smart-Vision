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
open_url "http://127.0.0.1:${SV_PORT}/police"
open_url "http://127.0.0.1:${SV_PORT}/hospital"
echo "These dashboard records are explicitly simulated, not detector outputs."
python simulate_incident.py police
sleep 2
python simulate_incident.py medical
sleep 2
python simulate_incident.py gps
wait "$SERVER_PID"
