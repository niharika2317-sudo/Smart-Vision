#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SV_PORT="${SV_PORT:-5050}"
exec python server.py --port "$SV_PORT" "$@"
