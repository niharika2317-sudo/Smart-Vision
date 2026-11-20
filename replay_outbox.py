"""Retry saved notifier POST/PATCH operations after the server returns."""
import json
import argparse
from pathlib import Path
from urllib.request import Request, urlopen
import config


def replay(outbox=None, base_url=config.SERVER_URL, opener=urlopen):
    path = Path(outbox or config.EVIDENCE_DIR / "outbox.jsonl")
    if not path.exists(): return 0, 0
    remaining, sent = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            method, route, payload = json.loads(line)
            body = json.dumps(payload).encode("utf-8")
            request = Request(base_url.rstrip("/") + route, data=body, method=method,
                              headers={"Content-Type": "application/json"})
            with opener(request, timeout=3) as response:
                if 200 <= response.status < 300: sent += 1
                else: remaining.append(line)
        except Exception:
            remaining.append(line)
    if remaining: path.write_text("\n".join(remaining) + "\n", encoding="utf-8")
    else: path.unlink(missing_ok=True)
    return sent, len(remaining)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=config.PORT, help="local server port (default: config.PORT / SV_PORT)")
    parser.add_argument("--server-url", help="override the configured local server URL")
    args = parser.parse_args()
    base_url = args.server_url or f"http://127.0.0.1:{args.port}"
    sent, pending = replay(base_url=base_url)
    print(f"[Outbox] replayed={sent} remaining={pending}")
