import json
import ssl
from pathlib import Path
from queue import Queue
from threading import Thread
from urllib.request import Request, urlopen
import config


class Notifier:
    def __init__(self, base_url=None):
        self.base_url = (base_url or config.SERVER_URL).rstrip("/")
        self.queue = Queue()
        self.worker = Thread(target=self._work, daemon=True)
        self.worker.start()

    def send(self, method, path, payload): self.queue.put((method, path, payload))

    def _work(self):
        while True:
            item = self.queue.get()
            if item is None: self.queue.task_done(); break
            method, path, payload = item
            body = json.dumps(payload).encode("utf-8")
            ok = False
            for _ in range(3):
                try:
                    req = Request(self.base_url + path, data=body, method=method, headers={"Content-Type":"application/json"})
                    options = {}
                    # The app's optional HTTPS mode uses Flask's generated local
                    # development certificate. Trust that certificate only for
                    # loopback dispatcher traffic; external TLS remains verified.
                    if self.base_url.startswith(("https://127.0.0.1", "https://localhost")):
                        options["context"] = ssl._create_unverified_context()
                    with urlopen(req, timeout=3, **options) as response: response.read()
                    ok = True; break
                except Exception: pass
            if not ok:
                outbox = config.EVIDENCE_DIR / "outbox.jsonl"
                outbox.parent.mkdir(parents=True, exist_ok=True)
                try:
                    with outbox.open("a", encoding="utf-8") as f: f.write(json.dumps(item) + "\n")
                except Exception: pass
            self.queue.task_done()

    def flush(self): self.queue.join()
    def close(self): self.flush(); self.queue.put(None); self.worker.join(timeout=1)
