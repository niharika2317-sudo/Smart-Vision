from collections import deque
from datetime import datetime
from pathlib import Path
import cv2
import config
import time


class EvidenceManager:
    def __init__(self, fps=config.EVIDENCE_FPS_FALLBACK, on_finish=None):
        self.images = config.EVIDENCE_DIR / "images"
        self.videos = config.EVIDENCE_DIR / "videos"
        self.incidents = config.EVIDENCE_DIR / "incidents"
        for folder in (self.images, self.videos, self.incidents): folder.mkdir(parents=True, exist_ok=True)
        self.fps = fps if fps > 0 else config.EVIDENCE_FPS_FALLBACK
        self.on_finish = on_finish
        self.buffer = deque()
        self.writer = None
        self.video_path = None
        self.recording = False
        self.last_active = None
        self.started = None
        self.last_imwrite_seconds = 0.0
        self.last_disk_check = 0.0

    def save_image(self, frame, tag="incident"):
        self._enforce_disk_limit()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.images / f"{stamp}_{tag}.jpg"
        started = time.perf_counter()
        cv2.imwrite(str(path), frame)
        self.last_imwrite_seconds += time.perf_counter() - started
        return path.relative_to(config.EVIDENCE_DIR).as_posix()

    def _start(self, frame, t):
        self._enforce_disk_limit(force=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        path = self.videos / f"{stamp}.mp4"
        h, w = frame.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        if not writer.isOpened():
            writer.release()
            return
        self.writer, self.video_path = writer, path
        self.recording, self.started, self.last_active = True, t, t
        for _, buffered in self.buffer: self.writer.write(buffered)
        print("[Evidence] Recording started")

    def update(self, frame, t=0.0, active=False):
        self.buffer.append((t, frame.copy()))
        while self.buffer and t - self.buffer[0][0] > config.EVIDENCE_PRE_SECONDS:
            self.buffer.popleft()
        if active and not self.recording: self._start(frame, t)
        if not self.recording: return
        if active: self.last_active = t
        if t - self.started >= config.EVIDENCE_MAX_SECONDS or (not active and t - self.last_active >= config.EVIDENCE_POST_SECONDS):
            self._finish()
            return
        self.writer.write(frame)

    def _finish(self):
        if not self.recording: return
        self.writer.release()
        relative = self.video_path.relative_to(config.EVIDENCE_DIR).as_posix()
        self.writer = None
        self.recording = False
        print(f"[Evidence] Saved {relative}")
        if self.on_finish:
            try: self.on_finish(relative)
            except Exception as exc: print(f"[Evidence] Finish callback failed: {exc}")
        self._enforce_disk_limit(force=True)

    def close(self): self._finish()

    def _enforce_disk_limit(self, force=False):
        """Remove oldest unreferenced evidence; incident-linked files are retained."""
        now = time.monotonic()
        if not force and now - self.last_disk_check < config.EVIDENCE_CAP_CHECK_SECONDS:
            return
        self.last_disk_check = now
        cap = max(0.0, float(config.EVIDENCE_MAX_GB)) * 1024 ** 3
        reports = list(self.incidents.glob("*.json"))
        referenced = set()
        for report_path in reports:
            try:
                import json
                report = json.loads(report_path.read_text(encoding="utf-8"))
                for key in ("image_path", "video_path"):
                    if report.get(key): referenced.add(Path(report[key]).as_posix())
            except Exception:
                continue
        files = [p for folder in (self.images, self.videos) for p in folder.glob("*") if p.is_file()]
        total = sum(p.stat().st_size for p in files)
        if total <= cap: return
        candidates = sorted((p for p in files
                             if p.relative_to(config.EVIDENCE_DIR).as_posix() not in referenced
                             and (self.video_path is None or p != self.video_path)),
                            key=lambda p:p.stat().st_mtime)
        for path in candidates:
            if total <= cap: break
            size = path.stat().st_size
            try:
                path.unlink()
                total -= size
            except OSError as exc:
                print(f"[WARN] Could not remove old evidence {path.name}: {exc}")
