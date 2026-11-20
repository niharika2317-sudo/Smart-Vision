"""Run only the staged, fixed 10-per-class SCVD test sample."""
import csv
import json
import re
import subprocess
import sys
from pathlib import Path
import cv2
import config

CLASSES = ("weaponized", "normal", "violence")
CAP_FRAMES = 300
RUN_RE = re.compile(r"\[RUN\] frames=(\d+) elapsed=([\d.]+)s achieved_fps=([\d.]+) max_persons=(\d+) weapons=(.*?) first_weapon_t=(\S+) behaviors=(.*?) max_threat=(LOW|MEDIUM|HIGH) police_alerts=(\d+) medical_alerts=(\d+) alert_times=(.*?) first_weapon_to_police=(\S+)")


def main():
    stage_root = config.BASE_DIR / "videos"
    output_root = config.BASE_DIR / "verification" / "scvd"
    log_root = config.BASE_DIR / "verification" / "scvd_logs"
    snapshot_root = config.BASE_DIR / "verification" / "scvd_snapshots"
    rows = []
    for class_name in CLASSES:
        staged = sorted((stage_root / class_name).glob("*"), key=lambda p: p.name.casefold())
        if len(staged) != 10:
            raise SystemExit(f"Expected exactly 10 staged files in videos/{class_name}; found {len(staged)}")
        for source in staged:
            destination = output_root / class_name / f"{source.stem}_annotated.mp4"
            destination.parent.mkdir(parents=True, exist_ok=True)
            log_path = log_root / class_name / f"{source.stem}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            command = [sys.executable, str(config.BASE_DIR / "main.py"), str(source), "--no-display",
                       "--max-frames", str(CAP_FRAMES), "--save-annotated", str(destination)]
            try:
                result = subprocess.run(command, cwd=config.BASE_DIR, capture_output=True, text=True,
                                        errors="replace", timeout=300)
                log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
                match = RUN_RE.search(result.stdout)
                if not match:
                    row = {"class": class_name, "clip": source.name, "exit_code": result.returncode,
                           "error": "No [RUN] summary; see per-clip log"}
                else:
                    values = match.groups()
                    row = {"class": class_name, "clip": source.name, "exit_code": result.returncode,
                           "frames": int(values[0]), "elapsed_s": float(values[1]), "fps": float(values[2]),
                           "max_persons": int(values[3]), "weapons": values[4], "first_weapon_t": values[5],
                           "behaviors": values[6], "max_threat": values[7], "police_alerts": int(values[8]),
                           "medical_alerts": int(values[9]), "alert_times": values[10],
                           "first_weapon_to_police": values[11], "error": ""}
                rows.append(row)
                if destination.exists():
                    cap = cv2.VideoCapture(str(destination))
                    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, count // 2))
                    ok, frame = cap.read()
                    cap.release()
                    if ok:
                        snap = snapshot_root / class_name / f"{source.stem}_mid.jpg"
                        snap.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(snap), frame)
                print(f"[{class_name}] {source.name}: {row}", flush=True)
            except subprocess.TimeoutExpired as exc:
                stdout = exc.stdout or ""
                stderr = exc.stderr or ""
                if isinstance(stdout, bytes):
                    stdout = stdout.decode("utf-8", errors="replace")
                if isinstance(stderr, bytes):
                    stderr = stderr.decode("utf-8", errors="replace")
                log_path.write_text(stdout + stderr, encoding="utf-8")
                row = {"class": class_name, "clip": source.name, "exit_code": "timeout", "error": "300s run timeout"}
                rows.append(row)
                print(f"[{class_name}] {source.name}: timeout", flush=True)
    csv_path = config.BASE_DIR / "verification" / "scvd_results.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    (config.BASE_DIR / "verification" / "scvd_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Saved {len(rows)} per-clip rows to {csv_path}")


if __name__ == "__main__":
    main()
