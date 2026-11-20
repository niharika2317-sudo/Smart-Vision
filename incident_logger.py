import json
from pathlib import Path
import config


class IncidentLogger:
    def __init__(self):
        self.folder = config.EVIDENCE_DIR / "incidents"
        self.folder.mkdir(parents=True, exist_ok=True)

    def log(self, payload):
        path = self.folder / f"{payload['id']}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return str(path)

    def update(self, incident_id, fields):
        path = self.folder / f"{incident_id}.json"
        if path.exists():
            report = json.loads(path.read_text(encoding="utf-8"))
            report.update(fields)
            path.write_text(json.dumps(report, indent=2), encoding="utf-8")
