from datetime import datetime, timezone
import threading
import uuid
import sys
import subprocess
import config


class Dispatcher:
    def __init__(self, notifier, logger, evidence, camera_name=None, on_incident=None,
                 analysis_mode="recorded clip analysed offline"):
        self.notifier, self.logger, self.evidence = notifier, logger, evidence
        self.camera_name = camera_name or config.CAMERA_NAME
        self.on_incident = on_incident
        self.analysis_mode = analysis_mode
        self.confirm = 0
        self.last_police = -1e9
        self.medical_sent = {}
        self.waiting_video = set()
        self._lock = threading.Lock()
        self.alert_counts = {"POLICE": 0, "MEDICAL": 0}
        self.alert_events = []

    def process(self, assessment, persons, weapons, behaviors, medical, frame, t):
        weapon_contact = any(
            float(w["box"][0]) <= float(p["box"][2]) and float(w["box"][2]) >= float(p["box"][0])
            and float(w["box"][1]) <= float(p["box"][3]) and float(w["box"][3]) >= float(p["box"][1])
            for w in weapons for p in persons)
        police_eligible = (not config.WEAPON_REQUIRE_PERSON_CONTACT_FOR_POLICE
                           or not weapons or weapon_contact)
        if assessment["level"] == "HIGH" and police_eligible: self.confirm += 1
        else: self.confirm = 0
        if self.confirm >= config.POLICE_CONFIRM_FRAMES and t - self.last_police >= config.POLICE_COOLDOWN_SECONDS:
            self.last_police = t
            incident = self._incident("POLICE", assessment, persons, weapons, behaviors, [], frame, t)
            self._send(incident)
            self.alert_counts["POLICE"] += 1
            self.alert_events.append({"type": "POLICE", "t": t})
            self.confirm = 0
        for case in medical:
            key = case["track_id"]
            old = self.medical_sent.get(key)
            if old is None or case["severity"] == "HIGH" and old != "HIGH":
                self.medical_sent[key] = case["severity"]
                self._send(self._incident("MEDICAL", assessment, persons, weapons, behaviors, [case], frame, t))
                self.alert_counts["MEDICAL"] += 1
                self.alert_events.append({"type": "MEDICAL", "t": t})

    def _incident(self, kind, assessment, persons, weapons, behaviors, medical, frame, video_time):
        incident_id = str(uuid.uuid4())
        image_path = self.evidence.save_image(frame, kind.lower())
        payload = {"id": incident_id, "type": kind, "level": assessment["level"], "score": assessment["score"],
                   "weapon_score": assessment["weapon_score"], "behavior_score": assessment["behavior_score"],
                   "reasons": assessment["reasons"] + [r for c in medical for r in c["reasons"]],
                   "behaviors": behaviors, "weapons": weapons, "camera_id": config.CAMERA_ID,
                   "camera_name": self.camera_name, "lat": config.CAMERA_LAT, "lon": config.CAMERA_LON,
                   "image_path": image_path, "video_path": None,
                   "timestamp": datetime.now(timezone.utc).isoformat(), "video_time":float(video_time),
                   "medical_cases": medical, "analysis_mode": self.analysis_mode}
        self.logger.log(payload)
        with self._lock: self.waiting_video.add(incident_id)
        return payload

    def _send(self, incident):
        if config.LOCAL_SIREN:
            print(f"[SIREN] local trigger called from dispatcher type={incident['type']} id={incident['id']}")
            threading.Thread(target=self._siren,args=(incident["type"],),daemon=True).start()
        self.notifier.send("POST", "/api/incidents", incident)
        print(f"[INCIDENT] POST /api/incidents type={incident['type']} id={incident['id']}")
        if self.on_incident:
            try: self.on_incident(incident)
            except Exception as exc: print(f"[WARN] Incident callback failed: {exc}")

    def video_finished(self, relative_path):
        with self._lock: ids = list(self.waiting_video); self.waiting_video.clear()
        for incident_id in ids:
            fields = {"video_path": relative_path}
            self.logger.update(incident_id, fields)
            self.notifier.send("PATCH", f"/api/incidents/{incident_id}", fields)

    @staticmethod
    def _siren(kind="POLICE"):
        if sys.platform == "darwin":
            try:
                for _ in range(3):
                    subprocess.run(["afplay", "/System/Library/Sounds/Sosumi.aiff"], check=False)
                return
            except Exception as exc:
                print(f"[SIREN] afplay unavailable ({exc}); using terminal bell", flush=True)
                print("\a", end="", flush=True)
                return
        if not sys.platform.startswith("win"):
            print("\a", end="", flush=True)
            return
        try:
            import winsound
            frequencies=(620,760) if kind=="MEDICAL" else (1100,1450)
            for _ in range(3):
                for frequency in frequencies: winsound.Beep(frequency,220)
        except Exception:
            print("\a", end="", flush=True)
