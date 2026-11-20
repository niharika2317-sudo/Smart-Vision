import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path
from threading import Event, Lock, Thread

import cv2
import config

from behavior_pipeline import BehaviorPipeline
from dispatcher import Dispatcher
from evidence_manager import EvidenceManager
from incident_logger import IncidentLogger
from medical_detector import MedicalDetector
from notifier import Notifier
from person_detector import PersonDetector
from threat_engine import ThreatEngine
from video_source import get_video_source
from weapon_detector import WeaponDetector, WeaponPersistence


_MODEL_CACHE = None
_MODEL_CACHE_LOCK = Lock()
_MODEL_CACHE_SECONDS = 0.0


class LatestFrameReader:
    """Continuously captures camera frames and keeps only the newest one."""

    def __init__(self, capture):
        self.capture = capture
        self.lock = Lock()
        self.stop_event = Event()
        self.frame = None
        self.sequence = 0
        self.finished = False
        self.error = None
        self.thread = Thread(
            target=self._read_frames,
            name="latest-camera-frame-reader",
            daemon=True,
        )

    def start(self):
        self.thread.start()

    def _read_frames(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()

                if not ok:
                    self.finished = True
                    self.error = "Camera stopped returning frames."
                    return

                if frame is None or getattr(frame, "ndim", 0) != 3 or frame.size == 0:
                    continue

                with self.lock:
                    self.frame = frame
                    self.sequence += 1

        except Exception as exc:
            self.finished = True
            self.error = str(exc)

    def newest_after(self, previous_sequence, timeout_seconds=0.25):
        deadline = time.monotonic() + timeout_seconds

        while not self.stop_event.is_set():
            with self.lock:
                if self.frame is not None and self.sequence > previous_sequence:
                    return self.sequence, self.frame.copy()

                finished = self.finished

            if finished or time.monotonic() >= deadline:
                return previous_sequence, None

            time.sleep(0.005)

        return previous_sequence, None

    def close(self):
        self.stop_event.set()

        try:
            self.capture.release()
        except Exception:
            pass

        if self.thread.is_alive():
            self.thread.join(timeout=1.0)


def build_detector_instances():
    """Load YOLO models once and create fresh detector wrappers for each job."""
    global _MODEL_CACHE, _MODEL_CACHE_SECONDS

    with _MODEL_CACHE_LOCK:
        if _MODEL_CACHE is None:
            started = time.monotonic()

            from ultralytics import YOLO

            _MODEL_CACHE = {
                "person": YOLO("yolov8n.pt"),
                "weapon": YOLO("weights/best.pt"),
                "pose": YOLO("yolov8n-pose.pt"),
            }

            _MODEL_CACHE_SECONDS = time.monotonic() - started
            print(f"[MODELS] loaded once in {_MODEL_CACHE_SECONDS:.2f}s")

        models = _MODEL_CACHE

    persons = PersonDetector(model=models["person"])
    weapons = WeaponDetector(model=models["weapon"])

    from pose_detector import PoseDetector
    from pose_behavior import PoseBehavior

    behavior = BehaviorPipeline(
        pose_behavior=PoseBehavior(
            PoseDetector(model=models["pose"])
        )
    )

    persons.reset()

    return persons, weapons, behavior


def model_cache_seconds():
    return _MODEL_CACHE_SECONDS


def main(
    argv=None,
    detectors=None,
    display=True,
    max_frames=None,
    save_annotated=None,
    stop_event=None,
    on_frame=None,
    source_name=None,
    on_incident=None,
    notifier_instance=None,
    logger_instance=None,
    evidence_instance=None,
):
    parser = argparse.ArgumentParser(description="VantaWatch CCTV safety demo")
    parser.add_argument("source", nargs="?", help="Webcam index or video path")
    parser.add_argument("--no-display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--save-annotated")

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    display = display and not args.no_display
    max_frames = args.max_frames if args.max_frames is not None else max_frames
    save_annotated = args.save_annotated or save_annotated

    import torch

    resolved_device = config.resolve_device()
    device_name = (
        f"cuda:{resolved_device}"
        if isinstance(resolved_device, int)
        else resolved_device
    )

    print(
        f"[START] device={device_name} torch={torch.__version__} "
        f"weapon_size={config.WEAPON_IMAGE_SIZE}"
    )

    if detectors is None:
        persons_detector, weapons_detector, behavior = build_detector_instances()
    else:
        persons_detector, weapons_detector, behavior = detectors

    source_arg = args.source
    cap = get_video_source(source_arg)

    if not cap.isOpened():
        return 1

    source_fps = cap.get(cv2.CAP_PROP_FPS) or config.EVIDENCE_FPS_FALLBACK
    is_file = source_arg is not None and not source_arg.isdigit()

    analysis_mode = (
        "recorded clip analysed offline"
        if is_file
        else "live webcam"
    )

    evidence = evidence_instance or EvidenceManager(fps=source_fps)
    logger = logger_instance or IncidentLogger()
    notifier = notifier_instance or Notifier()

    dispatcher = Dispatcher(
        notifier,
        logger,
        evidence,
        camera_name=source_name,
        on_incident=on_incident,
        analysis_mode=analysis_mode,
    )

    evidence.on_finish = dispatcher.video_finished

    medical_detector = MedicalDetector()
    weapon_persistence = WeaponPersistence()
    threat_engine = ThreatEngine()

    live_reader = None
    live_sequence = 0

    if not is_file:
        live_reader = LatestFrameReader(cap)
        live_reader.start()
        print("[LIVE] Newest-frame mode enabled. Old camera frames are dropped.")

    frame_number = 0
    processed_frames = 0
    video_writer = None
    max_persons = 0
    max_threat = "LOW"

    weapon_samples = {}
    behavior_labels = set()
    start_time = time.monotonic()
    exit_code = 0

    count = 0
    people = []
    weapons = []
    behaviors = []
    medical = []

    assessment = {
        "level": "LOW",
        "score": 0,
        "weapon_score": 0,
        "behavior_score": 0,
        "reasons": [],
    }

    threat_rank = {
        "LOW": 0,
        "MEDIUM": 1,
        "HIGH": 2,
    }

    try:
        while True:
            if stop_event is not None and stop_event.is_set():
                break

            if is_file:
                ok, raw = cap.read()

                if not ok:
                    break
            else:
                live_sequence, raw = live_reader.newest_after(
                    live_sequence,
                    timeout_seconds=0.25,
                )

                if raw is None:
                    if live_reader.finished:
                        print(
                            "[WARN] Camera ended: "
                            f"{live_reader.error or 'No camera frames available'}"
                        )
                        exit_code = 1
                        break

                    continue

            if raw is None or getattr(raw, "ndim", 0) != 3 or raw.size == 0:
                continue

            frame_number += 1

            scale = min(
                1.0,
                config.FRAME_WIDTH / max(1, raw.shape[1]),
            )

            try:
                frame = (
                    cv2.resize(
                        raw,
                        (
                            int(raw.shape[1] * scale),
                            int(raw.shape[0] * scale),
                        ),
                    )
                    if scale < 1.0
                    else raw
                )
            except cv2.error as exc:
                print(f"[WARN] Resize failed: {exc}")
                continue

            t = (
                frame_number / source_fps
                if is_file
                else time.monotonic() - start_time
            )

            person_behavior_frame = (
                (frame_number - 1)
                % max(1, config.PERSON_BEHAVIOR_EVERY_N_FRAMES)
                == 0
            )

            weapon_inference_frame = (
                (frame_number - 1)
                % max(1, config.WEAPON_EVERY_N_FRAMES)
                == 0
            )

            if person_behavior_frame:
                try:
                    _, count, people = persons_detector.detect(frame)
                    people = people or []
                    count = int(count or 0)
                except Exception as exc:
                    print(f"[WARN] Person detector failed: {exc}")
                    people = []
                    count = 0

            if weapon_inference_frame:
                try:
                    raw_weapons = weapons_detector.detect(frame) or []

                    # Supports every weapon detected in the frame.
                    weapons = weapon_persistence.update(raw_weapons)

                    for weapon in raw_weapons:
                        label = weapon.get("label", "?")
                        confidence = float(weapon.get("confidence", 0.0))

                        stats = weapon_samples.setdefault(
                            label,
                            [float("inf"), 0.0, 0],
                        )

                        stats[0] = min(stats[0], confidence)
                        stats[1] = max(stats[1], confidence)
                        stats[2] += 1

                except Exception as exc:
                    print(f"[WARN] Weapon detector failed: {exc}")
                    weapons = []

            if person_behavior_frame:
                try:
                    behaviors = behavior.detect(
                        frame,
                        people,
                        weapons,
                        t,
                    ) or []
                except Exception as exc:
                    print(f"[WARN] Behaviour detector failed: {exc}")
                    behaviors = ["Unknown"] * len(people)

                pose_labels = [
                    "Possible Fall"
                    if item == "Fall Detected"
                    else item
                    for item in behaviors
                ]

                try:
                    medical = medical_detector.detect(
                        frame,
                        people,
                        pose_labels,
                        t,
                    ) or []
                except Exception as exc:
                    print(f"[WARN] Medical detector failed: {exc}")
                    medical = []

                behavior_labels.update(behaviors)

            assessment = threat_engine.evaluate(
                people,
                weapons,
                behaviors,
            )

            max_persons = max(max_persons, count)

            if (
                threat_rank.get(assessment["level"], 0)
                > threat_rank.get(max_threat, 0)
            ):
                max_threat = assessment["level"]

            try:
                output = weapons_detector.draw(frame, weapons)
            except Exception:
                output = frame.copy()

            for person, label in zip(people, behaviors):
                x1, y1, x2, y2 = map(int, person["box"])

                color = (
                    (0, 0, 255)
                    if label in (
                        "Possible Attack",
                        "Armed Person",
                        "Aggressive Movement",
                    )
                    else (0, 220, 255)
                )

                cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

                display_label = {
                    "Armed Person": "Weapon candidate",
                    "Possible Attack": "Possible attack",
                }.get(label, label)

                cv2.putText(
                    output,
                    f"P{person.get('id', '?')} {display_label}",
                    (x1, max(18, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.43,
                    color,
                    2,
                )

            for case in medical:
                x1, y1, x2, y2 = map(int, case["box"])

                cv2.rectangle(
                    output,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 220),
                    3,
                )

            active = assessment["level"] != "LOW" or bool(medical)
            evidence.update(output, t, active)

            if person_behavior_frame or weapon_inference_frame:
                dispatcher.process(
                    assessment,
                    people,
                    weapons,
                    behaviors,
                    medical,
                    output,
                    t,
                )

            # Small overlay only.
            height, width = output.shape[:2]

            cv2.rectangle(
                output,
                (5, 5),
                (205, 35),
                (35, 35, 35),
                -1,
            )

            cv2.putText(
                output,
                f"{assessment['level']} | W:{len(weapons)} | P:{count}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
            )

            if assessment["level"] == "HIGH":
                cv2.rectangle(
                    output,
                    (0, max(0, height - 35)),
                    (width, height),
                    (0, 0, 210),
                    -1,
                )

                cv2.putText(
                    output,
                    "HIGH THREAT - POLICE ALERTED",
                    (10, max(22, height - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                )

            if save_annotated:
                if video_writer is None:
                    destination = Path(save_annotated).expanduser().resolve()
                    destination.parent.mkdir(parents=True, exist_ok=True)

                    video_writer = cv2.VideoWriter(
                        str(destination),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        source_fps,
                        (width, height),
                    )

                    if not video_writer.isOpened():
                        print(
                            "[WARN] Could not create output video: "
                            f"{destination}"
                        )
                        video_writer.release()
                        video_writer = False

                if video_writer:
                    video_writer.write(output)

            processed_frames += 1

            if on_frame is not None:
                try:
                    on_frame(
                        output,
                        t,
                        frame_number,
                        source_fps,
                        {
                            "threat": assessment,
                            "persons": count,
                            "weapons": len(weapons),
                            "behaviors": list(behaviors),
                            "medical": len(medical),
                            "people_details": [
                                {
                                    "id": person.get("id"),
                                    "behavior": person.get(
                                        "behavior_label",
                                        "Unknown",
                                    ),
                                    "pose": person.get(
                                        "pose_behavior",
                                        "Unknown",
                                    ),
                                    "speed": round(
                                        float(person.get("speed", 0.0)),
                                        2,
                                    ),
                                }
                                for person in people
                            ],
                            "processing_fps": (
                                processed_frames
                                / max(
                                    0.001,
                                    time.monotonic() - start_time,
                                )
                            ),
                        },
                    )
                except Exception as exc:
                    print(f"[WARN] Frame callback failed: {exc}")

            if display:
                cv2.imshow("VantaWatch AI", output)
                key = cv2.waitKey(1) & 0xFF

                try:
                    window_closed = (
                        cv2.getWindowProperty(
                            "VantaWatch AI",
                            cv2.WND_PROP_VISIBLE,
                        )
                        < 1
                    )
                except cv2.error:
                    window_closed = True

                if key in (27, ord("q")) or window_closed:
                    break

            if max_frames is not None and frame_number >= max_frames:
                break

    finally:
        try:
            evidence.close()
        except Exception as exc:
            print(f"[WARN] Evidence close failed: {exc}")

        try:
            notifier.close()
        except Exception as exc:
            print(f"[WARN] Notifier close failed: {exc}")

        try:
            if video_writer:
                video_writer.release()
        except Exception:
            pass

        try:
            if live_reader is not None:
                live_reader.close()
            else:
                cap.release()
        except Exception:
            pass

        if display:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

        elapsed = max(0.001, time.monotonic() - start_time)

        weapon_report = (
            "; ".join(
                f"{label}:{values[0]:.2f}-{values[1]:.2f}({values[2]})"
                for label, values in weapon_samples.items()
            )
            or "none"
        )

        print(
            f"[RUN] frames={processed_frames} "
            f"elapsed={elapsed:.2f}s "
            f"fps={processed_frames / elapsed:.2f} "
            f"max_persons={max_persons} "
            f"weapons={weapon_report} "
            f"behaviors={sorted(behavior_labels)} "
            f"max_threat={max_threat} "
            f"police_alerts={dispatcher.alert_counts['POLICE']} "
            f"medical_alerts={dispatcher.alert_counts['MEDICAL']}"
        )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
