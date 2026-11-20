"""Replay cached detector outputs through model-free project logic."""
import argparse
import copy
import json
from pathlib import Path

import cv2
import numpy as np

import config
from behavior_fusion import BehaviorFusion
from medical_detector import MedicalDetector
from threat_engine import ThreatEngine
from utils import point_in_box
from weapon_detector import WeaponPersistence


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _movement(speed):
    value = abs(float(speed or 0.0))
    if value >= config.AGGRESSIVE_SPEED:
        return "Aggressive Movement"
    if value >= config.RUNNING_SPEED:
        return "Running"
    if value >= config.WALKING_SPEED:
        return "Walking"
    return "Standing"


def _contact(box_a, box_b):
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    return ax1 <= bx2 and ax2 >= bx1 and ay1 <= by2 and ay2 >= by1


def _filter_weapons(raw, threshold, class_thresholds=None, max_area_ratio=None):
    class_thresholds = class_thresholds or {}
    out = []
    for weapon in raw:
        min_conf = float(class_thresholds.get(weapon["label"], threshold))
        if float(weapon["confidence"]) < min_conf:
            continue
        if max_area_ratio is not None:
            x1, y1, x2, y2 = map(float, weapon["box"])
            frame_area = max(1.0, float(weapon.get("frame_width", 0)) * float(weapon.get("frame_height", 0)))
            if frame_area == 1.0:
                # frame dimensions are attached by the replay loop below
                frame_area = 1.0
            if (x2-x1) * (y2-y1) / frame_area > float(max_area_ratio):
                continue
        out.append(weapon)
    return out


def _alerts_step(assessment, persons, weapons, medical, t, frame_number, state,
                 person_gate=False):
    gate_ok = any(_contact(w["box"], p["box"]) for w in weapons for p in persons)
    police_eligible = (not person_gate) or gate_ok
    if assessment["level"] == "HIGH" and police_eligible:
        state["confirm"] += 1
    else:
        state["confirm"] = 0
    if (state["confirm"] >= config.POLICE_CONFIRM_FRAMES
            and t - state["last_police"] >= config.POLICE_COOLDOWN_SECONDS):
        state["last_police"] = t
        state["police"].append({"frame": frame_number, "t": round(t, 4)})
        state["confirm"] = 0
    for case in medical:
        key, severity = str(case["track_id"]), case["severity"]
        old = state["medical_sent"].get(key)
        if old is None or (severity == "HIGH" and old != "HIGH"):
            state["medical_sent"][key] = severity
            state["medical"].append({"frame": frame_number, "t": round(t, 4),
                                     "track_id": int(case["track_id"]), "severity": severity,
                                     "reasons": list(case["reasons"])})


def replay_clip(cache, threshold=None, class_thresholds=None, max_area_ratio=None,
                confirm_frames=None, person_gate=False, legacy_blood_diagnostic=False,
                include_timeline=False):
    if threshold is None:
        threshold = float(config.WEAPON_CONFIDENCE)
        if class_thresholds is None:
            class_thresholds = getattr(config, "WEAPON_CLASS_CONFIDENCE", {})
    threshold = float(threshold)
    persistence = WeaponPersistence(confirm_frames=confirm_frames)
    fusion, threat, medical_detector = BehaviorFusion(), ThreatEngine(), MedicalDetector()
    persons, behaviors, weapons, medical = [], [], [], []
    state = {"confirm": 0, "last_police": -1e9, "medical_sent": {}, "police": [], "medical": []}
    detections_by_class = {}
    legacy_state, legacy_events = {}, []
    legacy_emitted = set()
    legacy_track_first = {}
    timeline = []

    for row in cache["records"]:
        t, frame_number = float(row["t"]), int(row["frame"])
        if row["weapon_inference_frame"]:
            raw = []
            frame_area = max(1, int(row["width"]) * int(row["height"]))
            for original in row["raw_weapons"]:
                w = copy.deepcopy(original)
                w["frame_width"], w["frame_height"] = int(row["width"]), int(row["height"])
                min_conf = float((class_thresholds or {}).get(w["label"], threshold))
                if float(w["confidence"]) < min_conf:
                    continue
                if max_area_ratio is not None:
                    x1, y1, x2, y2 = map(float, w["box"])
                    if (x2-x1) * (y2-y1) / frame_area > float(max_area_ratio):
                        continue
                raw.append(w)
                detections_by_class[w["label"]] = detections_by_class.get(w["label"], 0) + 1
            weapons = persistence.update(raw)

        if row["person_behavior_frame"]:
            persons = copy.deepcopy(row["persons"])
            behaviors = []
            for person in persons:
                holding = any(point_in_box(w["center"], person["box"], config.HOLD_MARGIN) for w in weapons)
                behaviors.append(fusion.fuse(_movement(person.get("speed", 0.0)),
                                             person.get("pose_behavior", "Normal"), holding))
            pose_labels = ["Possible Fall" if label == "Fall Detected" else label for label in behaviors]
            frame_stub = np.zeros((int(row["height"]), int(row["width"]), 3), dtype=np.uint8)
            medical = medical_detector.detect(frame_stub, persons, pose_labels, t) or []
            if legacy_blood_diagnostic:
                for person in persons:
                    track = int(person.get("id", -1))
                    if track < 0:
                        continue
                    legacy_track_first.setdefault(track, t)
                    ratio = float(person.get("blood_ratio", 0.0))
                    blood = config.MEDICAL_BLOOD_MIN_RATIO <= ratio <= config.MEDICAL_BLOOD_MAX_RATIO
                    when = legacy_state.get(track)
                    legacy_state[track] = (when if when is not None else t) if blood else None
                    if (blood and legacy_state[track] is not None
                            and t - legacy_state[track] >= config.MEDICAL_BLOOD_SECONDS
                            and track not in legacy_emitted):
                        x1, y1, x2, y2 = map(int, person["box"])
                        overlaps = []
                        for other in persons:
                            if other is person:
                                continue
                            ox1, oy1, ox2, oy2 = map(int, other["box"])
                            inter = max(0, min(x2, ox2)-max(x1, ox1)) * max(0, min(y2, oy2)-max(y1, oy1))
                            if inter:
                                overlaps.append(other.get("id"))
                        legacy_events.append({"clip": cache["clip"], "frame": frame_number, "t": round(t, 3),
                            "track_id": track, "pose_label": person.get("pose_behavior"),
                            "pose_fall_signal": person.get("pose_behavior") == "Possible Fall",
                            "aspect_ratio": round((x2-x1)/max(1,y2-y1), 4),
                            "aspect_fall_signal": (x2-x1)/max(1,y2-y1) >= config.LYING_ASPECT,
                            "blood_ratio": round(ratio, 5), "blood_heuristic_signal": blood,
                            "box_width_px": x2-x1, "box_height_px": y2-y1,
                            "box_height_fraction": round((y2-y1)/max(1,int(row["height"])), 4),
                            "touches_frame_edge": x1 <= 0 or y1 <= 0 or x2 >= int(row["width"]) or y2 >= int(row["height"]),
                            "overlapping_people_count": len(overlaps), "overlapping_track_ids": overlaps,
                            "track_age_s": round(t-legacy_track_first[track], 3),
                            "legacy_cause": "blood heuristic alone"})
                        legacy_emitted.add(track)

        assessment = threat.evaluate(persons, weapons, behaviors)
        # A nearby weaponless person cannot cause this override; weapon-only
        # detections still create at least MEDIUM activity/evidence.
        if weapons and not any(_contact(w["box"], p["box"]) for w in weapons for p in persons):
            if assessment["level"] == "LOW":
                assessment = {**assessment, "level": "MEDIUM", "score": max(config.SCORE_LOW, assessment["score"]),
                              "reasons": list(dict.fromkeys(assessment["reasons"] + ["Weapon without nearby person"]))}
        if row["person_behavior_frame"] or row["weapon_inference_frame"]:
            _alerts_step(assessment, persons, weapons, medical, t, frame_number, state, person_gate)
        if include_timeline:
            timeline.append({"frame": frame_number, "t": t, "width": row["width"], "height": row["height"],
                "persons": copy.deepcopy(persons), "weapons": copy.deepcopy(weapons),
                "behaviors": list(behaviors), "assessment": copy.deepcopy(assessment),
                "medical": copy.deepcopy(medical)})

    diag = []
    for event in state["medical"]:
        match = next((d for d in medical_detector.diagnostics if d["track_id"] == event["track_id"]), {})
        diag.append({**event, "signal_diagnostics_at_final_frame": match})
    result = {"clip": cache["clip"], "folder": cache["folder"], "threshold": threshold,
            "class_thresholds": class_thresholds or {}, "max_area_ratio": max_area_ratio,
            "confirm_frames": int(confirm_frames if confirm_frames is not None else config.WEAPON_CONFIRM_FRAMES),
            "person_gate": person_gate, "detections_by_class": detections_by_class,
            "police_alerts": state["police"], "medical_alerts": diag,
            "medical_count": len(diag), "legacy_blood_diagnostic": legacy_events}
    if include_timeline:
        result["timeline"] = timeline
    return result


def _cache_paths(cache_dir):
    return sorted((p for p in Path(cache_dir).glob("t_*.json")),
                  key=lambda p: p.name.casefold())


def _save_legacy_frames(events, cache_dir):
    target_dir = Path(cache_dir) / "medical_fp"
    target_dir.mkdir(parents=True, exist_ok=True)
    for event in events:
        source = config.BASE_DIR / "videos" / "normal" / event["clip"]
        cap = cv2.VideoCapture(str(source))
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(event["frame"]) - 1))
        ok, raw = cap.read()
        cap.release()
        if not ok:
            continue
        scale = min(1.0, config.FRAME_WIDTH / max(1, raw.shape[1]))
        frame = cv2.resize(raw, (int(raw.shape[1]*scale), int(raw.shape[0]*scale))) if scale < 1 else raw
        cache_file = Path(cache_dir) / f"{Path(event['clip']).stem}.json"
        cache = _load(cache_file)
        record = cache["records"][int(event["frame"]) - 1]
        person = next((p for p in record["persons"] if int(p["id"]) == int(event["track_id"])), None)
        if person:
            x1, y1, x2, y2 = map(int, person["box"])
            cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,220), 3)
            cv2.putText(frame, f"legacy blood-only candidate {event['blood_ratio']:.1%}",
                        (x1, max(20,y1-7)), cv2.FONT_HERSHEY_SIMPLEX, .5, (255,0,220), 2)
        filename = f"{Path(event['clip']).stem}_track{event['track_id']}_f{event['frame']}.jpg"
        event["example_frame"] = str((target_dir / filename).relative_to(config.BASE_DIR))
        cv2.imwrite(str(target_dir / filename), frame)


def _verify(cache_paths, results):
    checks = []
    for path, result in zip(cache_paths, results):
        cache = _load(path)
        ref = cache["reference_alerts"]
        actual_police = [e["t"] for e in result["police_alerts"]]
        actual_medical = [{k: e[k] for k in ("t", "track_id", "severity", "reasons")} for e in result["medical_alerts"]]
        match = actual_police == ref["police_times_s"] and actual_medical == ref["medical_events"]
        checks.append({"clip": cache["clip"], "police_match": actual_police == ref["police_times_s"],
                       "medical_match": actual_medical == ref["medical_events"], "match": match,
                       "reference_police": ref["police_times_s"], "replay_police": actual_police,
                       "reference_medical": ref["medical_events"], "replay_medical": actual_medical})
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", default=str(config.BASE_DIR / "verification" / "cache"))
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--class-thresholds", help="JSON object mapping class name to confidence")
    parser.add_argument("--max-area-ratio", type=float)
    parser.add_argument("--confirm-frames", type=int)
    parser.add_argument("--person-gate", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--verify-reference", action="store_true")
    parser.add_argument("--legacy-blood-diagnostic", action="store_true")
    parser.add_argument("--save-legacy-frames", action="store_true")
    parser.add_argument("--output", default=str(config.BASE_DIR / "verification" / "replay_results.json"))
    args = parser.parse_args()
    class_thresholds = json.loads(args.class_thresholds) if args.class_thresholds else None
    paths = _cache_paths(args.cache_dir)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"No cache JSON files in {args.cache_dir}")
    caches = [_load(path) for path in paths]
    results = [replay_clip(cache, args.threshold, class_thresholds, args.max_area_ratio,
                           args.confirm_frames, args.person_gate, args.legacy_blood_diagnostic)
               for cache in caches]
    payload = {"options": vars(args), "results": results}
    if args.verify_reference:
        payload["reference_checks"] = _verify(paths, results)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    for result in results:
        print(f"[{result['folder']}] {result['clip']}: detections={result['detections_by_class']} "
              f"police={len(result['police_alerts'])} medical={result['medical_count']}", flush=True)
    for check in payload.get("reference_checks", []):
        print(f"[VERIFY] {check['clip']}: police_match={check['police_match']} "
              f"medical_match={check['medical_match']} all_match={check['match']}", flush=True)
    if args.legacy_blood_diagnostic:
        events = [event for result in results for event in result["legacy_blood_diagnostic"]]
        if args.save_legacy_frames:
            _save_legacy_frames(events, args.cache_dir)
        Path(args.output).with_name("medical_legacy_diagnostics.json").write_text(json.dumps(events, indent=2), encoding="utf-8")
        print(f"[MEDICAL LEGACY] blood-only candidates={len(events)}", flush=True)


if __name__ == "__main__":
    main()
