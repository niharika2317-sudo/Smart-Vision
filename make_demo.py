"""Build or time-post a cached, recorded clip replay (never a live feed)."""
import argparse
import json
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import cv2
import config
from replay import _load, replay_clip


DEFAULT_CACHE = config.BASE_DIR / "verification" / "cache" / "t_n003_converted.json"
DEFAULT_OUTPUT = config.BASE_DIR / "verification" / "offline_demo" / "normal_quiet.mp4"


def _incident_events(cache, replayed, start_utc):
    by_frame = {item["frame"]: item for item in replayed["timeline"]}
    events = []
    for event in replayed["police_alerts"]:
        item = by_frame[event["frame"]]
        events.append({"type":"POLICE", "frame":event["frame"], "t":event["t"],
                       "assessment":item["assessment"], "persons":item["persons"],
                       "weapons":item["weapons"], "behaviors":item["behaviors"], "medical":[],
                       "timestamp":(start_utc+timedelta(seconds=event["t"])).isoformat()})
    for event in replayed["medical_alerts"]:
        item = by_frame[event["frame"]]
        case = next((c for c in item["medical"] if int(c["track_id"]) == int(event["track_id"])),
                    {"track_id":event["track_id"],"severity":event["severity"],"reasons":event["reasons"],"box":None})
        events.append({"type":"MEDICAL", "frame":event["frame"], "t":event["t"],
                       "assessment":item["assessment"], "persons":item["persons"],
                       "weapons":item["weapons"], "behaviors":item["behaviors"], "medical":[case],
                       "timestamp":(start_utc+timedelta(seconds=event["t"])).isoformat()})
    return sorted(events, key=lambda x:(x["t"],x["type"]))


def _build_video(cache_path, output_path):
    cache = _load(cache_path)
    replayed = replay_clip(cache, threshold=config.WEAPON_CONFIDENCE,
        class_thresholds=config.WEAPON_CLASS_CONFIDENCE,
        max_area_ratio=config.WEAPON_MAX_BOX_AREA_RATIO,
        confirm_frames=config.WEAPON_CONFIRM_FRAMES,
        person_gate=config.WEAPON_REQUIRE_PERSON_CONTACT_FOR_POLICE,
        include_timeline=True)
    source = config.BASE_DIR / "videos" / cache["folder"] / cache["clip"]
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open staged clip: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or config.EVIDENCE_FPS_FALLBACK)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, config.FRAME_WIDTH / max(1, width))
    out_size = (int(width*scale), int(height*scale))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size)
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create output video: {output_path}")
    event_frames = {int(e["frame"]): e for e in _incident_events(cache, replayed, datetime.now(timezone.utc))}
    frame_index = 0
    while frame_index < len(replayed["timeline"]):
        ok, raw = cap.read()
        if not ok:
            break
        frame_index += 1
        frame = cv2.resize(raw, out_size) if scale < 1 else raw
        row = replayed["timeline"][frame_index-1]
        for person in row["persons"]:
            x1,y1,x2,y2=map(int,person["box"])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,255),2)
            cv2.putText(frame,f"person {person['id']}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,220,255),1)
        for weapon in row["weapons"]:
            x1,y1,x2,y2=map(int,weapon["box"])
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,f"{weapon['label']} {weapon['confidence']:.2f}",(x1,max(18,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,255),1)
        for case in row["medical"]:
            if case.get("box"):
                x1,y1,x2,y2=map(int,case["box"])
                cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,220),2)
        cv2.rectangle(frame,(0,0),(frame.shape[1],30),(20,20,20),-1)
        cv2.putText(frame,"RECORDED CLIP ANALYSED OFFLINE",(8,21),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),1)
        if frame_index in event_frames:
            cv2.putText(frame,f"{event_frames[frame_index]['type']} alert at video t={row['t']:.2f}s",
                        (8,frame.shape[0]-12),cv2.FONT_HERSHEY_SIMPLEX,.52,(0,0,255),2)
        writer.write(frame)
    cap.release(); writer.release()
    anchor = datetime.now(timezone.utc)
    events = _incident_events(cache, replayed, anchor)
    manifest = {"label":"recorded clip analysed offline", "source_clip":cache["clip"],
        "folder":cache["folder"], "fps":fps, "frames_written":frame_index,
        "video_time_origin_utc":anchor.isoformat(), "events":events,
        "police_alert_count":len(replayed["police_alerts"]),
        "medical_alert_count":len(replayed["medical_alerts"]),
        "visually_confirmed_genuine_weapon":False}
    manifest_path = output_path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(f"Wrote {frame_index} frames at {fps:.3f} FPS to {output_path}")
    print(f"Alert events from cached detector pass: police={manifest['police_alert_count']} medical={manifest['medical_alert_count']}")
    print("Label: RECORDED CLIP ANALYSED OFFLINE")
    return manifest_path


def _post_timed(manifest_path):
    from notifier import Notifier
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    started = time.monotonic()
    video_origin_utc = datetime.now(timezone.utc)
    notifier = Notifier()
    for event in manifest["events"]:
        delay = max(0.0, float(event["t"]) - (time.monotonic() - started))
        if delay:
            time.sleep(delay)
        assessment = event["assessment"]
        payload = {"id":f"offline-{uuid.uuid4()}", "type":event["type"],
            "level":assessment["level"], "score":assessment["score"],
            "weapon_score":assessment["weapon_score"], "behavior_score":assessment["behavior_score"],
            "reasons":assessment["reasons"]+[r for c in event["medical"] for r in c["reasons"]],
            "behaviors":event["behaviors"], "weapons":event["weapons"],
            "camera_id":config.CAMERA_ID, "camera_name":config.CAMERA_NAME,
            "lat":config.CAMERA_LAT,"lon":config.CAMERA_LON,
            "image_path":None,"video_path":None,
            "timestamp":(video_origin_utc+timedelta(seconds=float(event["t"]))).isoformat(),
            "medical_cases":event["medical"],"analysis_mode":"recorded clip analysed offline"}
        notifier.send("POST","/api/incidents",payload)
        print(f"Posted {event['type']} event for recorded video time {event['t']:.2f}s")
    notifier.close()
    if not manifest["events"]:
        print("No cached alerts to post; no alert was fabricated.")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",default=str(DEFAULT_CACHE))
    parser.add_argument("--output",default=str(DEFAULT_OUTPUT))
    parser.add_argument("--post-only",action="store_true",help="time-post events from the adjacent manifest during video playback")
    parser.add_argument("--build-only",action="store_true",help="build the annotated clip and event manifest without posting")
    args=parser.parse_args()
    output=Path(args.output)
    manifest_path=output.with_suffix(".json")
    if not args.post_only:
        manifest_path=_build_video(Path(args.cache),output)
    if not args.build_only:
        _post_timed(manifest_path)


if __name__=="__main__":
    main()
