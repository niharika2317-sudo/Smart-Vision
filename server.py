"""Small demo dashboard and JSON API; intended for a trusted local network."""
import argparse
import atexit
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
from collections import deque
import uuid
import time
import json
import os
from urllib.request import Request, urlopen
from flask import Flask, request, jsonify, send_from_directory, render_template, Response, stream_with_context
from werkzeug.utils import secure_filename
import config
from video_source import camera_backend_candidates, camera_permission_hint

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024
DATA = config.BASE_DIR / "data"
DATA.mkdir(exist_ok=True)
STORE = DATA / "incidents.json"
LOCK = Lock()
incidents, gps_devices = {}, {}
PIPELINE_LOCK = Lock()
pipeline_thread = None
pipeline_stop = None
pipeline_state = {"status":"idle", "mode":None, "job_id":None, "filename":None,
                  "frame":0, "video_time":0.0, "fps":0.0, "error":None, "last_jpeg":None}
JOBS, JOB_QUEUE = {}, deque()
ACTIVE_JOB = None
MODEL_READY = False
CAMERA_STATUS = "not checked"
CAMERA_PROBE = {"ok":False,"first":None,"attempts":[],"reason":"not checked"}
TEST_DETECTORS_FACTORY = None  # Set only by selftest_pipeline's explicit mock fixture.
TEST_NOTIFIER_FACTORY = None
UPLOAD_DIR = config.BASE_DIR / "videos" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}

def _shutdown_pipeline():
    with PIPELINE_LOCK:
        for job in JOBS.values():
            if job.get("status") in ("queued", "running", "stopping"):
                job["stop_event"].set()
                if job.get("status")=="queued": job["status"]="cancelled"

atexit.register(_shutdown_pipeline)

def load_incidents():
    global incidents
    try:
        incidents = {x["id"]: x for x in json.loads(STORE.read_text(encoding="utf-8"))} if STORE.exists() else {}
    except Exception:
        incidents = {}
    return incidents

load_incidents()

def persist():
    temp = STORE.with_name(f"{STORE.name}.{uuid.uuid4().hex}.tmp")
    temp.write_text(json.dumps(list(incidents.values()), indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(temp, STORE)
            return
        except PermissionError:
            # Windows sync/virus scanners can briefly hold the destination open.
            if attempt == 4: raise
            time.sleep(0.05 * (attempt + 1))

    def telegram_push(item):
        """Optional best-effort push; network work never runs in the request handler."""
        token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat = os.environ.get(
            "TELEGRAM_POLICE_CHAT"
            if item.get("type") == "POLICE"
            else "TELEGRAM_HOSPITAL_CHAT"
        )
        if not token or not chat:
            return

        body = json.dumps(
            {
                "chat_id": chat,
                "text": (
                    f"VantaWatch {item['type']} incident: "
                    f"{item.get('level')} score {item.get('score')}"
                ),
            }
        ).encode()

        req = Request(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data=body,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urlopen(req, timeout=3) as response:
                response.read()
        except Exception:
            pass


@app.get("/api/incidents")
def get_incidents():
    kind = request.args.get("type", "").upper()

    with LOCK:
        # Reloads the saved incidents whenever the Alerts page refreshes.
        load_incidents()

        values = [
            item
            for item in incidents.values()
            if kind in ("", item.get("type"))
        ]

    return jsonify(
        sorted(
            values,
            key=lambda item: item.get(
                "created_at",
                item.get("timestamp", ""),
            ),
            reverse=True,
        )[:100]
    )

@app.post("/api/incidents")
def create_incident():
    data = request.get_json(silent=True) or {}
    if data.get("type") not in ("POLICE", "MEDICAL") or not data.get("id"): return jsonify(error="invalid incident"), 400
    with LOCK:
        if data["id"] in incidents: return jsonify(incidents[data["id"]]), 200
        allowed = {k:data[k] for k in ("id","type","level","score","weapon_score","behavior_score","reasons","behaviors","weapons","camera_id","camera_name","image_path","video_path","timestamp","video_time","medical_cases","analysis_mode","simulated") if k in data}
        fix = gps_devices.get(config.CAMERA_ID)
        live = fix and (datetime.now(timezone.utc).timestamp()-fix["timestamp"] < config.GPS_LIVE_SECONDS)
        allowed.update(status="NEW", created_at=datetime.now(timezone.utc).isoformat(), lat=fix["lat"] if live else config.CAMERA_LAT,
                       lon=fix["lon"] if live else config.CAMERA_LON, location_source="live-gps" if live else "camera-fixed", status_times={"NEW":datetime.now(timezone.utc).isoformat()})
        incidents[data["id"]] = allowed; persist()
        Thread(target=telegram_push, args=(allowed,), daemon=True).start()
    return jsonify(allowed), 201

@app.patch("/api/incidents/<incident_id>")
def patch_incident(incident_id):
    data=request.get_json(silent=True) or {}
    with LOCK:
        item=incidents.get(incident_id)
        if not item: return jsonify(error="not found"),404
        status=data.get("status")
        allowed_next={"NEW":"ACKNOWLEDGED","ACKNOWLEDGED":"DISPATCHED","DISPATCHED":"RESOLVED"}
        if status:
            if allowed_next.get(item["status"]) != status: return jsonify(error="invalid status transition"),400
            item["status"]=status; item.setdefault("status_times",{})[status]=datetime.now(timezone.utc).isoformat()
        if "video_path" in data: item["video_path"]=data["video_path"]
        persist(); return jsonify(item)

@app.post("/api/gps")
def post_gps():
    d=request.get_json(silent=True) or {}
    try: lat,lon,acc=float(d["lat"]),float(d["lon"]),float(d.get("accuracy",0))
    except (KeyError,TypeError,ValueError): return jsonify(error="invalid GPS"),400
    if not (-90<=lat<=90 and -180<=lon<=180 and acc>=0) or not d.get("device_id"): return jsonify(error="invalid GPS"),400
    gps_devices[str(d["device_id"]) ]={"device_id":str(d["device_id"]),"lat":lat,"lon":lon,"accuracy":acc,"timestamp":datetime.now(timezone.utc).timestamp()}
    return jsonify(ok=True),201

@app.get("/api/gps")
def get_gps():
    now=datetime.now(timezone.utc).timestamp()
    return jsonify([{**v,"age":now-v["timestamp"],"live":now-v["timestamp"]<config.GPS_LIVE_SECONDS} for v in gps_devices.values()])

@app.get("/api/system/status")
def system_status():
    now=datetime.now(timezone.utc).timestamp()
    live=sum(1 for d in gps_devices.values() if now-d["timestamp"]<config.GPS_LIVE_SECONDS)
    with PIPELINE_LOCK:
        active=JOBS.get(ACTIVE_JOB) if ACTIVE_JOB else None
        camera="running" if active and active.get("mode")=="camera" else CAMERA_STATUS
    return jsonify(server="online",model_ready=MODEL_READY,camera=camera,gps_live=live)

def probe_camera_devices():
    import cv2
    backends=camera_backend_candidates(cv2)
    attempts=[];first=None
    for index in range(4):
        for name,backend in backends:
            try:
                cap=cv2.VideoCapture(index,backend)
                opened=bool(cap.isOpened())
                cap.release()
            except Exception as exc:
                opened=False;attempts.append({"index":index,"backend":name,"opened":False,"error":str(exc)})
                continue
            attempts.append({"index":index,"backend":name,"opened":opened,"error":None if opened else "device did not open"})
            if opened and first is None:first={"index":index,"backend":name}
    reason="Camera opened." if first else camera_permission_hint()
    return {"ok":first is not None,"first":first,"attempts":attempts,"reason":reason}

@app.get("/api/camera/probe")
def camera_probe():
    global CAMERA_PROBE, CAMERA_STATUS
    CAMERA_PROBE=probe_camera_devices();CAMERA_STATUS="available" if CAMERA_PROBE["ok"] else "unavailable"
    return jsonify(CAMERA_PROBE)

@app.get("/evidence/<path:filename>")
def evidence(filename):
    return send_from_directory(config.EVIDENCE_DIR, filename)

@app.get("/")
def index(): return render_template("index.html",dev=request.args.get("dev")=="1")

@app.get("/police")
def police(): return render_template("dashboard.html",role="POLICE",center=[config.CAMERA_LAT,config.CAMERA_LON],offline=request.args.get("offline")=="1")

@app.get("/hospital")
def hospital(): return render_template("dashboard.html",role="MEDICAL",center=[config.CAMERA_LAT,config.CAMERA_LON],offline=request.args.get("offline")=="1")

@app.get("/gps")
def gps_page(): return render_template("gps.html")

@app.get("/monitoring")
def monitoring_page(): return render_template("monitoring.html")

@app.get("/alerts")
def alerts_page():
    role=request.args.get("type","POLICE").upper()
    if role not in ("POLICE","MEDICAL"): role="POLICE"
    return render_template("alerts.html",role=role)

@app.get("/cases")
def cases_page(): return render_template("workspace.html",section="cases",title="Case Management",subtitle="Review and advance incident response cases.")

@app.get("/evidence")
def evidence_page(): return render_template("workspace.html",section="evidence",title="Evidence Vault",subtitle="Browse image and video evidence attached to incidents.")

@app.get("/analytics")
def analytics_page(): return render_template("workspace.html",section="analytics",title="Analytics",subtitle="A summary of recorded incident data from this server.")

@app.get("/system")
def system_page(): return render_template("workspace.html",section="system",title="System Status",subtitle="Current dashboard, analysis and GPS service state.")

@app.get("/admin")
def admin_page(): return render_template("workspace.html",section="system",title="Admin Console",subtitle="Review the analysis service, camera status and incident storage.")

def _pipeline_snapshot():
    with PIPELINE_LOCK:
        return {**{k:v for k,v in pipeline_state.items() if k != "last_jpeg"},
                "has_frame":pipeline_state.get("last_jpeg") is not None}

def _public_job(job):
    total=job.get("total_frames") or 0
    frame=job.get("frame",0)
    target=job.get("target_frames") or total
    percent=min(100,round(frame*100/target)) if target else (100 if job["status"]=="completed" else 0)
    fps=job.get("fps",0) or 0
    return {k:job.get(k) for k in ("job_id","filename","mode","status","frame","video_time","max_frames","total_frames","fps","threat","max_threat","score","weapon_score","behavior_score","persons","weapons","behaviors","people_details","alerts","error","queued_position","police_alerts","medical_alerts","first_alert_time","first_frame_seconds","model_load_seconds","output_ready")} | {"percent":percent,"estimated_seconds_left":max(0,round((target-frame)/fps)) if fps else None,"output_url":f"/api/jobs/{job['job_id']}/output" if job.get("output_ready") else None}

def _start_pipeline(source, mode, label, max_frames=None):
    global pipeline_thread, pipeline_stop, pipeline_state, ACTIVE_JOB
    job_id=uuid.uuid4().hex; stop_event=Event(); total=0
    if mode=="upload":
        try:
            import cv2
            cap=cv2.VideoCapture(str(source)); total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); cap.release()
        except Exception: pass
    target=min(total,max_frames) if max_frames and total else (max_frames or total)
    output_dir=config.BASE_DIR/"videos"/"outputs";output_dir.mkdir(parents=True,exist_ok=True)
    output_path=output_dir/f"{job_id}.mp4"
    job={"job_id":job_id,"filename":label,"mode":mode,"status":"queued","frame":0,"video_time":0.0,"fps":0.0,"total_frames":total,"target_frames":target,"threat":"LOW","max_threat":"LOW","score":0,"weapon_score":0,"behavior_score":0,"persons":0,"weapons":0,"behaviors":[],"people_details":[],"alerts":[],"police_alerts":0,"medical_alerts":0,"first_alert_time":None,"first_frame_seconds":None,"model_load_seconds":None,"output_ready":False,"error":None,"last_jpeg":None,"stop_event":stop_event,"source":source,"max_frames":max_frames,"output_path":output_path,"created_monotonic":time.monotonic()}
    with PIPELINE_LOCK:
        JOBS[job_id]=job; JOB_QUEUE.append(job_id)
        job["queued_position"]=len(JOB_QUEUE)
        if pipeline_thread is None or not pipeline_thread.is_alive():
            pipeline_thread=Thread(target=_job_worker,name="analysis-queue",daemon=True); pipeline_thread.start()
    return job_id

def _job_worker():
    global pipeline_stop, pipeline_state, ACTIVE_JOB, MODEL_READY, CAMERA_STATUS
    while True:
        with PIPELINE_LOCK:
            while JOB_QUEUE and JOBS[JOB_QUEUE[0]]["status"]=="cancelled": JOB_QUEUE.popleft()
            if not JOB_QUEUE: ACTIVE_JOB=None; return
            job_id=JOB_QUEUE.popleft(); job=JOBS[job_id]; ACTIVE_JOB=job_id; job["status"]="running"; job["queued_position"]=0
            pipeline_stop=job["stop_event"]
            pipeline_state={k:job.get(k) for k in ("status","mode","job_id","filename","frame","video_time","fps","error","last_jpeg")}
        def publish(frame, video_time, frame_number, source_fps, metrics=None):
            global MODEL_READY, CAMERA_STATUS
            import cv2
            ok,encoded=cv2.imencode(".jpg",frame,[int(cv2.IMWRITE_JPEG_QUALITY),82])
            metrics=metrics or {}; threat=metrics.get("threat",{})
            with PIPELINE_LOCK:
                if job["first_frame_seconds"] is None: job["first_frame_seconds"]=round(time.monotonic()-job["created_monotonic"],2)
                MODEL_READY=True
                if job["mode"]=="camera": CAMERA_STATUS="available"
                rank={"LOW":0,"MEDIUM":1,"HIGH":2}; level=threat.get("level","LOW")
                if rank.get(level,0)>rank.get(job["max_threat"],0):job["max_threat"]=level
                job.update(status="stopping" if job["stop_event"].is_set() else "running",frame=int(frame_number),video_time=float(video_time),fps=float(metrics.get("processing_fps",0)),video_fps=float(source_fps or 0),last_jpeg=encoded.tobytes() if ok else None,threat=level,score=threat.get("score",0),weapon_score=threat.get("weapon_score",0),behavior_score=threat.get("behavior_score",0),persons=metrics.get("persons",0),weapons=metrics.get("weapons",0),behaviors=metrics.get("behaviors",[]),people_details=metrics.get("people_details",[]))
                pipeline_state={k:job.get(k) for k in ("status","mode","job_id","filename","frame","video_time","fps","error","last_jpeg")}
        def incident(item):
            with PIPELINE_LOCK:
                job["alerts"].append(item)
                job["police_alerts"]+=int(item.get("type")=="POLICE")
                job["medical_alerts"]+=int(item.get("type")=="MEDICAL")
                if job["first_alert_time"] is None: job["first_alert_time"]=float(item.get("video_time",job.get("video_time",0.0)))
        try:
            from main import main as run_pipeline
            argv=[str(job["source"]),"--no-display"]
            if job.get("max_frames"): argv += ["--max-frames",str(job["max_frames"])]
            argv += ["--save-annotated",str(job["output_path"])]
            from main import build_detector_instances,model_cache_seconds
            detectors=TEST_DETECTORS_FACTORY() if TEST_DETECTORS_FACTORY else build_detector_instances()
            if not TEST_DETECTORS_FACTORY: MODEL_READY=True
            job["model_load_seconds"]=model_cache_seconds()
            notifier=TEST_NOTIFIER_FACTORY() if TEST_NOTIFIER_FACTORY else None
            code=run_pipeline(argv=argv,detectors=detectors,display=False,stop_event=job["stop_event"],on_frame=publish,source_name=job["filename"],on_incident=incident,notifier_instance=notifier)
            status="cancelled" if job["stop_event"].is_set() else "completed" if code==0 else "error"
            with PIPELINE_LOCK:
                job["status"]=status
                job["output_ready"]=job["output_path"].exists() and job["output_path"].stat().st_size>0
                if job["mode"]=="camera" and status=="error": CAMERA_STATUS="unavailable"
                if code and status=="error": job["error"]=(camera_permission_hint() if job["mode"]=="camera" else "Could not open the video. Check its format and that the file is readable.")
                pipeline_state={k:job.get(k) for k in ("status","mode","job_id","filename","frame","video_time","fps","error","last_jpeg")}
        except Exception as exc:
            with PIPELINE_LOCK: job["status"]="error"; job["error"]=str(exc)
        finally:
            with PIPELINE_LOCK: ACTIVE_JOB=None

@app.get("/api/jobs")
def list_jobs():
    with PIPELINE_LOCK: values=sorted(JOBS.values(),key=lambda j:j.get("created_monotonic",0),reverse=True)
    return jsonify([_public_job(job) for job in values])

@app.post("/api/jobs/<job_id>/skip")
def skip_job(job_id): return cancel_job(job_id)

@app.post("/api/jobs/clear-queued")
def clear_queued_jobs():
    count=0
    with PIPELINE_LOCK:
        for job_id in list(JOB_QUEUE):
            job=JOBS.get(job_id)
            if job and job["status"]=="queued": job["status"]="cancelled";count+=1
    return jsonify(ok=True,skipped=count)

@app.get("/api/jobs/<job_id>/output")
def job_output(job_id):
    with PIPELINE_LOCK: job=JOBS.get(job_id)
    if not job:return jsonify(error="job not found"),404
    if not job.get("output_ready"):return jsonify(error="annotated video is not ready"),404
    return send_from_directory(job["output_path"].parent,job["output_path"].name,conditional=True)

@app.get("/api/jobs/<job_id>")
def job_status(job_id):
    with PIPELINE_LOCK: job=JOBS.get(job_id)
    return (jsonify(_public_job(job)),200) if job else (jsonify(error="job not found"),404)

@app.post("/api/jobs/<job_id>/cancel")
def cancel_job(job_id):
    with PIPELINE_LOCK:
        job=JOBS.get(job_id)
        if not job: return jsonify(error="job not found"),404
        if job["status"]=="queued": job["status"]="cancelled"
        elif job["status"]=="running": job["stop_event"].set(); job["status"]="stopping"
        else: return jsonify(error="job is not active"),409
    return jsonify(ok=True,status=job["status"]),202

@app.get("/api/jobs/<job_id>/preview")
def job_preview(job_id):
    with PIPELINE_LOCK: job=JOBS.get(job_id); frame=job.get("last_jpeg") if job else None
    if not job: return jsonify(error="job not found"),404
    if not frame: return jsonify(error="no annotated frame available yet"),404
    return Response(frame,mimetype="image/jpeg",headers={"Cache-Control":"no-store"})

@app.get("/api/pipeline/status")
def pipeline_status(): return jsonify(_pipeline_snapshot())

@app.get("/api/pipeline/frame.jpg")
def pipeline_frame():
    with PIPELINE_LOCK: frame=pipeline_state.get("last_jpeg")
    if not frame: return jsonify(error="no analysis frame available"),404
    return Response(frame,mimetype="image/jpeg",headers={"Cache-Control":"no-store"})

@app.post("/api/pipeline/stop")
def stop_pipeline():
    with PIPELINE_LOCK:
        if pipeline_stop is None or pipeline_thread is None or not pipeline_thread.is_alive():
            return jsonify(error="no analysis is running"),409
        pipeline_stop.set()
        pipeline_state["status"]="stopping"
    return jsonify(ok=True,status="stopping"),202

@app.post("/api/live/start")
def start_live_camera():
    data=request.get_json(silent=True) or {}
    source=str(data.get("camera_index",CAMERA_PROBE.get("first",{}).get("index",0)))
    if not source.isdigit() or int(source)>3: return jsonify(error="camera_index must be between 0 and 3"),400
    job_id=_start_pipeline(source,"camera",f"Live camera {source}")
    if not job_id: return jsonify(error="an analysis is already running"),409
    return jsonify(ok=True,job_id=job_id,status="starting"),202

@app.get("/api/live/stream")
def live_stream():
    @stream_with_context
    def frames():
        last=None
        while True:
            with PIPELINE_LOCK:
                current=pipeline_state.get("last_jpeg")
                active=pipeline_state.get("status") in ("starting","running","stopping")
            if current is not None and current is not last:
                last=current
                yield b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "+str(len(current)).encode()+b"\r\n\r\n"+current+b"\r\n"
            elif not active:
                return
            time.sleep(.08)
    return Response(frames(),mimetype="multipart/x-mixed-replace; boundary=frame",headers={"Cache-Control":"no-store","X-Accel-Buffering":"no"})

@app.post("/api/analysis/upload")
@app.post("/api/analyze")
def upload_analysis_video():
    video=request.files.get("video")
    if video is None or not video.filename: return jsonify(error="Choose a video file first."),400
    safe_name=secure_filename(video.filename)
    suffix=Path(safe_name).suffix.lower()
    if suffix not in VIDEO_EXTENSIONS: return jsonify(error="Use an MP4, M4V, MOV, AVI, MKV or WEBM video."),400
    job_id=uuid.uuid4().hex
    path=UPLOAD_DIR/f"{job_id[:10]}_{safe_name}"
    video.save(path)
    if path.stat().st_size == 0:
        path.unlink(missing_ok=True)
        return jsonify(error="The uploaded file was empty."),400
    mode=(request.form.get("mode") or "fast").lower()
    try:requested_frames=int(request.form.get("max_frames",300))
    except (TypeError,ValueError):return jsonify(error="max_frames must be a positive integer"),400
    max_frames=None if mode=="full" else requested_frames
    if mode not in ("fast","full"):
        path.unlink(missing_ok=True); return jsonify(error="mode must be fast or full"),400
    if mode=="fast" and not 1<=max_frames<=100000:
        path.unlink(missing_ok=True);return jsonify(error="fast mode max_frames must be between 1 and 100000"),400
    started=_start_pipeline(path,"upload",safe_name,max_frames=max_frames)
    return jsonify(ok=True,job_id=started,filename=safe_name,status="queued",mode=mode),202

@app.post("/api/demo/simulate")
def simulate_incident():
    if request.args.get("dev")!="1": return jsonify(error="developer simulation requires /?dev=1"),404
    data=request.get_json(silent=True) or {}; kind=str(data.get("type","" )).upper()
    if kind not in ("POLICE","MEDICAL"): return jsonify(error="type must be POLICE or MEDICAL"),400
    item={"id":"sim-"+uuid.uuid4().hex,"type":kind,"level":"HIGH" if kind=="POLICE" else "MEDIUM","score":0,"reasons":["Explicitly simulated for interface demonstration"],"behaviors":[],"weapons":[],"simulated":True,"analysis_mode":"simulated demonstration"}
    with app.test_request_context("/api/incidents",method="POST",json=item):
        return create_incident()

@app.errorhandler(413)
def upload_too_large(_error): return jsonify(error="Video exceeds the 500 MB upload limit."),413

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--https",action="store_true");parser.add_argument("--port",type=int,default=config.PORT);args=parser.parse_args()
    config.SERVER_URL=f"{'https' if args.https else 'http'}://127.0.0.1:{args.port}"
    app.run(host="0.0.0.0",port=args.port,threaded=True,ssl_context="adhoc" if args.https else None)
