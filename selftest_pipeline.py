"""Repeatable synthetic video-loop test; no YOLO weights are loaded."""
from unittest.mock import patch
import numpy as np
import main

class FakeNotifier:
    def __init__(self): self.messages=[]
    def send(self, method, path, payload): self.messages.append((method,path,payload))
    def close(self): pass

class FakeEvidence:
    def __init__(self, **kwargs): self.recording=False; self.on_finish=None
    def save_image(self, frame, tag): return "images/mock.jpg"
    def update(self, frame, t, active): self.recording=active
    def close(self): pass

class FakeLogger:
    def log(self, payload): pass
    def update(self, incident_id, fields): pass

class FakeCapture:
    def __init__(self): self.i=0
    def isOpened(self): return True
    def get(self, key): return 0  # Exercise the safe source-FPS fallback.
    def read(self):
        self.i+=1
        if self.i>100: return False,None
        if self.i==5: return True,None  # A corrupt frame must be skipped.
        return True,np.zeros((240,320,3),dtype=np.uint8)
    def release(self): pass

class Persons:
    def __init__(self): self.i=0
    def detect(self, frame):
        self.i+=1
        box=(20,80,220,180) if self.i>=13 else (50,20,150,220)
        return frame,1,[{"id":1,"center":(100,100),"box":box}]
class Weapons:
    def __init__(self): self.i=0
    def detect(self, frame):
        self.i+=1
        if self.i==2: raise RuntimeError("synthetic detector failure")
        return [{"label":"pistol","confidence":.9,"center":(100,100),"box":(95,95,105,105)}] if 3<=self.i<13 else []
    def draw(self, frame, weapons): return frame.copy()
class Behavior:
    def __init__(self): self.i=0
    def detect(self, frame, persons, weapons, t):
        self.i+=1
        for person in persons:
            person["pose_behavior"]="Possible Fall" if self.i>=13 else "Normal"
            person["pose_confident_keypoints"]=10
            person["speed"]=0.0
        if self.i<3: return ["Standing"]
        if self.i<13: return ["Possible Attack"]
        return ["Fall Detected"]

def run():
    cap=FakeCapture()
    notifier=FakeNotifier()
    captured=[]; siren_calls=[]
    with patch.object(main,"get_video_source",return_value=cap), patch.object(main,"Notifier",return_value=notifier), patch.object(main,"EvidenceManager",FakeEvidence), patch.object(main,"IncidentLogger",FakeLogger):
        with patch("dispatcher.Dispatcher._siren",new=staticmethod(lambda kind:siren_calls.append(kind))):
            code=main.main(["mock.mp4"],detectors=(Persons(),Weapons(),Behavior()),display=False,max_frames=100,on_incident=captured.append)
    assert code==0 and cap.i==100
    police=[m for m in notifier.messages if m[2].get("type")=="POLICE"]
    assert len(police)==1, f"expected one confirmed police incident, got {len(police)}"
    medical=[m for m in notifier.messages if m[2].get("type")=="MEDICAL"]
    assert medical, "expected a medical incident after the scripted fall duration"
    assert any(x.get("type")=="POLICE" and x.get("level")=="HIGH" for x in captured),"police alert callback did not fire"
    assert "POLICE" in siren_calls,"police local siren trigger was not called"
    print("PASS: mocked loop processed normal -> held pistol -> fall; police/medical incidents created and local siren triggers called")
    run_web_analysis_flow()

def run_web_analysis_flow():
    """End-to-end web queue test with a clearly marked mocked detector fixture."""
    import tempfile
    import server
    import config
    import time
    from pathlib import Path
    from threading import Event
    from dispatcher import Dispatcher
    siren_calls=[]
    old=(server.STORE,server.incidents,server.gps_devices,server.JOBS,server.JOB_QUEUE,
         server.pipeline_thread,server.pipeline_state,server.pipeline_stop,
         server.TEST_DETECTORS_FACTORY,server.TEST_NOTIFIER_FACTORY)
    temp=tempfile.TemporaryDirectory(dir=config.BASE_DIR)
    server.STORE=Path(temp.name)/"web-incidents.json";server.incidents={};server.gps_devices={};server.JOBS={};server.JOB_QUEUE=server.deque();server.pipeline_thread=None;server.pipeline_stop=None
    server.pipeline_state={"status":"idle","mode":None,"job_id":None,"filename":None,"frame":0,"video_time":0.0,"fps":0.0,"error":None,"last_jpeg":None}
    server.TEST_DETECTORS_FACTORY=lambda:(Persons(),Weapons(),Behavior())
    class IncidentPoster(FakeNotifier):
        def send(self,method,path,payload):
            super().send(method,path,payload)
            if method=="POST" and path=="/api/incidents":
                with server.app.test_request_context(path,method=method,json=payload):server.create_incident()
                print(f"[WEB-TEST] incident POSTED {payload['id']} type={payload['type']} level={payload['level']}")
            elif method=="PATCH":
                with server.app.test_request_context(path,method=method,json=payload):server.patch_incident(payload.get("id",path.rsplit('/',1)[-1]))
    server.TEST_NOTIFIER_FACTORY=IncidentPoster
    try:
        fixture=Path(config.BASE_DIR)/"verification"/"cli_smoke.mp4"
        if not fixture.exists():raise AssertionError("mocked web-flow video fixture is missing")
        client=server.app.test_client()
        with patch.object(main,"get_video_source",side_effect=lambda *_:FakeCapture()),patch.object(main,"EvidenceManager",FakeEvidence),patch.object(main,"IncidentLogger",FakeLogger),patch("dispatcher.Dispatcher._siren",new=staticmethod(lambda kind:siren_calls.append(kind))):
            with fixture.open("rb") as stream:
                response=client.post("/api/analyze",data={"video":(stream,"mocked_weapon_fixture.mp4"),"mode":"full"},content_type="multipart/form-data")
            assert response.status_code==202,response.get_json()
            job_id=response.get_json()["job_id"]
            deadline=time.monotonic()+20
            while time.monotonic()<deadline:
                status=client.get(f"/api/jobs/{job_id}").get_json()
                if status["status"] in ("completed","error","cancelled"):break
                time.sleep(.02)
        assert status["status"]=="completed",status
        assert status["people_details"] and status["people_details"][0]["behavior"] in ("Unknown","Standing","Walking","Running","Aggressive Movement","Fall Detected","Raised Hand","Armed Person","Possible Attack"),status["people_details"]
        police=next((a for a in status["alerts"] if a["type"]=="POLICE" and a["level"]=="HIGH"),None)
        assert police,"mocked high police incident was not created by the web job"
        assert "POLICE" in siren_calls,"web-dispatched incident did not call local siren"
        dashboard=client.get("/api/incidents?type=POLICE").get_json()
        assert any(x["id"]==police["id"] for x in dashboard),"dashboard API did not return the created incident"
        print(f"[WEB-TEST] alert created {police['id']} -> siren trigger called -> incident POSTed -> police dashboard API returned it")
    finally:
        upload=next((Path(job["source"]) for job in server.JOBS.values() if job.get("mode")=="upload"),None)
        output=next((Path(job["output_path"]) for job in server.JOBS.values() if job.get("output_path")),None)
        if upload:upload.unlink(missing_ok=True)
        if output:output.unlink(missing_ok=True)
        (server.STORE,server.incidents,server.gps_devices,server.JOBS,server.JOB_QUEUE,
         server.pipeline_thread,server.pipeline_state,server.pipeline_stop,
         server.TEST_DETECTORS_FACTORY,server.TEST_NOTIFIER_FACTORY)=old
        temp.cleanup()

if __name__=="__main__": run()
