"""Quick synthetic checks for model-independent project logic."""
import unittest
import json
import tempfile
import time
import io
from unittest.mock import patch
from pathlib import Path
from behavior_rules import BehaviorRules
from behavior_detector import BehaviorDetector
from threat_engine import ThreatEngine
from utils import box_iou, point_in_box
from medical_detector import MedicalDetector
from behavior_pipeline import BehaviorPipeline
from evidence_manager import EvidenceManager
from dispatcher import Dispatcher
from notifier import Notifier
from replay_outbox import replay
from weapon_detector import WeaponPersistence, filter_weapon_detections
import config
import numpy as np

def pose(lying=False, raised=False):
    p=[[0.,0.] for _ in range(17)]
    p[5]=[10 if lying else 40,100];p[6]=[30 if lying else 60,100];p[11]=[70 if lying else 45,100 if lying else 180];p[12]=[90 if lying else 55,100 if lying else 180]
    p[9]=[40,70 if raised else 140];p[10]=[60,140]
    return p

def medical_person(track=1, box=(20,20,80,60), speed=0.1, pose_label="Possible Fall", keypoints=8):
    return {"id":track,"box":box,"speed":speed,"pose_behavior":pose_label,
            "pose_confident_keypoints":keypoints}

class CoreTests(unittest.TestCase):
    def test_mps_fallback_and_precision_options(self):
        old_device=config._RUNTIME_DEVICE
        calls=[]
        def operation(device):
            calls.append(device)
            if device=="mps":raise RuntimeError("synthetic MPS operator failure")
            return "cpu result"
        try:
            with patch("config.resolve_device",return_value="mps"):
                self.assertEqual(config.run_model_with_device_fallback(operation),"cpu result")
            self.assertEqual(calls,["mps","cpu"])
            self.assertNotIn("half",config.model_options("mps",640))
            self.assertTrue(config.model_options(0,640)["half"])
        finally:
            config._RUNTIME_DEVICE=old_device

    def test_monitoring_layout_and_siren_auto_unlock_hooks(self):
        root=Path(__file__).resolve().parent
        html=(root/"templates"/"monitoring.html").read_text(encoding="utf-8")
        css=(root/"static"/"css"/"monitoring.css").read_text(encoding="utf-8")
        js=(root/"static"/"js"/"siren.js").read_text(encoding="utf-8")
        self.assertIn("Choose videos",html);self.assertIn("Threat engine",html);self.assertIn("fullscreen-video",html)
        self.assertIn('id="selected-files"',html);self.assertIn('class="video-file-input" type="file"',html)
        self.assertIn("object-fit:contain",css);self.assertIn("grid-template-columns:minmax(0,1fr) 300px",css)
        self.assertIn("@media(max-width:800px)",css);self.assertIn("@media(max-width:430px)",css)
        self.assertIn("document.addEventListener('pointerdown',unlock",js);self.assertIn("document.addEventListener('keydown',unlock",js)
        self.assertIn("CRITICAL - click anywhere to enable siren",js);self.assertIn("observeJobAlerts",js);self.assertIn("acknowledge",js)

    def test_pose_rules(self):
        r=BehaviorRules(); self.assertEqual(r.detect(pose()),"Normal");self.assertEqual(r.detect(pose(lying=True)),"Possible Fall");self.assertEqual(r.detect(pose(raised=True)),"Raised Hand")
        conf=[1.]*17;conf[5]=0;self.assertEqual(r.detect(pose(),conf),"Unknown")
    def test_speed_units_and_resolution(self):
        for scale in (1,2):
            d=BehaviorDetector(); result="Standing"
            for sec in range(2): result,speed=d.analyze(1,(sec*100*scale,0,(sec*100+50)*scale,100*scale),sec)
            self.assertAlmostEqual(speed,1.0,places=2);self.assertEqual(result,"Walking")
        for distance,label in ((0,"Standing"),(160,"Running"),(350,"Aggressive Movement")):
            d=BehaviorDetector()
            d.analyze(1,(0,0,100,100),0)
            result,_=d.analyze(1,(distance,0,distance+100,100),1)
            self.assertEqual(result,label)
    def test_motion_path_detects_direction_changes_without_lowering_speed_thresholds(self):
        detector=BehaviorDetector();label="Unknown"
        for t,x in ((0,0),(.25,70),(.5,0),(.75,70)):
            label,speed=detector.analyze(1,(x,0,x+40,100),t)
        self.assertEqual(label,"Running");self.assertGreaterEqual(speed,config.RUNNING_SPEED)
    def test_box_jitter_and_untracked_people_are_not_called_running_or_standing(self):
        detector=BehaviorDetector()
        self.assertEqual(detector.analyze(-1,(0,0,40,100),0),("Unknown",0.0))
        self.assertEqual(detector.analyze(1,(0,0,40,100),0)[0],"Unknown")
        for t,x in ((.1,5),(.2,-5),(.3,4)):
            label,speed=detector.analyze(1,(x,0,x+40,100),t)
        self.assertEqual(label,"Standing");self.assertLess(speed,config.WALKING_SPEED)
    def test_iou_and_holding(self):
        self.assertAlmostEqual(box_iou((0,0,10,10),(0,0,10,10)),1);self.assertTrue(point_in_box((11,5),(0,0,10,10),.15));self.assertFalse(point_in_box((20,5),(0,0,10,10),.15))
    def test_threat_expected_cases(self):
        e=ThreatEngine();p=[{"box":(0,0,100,200)}];w=[{"center":(50,100),"box":(45,95,55,105),"confidence":.5,"label":"pistol"}]
        self.assertEqual(e.evaluate(p,w,["Armed Person"])["level"],"HIGH")
        self.assertEqual(e.evaluate([],w,[])["level"],"MEDIUM")
        self.assertEqual(e.evaluate([],[],["Running"])["level"],"LOW")
        self.assertEqual(e.evaluate([],[],["Aggressive Movement"])["level"],"MEDIUM")
        candidate=e.evaluate(p,w,["Armed Person"])
        self.assertIn("Weapon model candidate · review required",candidate["reasons"])
        self.assertIn("Candidate overlaps a person",candidate["reasons"])
        self.assertEqual(e.evaluate(p,w,["Possible Attack"])["level"],"HIGH")
        self.assertEqual(e.evaluate([],[],[])["level"],"LOW")
        no_person=e.evaluate([], [{"center":(500,500),"box":(490,490,510,510),"confidence":.5,"label":"pistol"}], [])
        self.assertEqual(no_person["level"],"MEDIUM")
        self.assertIn("Weapon candidate without nearby person",no_person["reasons"])
    def test_medical_sitting_person_is_not_down(self):
        d=MedicalDetector();frame=np.zeros((100,100,3),dtype=np.uint8)
        sitting=medical_person(box=(20,10,65,95),pose_label="Normal",keypoints=12)
        d.detect(frame,[sitting],["Normal"],0)
        self.assertEqual(d.detect(frame,[sitting],["Normal"],3.2),[])

    def test_medical_heavy_crowd_overlap_is_ignored(self):
        d=MedicalDetector();frame=np.zeros((100,100,3),dtype=np.uint8)
        one=medical_person(track=1,box=(20,20,80,60))
        two=medical_person(track=2,box=(30,20,90,60))
        for t in (0,1.1,3.2):
            self.assertEqual(d.detect(frame,[one,two],["Possible Fall","Possible Fall"],t),[])
        self.assertGreater(d.diagnostics[0]["max_overlap_fraction"],config.MEDICAL_MAX_OCCLUSION_RATIO)

    def test_medical_frame_edge_person_is_ignored(self):
        d=MedicalDetector();frame=np.zeros((100,100,3),dtype=np.uint8)
        clipped=medical_person(box=(0,20,60,60))
        for t in (0,1.1,3.2):
            self.assertEqual(d.detect(frame,[clipped],["Possible Fall"],t),[])
        self.assertTrue(d.diagnostics[0]["touches_frame_edge"])

    def test_medical_true_lying_person_alerts_after_fall_confirm_sec(self):
        d=MedicalDetector();frame=np.zeros((100,100,3),dtype=np.uint8)
        lying=medical_person()
        self.assertEqual(d.detect(frame,[lying],["Possible Fall"],0),[])
        self.assertEqual(d.detect(frame,[lying],["Possible Fall"],1.1),[])
        self.assertEqual(d.detect(frame,[lying],["Possible Fall"],3.2)[0]["reasons"],["Person Down"])

    def test_medical_pose_fall_requires_confident_keypoints_and_stationarity(self):
        frame=np.zeros((100,100,3),dtype=np.uint8)
        weak=medical_person(box=(20,20,75,70),speed=0.1,keypoints=3)
        d=MedicalDetector()
        for t in (0,1.1,3.2):
            self.assertEqual(d.detect(frame,[weak],["Possible Fall"],t),[])
        moving=medical_person(speed=2.0)
        d=MedicalDetector()
        for t in (0,1.1,3.2):
            self.assertEqual(d.detect(frame,[moving],["Possible Fall"],t),[])

    def test_medical_minimum_box_height_and_track_age(self):
        d=MedicalDetector();frame=np.zeros((100,100,3),dtype=np.uint8)
        small=medical_person(box=(20,20,50,30))
        for t in (0,1.1,3.2):
            self.assertEqual(d.detect(frame,[small],["Possible Fall"],t),[])
        new_track=medical_person(track=2)
        d=MedicalDetector()
        self.assertEqual(d.detect(frame,[new_track],["Possible Fall"],0),[])
        self.assertEqual(d.detect(frame,[new_track],["Possible Fall"],0.5),[])

    def test_medical_red_shirt_does_not_alert_alone(self):
        d=MedicalDetector();frame=np.zeros((100,100,3),dtype=np.uint8)
        frame[10:95,20:65]=[0,0,150]
        upright=medical_person(box=(20,10,65,95),pose_label="Normal",keypoints=12)
        for t in (0,1.6,3.2):
            self.assertEqual(d.detect(frame,[upright],["Normal"],t),[])
        self.assertFalse(d.diagnostics[0]["blood_heuristic_signal"])

    def test_medical_red_patch_supports_lying_case_but_not_blood_only(self):
        frame=np.zeros((100,100,3),dtype=np.uint8)
        frame[25:35,30:50]=[0,0,150]
        lying=medical_person()
        d=MedicalDetector()
        self.assertEqual(d.detect(frame,[lying],["Normal"],0),[])
        self.assertEqual(d.detect(frame,[lying],["Normal"],1.1),[])
        self.assertEqual(d.detect(frame,[lying],["Normal"],3.2)[0]["reasons"],
                         ["Person Down","Possible Blood"])
        upright=medical_person(track=2,box=(20,10,65,95),speed=0.1,pose_label="Normal",keypoints=12)
        d=MedicalDetector()
        for t in (0,1.6,3.2):
            self.assertEqual(d.detect(frame,[upright],["Normal"],t),[])
    def test_pose_iou_matching_ignores_order(self):
        class FakePose:
            def detect(self, frame): return [{"box":(100,0,150,100),"behavior":"Raised Hand"},{"box":(0,0,50,100),"behavior":"Possible Fall"}]
        pipeline=BehaviorPipeline(pose_behavior=FakePose())
        people=[{"id":1,"box":(0,0,50,100),"center":(25,50)},{"id":2,"box":(100,0,150,100),"center":(125,50)}]
        self.assertEqual(pipeline.detect(None,people,[],0),["Fall Detected","Raised Hand"])
        class NoPose:
            def detect(self, frame): return []
        fallback=BehaviorPipeline(pose_behavior=NoPose())
        self.assertEqual(fallback.detect(None,[{"id":3,"box":(0,0,20,40),"center":(10,20)}],[],0),["Unknown"])
        self.assertEqual(fallback.detect(None,[],[],1),[])
    def test_weapon_requires_consecutive_overlapping_samples(self):
        tracker=WeaponPersistence();weapon={"label":"pistol","confidence":.43,"box":(10,10,30,30),"center":(20,20)}
        self.assertEqual(tracker.update([weapon]),[])
        self.assertEqual(len(tracker.update([weapon])),1)
        self.assertEqual(tracker.update([]),[])
        self.assertEqual(tracker.update([weapon]),[])

    def test_weapon_per_class_confidence_and_max_box_area_filters(self):
        detections=[
            {"label":"pistol","confidence":.45,"box":(10,10,20,20)},
            {"label":"knife","confidence":.45,"box":(30,30,40,40)},
            {"label":"pistol","confidence":.90,"box":(0,0,80,80)},
        ]
        kept=filter_weapon_detections(detections,(100,100,3),
            class_thresholds={"pistol":.40,"knife":.50},base_threshold=.40,max_area_ratio=.10)
        self.assertEqual(len(kept),1)
        self.assertEqual(kept[0]["label"],"pistol")

    def test_weapon_confirm_frames_is_configurable(self):
        weapon={"label":"pistol","confidence":.7,"box":(10,10,30,30),"center":(20,20)}
        tracker=WeaponPersistence(confirm_frames=3)
        self.assertEqual(tracker.update([weapon]),[])
        self.assertEqual(tracker.update([weapon]),[])
        self.assertEqual(len(tracker.update([weapon])),1)

class IntegrationTests(unittest.TestCase):
    def test_dispatcher_confirm_cooldown_and_medical_escalation(self):
        class N:
            def __init__(self): self.items=[]
            def send(self, method, path, payload): self.items.append(payload)
        class L:
            def log(self, payload): pass
            def update(self, *args): pass
        class E:
            def save_image(self, *args): return "images/test.jpg"
        n=N(); d=Dispatcher(n,L(),E())
        high={"level":"HIGH","score":90,"weapon_score":100,"behavior_score":90,"reasons":[]}
        for t in range(5): d.process(high,[],[],[],[],None,t)
        self.assertEqual(d.alert_counts["POLICE"],0)
        d.process(high,[],[],[],[],None,5)
        self.assertEqual(d.alert_counts["POLICE"],1)
        for t in range(6,12): d.process(high,[],[],[],[],None,t)
        self.assertEqual(d.alert_counts["POLICE"],1)
        for t in range(36,42): d.process(high,[],[],[],[],None,t)
        self.assertEqual(d.alert_counts["POLICE"],2)
        case={"track_id":7,"severity":"MEDIUM","reasons":["Person Down"],"box":(0,0,2,2)}
        d.process(high,[],[],[],[case],None,42);d.process(high,[],[],[],[case],None,43)
        self.assertEqual(d.alert_counts["MEDICAL"],1)
        case={**case,"severity":"HIGH","reasons":["Person Down","Possible Blood"]}
        d.process(high,[],[],[],[case],None,44);d.process(high,[],[],[],[case],None,45)
        self.assertEqual(d.alert_counts["MEDICAL"],2)

    def test_dispatcher_requires_weapon_person_contact_for_police(self):
        class N:
            def send(self,*args): pass
        class L:
            def log(self,*args): pass
            def update(self,*args): pass
        class E:
            def save_image(self,*args): return "images/test.jpg"
        high={"level":"HIGH","score":90,"weapon_score":100,"behavior_score":90,"reasons":[]}
        person={"box":(0,0,100,100)}
        distant={"label":"pistol","confidence":.9,"box":(120,120,140,140),"center":(130,130)}
        d=Dispatcher(N(),L(),E())
        for t in range(config.POLICE_CONFIRM_FRAMES+1):
            d.process(high,[person],[distant],[],[],None,float(t))
        self.assertEqual(d.alert_counts["POLICE"],0)
        touching={**distant,"box":(100,20,120,40),"center":(110,30)}
        d=Dispatcher(N(),L(),E())
        for t in range(config.POLICE_CONFIRM_FRAMES+1):
            d.process(high,[person],[touching],[],[],None,float(t))
        self.assertEqual(d.alert_counts["POLICE"],1)

    def test_evidence_pre_event_buffer_fps(self):
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as tmp:
            old=config.EVIDENCE_DIR;config.EVIDENCE_DIR=Path(tmp)
            try:
                manager=EvidenceManager(fps=10)
                for i in range(31):
                    frame=np.full((64,64,3),(i*5,20,40),dtype=np.uint8)
                    manager.update(frame,i/10,active=(i==30))
                manager.close()
                clips=list((Path(tmp)/"videos").glob("*.mp4"));self.assertEqual(len(clips),1)
                cap=__import__("cv2").VideoCapture(str(clips[0]))
                cv2=__import__("cv2")
                actual_fps=cap.get(cv2.CAP_PROP_FPS);frame_count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
                self.assertAlmostEqual(actual_fps,10,delta=.1)
                self.assertGreaterEqual(frame_count,30)
                self.assertGreaterEqual(frame_count/actual_fps,3.0)
                ok,first=cap.read();self.assertTrue(ok)
                last=first
                while True:
                    ok,frame=cap.read()
                    if not ok: break
                    last=frame
                cap.release()
                self.assertGreater(float(last.mean()),float(first.mean()))
            finally: config.EVIDENCE_DIR=old

    def test_evidence_cap_preserves_incident_files(self):
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as tmp:
            old_dir,old_cap=config.EVIDENCE_DIR,config.EVIDENCE_MAX_GB
            config.EVIDENCE_DIR=Path(tmp);config.EVIDENCE_MAX_GB=0
            try:
                manager=EvidenceManager(fps=10)
                keep=Path(tmp)/"videos"/"linked.mp4";drop=Path(tmp)/"videos"/"old.mp4"
                keep.write_bytes(b"linked evidence");drop.write_bytes(b"unlinked evidence")
                (Path(tmp)/"incidents"/"incident.json").write_text(json.dumps({"video_path":"videos/linked.mp4"}))
                manager._enforce_disk_limit(force=True)
                self.assertTrue(keep.exists());self.assertFalse(drop.exists())
            finally: config.EVIDENCE_DIR,config.EVIDENCE_MAX_GB=old_dir,old_cap

    def test_server_api_escape_gps_and_reload(self):
        import server
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as tmp:
            old_store,old_incidents,old_gps,old_jobs,old_queue=server.STORE,server.incidents,server.gps_devices,server.JOBS,server.JOB_QUEUE
            server.STORE=Path(tmp)/"incidents.json";server.incidents={};server.gps_devices={};server.JOBS={};server.JOB_QUEUE=server.deque()
            try:
                client=server.app.test_client()
                bad=client.post("/api/incidents",json={"id":"bad","type":"OTHER"});self.assertEqual(bad.status_code,400)
                gps=client.post("/api/gps",json={"device_id":config.CAMERA_ID,"lat":12.5,"lon":77.2,"accuracy":4});self.assertEqual(gps.status_code,201)
                payload={"id":"test-one","type":"POLICE","level":"HIGH","reasons":["<script>alert(1)</script>"],"image_path":None,
                         "analysis_mode":"recorded clip analysed offline"}
                response=client.post("/api/incidents",json=payload);self.assertEqual(response.status_code,201)
                self.assertEqual(response.get_json()["analysis_mode"],"recorded clip analysed offline")
                self.assertEqual(response.get_json()["location_source"],"live-gps")
                self.assertEqual(client.post("/api/incidents",json=payload).status_code,200)
                self.assertEqual(len(client.get("/api/incidents?type=POLICE").get_json()),1)
                self.assertEqual(client.patch("/api/incidents/test-one",json={"status":"ACKNOWLEDGED"}).status_code,200)
                patched=client.patch("/api/incidents/test-one",json={"video_path":"videos/a.mp4"})
                self.assertEqual(patched.get_json()["video_path"],"videos/a.mp4")
                self.assertTrue(client.get("/api/gps").get_json()[0]["live"])
                server.gps_devices[config.CAMERA_ID]["timestamp"]-=config.GPS_LIVE_SECONDS+1
                stale=client.post("/api/incidents",json={**payload,"id":"test-stale"}).get_json()
                self.assertEqual(stale["location_source"],"camera-fixed")
                self.assertNotEqual(client.get("/evidence/%2e%2e%2fconfig.py").status_code,200)
                page=client.get("/police").get_data(as_text=True)
                self.assertIn("d.textContent=String(v??'')",page)
                self.assertNotIn("<script>alert(1)</script>",page)
                offline_page=client.get("/police?offline=1").get_data(as_text=True)
                self.assertIn("RECORDED CLIP ANALYSED OFFLINE · NOT LIVE",offline_page)
                self.assertNotIn("RECORDED CLIP ANALYSED OFFLINE · NOT LIVE",page)
                for route in ("/", "/police", "/hospital", "/monitoring", "/alerts", "/alerts?type=MEDICAL", "/cases", "/evidence", "/analytics", "/system", "/admin", "/gps"):
                    self.assertEqual(client.get(route).status_code,200,route)
                self.assertEqual(client.get("/api/pipeline/status").status_code,200)
                self.assertEqual(client.get("/api/jobs/missing").status_code,404)
                self.assertEqual(client.get("/api/jobs/missing/preview").status_code,404)
                self.assertEqual(client.get("/api/system/status").get_json()["server"],"online")
                self.assertEqual(client.post("/api/demo/simulate",json={"type":"POLICE"}).status_code,404)
                sim=client.post("/api/demo/simulate?dev=1",json={"type":"POLICE"})
                self.assertEqual(sim.status_code,201)
                self.assertTrue(sim.get_json()["simulated"])
                self.assertIn("simulated",sim.get_json()["reasons"][0].lower())
                medsim=client.post("/api/demo/simulate?dev=1",json={"type":"MEDICAL"})
                self.assertEqual(medsim.status_code,201);self.assertEqual(medsim.get_json()["type"],"MEDICAL")
                home=client.get("/").get_data(as_text=True)
                self.assertIn("Police Dashboard",home);self.assertIn("Hospital Dashboard",home)
                self.assertNotIn("simulate-police",home)
                self.assertIn("simulate-police",client.get("/?dev=1").get_data(as_text=True))
                self.assertIn("js/siren.js",home)
                nav=Path(config.BASE_DIR)/"static"/"js"/"navigation.js"
                nav_text=nav.read_text(encoding="utf-8")
                for href in ("'/','Home / Analyse'","'/monitoring','Monitoring'","'/police','Police Dashboard'","'/hospital','Hospital Dashboard'","'/admin','Admin Console'","'/gps','GPS Sender'"):
                    self.assertIn(href,nav_text)
                import cv2
                from video_source import camera_backend_candidates
                selected_name,selected_backend=camera_backend_candidates(cv2)[0]
                class FakeCamera:
                    def __init__(self,index,backend):self.index=index;self.backend=backend
                    def isOpened(self):return self.index==2 and self.backend==selected_backend
                    def release(self):pass
                with patch("cv2.VideoCapture",side_effect=lambda index,backend:FakeCamera(index,backend)):
                    camera=client.get("/api/camera/probe").get_json()
                self.assertTrue(camera["ok"]);self.assertEqual(camera["first"],{"index":2,"backend":selected_name})
                queued={"job_id":"queued-check","filename":"q.mp4","mode":"upload","status":"queued","frame":0,"target_frames":10,"total_frames":10,"fps":0,"alerts":[],"stop_event":server.Event(),"created_monotonic":time.monotonic()}
                server.JOBS["queued-check"]=queued;server.JOB_QUEUE.append("queued-check")
                self.assertEqual(client.get("/api/jobs").get_json()[0]["job_id"],"queued-check")
                self.assertEqual(client.post("/api/jobs/queued-check/skip").status_code,202)
                self.assertEqual(client.get("/api/jobs/queued-check").get_json()["status"],"cancelled")
                queued2={**queued,"job_id":"clear-check","status":"queued","stop_event":server.Event()}
                server.JOBS["clear-check"]=queued2;server.JOB_QUEUE.append("clear-check")
                self.assertEqual(client.post("/api/jobs/clear-queued").get_json()["skipped"],1)
                import cv2
                ok,preview=cv2.imencode(".jpg",np.zeros((8,8,3),dtype=np.uint8));self.assertTrue(ok)
                old_preview_jobs=server.JOBS;server.JOBS={"preview-check":{"last_jpeg":preview.tobytes()}}
                try:self.assertEqual(client.get("/api/jobs/preview-check/preview").status_code,200)
                finally:server.JOBS=old_preview_jobs
                self.assertEqual(client.post("/api/analysis/upload").status_code,400)
                self.assertEqual(client.post("/api/live/start",json={"camera_index":"invalid"}).status_code,400)
                invalid_video=client.post("/api/analysis/upload",data={"video":(io.BytesIO(b"x"),"bad.txt")},content_type="multipart/form-data")
                self.assertEqual(invalid_video.status_code,400)
                saved=server.incidents.copy();server.incidents={};server.load_incidents()
                self.assertIn("test-one",server.incidents)
                server.incidents=saved
            finally:
                server.STORE,server.incidents,server.gps_devices,server.JOBS,server.JOB_QUEUE=old_store,old_incidents,old_gps,old_jobs,old_queue

    def test_server_down_outbox_and_replay(self):
        with tempfile.TemporaryDirectory(dir=config.BASE_DIR) as tmp:
            old=config.EVIDENCE_DIR;config.EVIDENCE_DIR=Path(tmp)
            try:
                notifier=Notifier("http://127.0.0.1:1")
                before=time.monotonic();notifier.send("POST","/api/incidents",{"id":"offline","type":"POLICE"})
                self.assertLess(time.monotonic()-before,.5)
                notifier.close()
                outbox=Path(tmp)/"outbox.jsonl";self.assertTrue(outbox.exists())
                class Response:
                    status=201
                    def __enter__(self): return self
                    def __exit__(self,*args): pass
                sent,pending=replay(outbox,opener=lambda *a,**k:Response())
                self.assertEqual((sent,pending),(1,0));self.assertFalse(outbox.exists())
            finally: config.EVIDENCE_DIR=old

if __name__=="__main__":
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(CoreTests),unittest.defaultTestLoader.loadTestsFromTestCase(IntegrationTests)])
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        try:
            from selftest_pipeline import run
            run()
        except Exception as exc:
            print(f"FAIL: mocked pipeline integration: {exc}")
            raise SystemExit(1)
    raise SystemExit(not result.wasSuccessful())
