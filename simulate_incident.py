"""Create demo dashboard records without loading any vision models."""
import json
import time
import urllib.request
import argparse
from pathlib import Path
import config

def post(path, payload):
    req=urllib.request.Request(config.SERVER_URL+path,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    with urllib.request.urlopen(req,timeout=3) as r: return r.read()

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode",nargs="?",choices=("police","medical","gps"),default="police")
    parser.add_argument("--port",type=int,default=config.PORT)
    parser.add_argument("--server-url",help="override the configured server URL")
    args=parser.parse_args()
    mode=args.mode.lower()
    config.SERVER_URL=args.server_url or f"http://127.0.0.1:{args.port}"
    if mode=="gps":
        for i in range(30):
            post("/api/gps",{"device_id":config.CAMERA_ID,"lat":config.CAMERA_LAT+i*.00001,"lon":config.CAMERA_LON,"accuracy":5})
            time.sleep(2)
    elif mode in ("police","medical"):
        images=sorted((config.EVIDENCE_DIR/"images").glob("*"),key=lambda p:p.stat().st_mtime,reverse=True)
        kind="POLICE" if mode=="police" else "MEDICAL"
        payload={"id":"demo-"+str(time.time_ns()),"type":kind,"level":"HIGH" if mode=="police" else "MEDIUM","score":86 if mode=="police" else 35,"reasons":["Explicitly simulated for interface demonstration"],"behaviors":["Possible Attack"] if mode=="police" else ["Fall Detected"],"weapons":[],"camera_id":config.CAMERA_ID,"camera_name":config.CAMERA_NAME,"image_path":"images/"+images[0].name if images else None,"video_path":None,"timestamp":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"analysis_mode":"simulated demonstration","simulated":True}
        post("/api/incidents",payload)
    else: raise SystemExit("usage: python simulate_incident.py police|medical|gps")
