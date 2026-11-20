# Smart Vision front-end handoff

## Pages

- `/` — Home overview with real incident counts and links to separate workspaces.
- `/police` and `/hospital` — role-specific overview dashboards.
- `/alerts?type=POLICE` and `/alerts?type=MEDICAL` — separate, searchable, filterable response queues with evidence details and sequential status actions.
- `/monitoring` — starts the server computer's webcam inference, or accepts an uploaded clip for full-pipeline inference; shows the annotated frame preview, progress and stop action.
- `/cases`, `/evidence`, `/analytics`, `/system` — separate case, evidence, recorded-data analytics and service-status views.
- `/gps` — browser geolocation field unit. On a phone over LAN, use HTTPS (`python server.py --https`) because browsers deny geolocation on ordinary HTTP LAN pages.

## Incident and GPS API

- `GET /api/incidents?type=POLICE|MEDICAL` supplies `id`, `type`, `level`, `status`, `created_at`, `timestamp`, `camera_id`, `camera_name`, `score`, `weapon_score`, `behavior_score`, `reasons`, `behaviors`, `weapons`, `medical_cases`, `image_path`, `video_path`, `lat`, `lon`, `location_source`, and `status_times`.
- All dashboards derive counts/categories from these records. They do not report browser-only people counts, fake cameras, or model-health metrics.
- `PATCH /api/incidents/<id>` sends `{ "status": "ACKNOWLEDGED" | "DISPATCHED" | "RESOLVED" }` in the existing sequential order.
- `GET /api/gps` supplies `device_id`, `lat`, `lon`, `accuracy`, `age`, and `live`; `POST /api/gps` sends `{ "device_id": "camera-01", "lat": number, "lon": number, "accuracy": number }`.
- Evidence uses `/evidence/<path>` with `image_path` and `video_path`.

## Analysis controls

- `POST /api/live/start` with `{ "camera_index": 0 }` starts the Python pipeline on the server's webcam. `GET /api/live/stream` serves its annotated MJPEG frames; `/api/pipeline/status` supplies job state, frame count, source timestamp and source FPS. `POST /api/pipeline/stop` stops at a frame boundary.
- `POST /api/analysis/upload` accepts multipart field `video` (MP4, M4V, MOV, AVI, MKV or WEBM, maximum 500 MB) and runs the same local Python pipeline. The selected clip is kept under `videos/uploads/`; generated incidents/evidence use the existing dispatcher and evidence paths.
- Only one webcam or video analysis can run at a time. CPU inference may be slower than playback. Webcam access is from the computer running Flask, not the remote browser device.

## Behaviours to preserve

- Incident and GPS dashboard polling every 2 seconds; monitoring status polling every 1 second.
- Siren starts only after an explicit user gesture. New incidents show a visual notification cue.
- Status actions follow `NEW → ACKNOWLEDGED → DISPATCHED → RESOLVED`.
- Browser GPS is permission-based; periodically resend the last fix while sharing is enabled so the server's live window does not expire when a stationary device emits no new position event.
- Treat API fields as untrusted. Create DOM nodes and assign `textContent`; never inject incident strings as HTML.
- Keep uploaded/live video clearly distinguished, preserve the offline-analysis label, and do not call uploaded recordings live.
- Leaflet and fonts are bundled locally. Map tiles need internet; queues and coordinates remain usable without them.
