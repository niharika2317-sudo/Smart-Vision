# Final fixes and demo checklist

## Behavior feedback follow-up (2026-11-20)

- Untracked/first-seen people no longer default to Standing; behavior stays Unknown until motion history exists.
- Motion scoring now includes filtered path speed in addition to net displacement, to reduce misses when direction changes.
- The monitoring threat panel lists each person's behavior, pose, and normalized speed.
- Weapon output says candidate/review rather than asserting that the model has confirmed a weapon. No model weights or weapon filters changed.
- Verification: `python selftest.py` passed 25 tests. One CPU diagnostic on the first 60 frames of `t_v001_converted.avi` took 222.73 s (0.27 FPS); reviewed sample frames did not clearly show the pistol candidate. Real-video behavior after the code changes is unverified. See `VERIFICATION.md`.

## What changed

- The monitoring workspace puts the uncropped annotated video first and uses a compact, collapsible threat column. It has multi-file queue controls, a results table, fullscreen video, configurable fast-mode frame cap, and camera probe/start/stop controls. CSS stacks the threat panel below the video on narrow screens.
- Added an explicit `/admin` route and **Admin Console** navigation link (the same system-status workspace remains available at `/system`). The updated Monitoring page uses a visible native file picker and shows the selected filename/count before queuing.
- Models are constructed once on first real job and cached for later jobs. Person tracking wrappers are recreated per job. The job table reports processing FPS rather than the video's encoded FPS.
- A page-wide first click or keypress unlocks WebAudio on Home, Monitoring, Police and Hospital. Sound is on after unlock; the speaker toggle mutes. HIGH police alerts show the locked-audio banner and flash the title until unlocked. Medical alerts use a softer tone. Acknowledged incidents are not restarted. The dispatcher calls the local siren asynchronously for both police and medical alerts.
- Incident records carry video time and distinguish `recorded clip analysed offline` from `live webcam`. Developer simulations are hidden except at `/?dev=1` and remain labelled SIMULATED.
- Before editing, `templates/` and `static/` were copied to `templates_backup_before_finalfix_20261120/` and `static_backup_before_finalfix_20261120/`.

## Verification

- `python selftest.py`: **23 tests passed**, including queue endpoints, a mocked camera-backend selection, layout/siren hooks, and a test-fixture web analysis that logged: HIGH alert created → local siren trigger called → incident POSTed → Police dashboard API returned it. The mocked held pistol is a test fixture only; it is not a real detection. The selftest also exercises medical alert dispatch. No detection filters were weakened.
- Real model queue run: `verification/cli_smoke.mp4` (2 decoded frames), `videos/weaponized/t_w001_converted.avi` (8 frames), and `videos/normal/t_n001_converted.avi` (8 frames), all fast mode. All completed with zero police and medical alerts. Pipeline achieved FPS was approximately **0.16, 1.16, and 0.80**. Model construction took **0.53 s** once and the same cached models were reused. Request-to-first-frame measured **12.26, 14.33, and 21.25 s**; queued jobs include queue wait in that measurement. The earlier tiny-fixture first-frame time was **88.64 s**, compared with **12.26 s** here. No more YOLO runs were made.
- The actual camera probe tried indexes 0–3 with DirectShow, Media Foundation and default backends; **none opened**. OpenCV reported “device did not open” for all 12 attempts and an index-out-of-range diagnostic. OpenCV cannot determine from this result whether the device is absent, busy, or blocked by permission, so the exact underlying cause remains unknown.
- Restarted the stale relative-path server and confirmed the refreshed browser page shows the new Monitoring layout, Admin Console link, visible native file input and “No files selected” state. `/admin` returns the Admin Console page. The upload API itself has passed real-clip and mocked-pipeline tests, but I did not select a file through the browser dialog, so the browser's post-selection filename update is not manually verified. The 1920×1080, 1366×768 and 390×844 layout checks and screenshots were not completed.
- The mocked web test verifies the server-side siren trigger and dashboard API path. Browser audio playback, first-click WebAudio behavior, acknowledgement sound stop, live webcam alerts, and actual camera start/stop were not verified in a browser/device session.

## Short demo checklist

1. Start the intended workspace with `python server.py` and open `/monitoring`; confirm the page says **Monitoring** and has **Choose videos** and **Threat engine**. If it instead shows the older “Analyse a video” page, stop the stale `server.py` and restart from this project folder.
2. Click once anywhere on the page to unlock browser sound. Use the speaker control to mute/unmute; no Test siren button is required.
3. Choose multiple clips, leave Fast mode selected, adjust the frame cap if needed, and add them to the queue. The results include processing FPS and alert counts.
4. Use **Check cameras 0–3** before starting the webcam. This computer's probe found no available camera.
5. For a real incident workflow, review the Police or Hospital dashboard and acknowledge the incident. Do not present the mocked pistol test as a real detection.
6. Use `/?dev=1` only for the explicitly SIMULATED controls; hide that query in a normal demo.

## Remaining limits

CPU processing is below real time on the tested clips. Browser layout screenshots and actual browser audio are unverified. The weapon weights are third-party and have known false positives; dataset labels are clip-level, and blood, fall and running logic are heuristics without real-world validation. Human review is required.
