# Phase 5 — final verification, SCVD sample, and speed

Run date: 2026-11-19. This section records two passes over the same fixed 30-clip sample: initial `WEAPON_CONFIDENCE=0.40`, then one tuned pass at `0.75`. No second sample and no full-dataset run were performed.

## Environment and setup

- Python 3.12.6; PyTorch 2.12.1+cpu; `torch.cuda.is_available() == False`; 2 PyTorch CPU threads.
- CPU: Intel Core i3-1005G1 @ 1.20 GHz. GPU: none reported. `DEVICE="auto"` resolved to CPU. The GPU/half-precision path was not available to test.
- OpenCV 5.0. A fresh `python server.py` process was started separately; `http://127.0.0.1:5000/` returned HTTP 200 before the sample.
- No model files were changed. All sample runs used `--no-display --max-frames 300 --save-annotated ...` against only the staged `videos/` sample.

## Fixed sample and copied files

Source: `C:/Users/saniy/Downloads/archive/SCVD/SCVD_converted/Test`. Selection was the first 10 `.avi` files sorted case-insensitively by filename in each source folder. These were copied to the lowercase staging folders shown below. The 30 source names are exact; the same set was used for both passes.

**Weaponized → `videos/weaponized/`**

- `t_w001_converted.avi`
- `t_w002_converted.avi`
- `t_w003_converted.avi`
- `t_w004_converted.avi`
- `t_w005_converted.avi`
- `t_w006_converted.avi`
- `t_w007_converted.avi`
- `t_w008_converted.avi`
- `t_w009_converted.avi`
- `t_w010_converted.avi`

**Normal → `videos/normal/`**

- `t_n001_converted.avi`
- `t_n002_converted.avi`
- `t_n003_converted.avi`
- `t_n004_converted.avi`
- `t_n005_converted.avi`
- `t_n006_converted.avi`
- `t_n007_converted.avi`
- `t_n008_converted.avi`
- `t_n009_converted.avi`
- `t_n010_converted.avi`

**Violence → `videos/violence/`**

- `t_v001_converted.avi`
- `t_v002_converted.avi`
- `t_v003_converted.avi`
- `t_v004_converted.avi`
- `t_v005_converted.avi`
- `t_v006_converted.avi`
- `t_v007_converted.avi`
- `t_v008_converted.avi`
- `t_v009_converted.avi`
- `t_v010_converted.avi`

Twelve source clips exceed 300 frames (Weaponized 004/009/010; Violence 001–004 and 006–010); completed runs stopped at 300 frames. Baseline Violence 002 and tuned Violence 010 hit the runner’s separate 300-second subprocess timeout before a summary, so their metrics are incomplete. No clip exceeded the requested 300-frame cap.

## Detection and alert results

The percentages below are clip-level **police-alert rates by SCVD folder**, not detector precision/recall: the dataset folders provide clip classes but no frame-level weapon boxes or event timestamps. Visual checks found likely false weapon boxes, so an alert in Weaponized is not automatically a verified true positive.

| Folder / measure | Baseline, confidence 0.40 | Tuned, confidence 0.75 |
|---|---:|---:|
| Weaponized police-alert clips / 10 | 2/10 | 0/10 |
| Normal police-alert clips / 10 (false-alarm rate by class) | 2/10 | 0/10 |
| Violence police-alert clips / completed | 2/9 (one timeout) | 0/9 (one timeout) |
| Weapon-positive clips / completed, raw model summary | 24/29 | 7/29 |
| Mean first-weapon-to-police time, Weaponized alerts | 1.95 s across 2 alerted clips | n/a; no police alerts |

At baseline, the Weaponized police-alert rate was 2/10 (20%); the Normal police-alert rate was also 2/10 (20%). The 0.75 setting reduced Normal police alerts to 0/10 but also reduced Weaponized police alerts to 0/10. This is a poor detection/false-alarm tradeoff for deployment; the higher threshold is retained as a conservative demonstration setting, not presented as a successful accuracy fix. Do not dispatch from these outputs without a human review.

The baseline’s two Weaponized alerts were `t_w003` at 6.80 s and `t_w006` at 1.60 s; their first-weapon-to-alert deltas were 2.93 s and 0.97 s (mean 1.95 s). Across all six baseline alerts, including Normal and Violence clips, the mean was 1.82 s. The Normal alerts at 3.20 s and 3.70 s are false-alarm candidates. At confidence 0.75 there were no police alerts, so there is no final mean latency.

### Per-clip results — baseline 0.40 → tuned 0.75

Each cell lists processed frames, achieved FPS, raw weapon labels/confidence range, behaviors, maximum threat, police count/time, and medical alert count. `TIMEOUT` means no complete run summary was captured.

#### Weaponized

| Clip | Baseline 0.40 | Tuned 0.75 |
|---|---|---|
| `t_w001_converted.avi` | 176f, 1.74 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) | 176f, 1.12 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_w002_converted.avi` | 237f, 2.22 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) | 237f, 2.98 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_w003_converted.avi` | 229f, 1.98 FPS; pistol:0.43-0.81(42); ['Armed Person', 'Standing']; HIGH; police 1 (POLICE@6.80s); medical 3 (MEDICAL@4.73s,MEDICAL@6.13s,MEDICAL@6.63s) | 229f, 1.07 FPS; pistol:0.75-0.81(3); ['Standing']; LOW; police 0 (none); medical 3 (MEDICAL@4.73s,MEDICAL@6.13s,MEDICAL@6.63s) |
| `t_w004_converted.avi` | 300f, 2.98 FPS; pistol:0.40-0.57(32); ['Raised Hand', 'Standing']; MEDIUM; police 0 (none); medical 0 (none) | 300f, 2.47 FPS; none; ['Raised Hand', 'Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_w005_converted.avi` | 172f, 2.47 FPS; knife:0.43-0.53(6);pistol:0.40-0.59(24); ['Armed Person', 'Raised Hand', 'Standing', 'Walking']; HIGH; police 0 (none); medical 0 (none) | 172f, 2.70 FPS; none; ['Raised Hand', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 (none) |
| `t_w006_converted.avi` | 225f, 2.21 FPS; pistol:0.40-0.76(55); ['Armed Person', 'Standing']; HIGH; police 1 (POLICE@1.60s); medical 1 (MEDICAL@3.33s) | 225f, 2.33 FPS; pistol:0.76-0.76(1); ['Standing']; LOW; police 0 (none); medical 1 (MEDICAL@3.33s) |
| `t_w007_converted.avi` | 225f, 3.06 FPS; knife:0.40-0.86(394);pistol:0.40-0.56(4); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) | 225f, 1.66 FPS; knife:0.75-0.86(224); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) |
| `t_w008_converted.avi` | 225f, 2.46 FPS; knife:0.40-0.87(330); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) | 225f, 1.70 FPS; knife:0.79-0.87(236); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) |
| `t_w009_converted.avi` | 300f, 1.75 FPS; pistol:0.44-0.68(8); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) | 300f, 3.54 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_w010_converted.avi` | 300f, 1.65 FPS; pistol:0.40-0.70(167); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) | 300f, 3.26 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |

#### Normal

| Clip | Baseline 0.40 | Tuned 0.75 |
|---|---|---|
| `t_n001_converted.avi` | 125f, 0.52 FPS; none; ['Standing']; LOW; police 0 (none); medical 1 (MEDICAL@1.73s) | 125f, 1.70 FPS; none; ['Standing']; LOW; police 0 (none); medical 1 (MEDICAL@1.73s) |
| `t_n002_converted.avi` | 144f, 0.83 FPS; pistol:0.40-0.40(1); ['Standing']; LOW; police 0 (none); medical 1 (MEDICAL@1.53s) | 144f, 1.45 FPS; none; ['Standing']; LOW; police 0 (none); medical 1 (MEDICAL@1.53s) |
| `t_n003_converted.avi` | 117f, 1.57 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) | 117f, 1.44 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_n004_converted.avi` | 127f, 0.89 FPS; none; ['Standing']; LOW; police 0 (none); medical 2 (MEDICAL@1.53s,MEDICAL@1.63s) | 127f, 0.76 FPS; none; ['Standing']; LOW; police 0 (none); medical 2 (MEDICAL@1.53s,MEDICAL@1.63s) |
| `t_n005_converted.avi` | 119f, 1.12 FPS; pistol:0.45-0.45(1); ['Running', 'Standing', 'Walking']; LOW; police 0 (none); medical 2 (MEDICAL@1.53s,MEDICAL@1.53s) | 119f, 1.10 FPS; none; ['Running', 'Standing', 'Walking']; LOW; police 0 (none); medical 2 (MEDICAL@1.53s,MEDICAL@1.53s) |
| `t_n006_converted.avi` | 127f, 0.99 FPS; pistol:0.40-0.65(78); ['Armed Person', 'Raised Hand', 'Running', 'Standing', 'Walking']; HIGH; police 0 (none); medical 3 (MEDICAL@2.33s,MEDICAL@3.63s,MEDICAL@3.93s) | 127f, 1.06 FPS; none; ['Raised Hand', 'Running', 'Standing', 'Walking']; LOW; police 0 (none); medical 3 (MEDICAL@2.33s,MEDICAL@3.63s,MEDICAL@3.93s) |
| `t_n007_converted.avi` | 117f, 1.16 FPS; pistol:0.40-0.60(14); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 1 (MEDICAL@2.63s) | 117f, 1.10 FPS; none; ['Running', 'Standing', 'Walking']; LOW; police 0 (none); medical 1 (MEDICAL@2.63s) |
| `t_n008_converted.avi` | 119f, 0.78 FPS; pistol:0.40-0.73(79); ['Armed Person', 'Standing', 'Walking']; HIGH; police 1 (POLICE@3.20s); medical 1 (MEDICAL@2.13s) | 119f, 1.15 FPS; none; ['Standing', 'Walking']; LOW; police 0 (none); medical 1 (MEDICAL@2.13s) |
| `t_n009_converted.avi` | 129f, 1.12 FPS; pistol:0.43-0.74(37); ['Armed Person', 'Raised Hand', 'Standing', 'Walking']; HIGH; police 1 (POLICE@3.70s); medical 0 (none) | 129f, 1.37 FPS; none; ['Raised Hand', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 (none) |
| `t_n010_converted.avi` | 138f, 1.17 FPS; pistol:0.41-0.59(87); ['Standing', 'Walking']; MEDIUM; police 0 (none); medical 1 (MEDICAL@2.33s) | 138f, 1.04 FPS; none; ['Standing', 'Walking']; LOW; police 0 (none); medical 1 (MEDICAL@2.33s) |

#### Violence

| Clip | Baseline 0.40 | Tuned 0.75 |
|---|---|---|
| `t_v001_converted.avi` | 300f, 1.14 FPS; pistol:0.40-0.65(168); ['Armed Person', 'Standing']; HIGH; police 1 (POLICE@0.50s); medical 0 (none) | 300f, 1.29 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_v002_converted.avi` | TIMEOUT / no summary | 300f, 1.56 FPS; pistol:0.77-0.77(1); ['Raised Hand', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 (none) |
| `t_v003_converted.avi` | 300f, 1.73 FPS; pistol:0.41-0.51(7); ['Armed Person', 'Standing', 'Walking']; HIGH; police 1 (POLICE@8.40s); medical 0 (none) | 300f, 1.60 FPS; none; ['Standing', 'Walking']; LOW; police 0 (none); medical 0 (none) |
| `t_v004_converted.avi` | 300f, 3.17 FPS; pistol:0.40-0.63(55); ['Armed Person', 'Standing']; HIGH; police 0 (none); medical 0 (none) | 300f, 1.72 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_v005_converted.avi` | 260f, 3.07 FPS; pistol:0.45-0.46(2); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) | 260f, 1.78 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_v006_converted.avi` | 300f, 1.76 FPS; pistol:0.40-0.85(121);knife:0.40-0.58(14); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 (none) | 300f, 1.67 FPS; pistol:0.77-0.85(7); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 (none) |
| `t_v007_converted.avi` | 300f, 1.83 FPS; pistol:0.40-0.73(27); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 (none) | 300f, 2.62 FPS; none; ['Running', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 (none) |
| `t_v008_converted.avi` | 300f, 1.60 FPS; pistol:0.40-0.76(136); ['Standing']; MEDIUM; police 0 (none); medical 0 (none) | 300f, 2.54 FPS; pistol:0.76-0.76(2); ['Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_v009_converted.avi` | 300f, 1.54 FPS; pistol:0.40-0.64(39); ['Armed Person', 'Raised Hand', 'Standing']; HIGH; police 0 (none); medical 0 (none) | 300f, 2.43 FPS; none; ['Raised Hand', 'Standing']; LOW; police 0 (none); medical 0 (none) |
| `t_v010_converted.avi` | 300f, 1.12 FPS; pistol:0.40-0.67(28); ['Raised Hand', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 (none) | TIMEOUT / no summary |

Complete machine-readable records: [baseline CSV](verification/scvd_baseline_results.csv), [final CSV](verification/scvd_results.csv), [final JSON](verification/scvd_results.json). Annotated outputs from the initial pass are in `verification/scvd_baseline/`; current 0.75 outputs are in `verification/scvd/`. The timeout clips have partial/no complete output and must not be treated as successful runs.

## Visual review and false positives

- Initial Normal clip `t_n008` produced a `pistol` box over a pedestrian and a police alert at 3.20 s; `t_n009` did the same at 3.70 s. The reviewed event frames show ordinary street pedestrians and no clearly visible weapon inside the red boxes. These are likely false positives. See [t_n008 at 3.20 s](verification/scvd_snapshots/review/t_n008_converted_3.20s.jpg) and [t_n009 at 3.70 s](verification/scvd_snapshots/review/t_n009_converted_3.70s.jpg). The 0.75 cutoff removed both.
- Other common false candidate: `knife` on Weaponized `t_w007` and `t_w008`, confidence up to 0.86/0.87. Reviewed midpoint frames show a large box over a white diagonal rail/structure, not a clearly visible knife. See [t_w007 midpoint](verification/scvd_snapshots/weaponized/t_w007_converted_mid.jpg) and [t_w008 midpoint](verification/scvd_snapshots/weaponized/t_w008_converted_mid.jpg). These boxes persisted above 0.75, so confidence alone does not solve the model’s localization errors.
- Baseline Weaponized alert frames `t_w003` and `t_w006` show red boxes over people; the reviewed frames do not clearly reveal a weapon. The clip-level folder label is not enough to verify the box. See [t_w003 at 6.80 s](verification/scvd_snapshots/review/t_w003_converted_6.80s.jpg) and [t_w006 at 1.60 s](verification/scvd_snapshots/review/t_w006_converted_1.60s.jpg). These are alerts, not confirmed true positives.
- The SCVD Violence clips contain visible altercations in some sampled frames, but weapon localization was not visually confirmed. At baseline, two Violence clips generated police alerts; at 0.75, none of the completed clips did. Violence clips should stay at MEDIUM or lower when no verified weapon is present; the output is not reliable enough to make that distinction consistently.
- Medical heuristics also fired in 8/10 Normal clips on both passes (multiple repeated alerts in some clips). This indicates a serious blood/fall false-alarm limitation; red color and scene geometry are not ground-truth evidence of injury.

## Stage timing and speed

The final dedicated headless 100-frame benchmark (`verification/stage_benchmark_100f.mp4`, 640×360, CPU, current 0.75 config) processed 100 frames in 88.75 s = **1.13 FPS**. It had no police alert. The printed values below are averages per processed source frame. `imshow` is 0 because the run was headless; `imwrite` was 0 because no image snapshot was saved in that benchmark.

| Stage | ms/frame |
|---|---:|
| Person detect + track | 326.85 |
| Weapon detect | 483.10 |
| Pose | 64.57 |
| Behavior logic | 0.06 |
| Medical / blood | 0.17 |
| Threat fusion | 0.08 |
| Evidence write | 0.31 |
| `imwrite` | 0.00 |
| `imshow` | 0.00 (headless) |
| Dispatcher / logging | 0.00 |
| Drawing | 1.62 |

On the 29 complete SCVD clips per pass, the baseline mean stage times were person/track 209.65 ms, weapon 393.86 ms, pose 86.43 ms, behavior 0.11 ms, medical/blood 0.24 ms, threat fusion 0.06 ms, evidence 6.01 ms, imwrite 0.10 ms, dispatcher/logging 0.63 ms, and drawing 7.64 ms. The 0.75 pass measured 198.03 / 347.71 / 79.91 / 0.11 / 0.24 / 0.05 / 3.89 / 0.09 / 0.32 / 5.12 ms respectively. `imshow` is disabled in both passes. Weapon and person inference dominate.

The best full-pipeline clip result was 3.54 FPS (one 300-frame clip); the best dedicated 100-frame benchmark before the last run was 2.47 FPS, while the final 100-frame measurement was 1.13 FPS. On the same benchmark file, earlier measured runs were 2.47 and 2.02 FPS; timings varied substantially under CPU load. The current result is not real-time and does not meet 8 FPS.

Tuning experiments and tradeoffs:

- Weapon model at `imgsz=640`: 281.93 ms/inference (3.55 FPS) in a 100-frame isolated timing. `imgsz=416`: 102.71 ms (9.74 FPS); `imgsz=320`: 77.36 ms (12.93 FPS), but 320 missed the observed knife candidate. Integrated 416 runs measured 2.15/1.85 FPS with BoT-SORT and 2.16 FPS with ByteTrack, versus a 2.02 FPS 640/BoT-SORT run; no consistent overall speed gain was demonstrated. Final sizes remain 640 to avoid unverified accuracy loss.
- Person-only detector timing improved from 260.19 ms at 640 to 160.69 ms at 416 and 129.67 ms at 320, but detections on the dark overlaid benchmark rose 37→90→112; final person size remains 640.
- Throttling the evidence disk-cap check reduced measured evidence-save time from 47.58 ms to 5.55 ms per snapshot (about 8.6×); this small component is not the main pipeline bottleneck.
- Final cadence: weapon inference every frame; person tracking and pose/behavior sampling every third frame where applicable; pose is skipped when no person is present. `DEVICE="auto"` chooses CPU here. Half precision is only enabled for non-CPU devices and was not exercised.
- Four-frame transient clip `163318`: stride-three baseline missed the sole raw candidate (1/1 missed = 100% raw-candidate miss rate); weapon inference every frame saw it (0/1 missed = 0% raw-candidate miss rate). The two-frame filter still rejected the single sample, so accepted weapon detections/alerts were 0/1 both before and after. This is why weapon inference stays every frame.

## Final configuration and filter effect

| Setting | Final value | Reason / evidence |
|---|---:|---|
| `DEVICE` | `auto` | Uses CUDA if available; CPU here. |
| Person / weapon / pose image size | 640 / 640 / 640 | Lower isolated sizes did not improve integrated throughput reliably; lower sizes reduced candidate detections. |
| `PERSON_TRACKER` | `botsort.yaml` | ByteTrack trial did not improve integrated FPS. |
| `PERSON_BEHAVIOR_EVERY_N_FRAMES` | 3 | Reduce person/pose work while preserving source-video timestamps. |
| `WEAPON_EVERY_N_FRAMES` | 1 | Avoid skipping brief weapon samples. |
| `WEAPON_CONFIDENCE` | 0.75 | Raised from 0.40 to suppress observed Normal pistol boxes (max 0.74); full rerun removed Normal police alerts but also Weaponized police detections (2/10→0/10). |
| `WEAPON_CONFIRM_FRAMES` | 2 | Require two overlapping samples. Suppressed single high-confidence candidates (for example Violence 002 at 0.77); did not prevent repeated low-threshold Normal false alerts. |
| `WEAPON_TRACK_IOU` | 0.30 | Existing overlap threshold for the persistence filter. |
| Evidence cap check | 30 s | Avoid a disk-cap scan on every evidence frame. |

No more config changes were made after the 0.75 all-folder rerun. No optional second 10-clip sample was run because the first sample required long CPU processing and the current result remains below target speed.

## Demo plan

No Weaponized clip produced a police alert with the final `WEAPON_CONFIDENCE=0.75` setting. `demo.bat` therefore launches `demo_dashboard_only.bat` and opens annotated `t_w003` and `t_n003` videos; the simulated dashboard incidents are labeled as simulations. The Weaponized annotated clip is not a verified alarm demonstration. The Normal clip shows LOW/no weapon output. The batch launcher itself was not executed in this verification session.

## Regression tests and remaining verification

`python selftest.py` passed after the parser correction, after the confidence-threshold change, and after the demo launcher edit: 12 unittest checks passed each time, plus the mocked 100-frame frame-loop scenario. The tests cover Flask API idempotency/status/GPS/path checks, evidence pre-event order/FPS/cap, dispatcher persistence/cooldowns, and mocked server-down outbox/replay behavior.

- The server-down self-test used a closed local port and mocked replay success. It did **not** stop the live Flask process during an actual YOLO run or deliver a real alert back to the restarted server. To verify manually: start a long sample with `python main.py videos/weaponized/t_w003_converted.avi --no-display --max-frames 300`; stop the local server while it runs; confirm the video loop continues and `evidence/outbox.jsonl` gains a POST; restart `python server.py`; run `python replay_outbox.py`; query `/api/incidents` and confirm exactly one incident ID.
- A bounded normal-EOF smoke run (`t_n003`, 3 frames) wrote a readable 3-frame MP4 at 30 FPS. A Ctrl+C attempt in the console terminated the Windows process before a readable output file was finalized; clean Ctrl+C flush/release is therefore **not verified**. Manual check: run a longer clip with `--no-display --save-annotated verification/shutdown_test.mp4`, press Ctrl+C after several seconds, then open the file in OpenCV and confirm the last frames are readable.
- `demo.bat` and `demo_dashboard_only.bat` were not launched interactively; their server-window, browser-tab, default-video-player and simulation behavior is not verified. Run `demo.bat` on the presentation PC and confirm both dashboard tabs and both MP4s open.
- The user previously verified dashboards, siren, map, live-GPS dot and status buttons on their PC; this session only confirmed the fresh server root returned HTTP 200. Phone GPS over HTTPS, Telegram delivery, webcam capture, and live-model behavior on a webcam were not tested here.
- Falls/running/blood have no frame-level labels in this sample. The medical heuristic raised alerts in 8/10 Normal clips. Tracking ID stability and true weapon localization are not validated.

### Dataset citation placeholder

SCVD (Kaggle): exact dataset title: **[user fill in]**; author/uploader: **[user fill in]**; license: **[user fill in from the Kaggle dataset page]**; dataset page URL: **[user fill in]**. Do not present a guessed license as fact.

### Responsible use

Alerts are decision support only. A human must inspect the scene and evidence before any dispatch. Use the system only where surveillance is lawful and disclosed; restrict access to evidence, set a documented retention period, and securely delete recordings when no longer needed. The evidence cap deletes only unreferenced media, so incident-linked files may remain beyond the cap. The dashboard has no authentication and is for a controlled demo, not deployment.

### Viva Q&A additions

- **Does it meet real-time speed?** No. On this CPU-only i3-1005G1, the final 100-frame run measured 1.13 FPS; best observed full-pipeline clip rate was 3.54 FPS. It does not meet 8 FPS.
- **What is the weapon detection rate?** At the initial 0.40 threshold, 2/10 Weaponized clips produced police alerts and 2/10 Normal clips also did. At 0.75, both rates were 0/10. These are class-level alert rates, not ground-truth detector recall.
- **Why raise confidence to 0.75 if it misses detections?** It removed the observed Normal false police alerts, but also removed Weaponized alerts. The result exposes the model’s poor separation; labeled boxes and further model work are needed before claiming reliable alerts.
- **What are the common false positives?** Pistol boxes over ordinary pedestrians at roughly 0.40–0.74 confidence, and high-confidence knife boxes over a diagonal rail/structure. Visual review found no clearly visible weapon in those boxes.
- **Why use two-frame persistence?** It rejects isolated samples, but can miss brief weapons: one high-confidence sample at a time is suppressed. On the Normal sample it did not remove repeated detections at the original 0.40 threshold.

### Three weakest points

1. Weapon model localization/classification is unreliable: Normal false alerts at 0.40, zero Weaponized alerts at 0.75, and repeated knife boxes on scene geometry.
2. CPU throughput is 1.13 FPS on the final 100-frame run, below the 8 FPS target and unsuitable for real-time use.
3. Medical/fall/blood heuristics are weakly validated; Normal clips produced medical alerts in 8/10 cases, and this sample has no ground truth to establish injury recall or precision.

### README claims not confirmed in this session

The prior user-reported dashboard/siren/map/status behavior was not retested in the browser. This session did not verify the launchers, phone GPS over HTTPS, Telegram, webcam capture, or any real police/hospital dispatch integration. No real automatic dispatch is implemented.
# Phase 5 — final verification, SCVD sample, and speed

Run date: 2026-11-19. This section records two passes over the same fixed 30-clip sample: initial `WEAPON_CONFIDENCE=0.40`, then one tuned pass at `0.75`. No second sample and no full-dataset run were performed.

## Environment and setup

- Python 3.12.6; PyTorch 2.12.1+cpu; `torch.cuda.is_available() == False`; 2 PyTorch CPU threads.
- CPU: Intel Core i3-1005G1 @ 1.20 GHz. GPU: none reported. `DEVICE="auto"` resolved to CPU. The GPU/half-precision path was not available to test.
- OpenCV 5.0. A fresh `python server.py` process was started separately; `http://127.0.0.1:5000/` returned HTTP 200 before the sample.
- No model files were changed. All sample runs used `--no-display --max-frames 300 --save-annotated ...` against only the staged `videos/` sample.

## Fixed sample and copied files

Source: `C:/Users/saniy/Downloads/archive/SCVD/SCVD_converted/Test`. Selection was the first 10 `.avi` files sorted case-insensitively by filename in each source folder. These were copied to the lowercase staging folders shown below. The 30 source names are exact; the same set was used for both passes.

**Weaponized → `videos/weaponized/`**

- `t_w001_converted.avi`
- `t_w002_converted.avi`
- `t_w003_converted.avi`
- `t_w004_converted.avi`
- `t_w005_converted.avi`
- `t_w006_converted.avi`
- `t_w007_converted.avi`
- `t_w008_converted.avi`
- `t_w009_converted.avi`
- `t_w010_converted.avi`

**Normal → `videos/normal/`**

- `t_n001_converted.avi`
- `t_n002_converted.avi`
- `t_n003_converted.avi`
- `t_n004_converted.avi`
- `t_n005_converted.avi`
- `t_n006_converted.avi`
- `t_n007_converted.avi`
- `t_n008_converted.avi`
- `t_n009_converted.avi`
- `t_n010_converted.avi`

**Violence → `videos/violence/`**

- `t_v001_converted.avi`
- `t_v002_converted.avi`
- `t_v003_converted.avi`
- `t_v004_converted.avi`
- `t_v005_converted.avi`
- `t_v006_converted.avi`
- `t_v007_converted.avi`
- `t_v008_converted.avi`
- `t_v009_converted.avi`
- `t_v010_converted.avi`

Twelve source clips exceed 300 frames (Weaponized 004/009/010; Violence 001–004 and 006–010); completed runs stopped at 300 frames. Baseline Violence 002 and tuned Violence 010 hit the runner’s separate 300-second subprocess timeout before a summary, so their metrics are incomplete. No clip exceeded the requested 300-frame cap.

## Detection and alert results

The percentages below are clip-level **police-alert rates by SCVD folder**, not detector precision/recall: the dataset folders provide clip classes but no frame-level weapon boxes or event timestamps. Visual checks found likely false weapon boxes, so an alert in Weaponized is not automatically a verified true positive.

| Folder / measure | Baseline, confidence 0.40 | Tuned, confidence 0.75 |
|---|---:|---:|
| Weaponized police-alert clips / 10 | 2/10 | 0/10 |
| Normal police-alert clips / 10 (false-alarm rate by class) | 2/10 | 0/10 |
| Violence police-alert clips / completed | 2/9 (one timeout) | 0/9 (one timeout) |
| Weapon-positive clips / completed, raw model summary | 24/29 | 7/29 |
| Mean first-weapon-to-police time, Weaponized alerts | 1.95 s across 2 alerted clips | n/a; no police alerts |

At baseline, the Weaponized police-alert rate was 2/10 (20%); the Normal police-alert rate was also 2/10 (20%). The 0.75 setting reduced Normal police alerts to 0/10 but also reduced Weaponized police alerts to 0/10. This is a poor detection/false-alarm tradeoff for deployment; the higher threshold is retained as a conservative demonstration setting, not presented as a successful accuracy fix. Do not dispatch from these outputs without a human review.

The baseline’s two Weaponized alerts were `t_w003` at 6.80 s and `t_w006` at 1.60 s; their first-weapon-to-alert deltas were 2.93 s and 0.97 s (mean 1.95 s). Across all six baseline alerts, including Normal and Violence clips, the mean was 1.82 s. The Normal alerts at 3.20 s and 3.70 s are false-alarm candidates. At confidence 0.75 there were no police alerts, so there is no final mean latency.

### Per-clip results — baseline 0.40 → tuned 0.75

Each cell lists processed frames, achieved FPS, raw weapon labels/confidence range, behaviors, maximum threat, police count/time, and medical alert count. `TIMEOUT` means no complete run summary was captured.

#### Weaponized

| Clip | Baseline 0.40 | Tuned 0.75 |
|---|---|---|
| `t_w001_converted.avi` | 176f, 1.74 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 | 176f, 1.12 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_w002_converted.avi` | 237f, 2.22 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 | 237f, 2.98 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_w003_converted.avi` | 229f, 1.98 FPS; pistol:0.43-0.81(42); ['Armed Person', 'Standing']; HIGH; police 1 (MEDICAL@4.73s;MEDICAL@6.13s;MEDICAL@6.63s;POLICE@6.80s); medical 3 | 229f, 1.07 FPS; pistol:0.75-0.81(3); ['Standing']; LOW; police 0 (MEDICAL@4.73s;MEDICAL@6.13s;MEDICAL@6.63s); medical 3 |
| `t_w004_converted.avi` | 300f, 2.98 FPS; pistol:0.40-0.57(32); ['Raised Hand', 'Standing']; MEDIUM; police 0 (none); medical 0 | 300f, 2.47 FPS; none; ['Raised Hand', 'Standing']; LOW; police 0 (none); medical 0 |
| `t_w005_converted.avi` | 172f, 2.47 FPS; knife:0.43-0.53(6);pistol:0.40-0.59(24); ['Armed Person', 'Raised Hand', 'Standing', 'Walking']; HIGH; police 0 (none); medical 0 | 172f, 2.70 FPS; none; ['Raised Hand', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 |
| `t_w006_converted.avi` | 225f, 2.21 FPS; pistol:0.40-0.76(55); ['Armed Person', 'Standing']; HIGH; police 1 (POLICE@1.60s;MEDICAL@3.33s); medical 1 | 225f, 2.33 FPS; pistol:0.76-0.76(1); ['Standing']; LOW; police 0 (MEDICAL@3.33s); medical 1 |
| `t_w007_converted.avi` | 225f, 3.06 FPS; knife:0.40-0.86(394);pistol:0.40-0.56(4); ['Standing']; MEDIUM; police 0 (none); medical 0 | 225f, 1.66 FPS; knife:0.75-0.86(224); ['Standing']; MEDIUM; police 0 (none); medical 0 |
| `t_w008_converted.avi` | 225f, 2.46 FPS; knife:0.40-0.87(330); ['Standing']; MEDIUM; police 0 (none); medical 0 | 225f, 1.70 FPS; knife:0.79-0.87(236); ['Standing']; MEDIUM; police 0 (none); medical 0 |
| `t_w009_converted.avi` | 300f, 1.75 FPS; pistol:0.44-0.68(8); ['Standing']; MEDIUM; police 0 (none); medical 0 | 300f, 3.54 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_w010_converted.avi` | 300f, 1.65 FPS; pistol:0.40-0.70(167); ['Standing']; MEDIUM; police 0 (none); medical 0 | 300f, 3.26 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |

#### Normal

| Clip | Baseline 0.40 | Tuned 0.75 |
|---|---|---|
| `t_n001_converted.avi` | 125f, 0.52 FPS; none; ['Standing']; LOW; police 0 (MEDICAL@1.73s); medical 1 | 125f, 1.70 FPS; none; ['Standing']; LOW; police 0 (MEDICAL@1.73s); medical 1 |
| `t_n002_converted.avi` | 144f, 0.83 FPS; pistol:0.40-0.40(1); ['Standing']; LOW; police 0 (MEDICAL@1.53s); medical 1 | 144f, 1.45 FPS; none; ['Standing']; LOW; police 0 (MEDICAL@1.53s); medical 1 |
| `t_n003_converted.avi` | 117f, 1.57 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 | 117f, 1.44 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_n004_converted.avi` | 127f, 0.89 FPS; none; ['Standing']; LOW; police 0 (MEDICAL@1.53s;MEDICAL@1.63s); medical 2 | 127f, 0.76 FPS; none; ['Standing']; LOW; police 0 (MEDICAL@1.53s;MEDICAL@1.63s); medical 2 |
| `t_n005_converted.avi` | 119f, 1.12 FPS; pistol:0.45-0.45(1); ['Running', 'Standing', 'Walking']; LOW; police 0 (MEDICAL@1.53s;MEDICAL@1.53s); medical 2 | 119f, 1.10 FPS; none; ['Running', 'Standing', 'Walking']; LOW; police 0 (MEDICAL@1.53s;MEDICAL@1.53s); medical 2 |
| `t_n006_converted.avi` | 127f, 0.99 FPS; pistol:0.40-0.65(78); ['Armed Person', 'Raised Hand', 'Running', 'Standing', 'Walking']; HIGH; police 0 (MEDICAL@2.33s;MEDICAL@3.63s;MEDICAL@3.93s); medical 3 | 127f, 1.06 FPS; none; ['Raised Hand', 'Running', 'Standing', 'Walking']; LOW; police 0 (MEDICAL@2.33s;MEDICAL@3.63s;MEDICAL@3.93s); medical 3 |
| `t_n007_converted.avi` | 117f, 1.16 FPS; pistol:0.40-0.60(14); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (MEDICAL@2.63s); medical 1 | 117f, 1.10 FPS; none; ['Running', 'Standing', 'Walking']; LOW; police 0 (MEDICAL@2.63s); medical 1 |
| `t_n008_converted.avi` | 119f, 0.78 FPS; pistol:0.40-0.73(79); ['Armed Person', 'Standing', 'Walking']; HIGH; police 1 (MEDICAL@2.13s;POLICE@3.20s); medical 1 | 119f, 1.15 FPS; none; ['Standing', 'Walking']; LOW; police 0 (MEDICAL@2.13s); medical 1 |
| `t_n009_converted.avi` | 129f, 1.12 FPS; pistol:0.43-0.74(37); ['Armed Person', 'Raised Hand', 'Standing', 'Walking']; HIGH; police 1 (POLICE@3.70s); medical 0 | 129f, 1.37 FPS; none; ['Raised Hand', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 |
| `t_n010_converted.avi` | 138f, 1.17 FPS; pistol:0.41-0.59(87); ['Standing', 'Walking']; MEDIUM; police 0 (MEDICAL@2.33s); medical 1 | 138f, 1.04 FPS; none; ['Standing', 'Walking']; LOW; police 0 (MEDICAL@2.33s); medical 1 |

#### Violence

| Clip | Baseline 0.40 | Tuned 0.75 |
|---|---|---|
| `t_v001_converted.avi` | 300f, 1.14 FPS; pistol:0.40-0.65(168); ['Armed Person', 'Standing']; HIGH; police 1 (POLICE@0.50s); medical 0 | 300f, 1.29 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_v002_converted.avi` | TIMEOUT / no summary | 300f, 1.56 FPS; pistol:0.77-0.77(1); ['Raised Hand', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 |
| `t_v003_converted.avi` | 300f, 1.73 FPS; pistol:0.41-0.51(7); ['Armed Person', 'Standing', 'Walking']; HIGH; police 1 (POLICE@8.40s); medical 0 | 300f, 1.60 FPS; none; ['Standing', 'Walking']; LOW; police 0 (none); medical 0 |
| `t_v004_converted.avi` | 300f, 3.17 FPS; pistol:0.40-0.63(55); ['Armed Person', 'Standing']; HIGH; police 0 (none); medical 0 | 300f, 1.72 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_v005_converted.avi` | 260f, 3.07 FPS; pistol:0.45-0.46(2); ['Standing']; MEDIUM; police 0 (none); medical 0 | 260f, 1.78 FPS; none; ['Standing']; LOW; police 0 (none); medical 0 |
| `t_v006_converted.avi` | 300f, 1.76 FPS; pistol:0.40-0.85(121);knife:0.40-0.58(14); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 | 300f, 1.67 FPS; pistol:0.77-0.85(7); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 |
| `t_v007_converted.avi` | 300f, 1.83 FPS; pistol:0.40-0.73(27); ['Running', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 | 300f, 2.62 FPS; none; ['Running', 'Standing', 'Walking']; LOW; police 0 (none); medical 0 |
| `t_v008_converted.avi` | 300f, 1.60 FPS; pistol:0.40-0.76(136); ['Standing']; MEDIUM; police 0 (none); medical 0 | 300f, 2.54 FPS; pistol:0.76-0.76(2); ['Standing']; LOW; police 0 (none); medical 0 |
| `t_v009_converted.avi` | 300f, 1.54 FPS; pistol:0.40-0.64(39); ['Armed Person', 'Raised Hand', 'Standing']; HIGH; police 0 (none); medical 0 | 300f, 2.43 FPS; none; ['Raised Hand', 'Standing']; LOW; police 0 (none); medical 0 |
| `t_v010_converted.avi` | 300f, 1.12 FPS; pistol:0.40-0.67(28); ['Raised Hand', 'Standing', 'Walking']; MEDIUM; police 0 (none); medical 0 | TIMEOUT / no summary |

Complete machine-readable records: [baseline CSV](verification/scvd_baseline_results.csv), [final CSV](verification/scvd_results.csv), [final JSON](verification/scvd_results.json). Annotated outputs from the initial pass are in `verification/scvd_baseline/`; current 0.75 outputs are in `verification/scvd/`. The timeout clips have partial/no complete output and must not be treated as successful runs.

## Visual review and false positives

- Initial Normal clip `t_n008` produced a `pistol` box over a pedestrian and a police alert at 3.20 s; `t_n009` did the same at 3.70 s. The reviewed event frames show ordinary street pedestrians and no clearly visible weapon inside the red boxes. These are likely false positives. See [t_n008 at 3.20 s](verification/scvd_snapshots/review/t_n008_converted_3.20s.jpg) and [t_n009 at 3.70 s](verification/scvd_snapshots/review/t_n009_converted_3.70s.jpg). The 0.75 cutoff removed both.
- Other common false candidate: `knife` on Weaponized `t_w007` and `t_w008`, confidence up to 0.86/0.87. Reviewed midpoint frames show a large box over a white diagonal rail/structure, not a clearly visible knife. See [t_w007 midpoint](verification/scvd_snapshots/weaponized/t_w007_converted_mid.jpg) and [t_w008 midpoint](verification/scvd_snapshots/weaponized/t_w008_converted_mid.jpg). These boxes persisted above 0.75, so confidence alone does not solve the model’s localization errors.
- Baseline Weaponized alert frames `t_w003` and `t_w006` show red boxes over people; the reviewed frames do not clearly reveal a weapon. The clip-level folder label is not enough to verify the box. See [t_w003 at 6.80 s](verification/scvd_snapshots/review/t_w003_converted_6.80s.jpg) and [t_w006 at 1.60 s](verification/scvd_snapshots/review/t_w006_converted_1.60s.jpg). These are alerts, not confirmed true positives.
- The SCVD Violence clips contain visible altercations in some sampled frames, but weapon localization was not visually confirmed. At baseline, two Violence clips generated police alerts; at 0.75, none of the completed clips did. Violence clips should stay at MEDIUM or lower when no verified weapon is present; the output is not reliable enough to make that distinction consistently.
- Medical heuristics also fired in 8/10 Normal clips on both passes (multiple repeated alerts in some clips). This indicates a serious blood/fall false-alarm limitation; red color and scene geometry are not ground-truth evidence of injury.

## Stage timing and speed

The final dedicated headless 100-frame benchmark (`verification/stage_benchmark_100f.mp4`, 640×360, CPU, current 0.75 config) processed 100 frames in 88.75 s = **1.13 FPS**. It had no police alert. The printed values below are averages per processed source frame. `imshow` is 0 because the run was headless; `imwrite` was 0 because no image snapshot was saved in that benchmark.

| Stage | ms/frame |
|---|---:|
| Person detect + track | 326.85 |
| Weapon detect | 483.10 |
| Pose | 64.57 |
| Behavior logic | 0.06 |
| Medical / blood | 0.17 |
| Threat fusion | 0.08 |
| Evidence write | 0.31 |
| `imwrite` | 0.00 |
| `imshow` | 0.00 (headless) |
| Dispatcher / logging | 0.00 |
| Drawing | 1.62 |

On the 29 complete SCVD clips per pass, the baseline mean stage times were person/track 209.65 ms, weapon 393.86 ms, pose 86.43 ms, behavior 0.11 ms, medical/blood 0.24 ms, threat fusion 0.06 ms, evidence 6.01 ms, imwrite 0.10 ms, dispatcher/logging 0.63 ms, and drawing 7.64 ms. The 0.75 pass measured 198.03 / 347.71 / 79.91 / 0.11 / 0.24 / 0.05 / 3.89 / 0.09 / 0.32 / 5.12 ms respectively. `imshow` is disabled in both passes. Weapon and person inference dominate.

The best full-pipeline clip result was 3.54 FPS (one 300-frame clip); the best dedicated 100-frame benchmark before the last run was 2.47 FPS, while the final 100-frame measurement was 1.13 FPS. On the same benchmark file, earlier measured runs were 2.47 and 2.02 FPS; timings varied substantially under CPU load. The current result is not real-time and does not meet 8 FPS.

Tuning experiments and tradeoffs:

- Weapon model at `imgsz=640`: 281.93 ms/inference (3.55 FPS) in a 100-frame isolated timing. `imgsz=416`: 102.71 ms (9.74 FPS); `imgsz=320`: 77.36 ms (12.93 FPS), but 320 missed the observed knife candidate. Integrated 416 runs measured 2.15/1.85 FPS with BoT-SORT and 2.16 FPS with ByteTrack, versus a 2.02 FPS 640/BoT-SORT run; no consistent overall speed gain was demonstrated. Final sizes remain 640 to avoid unverified accuracy loss.
- Person-only detector timing improved from 260.19 ms at 640 to 160.69 ms at 416 and 129.67 ms at 320, but detections on the dark overlaid benchmark rose 37→90→112; final person size remains 640.
- Throttling the evidence disk-cap check reduced measured evidence-save time from 47.58 ms to 5.55 ms per snapshot (about 8.6×); this small component is not the main pipeline bottleneck.
- Final cadence: weapon inference every frame; person tracking and pose/behavior sampling every third frame where applicable; pose is skipped when no person is present. `DEVICE="auto"` chooses CPU here. Half precision is only enabled for non-CPU devices and was not exercised.
- Four-frame transient clip `163318`: stride-three baseline missed the sole raw candidate (1/1 missed = 100% raw-candidate miss rate); weapon inference every frame saw it (0/1 missed = 0% raw-candidate miss rate). The two-frame filter still rejected the single sample, so accepted weapon detections/alerts were 0/1 both before and after. This is why weapon inference stays every frame.

## Final configuration and filter effect

| Setting | Final value | Reason / evidence |
|---|---:|---|
| `DEVICE` | `auto` | Uses CUDA if available; CPU here. |
| Person / weapon / pose image size | 640 / 640 / 640 | Lower isolated sizes did not improve integrated throughput reliably; lower sizes reduced candidate detections. |
| `PERSON_TRACKER` | `botsort.yaml` | ByteTrack trial did not improve integrated FPS. |
| `PERSON_BEHAVIOR_EVERY_N_FRAMES` | 3 | Reduce person/pose work while preserving source-video timestamps. |
| `WEAPON_EVERY_N_FRAMES` | 1 | Avoid skipping brief weapon samples. |
| `WEAPON_CONFIDENCE` | 0.75 | Raised from 0.40 to suppress observed Normal pistol boxes (max 0.74); full rerun removed Normal police alerts but also Weaponized police detections (2/10→0/10). |
| `WEAPON_CONFIRM_FRAMES` | 2 | Require two overlapping samples. Suppressed single high-confidence candidates (for example Violence 002 at 0.77); did not prevent repeated low-threshold Normal false alerts. |
| `WEAPON_TRACK_IOU` | 0.30 | Existing overlap threshold for the persistence filter. |
| Evidence cap check | 30 s | Avoid a disk-cap scan on every evidence frame. |

No more config changes were made after the 0.75 all-folder rerun. No optional second 10-clip sample was run because the first sample required long CPU processing and the current result remains below target speed.

## Demo plan

No Weaponized clip produced a police alert with the final `WEAPON_CONFIDENCE=0.75` setting. `demo.bat` therefore launches `demo_dashboard_only.bat` and opens annotated `t_w003` and `t_n003` videos; the simulated dashboard incidents are labeled as simulations. The Weaponized annotated clip is not a verified alarm demonstration. The Normal clip shows LOW/no weapon output. The batch launcher itself was not executed in this verification session.

## Regression tests and remaining verification

`python selftest.py` passed after the parser correction, after the confidence-threshold change, and after the demo launcher edit: 12 unittest checks passed each time, plus the mocked 100-frame frame-loop scenario. The tests cover Flask API idempotency/status/GPS/path checks, evidence pre-event order/FPS/cap, dispatcher persistence/cooldowns, and mocked server-down outbox/replay behavior.

- The server-down self-test used a closed local port and mocked replay success. It did **not** stop the live Flask process during an actual YOLO run or deliver a real alert back to the restarted server. To verify manually: start a long sample with `python main.py videos/weaponized/t_w003_converted.avi --no-display --max-frames 300`; stop the local server while it runs; confirm the video loop continues and `evidence/outbox.jsonl` gains a POST; restart `python server.py`; run `python replay_outbox.py`; query `/api/incidents` and confirm exactly one incident ID.
- A bounded normal-EOF smoke run (`t_n003`, 3 frames) wrote a readable 3-frame MP4 at 30 FPS. A Ctrl+C attempt in the console terminated the Windows process before a readable output file was finalized; clean Ctrl+C flush/release is therefore **not verified**. Manual check: run a longer clip with `--no-display --save-annotated verification/shutdown_test.mp4`, press Ctrl+C after several seconds, then open the file in OpenCV and confirm the last frames are readable.
- `demo.bat` and `demo_dashboard_only.bat` were not launched interactively; their server-window, browser-tab, default-video-player and simulation behavior is not verified. Run `demo.bat` on the presentation PC and confirm both dashboard tabs and both MP4s open.
- The user previously verified dashboards, siren, map, live-GPS dot and status buttons on their PC; this session only confirmed the fresh server root returned HTTP 200. Phone GPS over HTTPS, Telegram delivery, webcam capture, and live-model behavior on a webcam were not tested here.
- Falls/running/blood have no frame-level labels in this sample. The medical heuristic raised alerts in 8/10 Normal clips. Tracking ID stability and true weapon localization are not validated.

### Dataset citation placeholder

SCVD (Kaggle): exact dataset title: **[user fill in]**; author/uploader: **[user fill in]**; license: **[user fill in from the Kaggle dataset page]**; dataset page URL: **[user fill in]**. Do not present a guessed license as fact.

### Responsible use

Alerts are decision support only. A human must inspect the scene and evidence before any dispatch. Use the system only where surveillance is lawful and disclosed; restrict access to evidence, set a documented retention period, and securely delete recordings when no longer needed. The evidence cap deletes only unreferenced media, so incident-linked files may remain beyond the cap. The dashboard has no authentication and is for a controlled demo, not deployment.

### Viva Q&A additions

- **Does it meet real-time speed?** No. On this CPU-only i3-1005G1, the final 100-frame run measured 1.13 FPS; best observed full-pipeline clip rate was 3.54 FPS. It does not meet 8 FPS.
- **What is the weapon detection rate?** At the initial 0.40 threshold, 2/10 Weaponized clips produced police alerts and 2/10 Normal clips also did. At 0.75, both rates were 0/10. These are class-level alert rates, not ground-truth detector recall.
- **Why raise confidence to 0.75 if it misses detections?** It removed the observed Normal false police alerts, but also removed Weaponized alerts. The result exposes the model’s poor separation; labeled boxes and further model work are needed before claiming reliable alerts.
- **What are the common false positives?** Pistol boxes over ordinary pedestrians at roughly 0.40–0.74 confidence, and high-confidence knife boxes over a diagonal rail/structure. Visual review found no clearly visible weapon in those boxes.
- **Why use two-frame persistence?** It rejects isolated samples, but can miss brief weapons: one high-confidence sample at a time is suppressed. On the Normal sample it did not remove repeated detections at the original 0.40 threshold.

### Three weakest points

1. Weapon model localization/classification is unreliable: Normal false alerts at 0.40, zero Weaponized alerts at 0.75, and repeated knife boxes on scene geometry.
2. CPU throughput is 1.13 FPS on the final 100-frame run, below the 8 FPS target and unsuitable for real-time use.
3. Medical/fall/blood heuristics are weakly validated; Normal clips produced medical alerts in 8/10 cases, and this sample has no ground truth to establish injury recall or precision.

### README claims not confirmed in this session

The prior user-reported dashboard/siren/map/status behavior was not retested in the browser. This session did not verify the launchers, phone GPS over HTTPS, Telegram, webcam capture, or any real police/hospital dispatch integration. No real automatic dispatch is implemented.
# VantaWatch Phase 4 Verification

Run date: 2026-11-19. Runtime: Python 3.12, Windows, OpenCV 5.0, PyTorch CPU-only (`cuda_available=False`, 2 PyTorch threads). The local Flask server was started for the video sweep and returned HTTP 200 on `/`.

## Real-video sweep

`videos/` had no MP4 files. `evidence/videos/` contained the 14 clips below. Each was run through the actual person, weapon, and pose pipeline without a display, saving annotated MP4s under `verification/annotated/`. Original model thresholds were retained. Durations/FPS below are from video metadata and whole-run elapsed time (including model initialization), so the shortest clips are heavily affected by startup. The first sweep used inference on every frame; the second used `INFERENCE_EVERY_N_FRAMES=3`. The weapon persistence rule was added after that sweep and was verified with a repeat run of the affected clip.

| Clip | Duration | FPS before → stride 3 | Max people | Raw weapon result (stride 3) | Behavior seen | Final max threat / alerts | Notes |
|---|---:|---:|---:|---|---|---|---|
| 20260806_163300 | 0.95 s | 0.70 → 0.83 | 0 | none | none | LOW; 0 police, 0 medical | Low-light; no detections |
| 20260806_163305 | 0.95 s | 0.58 → 1.74 | 0 | none | none | LOW; 0 / 0 | Low-light; no detections |
| 20260806_163310 | 0.20 s | 0.45 → 0.32 | 0 | none | none | LOW; 0 / 0 | Startup dominates |
| 20260806_163311 | 0.15 s | 0.32 → 0.20 | 0 | none | none | LOW; 0 / 0 | Startup dominates |
| 20260806_163312 | 0.95 s | 0.97 → 1.28 | 0 | none | none | LOW; 0 / 0 | No detections |
| 20260806_163318 | 0.20 s | 0.41 → 0.40 | 0 | none | none | LOW; 0 / 0 | Full-frame baseline opened evidence; stride 3 missed that transient trigger. It may be a false positive or a missed short event; the clip is too short to decide. |
| 20260806_163319 | 0.15 s | 0.37 → 0.25 | 0 | none | none | LOW; 0 / 0 | Startup dominates |
| 20260806_163320 | 0.40 s | 0.67 → 0.52 | 0 | none | none | LOW; 0 / 0 | No detections |
| 20260806_163322 | 3.50 s | 1.09 → 2.16 | 1 | none | Standing | LOW; 0 / 0 | Person appears around 2.0 s; no fall or threat event |
| 20260806_163343 | 1.15 s | 0.78 → 1.73 | 1 | pistol 0.428, one sample; accepted after persistence: 0 | Standing | LOW after filter; 0 / 0 | Frame 22 at 1.10 s was a clear full-frame-box false positive. A filtered repeat stayed LOW; whole-run rate varied to 0.55 FPS. |
| 20260806_163350 | 3.50 s | 0.76 → 2.00 | 3 | none | Standing | LOW; 0 / 0 | No clear incident; people are partly framed |
| 20260806_163409 | 0.10 s | 0.21 → 0.23 | 0 | none | none | LOW; 0 / 0 | Startup dominates |
| 20260806_163410 | 0.25 s | 0.43 → 0.45 | 0 | none | none | LOW; 0 / 0 | Startup dominates |
| 20260806_163411 | 0.55 s | 0.34 → 0.81 | 0 | none | none | LOW; 0 / 0 | No detections |

The clips are short, dim recordings and several contain colored boxes/labels baked into earlier output. They are not clean raw-camera validation footage. No clip provides a known weapon, running event, or fall ground truth. I therefore cannot calculate detection recall, assert ID stability, or call the standing detections accurate for every person. No police or medical incidents were produced by the real clip sweep.

## Regression and tuning record

| Check | Before | After | Result |
|---|---|---|---|
| Long clip 163322 throughput | 1.09 FPS, all frames inferred | 2.16 FPS, every third frame inferred | About 1.98× for this run; still far below 8 FPS |
| Long clip 163350 throughput | 0.76 FPS | 2.00 FPS | About 2.63× for this run; still far below 8 FPS |
| Weapon false positive, 163343 | `pistol`, confidence 0.428, frame 22, bbox `(0,57,640,474)` | Repeated-sample filter rejected it; filtered run max threat LOW | Confirmed by direct weapon-model replay and annotated frame |
| Short transient, 163318 | Evidence recording began in the full-frame sweep | No detection at stride 3 | Sampling regression; do not use this cadence where sub-0.15-second events matter |
| Motion/fall/medical thresholds | unchanged | unchanged | Footage had no labelled running/fall/blood examples, so no threshold tuning was justified |

The annotated contact sheet is [verification/contact_sheet.png](verification/contact_sheet.png). The false-positive frame is [verification/weapon_false_positive_frame22.jpg](verification/weapon_false_positive_frame22.jpg). Annotated output clips are in `verification/annotated/`. The `pistol` box covered essentially the entire 640×480 image and did not localize a weapon. Requiring two overlapping samples removes this single-frame error; it can also delay or miss brief real weapon views.

## Automated tests

`python selftest.py`: **12 unittest checks passed, 0 failed**, plus the mocked 100-frame main-loop integration passed. The suite covers pose geometry/confidence, motion normalization, IoU/weapon association, weapon persistence, threat cases, medical persistence, dispatcher confirmation/cooldown/severity escalation, evidence pre-event frame order and FPS, retention protection for incident-linked evidence, Flask idempotency/status/video/GPS/live and stale location/path traversal/template escaping/reload, and asynchronous notifier fallback plus replay. The mocked loop injected a zero source FPS, a corrupted frame, and a detector exception; it continued, then produced exactly one police and one medical notification.

The server-down check sent to a closed local port, verified `send()` returned immediately, found the queued operation in `outbox.jsonl`, and replayed it with a mocked successful response. It did not kill the already-running Flask process in the middle of an actual YOLO run. The persistence check called the same JSON load function after clearing in-memory state; it did not stop and relaunch the server process.

## Configuration changes this phase

| Setting | Value | Reason |
|---|---:|---|
| `INFERENCE_EVERY_N_FRAMES` | 3 | Configurable throughput improvement; tradeoff and missed short event are recorded above |
| `WEAPON_CONFIRM_FRAMES` | 2 | Suppress the observed one-sample full-frame pistol false positive |
| `WEAPON_TRACK_IOU` | 0.30 | Match overlapping same-class weapon boxes between inference samples |
| `EVIDENCE_MAX_GB` | 5.0 | Bound unreferenced evidence; referenced incident files are protected |
| `PERSON_CONFIDENCE` | 0.30 | Moved the existing person confidence literal into config without tuning it |
| `POSE_DETECT_CONFIDENCE` | 0.40 | Moved the existing pose confidence literal into config without tuning it |
| `BLOOD_HUE_LOW` / `BLOOD_HUE_HIGH` | 8 / 172 | Moved existing HSV hue limits into config |
| `BLOOD_SAT_MIN` / `BLOOD_VALUE_MIN` / `BLOOD_VALUE_MAX` | 110 / 40 / 190 | Moved existing HSV saturation/value limits into config |

Existing person/pose, motion, fall, blood, and threat thresholds were not tuned because these videos provide no reliable positive/negative ground-truth cases. Person/pose confidence and blood HSV values were moved into `config.py` without changing their values so they can be tuned there.

## Not verified on real hardware/services

- **Webcam:** run `python main.py 0 --no-display --max-frames 300`; then run `python main.py 0` to verify the window and q/Esc/close behavior.
- **Phone GPS/HTTPS:** run `python server.py --https`, open `https://<laptop-ip>:5000/gps` on the same Wi-Fi, accept the local certificate, start location, and verify the blue dot moves on both dashboards.
- **Telegram:** set `TELEGRAM_BOT_TOKEN`, `TELEGRAM_POLICE_CHAT`, and `TELEGRAM_HOSPITAL_CHAT`; create a simulated police and medical incident and confirm the respective chats receive one message.
- **Windows demo launchers:** run `demo.bat` and `demo_dashboard_only.bat` on the presentation laptop; the batch files were created but not launched in this session.
- **Real fall/blood/weapon events:** test with labelled clips or staged objects/poses in a controlled environment. Current videos do not establish these detections.

The live dashboard, browser siren/map, and phone GPS were reported as working by the user before this phase; they were not rechecked in a browser here. HTTPS, Telegram, and a real police connection remain unverified.

The police and medical `simulate_incident.py` modes were posted to the running local Flask server. The fake `gps` mode also ran to completion for its 60-second update sequence. This verifies the simulator clients and local API path, not phone geolocation or the live browser map.

CLI verification: `python main.py --help` displayed all three requested flags. A real-model smoke run with `--no-display --max-frames 2 --save-annotated verification/cli_smoke.mp4` completed successfully (2 frames, 13.77 seconds including model startup); it produced no alert.

---

# Final time-boxed round (authoritative current result)

Date: 2026-11-20. This section supersedes the earlier Phase 5 settings wherever they describe the current configuration. The model files were not changed, no training was run, and all replay sweeps below use cached detections without model inference.

## One-pass detection cache

The first five filenames, sorted case-insensitively, were selected from each already-staged tuning folder. Each was capped at 150 decoded frames. The cache collector ran person tracking, pose/behavior, and weapon inference once on the selected set; weapon inference used a 0.25 confidence floor so all requested thresholds could be replayed. Per-frame cache records include video timestamp, frame size, person IDs/boxes, pose labels and keypoint confidence arrays, weapon class/confidence/box/center, speed, and red-pixel ratio per person box. Files are in `verification/cache/`.

| Folder | Cached clips | Decoded frames | Max per clip |
|---|---|---:|---:|
| Weaponized | `t_w001`–`t_w005` | 750 | 150 |
| Normal | `t_n001`–`t_n005` | 632 | 150 |
| Violence | `t_v001`–`t_v005` | 750 | 150 |
| **Total** | **15** | **2,132** | **150** |

Summed per-clip model processing time was **896.9 s (14 min 57 s)**, averaging **2.38 frames/s**. The per-clip timer excludes model construction/startup and script startup, so total wall time was longer. The sweep/filter comparisons and demo generation made no new model calls.

Transparency: before the revised 90-minute instruction arrived, an incomplete medical-only pass had already processed the first three Normal clips and was canceled. Those three clips were then included in the one-pass cache, so their model detections were repeated once during this work. No full-dataset run occurred.

## Model-free replay validation

`replay.py` replays cached data through movement/pose fusion, threat scoring, medical logic, weapon persistence, and police/medical confirmation/cooldowns. It does not load a model. On the original captured configuration (`WEAPON_CONFIDENCE=0.75`, before the final filters), police and medical event lists matched the model-backed cache reference exactly on all five cached Normal clips (`t_n001`–`t_n005`). Each had zero police and zero medical events. The check is recorded in `verification/cache/normal_replay.json`; the initial two-clip check is in `verification/replay_results.json`.

The final filters were evaluated by replay only; their exact outputs were not cross-checked by a second live model pass because the one-pass cache rule prohibits re-running YOLO for tuning. One current police-alert snapshot is saved at `verification/replay_alert_frames/violence_t_v001_converted_f21.jpg` and its manifest is next to it. Visual review: the `pistol 0.42` box covers an unclear object near pedestrians' feet. Label: **uncertain**. It is not counted as a verified weapon.

## Threshold sweep on cached tuning clips

Detection values are counts of per-frame boxes across five clips per folder (up to 750 decoded frames per folder), not unique objects. `best.pt` reports `model.names={0: pistol, 1: knife}`. Police values count alerts after the existing two-frame persistence and six-high-sample police confirmation, before the additional final filters.

| Confidence | Weaponized detections (knife / pistol) | Weaponized police | Normal detections (knife / pistol) | Normal police | Violence detections (knife / pistol) | Violence police |
|---:|---:|---:|---:|---:|---:|---:|
| 0.40 | 6 / 35 | 0 | 0 / 2 | 0 | 0 / 148 | 1 |
| 0.50 | 3 / 11 | 0 | 0 / 0 | 0 | 0 / 28 | 0 |
| 0.60 | 0 / 0 | 0 | 0 / 0 | 0 | 0 / 2 | 0 |
| 0.70 | 0 / 0 | 0 | 0 / 0 | 0 | 0 / 0 | 0 |

There is no threshold in this reduced sample that produced a Weaponized police alert. I retained the 0.40 base point because it retains the most Weaponized-folder detection evidence among the requested sweep and still produced no Normal police alert after person-contact gating; I did **not** select a high threshold just to suppress alerts. This is not a satisfactory operating point: it has zero Weaponized police alerts and one uncertain Violence alert. The alert-rate trade-off requirement is unmet by the available evidence.

## Filter effects and final tuning-set replay

Counts below are from the same cached 5-per-folder sample, replayed with no model calls. `P/K` means pistol/knife box counts.

| Replay scenario | Weaponized P/K; police | Normal P/K; police | Violence P/K; police | Change observed |
|---|---:|---:|---:|---|
| 0.40 scalar threshold, no extra filters | 35 / 6; 0 | 2 / 0; 0 | 148 / 0; 1 | Baseline |
| Per-class: pistol 0.40, knife 0.50 | 35 / 3; 0 | 2 / 0; 0 | 148 / 0; 1 | Removed 3 knife boxes; police count unchanged |
| Maximum box area 10% | 35 / 6; 0 | 2 / 0; 0 | 147 / 0; 1 | Removed 1 large box; police count unchanged |
| Person box contact required | 35 / 6; 0 | 2 / 0; 0 | 148 / 0; 1 | Alert unchanged because its box touches/overlaps a person box |
| Persistence raised to 3 frames | 35 / 6; 0 | 2 / 0; 0 | 148 / 0; 1 | Alert unchanged; candidate persisted long enough |
| **Combined final configuration** | **35 / 3; 0** | **2 / 0; 0** | **147 / 0; 1** | **Same one uncertain Violence alert** |

When a confirmed weapon has no touching/overlapping person box, the threat engine keeps at least MEDIUM activity so evidence can be recorded; the dispatcher blocks its police alert. Synthetic unit tests cover touching contact, no contact, configurable per-class thresholds, box-area rejection, and persistence. On these clips the only police alert passed the person gate, so the gate did not reduce the observed alert count.

## Medical diagnosis and result

The former blood-only trigger replayed on the five cached Normal clips would have raised **6 false medical candidates**. The current medical detector raised **0/5 medical alerts** in those clips. Saved diagnostics and inspected example frames are under `verification/cache/medical_fp/`; the complete candidate list is `verification/cache/medical_legacy_diagnostics.json`.

| Clip | Blood-only candidates | Box dimensions (px) | Box height (% frame) | Edge touches | Overlapping people |
|---|---:|---|---:|---:|---:|
| `t_n001` | 1 | 49×107 | 29.72% | 0 | 0 |
| `t_n002` | 1 | 54×119 | 33.06% | 0 | 0 |
| `t_n003` | 0 | — | — | — | — |
| `t_n004` | 2 | 76×91; 50×118 | 25.28%; 32.78% | 0 | 0 |
| `t_n005` | 2 | 31×106; 53×121 | 29.44%; 33.61% | 0 | 1 (on one target) |
| **Total** | **6** |  |  | **0/6** | **1/6** |

Cause breakdown for all six legacy candidates: pose-fall signal **0/6**, lying-aspect signal **0/6**, blood-color signal **6/6**, blood-only cause **6/6**. Visual review showed upright shoppers; red clothing/scene color drove the heuristic. None of the six was edge-clipped. The new rules require sufficient track age and height, reject near-edge or heavily overlapped boxes, require stillness plus fall/aspect evidence, require at least six confident keypoints for pose-fall evidence, preserve `MEDICAL_FALL_SECONDS=2.0`, and permit blood only as support for a confirmed down-person alert. The target of at most one false alarm was met on these five clips (0/5); this does not revalidate the earlier 10-clip set.

## Held-out check

The source Test split contained 24 Weaponized, 46 Normal, and 12 Violence video files. The intended clips 11–13 were `t_w011`–`t_w013`, `t_n011`–`t_n013`, and `t_v011`–`t_v013`; the supplied Violence folder has no `t_v013`, so only `t_v011` and `t_v012` were available. A real `main.py` run on `t_w011_converted.avi` with the final configuration and `--max-frames 150` exceeded the runner's 240-second limit without a complete `[RUN]` summary. To stay inside the time cap, the remaining seven clips were skipped. **Held-out police/medical counts are unverified; no tuning-versus-held-out comparison can be claimed.** See `verification/heldout/sample.json`, `results.json`, and the timeout log.

## Final configuration

| Setting | Value | Reason |
|---|---:|---|
| `DEVICE` | `auto` (resolved to CPU) | CUDA unavailable in the installed PyTorch build |
| Person/weapon/pose image size | 640 / 640 / 640 | Existing tested sizes; weights unchanged |
| `PERSON_BEHAVIOR_EVERY_N_FRAMES` | 3 | Existing cadence; cache preserves timestamps |
| `WEAPON_EVERY_N_FRAMES` | 1 | Avoid skipping brief candidates |
| `WEAPON_CONFIDENCE` | 0.40 | Lowest requested sweep threshold; retains more candidate signal |
| `WEAPON_CLASS_CONFIDENCE` | pistol 0.40; knife 0.50 | Removes 3 of 6 low-threshold knife boxes on Weaponized sample; class-dependent tuning remains provisional |
| `WEAPON_MAX_BOX_AREA_RATIO` | 0.10 | Rejects very large boxes; removed one box in this cache |
| `WEAPON_REQUIRE_PERSON_CONTACT_FOR_POLICE` | `True` | Police needs a weapon/person box contact; personless weapons can still be MEDIUM/evidence |
| `WEAPON_CONFIRM_FRAMES` | 2 | Baseline persistence; three frames did not change the one alert and could further reduce sensitivity |
| `POLICE_CONFIRM_FRAMES` / cooldown | 6 / 30 s | Existing values |
| `MEDICAL_MIN_TRACK_AGE_SECONDS` | 1.0 | Avoid immediate one-frame track alerts |
| `MEDICAL_MIN_BOX_HEIGHT_RATIO` | 0.15 | Ignore very small person boxes |
| `MEDICAL_EDGE_MARGIN_RATIO` | 0.01 | Ignore people at frame edges |
| `MEDICAL_MAX_OCCLUSION_RATIO` | 0.45 | Ignore heavy overlap |
| `MEDICAL_STATIONARY_SPEED` | 0.35 | Require near-stationary person while down |
| Pose fall keypoints | ≥6 at confidence ≥0.50 | Require enough reliable joints |
| `LYING_ASPECT` / `MEDICAL_FALL_SECONDS` | 1.25 / 2.0 s | Keep fall confirmation duration and existing aspect rule |
| Blood confirmation | 1.5 s; down person required | Blood pixels are supporting evidence only |

No weights were trained or edited. `torch.cuda.is_available()` was false; GPU performance was not tested. Historical 300-frame Phase 5 results remain in the earlier section and are not measurements of this final configuration.

## Offline demo status

`make_demo.py --build-only` generated `verification/offline_demo/normal_quiet.mp4`: 117 frames at the source rate of 30 FPS, zero police events, zero medical events. The MP4 overlay, manifest, `play_demo.bat`, dashboard query banner (`?offline=1`), and incident `analysis_mode` are labeled **RECORDED CLIP ANALYSED OFFLINE**. This video was built from cached detections and replayed logic; it made no new model calls. `play_demo.bat`, live server posting, and timed delivery for a nonempty alert schedule were not run. No alert candidate was visually confirmed as a genuine weapon, so a weapon-positive demo was not created. Do not present the uncertain Violence alert as genuine.

## Final self-test and unverified items

The final `python selftest.py` run passed **22 unittest checks, 0 failures**, plus the mocked 100-frame pipeline integration. Tests include the six medical safeguards, red-shirt/red-patch cases, person-contact gating, per-class confidence filtering, max box area, and configurable weapon persistence. A transient test failure while updating the threat test fixture was corrected; the final run passed.

Still unverified: the full held-out set, live dashboard POST/timed alert playback from `make_demo.py`, running `play_demo.bat` on the presentation PC, CUDA/GPU behavior, real weapon correctness, running/fall behavior on labeled clips, and any real dispatch/GPS/Telegram hardware path.

## Model provenance

`weights/best.pt` came from a third-party GitHub repository; it was not trained by the user. Complete before submission:

- Repository URL: **[user fill in]**
- Author/maintainer: **[user fill in]**
- License: **[user fill in from repository]**
- Training dataset: **[user fill in from repository/model card]**

The project contribution is the detector orchestration, tracking and pose/motion rules, threat/medical logic, filters, evidence flow, cached replay, and dashboards. The pretrained weapon model is an external dependency.

## Responsible use and viva answers

Use this as a human-reviewed decision-support prototype only. Do not dispatch from a model score alone. Protect and limit retention of CCTV evidence, disclose monitoring, and do not use the unauthenticated dashboard for deployment.

- **Threshold trade-off:** lowering the threshold preserves more detections but increases false candidates; on this reduced sweep no threshold yielded a Weaponized police alert, so 0.40 is provisional and not a solved trade-off.
- **Weapon-person gating:** box contact stops a personless weapon-like object from producing police dispatch while still allowing MEDIUM evidence capture. It cannot reject a false box that overlaps a person.
- **Held-out testing:** new clips check whether tuning generalizes. This run timed out on its first held-out clip, so generalization was not measured.
- **Project contribution vs pretrained model:** the project supplies the integration, tracking/pose/motion/medical/threat logic, filters, replay, and dashboard; `best.pt` is third-party and was not trained here.
- **Why not real-time:** measured cache processing averaged 2.38 FPS on CPU, and a capped live main run timed out at 240 seconds; the system needs offline playback and human review.

## Three weakest points and demonstration guidance

1. The weapon detector has no verified Weaponized police alert in this reduced sample; its only alert is an uncertain object in a Violence clip.
2. The full held-out check timed out before producing a valid result.
3. CPU speed and clip-level labels do not support real-time or accuracy claims; medical results only cover five cached Normal clips.

Show the quiet Normal offline video, the clear offline dashboard banner, simulator workflows, and unit-test rule cases. Avoid claiming that the uncertain Violence object is a weapon, avoid any claim of held-out success, and do not describe the pipeline as real-time or validated for dispatch.
# Behavior feedback follow-up — 2026-11-20

## What was observed and changed

A single real-model diagnostic was run on `videos/violence/t_v001_converted.avi`, capped at 60 frames, on CPU. It processed 60 frames in 222.73 seconds (0.27 FPS). This was not a full clip evaluation and no model was retrained or changed. The raw pistol candidate appeared in 47 frames with confidence 0.41–0.53; the run produced one HIGH police alert at 0.70 s. I reviewed the extracted raw and annotated frames around 21 and 30 and did not see a clearly identifiable weapon. This candidate/alert is therefore unconfirmed and should be treated as a likely false positive in the reviewed scene, not a true weapon detection. The clip is labeled Violence at clip level; that alone does not verify any individual detection.

The same short run showed people listed as Standing on the initial frame and after tracking gaps. The old behavior logic assigned Standing even when there was no usable motion history, and its endpoint-only displacement could miss direction-changing motion. Changes now return Unknown until there are at least two motion samples, calculate filtered path speed alongside net speed, and expose per-person behavior, pose, and normalized speed in the monitoring panel. Weapon boxes and summaries are now labeled as candidates requiring review. Confidence, person-association, persistence, and police-alert thresholds were not weakened or otherwise changed in this follow-up.

`python selftest.py`: **25 tests passed** after these changes. The path-speed and unknown-state behavior is covered with synthetic tests. The 60-frame real-model diagnostic was not rerun after the code change because it took over 3.5 minutes for 60 frames; the improvement on real footage is therefore unverified. Running detection remains a heuristic, not a trained violence/action classifier. This diagnostic supports no claim about running elsewhere in the video.

# User supplied fall video check — 2026-11-20

Three supplied MP4s were copied to `verification/fall_samples/` and each was run once through the real CPU pipeline with annotated output. The clips are 640×240 at 30 FPS: `fall-01-cam0.mp4` (160 frames / 5.33 s), `fall-01-cam1.mp4` (160 / 5.33 s), and `fall-02-cam0.mp4` (110 / 3.67 s). The two GIFs depict an altercation and a crowded scene; they were not used as fall test clips.

| Clip | Processed | FPS | Observed behavior labels | Police | Medical | Siren |
|---|---:|---:|---|---:|---:|---|
| `fall-01-cam0.mp4` | 160 frames | 0.56 | Fall Detected appeared in clip summary; reviewed down-person frame 120 was labeled Standing | 0 | 0 | Not triggered |
| `fall-01-cam1.mp4` | 160 frames | 1.34 | Fall Detected appeared in clip summary; reviewed frame 120 was labeled Raised Hand | 0 | 0 | Not triggered |
| `fall-02-cam0.mp4` | 110 frames | 1.00 | Fall Detected appeared in clip summary; reviewed frame 90 showed Walking / Unknown | 0 | 0 | Not triggered |

All three runs had zero medical alerts. The `Fall Detected` summary means the pose heuristic emitted that label at some point; it does not mean the detector confirmed a sustained medical event. The medical rule separately requires a sufficiently old, large, in-frame, non-crowded, nearly stationary track and persistent fall/lying evidence. The first clip's reviewed down-person frame is partly clipped at the bottom edge, which the current safety filter rejects. On the other reviewed frames the person was not labeled as down. The annotated MP4s and inspected frames are in `verification/fall_samples/`.

The dispatcher already starts the softer local siren asynchronously when it sends a MEDICAL incident, and the browser incident path has self-test coverage. This dataset run did **not** exercise that path because no medical cases were emitted. So it is verified that a confirmed MEDICAL incident calls the siren, but **not** that this detector currently recognizes the supplied falls. I did not weaken the edge, confidence, track-age, overlap, stationarity, or persistence filters. A more permissive trigger would risk sounding on sitting, occlusion, or transient pose errors and requires an explicit operating trade-off. `python selftest.py` passed 25 tests after the three runs.

