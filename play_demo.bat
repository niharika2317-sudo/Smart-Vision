@echo off
setlocal
cd /d "%~dp0"
if not defined SV_PORT set "SV_PORT=5050"
set "PORT=%SV_PORT%"
echo RECORDED CLIP ANALYSED OFFLINE - this is not a live camera feed.
start "VantaWatch server" /min python server.py --port %PORT%
timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:%PORT%/police?offline=1"
start "" "http://127.0.0.1:%PORT%/hospital?offline=1"
python make_demo.py --build-only
if errorlevel 1 (
  echo Offline demo video generation failed.
  pause
  exit /b 1
)
start "" "%~dp0verification\offline_demo\normal_quiet.mp4"
timeout /t 2 /nobreak >nul
python make_demo.py --post-only
echo RECORDED CLIP ANALYSED OFFLINE - review the clip beside the dashboards.
pause
