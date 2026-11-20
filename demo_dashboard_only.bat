@echo off
cd /d "%~dp0"
if not defined SV_PORT set "SV_PORT=5050"
set "PORT=%SV_PORT%"
start "VantaWatch server" cmd /k python server.py --port %PORT%
timeout /t 3 /nobreak >nul
start "" http://localhost:%PORT%/police
start "" http://localhost:%PORT%/hospital
python simulate_incident.py police
timeout /t 4 /nobreak >nul
python simulate_incident.py medical
timeout /t 4 /nobreak >nul
python simulate_incident.py gps
pause
