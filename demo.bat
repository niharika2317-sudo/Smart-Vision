@echo off
cd /d "%~dp0"
start "" "%~dp0demo_dashboard_only.bat"
timeout /t 12 /nobreak >nul
start "" "%~dp0verification\scvd\weaponized\t_w003_converted_annotated.mp4"
start "" "%~dp0verification\scvd\normal\t_n003_converted_annotated.mp4"
echo No Weaponized sample produced a police alert at the final threshold.
echo The dashboard incidents are simulations; the videos are annotated SCVD samples.
pause
