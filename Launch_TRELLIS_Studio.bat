@echo off
title TRELLIS 3D Studio Launcher
cd /d "%~dp0"
echo =====================================================================
echo Launching TRELLIS 3D Studio (Standalone Windows Desktop App)
echo GPU: NVIDIA GeForce RTX 4060 (CUDA Enabled)
echo =====================================================================
if exist "TRELLIS-for-windows\.venv\Scripts\pythonw.exe" (
    start "" "TRELLIS-for-windows\.venv\Scripts\pythonw.exe" desktop_studio_app.py
) else (
    start "" "TRELLIS_3D_Studio.exe"
)
exit
