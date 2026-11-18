@echo off
SETLOCAL EnableDelayedExpansion
title AeroTwin Mission Control -- DNAA to DNMM
color 0B

echo.
echo  ============================================================
echo   AEROTWIN PREFLIGHT -- DNAA (Abuja) to DNMM (Lagos)
echo  ============================================================
echo.

:: ─── [1/3] POSTGRESQL ────────────────────────────────────────────────────────
echo [1/3] Starting PostgreSQL...
net stop postgresql-x64-13 >nul 2>&1
timeout /t 2 /nobreak >nul
net start postgresql-x64-13 >nul 2>&1
if %errorlevel% neq 0 (
    echo        WARNING: PostgreSQL may not have started.
    echo        Check service name in services.msc
) else (
    echo        PostgreSQL OK.
)
timeout /t 3 /nobreak >nul

:: ─── [2/3] PYTHON AI HUB ─────────────────────────────────────────────────────
:: Python MUST start first — it is the TCP server on port 5000.
:: FlightGear connects TO Python, so Python must be listening before FG starts.
echo [2/3] Starting Python AI Co-Pilot Hub (server on port 5000)...
start "AeroTwin_AI_Hub" cmd /k "conda activate aerospace && python %~dp0live_diagnostics.py"
echo        Waiting 10s for Python to bind port 5000...
timeout /t 10 /nobreak >nul

:: ─── [3/3] FLIGHTGEAR ────────────────────────────────────────────────────────
:: FG starts LAST. It connects TO Python (port 5000) and MATLAB (port 5002).
:: FG also opens port 5001 as server (Python will connect to it for commands).
::
:: IMPORTANT: aerotwin_proto.xml must be in FlightGear's data\Protocol\ folder
::
echo [3/3] Launching Dornier 228-212 at Abuja (DNAA)...
start "FlightGear_Do228" "C:\Program Files\FlightGear 2024.1\bin\fgfs.exe" ^
    --aircraft=do228-212 ^
    --airport=DNAA ^
    --nav1=109.70 ^
    --timeofday=noon ^
    --prop:/sim/sound/working=true ^
    --generic=socket,out,20,localhost,5000,tcp,aerotwin_proto ^
    --generic=socket,in,20,localhost,5001,tcp,aerotwin_proto ^
    --generic=socket,out,20,localhost,5002,tcp,aerotwin_proto

echo.
echo  ============================================================
echo   ALL SYSTEMS LAUNCHED
echo.
echo   PORT LAYOUT:
echo     5000 - FG sends telemetry TO Python (Python is server)
echo     5001 - Python sends commands TO FG  (FG is server)
echo     5002 - FG sends physics TO MATLAB   (MATLAB is server)
echo.
echo   MATLAB: Open manually, then run launch_twin.m
echo.
echo   FLIGHT PROCEDURE:
echo     1. Wait for FG to fully load at Abuja runway
echo     2. Start engines, taxi, and take off MANUALLY
echo     3. AI Co-Pilot engages automatically at 500ft AGL
echo     4. Co-Pilot flies to Lagos DNMM, ILS runway 18L
echo     5. AI diagnostic fires at touchdown
echo.
echo   GRAFANA: http://localhost:3000 (PostgreSQL telemetry)
echo  ============================================================
echo.
pause