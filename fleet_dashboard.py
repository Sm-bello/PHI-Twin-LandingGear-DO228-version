"""
╔════════════════════════════════════════════════════════════════════════╗
║  PHI-FLEET · FEDERATED LANDING GEAR PROGNOSTIC SYSTEM                 ║
║  AeroTwin v4.0 · Dornier 228-212 · UNIFIED SERVER                     ║
║                                                                        ║
║  Includes: FlightGear UDP Bridge, PostgreSQL, FastAPI, WebSocket      ║
║  Broadcast, TF Inference, PyVista 3D — one process, one command.      ║
║                                                                        ║
║  Run:  python fleet_dashboard.py                                       ║
║  Then open http://<this-machine-ip>:8765 — the "FG BRIDGE LINK" and  ║
║  "SUPERVISORS CONNECTED" panels replace the old terminal printouts.   ║
╚════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import socket
import asyncio
import threading
import psycopg2
import tensorflow as tf
import joblib

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import uvicorn
from datetime import datetime, timezone
from typing import List

from aerotwin_config import (
    MODEL_PATH,
    SCALER_SCALAR_PATH,
    SCALER_SEQ_PATH,
    get_db_config,
    get_fg_config,
)
from aerotwin_runtime import (
    ensure_telemetry_schema,
    fetch_fleet_snapshot,
    run_touchdown_diagnostics,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION & ML GLOBALS
# ─────────────────────────────────────────────────────────────────────────────
DB = get_db_config()

SCALER_SEQ    = SCALER_SEQ_PATH
SCALER_SCALAR = SCALER_SCALAR_PATH

POLL_S  = 1.0       
TREND_N = 24        

# ── FlightGear UDP Bridge configuration (formerly fg_connect.py) ──────────
_FG = get_fg_config()
FG_UDP_LISTEN_PORT  = _FG["listen_port"]   # Port this process listens on (FG sends here)
FG_UDP_SEND_PORT    = _FG["send_port"]     # Port FG listens on (gear cmd sent here, unused unless gear_sender started)
FG_HOST             = _FG["host"]
BRIDGE_DB_WRITE_HZ  = 5       # Bridge write rate to PostgreSQL
GEAR_UP_ALT_FT      = 1000
GEAR_DOWN_ALT_FT    = 1200
APPROACH_VS_FPM     = -200
TAKEOFF_GS_KT       = 40
LINK_TIMEOUT_S      = 3.0    # No UDP frame for this long => bridge shows "NO SIGNAL"

# We will load these during the lifespan startup
ai_model = None
s_seq = None
s_scl = None
active_plotter = None
LAST_WOW_STATUS = False


# ─────────────────────────────────────────────────────────────────────────────
# FLIGHTGEAR UDP BRIDGE  (formerly fg_connect.py — now runs in-process)
# ─────────────────────────────────────────────────────────────────────────────
class BridgeState:
    """Shared state written by the UDP thread, read by the DB thread and by
    fetch_all() for the dashboard. Nothing here goes to the terminal —
    it all surfaces in the 'FG BRIDGE LINK' panel instead."""
    lock         = threading.Lock()
    raw          = None            # latest parsed telemetry dict
    prev_wow     = False
    gear_cmd     = "DOWN"
    last_td      = 0.0             # time.time() of last touchdown (debounce)
    last_td_iso  = None            # ISO string for the dashboard
    phase        = "GROUND"
    frames       = 0
    dropped      = 0
    last_frame_t = 0.0             # time.time() of the last UDP packet received
    db_connected = False
    db_attempts  = 0

bridge = BridgeState()


def derive_phase(alt_agl: float, vs_fps: float, gs_kt: float,
                 wow: bool, gear_norm: float) -> str:
    """Derive a human-readable flight phase from raw FG telemetry."""
    vs_fpm = vs_fps * 60.0
    if wow and gs_kt < TAKEOFF_GS_KT: return "GROUND"
    if wow and gs_kt >= TAKEOFF_GS_KT: return "LANDING ROLL"
    if not wow and alt_agl < 500: return "TAKEOFF"
    if not wow and alt_agl < 10000 and vs_fpm > 400: return "CLIMB"
    if not wow and alt_agl > 10000 and abs(vs_fpm) < 400: return "CRUISE"
    if not wow and alt_agl > 10000 and vs_fpm < -400: return "DESCENT"
    if not wow and alt_agl < 5000 and vs_fpm < APPROACH_VS_FPM: return "APPROACH"
    if not wow and alt_agl < 500 and vs_fpm < 0: return "FINAL"
    return "FLIGHT"


def decide_gear(alt_agl: float, vs_fps: float, wow: bool, phase: str) -> str:
    """Autonomous gear command for the Do228. Returns 'UP' or 'DOWN'."""
    vs_fpm = vs_fps * 60.0
    if wow or alt_agl < 200: return "DOWN"
    if alt_agl > GEAR_UP_ALT_FT and vs_fpm > 0 and phase in ("TAKEOFF", "CLIMB"): return "UP"
    if alt_agl < GEAR_DOWN_ALT_FT and vs_fpm < APPROACH_VS_FPM: return "DOWN"
    if phase in ("CRUISE", "DESCENT") and alt_agl > GEAR_DOWN_ALT_FT: return "UP"
    return bridge.gear_cmd


def udp_listener():
    """Listens for 16-field CSV UDP datagrams from FlightGear at up to 20Hz.
    No terminal output — link state is read off bridge.last_frame_t by the
    dashboard's 'FG BRIDGE LINK' panel."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", FG_UDP_LISTEN_PORT))
    sock.settimeout(5.0)

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            raw = data.decode("utf-8", errors="replace").strip()
            parts = raw.split(",")
            if len(parts) < 16:
                with bridge.lock: bridge.dropped += 1
                continue

            f = [float(x) for x in parts[:16]]
            wow_any = (f[4] > 0.5) or (f[5] > 0.5) or (f[6] > 0.5)
            phase   = derive_phase(f[3], f[0], f[8], wow_any, f[4])
            gear    = decide_gear(f[3], f[0], wow_any, phase)

            raw_vs_fps = f[0]
            mass_lbs_raw = f[1]
            if mass_lbs_raw < 100.0:
                mass_lbs_raw = 12125.0  # Do-228 fallback mass (~5500 kg) if FG sends 0

            parsed = {
                "sink_rate_fps": raw_vs_fps, "sink_rate_ms": abs(raw_vs_fps) * 0.3048,
                "mass_lbs": mass_lbs_raw, "mass_kg": mass_lbs_raw * 0.453592,
                "nose_steer": f[2], "alt_agl_ft": f[3],
                "wow_nose": f[4] > 0.5, "wow_left": f[5] > 0.5, "wow_right": f[6] > 0.5,
                "wow_any": wow_any, "compression_ft": f[7], "groundspeed_kt": f[8],
                "prop_rpm": f[9], "engine_torque": f[10], "airspeed_kt": f[11],
                "latitude": f[12], "longitude": f[13], "vsi_fpm": f[14],
                "elevator_trim": f[15], "flight_phase": phase, "gear_cmd": gear,
                "ts": datetime.now(timezone.utc),
            }

            with bridge.lock:
                bridge.raw          = parsed
                bridge.gear_cmd     = gear
                bridge.phase        = phase
                bridge.frames      += 1
                bridge.last_frame_t = time.time()

        except socket.timeout:
            continue
        except ValueError:
            with bridge.lock: bridge.dropped += 1
        except Exception:
            time.sleep(0.1)


def _bridge_db_connect():
    for attempt in range(1, 6):
        try:
            c = psycopg2.connect(**DB)
            with bridge.lock:
                bridge.db_connected = True
                bridge.db_attempts  = attempt
            return c
        except Exception:
            with bridge.lock:
                bridge.db_connected = False
                bridge.db_attempts  = attempt
            time.sleep(3)
    raise RuntimeError("Cannot connect to PostgreSQL after 5 attempts")


def _ensure_bridge_schema(conn):
    """Adds any missing columns/tables. Safe to run on every startup."""
    ensure_telemetry_schema(conn)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS touchdown_events (
            id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT NOW(),
            sink_rate_ms FLOAT, compression_ft FLOAT, mass_kg FLOAT,
            groundspeed_kt FLOAT, flight_phase VARCHAR(20), matlab_seen BOOLEAN DEFAULT FALSE
        )""")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gear_commands (
            id SERIAL PRIMARY KEY, ts TIMESTAMPTZ DEFAULT NOW(), command VARCHAR(10)
        )""")
    conn.commit()
    cur.close()


def bridge_db_writer():
    """Writes latest telemetry to PostgreSQL at BRIDGE_DB_WRITE_HZ, handles
    touchdown detection. All status surfaces via BridgeState, not print()."""
    interval = 1.0 / BRIDGE_DB_WRITE_HZ
    conn = _bridge_db_connect()
    _ensure_bridge_schema(conn)

    while True:
        time.sleep(interval)
        with bridge.lock:
            d, gear, prev_wow = bridge.raw, bridge.gear_cmd, bridge.prev_wow
        if d is None:
            continue

        try:
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO telemetry (
                    timestamp, sink_rate_fps, compression_ft, wow_status,
                    groundspeed_kt, alt_agl_ft, mass_lbs, engine_torque, prop_rpm,
                    flight_phase, airspeed_kt, latitude_deg, longitude_deg,
                    wow_nose, wow_left, wow_right, vsi_fpm, mass_kg, sink_rate_ms
                ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """, (
                d["ts"], d["sink_rate_fps"], d["compression_ft"], d["wow_any"],
                d["groundspeed_kt"], d["alt_agl_ft"], d["mass_lbs"], d["engine_torque"],
                d["prop_rpm"], d["flight_phase"], d["airspeed_kt"], d["latitude"],
                d["longitude"], d["wow_nose"], d["wow_left"], d["wow_right"],
                d["vsi_fpm"], d["mass_kg"], d["sink_rate_ms"],
            ))
            cur.execute("INSERT INTO gear_commands (ts, command) VALUES (%s,%s)", (d["ts"], gear))

            now = time.time()
            wow_rising = d["wow_any"] and not prev_wow
            if wow_rising and (now - bridge.last_td > 5.0):
                with bridge.lock:
                    bridge.last_td     = now
                    bridge.last_td_iso = d["ts"].isoformat()
                cur.execute("""
                    INSERT INTO touchdown_events (
                        ts, sink_rate_ms, compression_ft, mass_kg, groundspeed_kt, flight_phase
                    ) VALUES (%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING
                """, (d["ts"], d["sink_rate_ms"], d["compression_ft"],
                      d["mass_kg"], d["groundspeed_kt"], d["flight_phase"]))

            with bridge.lock:
                bridge.prev_wow = d["wow_any"]
            conn.commit()

        except psycopg2.OperationalError:
            with bridge.lock: bridge.db_connected = False
            try: conn.close()
            except Exception: pass
            conn = _bridge_db_connect()
            _ensure_bridge_schema(conn)

        except Exception:
            try: conn.rollback()
            except Exception: pass

# ─────────────────────────────────────────────────────────────────────────────
# BACKGROUND INFERENCE THREAD (Watches DB for WOW 0->1)
# ─────────────────────────────────────────────────────────────────────────────
def inference_worker_loop():
    global LAST_WOW_STATUS, ai_model, s_seq, s_scl, active_plotter
    
    while True:
        try:
            conn = psycopg2.connect(**DB)
            cur = conn.cursor()
            cur.execute("""
                SELECT wow_status, sink_rate_ms, compression_ft, mass_kg, vsi_fpm 
                FROM telemetry ORDER BY timestamp DESC LIMIT 1
            """)
            row = cur.fetchone()
            cur.close()
            
            if row:
                wow_status = row[0]
                # Detect Touchdown Transition
                if wow_status and not LAST_WOW_STATUS:
                    LAST_WOW_STATUS = True
                    active_plotter = run_touchdown_diagnostics(
                        conn, row, ai_model, s_seq, s_scl, active_plotter
                    )
                elif not wow_status:
                    LAST_WOW_STATUS = False

            conn.close()
        except Exception:
            pass # Suppress DB poll errors to keep terminal clean
            
        time.sleep(0.5)

# ─────────────────────────────────────────────────────────────────────────────
# FASTAPI LIFESPAN & WEBSOCKET SETUP
# ─────────────────────────────────────────────────────────────────────────────
def ensure_schema():
    try:
        c = psycopg2.connect(**DB)
        ensure_telemetry_schema(c)
        c.close()
        print("  [SCHEMA] tables verified ✓")
    except Exception:
        pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    global ai_model, s_seq, s_scl
    ensure_schema()
    
    print("\n  [INIT] Loading AeroTwin V16 Neural Networks...")
    try:
        s_seq = joblib.load(SCALER_SEQ)
        s_scl = joblib.load(SCALER_SCALAR)
        ai_model = tf.keras.models.load_model(MODEL_PATH)
        print("  [INIT] ✓ Models & Scalers loaded successfully.")
    except Exception as e:
        print(f"  [INIT] ⚠ Model load failed: {e}. Will use physics heuristics fallback.")
    
    # Start background tasks
    asyncio.create_task(broadcast_loop())
    
    # Start isolated inference thread for PyVista and TF
    inf_thread = threading.Thread(target=inference_worker_loop, daemon=True)
    inf_thread.start()

    # Start the FlightGear UDP bridge threads (formerly a separate fg_connect.py
    # process). Their status now lives in the dashboard's "FG BRIDGE LINK" panel.
    threading.Thread(target=udp_listener, daemon=True, name="FG-UDP-RX").start()
    time.sleep(0.1)
    threading.Thread(target=bridge_db_writer, daemon=True, name="FG-DB-WR").start()

    try: ip = socket.gethostbyname(socket.gethostname())
    except: ip = "localhost"
    print(f"\n  ┌─────────────────────────────────────────────┐")
    print(f"  │  PHI-FLEET  ·  Unified Supervisor Dashboard  │")
    print(f"  │  LAN    →  http://{ip}:8765               │")
    print(f"  │  FlightGear UDP bridge listening on :{FG_UDP_LISTEN_PORT}    │")
    print(f"  │  (connection + user status now on-screen)    │")
    print(f"  └─────────────────────────────────────────────┘\n")
    yield

app = FastAPI(title="PHI-FLEET", lifespan=lifespan)

class Hub:
    """Tracks connected dashboard clients. This replaces terminal connect/
    disconnect prints — the client list is broadcast to the dashboard's
    'SUPERVISORS CONNECTED' panel instead."""
    def __init__(self): self.clients: dict = {}   # id(ws) -> {ws, ip, connected_at}

    async def connect(self, ws: WebSocket):
        await ws.accept()
        ip = ws.client.host if ws.client else "unknown"
        self.clients[id(ws)] = {"ws": ws, "ip": ip, "connected_at": datetime.utcnow()}

    def disconnect(self, ws: WebSocket):
        self.clients.pop(id(ws), None)

    async def broadcast(self, payload: dict):
        msg, dead = json.dumps(payload, default=str), []
        for key, c in self.clients.items():
            try: await c["ws"].send_text(msg)
            except Exception: dead.append(key)
        for key in dead: self.clients.pop(key, None)

    @property
    def count(self): return len(self.clients)

    def client_list(self):
        now = datetime.utcnow()
        rows = [{
            "ip": c["ip"],
            "since": c["connected_at"].isoformat(),
            "duration_s": int((now - c["connected_at"]).total_seconds()),
        } for c in self.clients.values()]
        return sorted(rows, key=lambda r: r["since"])

hub = Hub()

def fetch_all():
    out = fetch_fleet_snapshot(DB, TREND_N, hub.count)
    if out is None:
        return None

    def f(v, n=3):
        return round(float(v or 0), n)

    # ── FG Bridge link status (replaces terminal heartbeat) ────────────────
    with bridge.lock:
        b = bridge.raw
        link_up = bool(bridge.last_frame_t) and (time.time() - bridge.last_frame_t) < LINK_TIMEOUT_S
        out["bridge"] = {
            "link": link_up,
            "phase": bridge.phase,
            "gear": bridge.gear_cmd,
            "frames": bridge.frames,
            "dropped": bridge.dropped,
            "db_connected": bridge.db_connected,
            "last_td": bridge.last_td_iso or "—",
            "sink_ms": f(b["sink_rate_ms"], 3) if b else 0,
            "alt_ft": f(b["alt_agl_ft"], 0) if b else 0,
            "wow": bool(b["wow_any"]) if b else False,
        }

    # ── Connected supervisors (replaces terminal connect/disconnect logs) ──
    out["clients"] = hub.client_list()
    return out

async def broadcast_loop():
    last_ts = None
    while True:
        try:
            if hub.count > 0:
                d = fetch_all()
                if d:
                    new_ts = d.get("diag", {}).get("ts") if d.get("diag") else None
                    if new_ts != last_ts or True:
                        last_ts = new_ts
                        await hub.broadcast({"type": "fleet", **d})
                else:
                    await hub.broadcast({"type": "waiting", "supervisors": hub.count})
        except: pass
        await asyncio.sleep(POLL_S)

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await hub.connect(ws)
    try:
        d = fetch_all()
        if d: await ws.send_text(json.dumps({"type": "fleet", **d}, default=str))
        while True: await ws.receive_text()
    except WebSocketDisconnect: hub.disconnect(ws)

# ─────────────────────────────────────────────────────────────────────────────
# HTML DASHBOARD (Same as before)
# ─────────────────────────────────────────────────────────────────────────────
HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PHI-FLEET · Federated Landing Gear PHM</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@300;400;500&display=swap" rel="stylesheet">
<style>
  :root{
  --ink:#0f1419; --ink2:#1a2332; --ink3:#2d3e52; --ink4:#3d505a;
  --gold:#ffd700; --gold-dim:#d4af37; --gold-dark:#b89e00;
  --teal:#0cf2d6; --teal2:rgba(12,242,214,.15); --teal3:rgba(12,242,214,.08);
  --blue:#1a7bff; --blue2:rgba(26,123,255,.15);
  --ok:#05e87a;   --ok2:rgba(5,232,122,.1);
  --warn:#f5b800; --warn2:rgba(245,184,0,.1);
  --crit:#ff4436; --crit2:rgba(255,68,54,.1);
  --aog:#d94fff;  --aog2:rgba(217,79,255,.1);
  --muted:#3a4a5a; --muted2:#1a2a3a; --dim:#7a8a9a;
  --raj:'Rajdhani',sans-serif;
  --mono:'IBM Plex Mono',monospace;
}
*{box-sizing:border-box;margin:0;padding:0;}
html,body{height:100%;background:var(--ink);color:var(--gold);font-family:var(--raj);}
header{background:linear-gradient(180deg,#142230 0%,var(--ink2) 100%);border-bottom:1px solid var(--muted);padding:.6rem 1.6rem;display:flex;align-items:center;justify-content:space-between;position:sticky;top:0;z-index:200;}
.brand{display:flex;align-items:center;gap:1rem;}
.brand-hex{width:38px;height:38px;background:var(--teal2);border:1px solid var(--teal);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-size:.65rem;font-family:var(--mono);color:var(--teal);font-weight:500;}
.brand-title{font-size:1.35rem;font-weight:700;letter-spacing:.1em;color:var(--gold);font-family:var(--raj);}
.brand-sub{font-size:.65rem;color:var(--dim);letter-spacing:.15em;margin-top:.1rem;}
.header-right{display:flex;align-items:center;gap:1.8rem;}
.sup-count{display:flex;align-items:center;gap:.5rem;font-size:.72rem;font-family:var(--mono);color:var(--dim);}
.sup-num{font-size:1.5rem;font-weight:500;color:var(--teal);font-family:var(--mono);}
.live-pill{display:flex;align-items:center;gap:.4rem;font-size:.65rem;font-family:var(--mono);padding:.25rem .7rem;border-radius:20px;background:var(--ink3);border:1px solid var(--muted);color:var(--dim);transition:all .4s;}
.live-pill.on{border-color:var(--ok);color:var(--ok);background:var(--ok2);box-shadow:0 0 12px rgba(5,232,122,.15);}
.live-dot{width:7px;height:7px;border-radius:50%;background:var(--muted);}
.live-pill.on .live-dot{background:var(--ok);animation:blink 1.6s ease-in-out infinite;}
@keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}
.fleet-strip{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--muted);border-bottom:1px solid var(--muted);}
.node{background:var(--ink2);padding:.9rem 1.4rem;display:flex;align-items:center;gap:1rem;transition:background .3s;cursor:default;}
.node.active{background:var(--ink3);}
.node:hover{background:var(--ink4);}
.node-icon{width:52px;height:52px;flex-shrink:0;position:relative;}
.node-icon svg{width:100%;height:100%;}
.node-body{flex:1;min-width:0;}
.node-reg{font-size:1.05rem;font-weight:700;color:var(--gold);letter-spacing:.06em;}
.node-type{font-size:.63rem;color:var(--dim);letter-spacing:.12em;margin-top:.05rem;}
.node-status{display:inline-block;font-size:.6rem;padding:.15rem .5rem;border-radius:3px;margin-top:.4rem;font-family:var(--mono);letter-spacing:.08em;font-weight:500;}
.node-kv{display:flex;gap:1rem;margin-top:.35rem;flex-wrap:wrap;}
.nkv{display:flex;flex-direction:column;gap:.05rem;}
.nkv .k{font-size:.55rem;color:var(--dim);letter-spacing:.12em;}
.nkv .v{font-size:.8rem;font-family:var(--mono);color:var(--gold);}
.health-bar{width:90px;height:3px;background:var(--muted);border-radius:2px;margin-top:.3rem;}
.health-fill{height:100%;border-radius:2px;transition:width .8s;}
.main{display:grid;grid-template-columns:1fr 1.8fr 1fr;gap:1px;background:var(--muted);min-height:calc(100vh - 140px);}
.col{display:flex;flex-direction:column;gap:1px;background:var(--muted);}
.panel{background:var(--ink2);padding:1.1rem 1.3rem;position:relative;flex:1;}
.ptitle{font-size:.6rem;font-weight:600;letter-spacing:.2em;color:var(--dim);text-transform:uppercase;border-bottom:1px solid var(--muted);padding-bottom:.4rem;margin-bottom:.85rem;display:flex;align-items:center;justify-content:space-between;}
.ptitle-badge{font-size:.55rem;padding:.1rem .4rem;border-radius:3px;font-weight:500;letter-spacing:.08em;font-family:var(--mono);}
.tel-grid{display:grid;grid-template-columns:1fr 1fr;gap:.5rem;}
.tv{background:var(--ink3);border:1px solid var(--muted);border-radius:4px;padding:.55rem .7rem;}
.tv-label{font-size:.55rem;color:var(--dim);letter-spacing:.14em;margin-bottom:.15rem;}
.tv-val{font-size:1.2rem;font-family:var(--mono);font-weight:500;color:var(--gold);}
.tv-unit{font-size:.58rem;color:var(--dim);margin-left:.2rem;}
.tv.wow-active{border-color:var(--ok);background:var(--ok2);}
.tv.wow-active .tv-val{color:var(--ok);}
.tv.warn .tv-val{color:var(--warn);}
.tv.crit .tv-val{color:var(--crit);}
.phase-row{background:var(--ink4);border:1px solid var(--muted2);border-radius:4px;padding:.5rem .8rem;margin-top:.5rem;display:flex;align-items:center;justify-content:space-between;}
.phase-label{font-size:.58rem;color:var(--dim);}
.phase-val{font-size:.85rem;font-family:var(--mono);color:var(--gold);letter-spacing:.1em;}
.cls-block{padding:.8rem 1rem;border-radius:6px;margin-bottom:.8rem;border:1px solid var(--muted);transition:all .4s;}
.cls-name{font-size:1.45rem;font-weight:700;letter-spacing:.04em;line-height:1;font-family:var(--raj);color:var(--gold);}
.cls-conf-row{display:flex;align-items:center;gap:.7rem;margin-top:.5rem;}
.cls-conf-val{font-size:.72rem;font-family:var(--mono);color:var(--dim);}
.conf-track{flex:1;height:4px;background:var(--muted);border-radius:2px;}
.conf-fill{height:100%;border-radius:2px;background:var(--gold);transition:width .7s;}
.meta-row{display:flex;gap:1.5rem;margin-top:.7rem;flex-wrap:wrap;}
.mkv .k{font-size:.55rem;color:var(--dim);letter-spacing:.12em;}
.mkv .v{font-size:.78rem;font-family:var(--mono);color:#c8dce8;margin-top:.1rem;}
.tag{display:inline-block;font-size:.58rem;padding:.12rem .5rem;border-radius:3px;font-family:var(--mono);letter-spacing:.06em;}
.cs23-panel{border-radius:6px;padding:.8rem 1rem;border:1px solid var(--muted);margin-bottom:.8rem;transition:all .4s;}
.cs23-panel.clear{border-color:var(--ok);background:var(--ok2);}
.cs23-panel.triggered{border-color:var(--crit);background:var(--crit2);animation:cs-pulse 1.2s ease-in-out infinite;}
@keyframes cs-pulse{0%,100%{box-shadow:none}50%{box-shadow:0 0 18px rgba(255,68,54,.3)}}
.cs23-top{display:flex;align-items:center;justify-content:space-between;}
.cs23-title{font-size:.9rem;font-weight:700;letter-spacing:.06em;}
.cs23-vals{font-size:.65rem;font-family:var(--mono);color:var(--dim);margin-top:.3rem;}
.dispatch{font-size:.7rem;font-weight:700;padding:.2rem .6rem;border-radius:4px;font-family:var(--mono);letter-spacing:.08em;}
.gauges-row{display:flex;gap:.6rem;flex-wrap:wrap;margin-bottom:.6rem;}
.gauge{flex:1;min-width:90px;}
.gauge-lbl{font-size:.55rem;color:var(--dim);letter-spacing:.12em;margin-bottom:.25rem;}
.gauge-val{font-size:1.15rem;font-family:var(--mono);font-weight:500;line-height:1;color:var(--gold);}
.gauge-unit{font-size:.55rem;color:var(--dim);margin-left:.15rem;}
.gauge-track{width:100%;height:3px;background:var(--muted);border-radius:2px;margin-top:.3rem;}
.gauge-fill{height:100%;border-radius:2px;transition:width .7s cubic-bezier(.22,1,.36,1);}
.sparkline-wrap{margin-top:.3rem;}
.spark-label{font-size:.55rem;color:var(--dim);letter-spacing:.12em;display:flex;justify-content:space-between;margin-bottom:.3rem;}
canvas#spark{width:100%;height:64px;display:block;border-radius:4px;background:var(--ink3);border:1px solid var(--muted);}
.health-row{display:flex;align-items:center;gap:1.2rem;margin-bottom:.8rem;}
svg.big-ring{flex-shrink:0;}
.ring-track2{fill:none;stroke:var(--muted);stroke-width:8;}
.ring-fill2{fill:none;stroke-width:8;stroke-linecap:round;transform:rotate(-90deg);transform-origin:50% 50%;transition:stroke-dashoffset .9s;}
.ring-center{text-anchor:middle;dominant-baseline:middle;font-family:var(--mono);font-weight:500;}
.ring-pct-text{font-size:14px;}
.ring-sub-text{font-size:7px;fill:var(--dim);}
.rul-big{font-size:2.2rem;font-weight:700;font-family:var(--mono);line-height:1;color:var(--gold);}
.rul-lbl{font-size:.58rem;color:var(--dim);letter-spacing:.14em;margin-top:.2rem;}
.rul-sub{font-size:.65rem;color:var(--dim);margin-top:.1rem;}
.event-list{display:flex;flex-direction:column;gap:.4rem;}
.event{background:var(--ink3);border-left:3px solid var(--muted);border-radius:0 4px 4px 0;padding:.55rem .75rem;transition:background .2s;}
.event:first-child{background:var(--ink4);}
.event-top{display:flex;align-items:center;justify-content:space-between;gap:.5rem;}
.event-cls{font-size:.75rem;font-weight:600;flex:1;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.event-g{font-size:.65rem;font-family:var(--mono);color:var(--dim);}
.event-bottom{display:flex;align-items:center;gap:.5rem;margin-top:.2rem;font-size:.58rem;color:var(--dim);font-family:var(--mono);}
.dot-tag{width:6px;height:6px;border-radius:50%;flex-shrink:0;display:inline-block;margin-right:.2rem;}
.aog-badge{font-size:.55rem;padding:.05rem .3rem;border-radius:2px;background:var(--aog2);color:var(--aog);margin-left:.3rem;}
.kb-table{width:100%;border-collapse:collapse;font-size:.65rem;}
.kb-table tr{border-bottom:1px solid var(--muted2);}
.kb-table tr:last-child{border:none;}
.kb-table td{padding:.3rem .2rem;}
.kb-table td:first-child{color:var(--dim);letter-spacing:.1em;}
.kb-table td:last-child{font-family:var(--mono);color:var(--gold);text-align:right;}
footer{background:var(--ink2);border-top:1px solid var(--muted);padding:.4rem 1.6rem;font-size:.58rem;color:var(--dim);display:flex;justify-content:space-between;font-family:var(--mono);}
@keyframes slide-in{from{opacity:0;transform:translateX(-6px)}to{opacity:1;transform:none}}
.slide-in{animation:slide-in .25s ease-out;}

/* ── Premium retouch ──────────────────────────────────────────────────── */
body{-webkit-font-smoothing:antialiased;-moz-osx-font-smoothing:grayscale;}
body::before{content:'';position:fixed;top:0;left:0;right:0;height:2px;z-index:300;
  background:linear-gradient(90deg,transparent,var(--teal) 20%,var(--gold) 50%,var(--teal) 80%,transparent);
  opacity:.55;}
header{box-shadow:0 2px 28px rgba(0,0,0,.4);}
.brand-hex{box-shadow:0 0 16px rgba(12,242,214,.28);}
.panel{box-shadow:inset 0 1px 0 rgba(255,255,255,.025);transition:box-shadow .3s;}
.panel:hover{box-shadow:inset 0 1px 0 rgba(255,255,255,.025), inset 0 0 0 1px rgba(12,242,214,.06);}
.node{position:relative;}
.node.active::after{content:'';position:absolute;inset:0;pointer-events:none;box-shadow:inset 0 0 0 1px rgba(12,242,214,.10);}
.tv, .event, .cls-block, .cs23-panel{transition:background .25s ease, border-color .25s ease, box-shadow .25s ease;}
.live-pill.on{box-shadow:0 0 16px rgba(5,232,122,.18);}
::selection{background:rgba(12,242,214,.28);color:#eafffb;}
::-webkit-scrollbar{width:8px;height:8px;}
::-webkit-scrollbar-track{background:var(--ink2);}
::-webkit-scrollbar-thumb{background:var(--muted);border-radius:4px;}
::-webkit-scrollbar-thumb:hover{background:var(--ink4);}

/* FG Bridge Link + Supervisors panels */
.link-row{display:flex;justify-content:space-between;align-items:center;padding:.4rem 0;
  border-bottom:1px solid var(--muted2);font-size:.65rem;font-family:var(--mono);color:var(--dim);}
.link-row:last-child{border:none;}
.link-status{display:flex;align-items:center;gap:.4rem;font-weight:600;letter-spacing:.06em;}
.link-dot{width:7px;height:7px;border-radius:50%;flex-shrink:0;}
.link-dot.up{background:var(--ok);box-shadow:0 0 8px rgba(5,232,122,.6);animation:blink 1.8s ease-in-out infinite;}
.link-dot.down{background:var(--crit);box-shadow:0 0 8px rgba(255,68,54,.45);}
.client-list-wrap{max-height:170px;overflow-y:auto;}
.client-row{display:flex;justify-content:space-between;align-items:center;padding:.4rem .1rem;
  font-size:.62rem;font-family:var(--mono);color:var(--dim);border-bottom:1px solid var(--muted2);}
.client-row:last-child{border:none;}
.client-ip{color:var(--gold);font-weight:500;}
.empty-hint{font-size:.62rem;color:var(--dim);padding:.5rem 0;}
</style>
</head>
<body>

<header>
  <div class="brand">
    <div class="brand-hex">PHI</div>
    <div>
      <div class="brand-title">PHI-FLEET</div>
      <div class="brand-sub">FEDERATED LANDING GEAR PROGNOSTIC SYSTEM · ATA 32 · SM BELLO</div>
    </div>
  </div>
  <div class="header-right">
    <div class="sup-count">
      <span class="sup-num" id="sup-num">0</span>
      <div>
        <div style="font-size:.58rem;letter-spacing:.12em;">SUPERVISORS</div>
        <div style="font-size:.55rem;color:#0d2030;">CONNECTED</div>
      </div>
    </div>
    <div class="live-pill" id="live-pill">
      <div class="live-dot"></div>
      <span id="live-txt">CONNECTING</span>
    </div>
  </div>
</header>

<div class="fleet-strip">
  <div class="node active" id="node-0">
    <div class="node-icon">
      <svg viewBox="0 0 52 52" fill="none" xmlns="http://www.w3.org/2000/svg">
        <ellipse cx="26" cy="26" rx="22" ry="22" fill="rgba(12,242,214,.04)" stroke="rgba(12,242,214,.2)" stroke-width=".5"/>
        <rect x="8" y="23" width="36" height="6" rx="3" fill="rgba(12,242,214,.25)"/>
        <polygon points="18,26 34,26 38,20 14,20" fill="rgba(12,242,214,.18)"/>
        <polygon points="8,24 8,29 4,22" fill="rgba(12,242,214,.18)"/>
        <rect x="8" y="19" width="2" height="6" fill="rgba(12,242,214,.18)"/>
        <circle cx="22" cy="33" r="2.5" fill="rgba(5,232,122,.8)" id="ni-gear-0"/>
        <circle cx="30" cy="33" r="2.5" fill="rgba(5,232,122,.8)"/>
        <circle cx="14" cy="33" r="2" fill="rgba(5,232,122,.6)"/>
      </svg>
    </div>
    <div class="node-body">
      <div class="node-reg">5N-ATW</div>
      <div class="node-type">DO 228-212 · AFIT ABUJA</div>
      <span class="node-status" id="ns-0" style="background:var(--ok2);color:var(--ok)">● LIVE DATA</span>
      <div class="node-kv">
        <div class="nkv"><span class="k">HEALTH</span><span class="v" id="nv-health">—%</span></div>
        <div class="nkv"><span class="k">PHASE</span><span class="v" id="nv-phase">—</span></div>
        <div class="nkv"><span class="k">LAST G</span><span class="v" id="nv-g">— g</span></div>
        <div class="nkv"><span class="k">SEV</span><span class="v" id="nv-sev">—</span></div>
      </div>
      <div class="health-bar"><div class="health-fill" id="nv-hbar" style="width:0%;background:var(--ok)"></div></div>
    </div>
  </div>

  <div class="node">
    <div class="node-icon" style="opacity:.3">
      <svg viewBox="0 0 52 52" fill="none">
        <ellipse cx="26" cy="26" rx="22" ry="22" stroke="#1a3349" stroke-width=".5"/>
        <rect x="8" y="23" width="36" height="6" rx="3" fill="#1a3349"/>
        <polygon points="18,26 34,26 38,20 14,20" fill="#0d2030"/>
        <polygon points="8,24 8,29 4,22" fill="#0d2030"/>
        <rect x="8" y="19" width="2" height="6" fill="#0d2030"/>
        <circle cx="22" cy="33" r="2.5" fill="#1a3349"/>
        <circle cx="30" cy="33" r="2.5" fill="#1a3349"/>
        <circle cx="14" cy="33" r="2" fill="#1a3349"/>
      </svg>
    </div>
    <div class="node-body" style="opacity:.35">
      <div class="node-reg">5N-BTW</div>
      <div class="node-type">DO 228-212 · STANDBY</div>
      <span class="node-status" style="background:var(--ink4);color:var(--dim)">○ NO DATA</span>
    </div>
  </div>

  <div class="node">
    <div class="node-icon" style="opacity:.3">
      <svg viewBox="0 0 52 52" fill="none">
        <ellipse cx="26" cy="26" rx="22" ry="22" stroke="#1a3349" stroke-width=".5"/>
        <rect x="8" y="23" width="36" height="6" rx="3" fill="#1a3349"/>
        <polygon points="18,26 34,26 38,20 14,20" fill="#0d2030"/>
        <polygon points="8,24 8,29 4,22" fill="#0d2030"/>
        <rect x="8" y="19" width="2" height="6" fill="#0d2030"/>
        <circle cx="22" cy="33" r="2.5" fill="#1a3349"/>
        <circle cx="30" cy="33" r="2.5" fill="#1a3349"/>
        <circle cx="14" cy="33" r="2" fill="#1a3349"/>
      </svg>
    </div>
    <div class="node-body" style="opacity:.35">
      <div class="node-reg">5N-CTW</div>
      <div class="node-type">DO 228-212 · OFFLINE</div>
      <span class="node-status" style="background:var(--ink4);color:var(--dim)">○ OFFLINE</span>
    </div>
  </div>
</div>

<div class="main">

  <div class="col">
    <div class="panel">
      <div class="ptitle">
        LIVE TELEMETRY
        <span class="ptitle-badge" id="tel-src-badge" style="background:var(--teal2);color:var(--teal)">PYTHON SIM</span>
      </div>
      <div class="tel-grid">
        <div class="tv" id="tv-alt">
          <div class="tv-label">ALT AGL</div>
          <div><span class="tv-val" id="t-alt">—</span><span class="tv-unit">ft</span></div>
        </div>
        <div class="tv" id="tv-gs">
          <div class="tv-label">GROUNDSPEED</div>
          <div><span class="tv-val" id="t-gs">—</span><span class="tv-unit">kt</span></div>
        </div>
        <div class="tv" id="tv-sink">
          <div class="tv-label">SINK RATE</div>
          <div><span class="tv-val" id="t-sink">—</span><span class="tv-unit">m/s</span></div>
        </div>
        <div class="tv" id="tv-mass">
          <div class="tv-label">AIRCRAFT MASS</div>
          <div><span class="tv-val" id="t-mass">—</span><span class="tv-unit">kg</span></div>
        </div>
        <div class="tv" id="tv-torque">
          <div class="tv-label">ENG TORQUE</div>
          <div><span class="tv-val" id="t-torque">—</span><span class="tv-unit">%</span></div>
        </div>
        <div class="tv" id="tv-rpm">
          <div class="tv-label">PROP RPM</div>
          <div><span class="tv-val" id="t-rpm">—</span><span class="tv-unit">rpm</span></div>
        </div>
        <div class="tv" id="tv-comp">
          <div class="tv-label">COMPRESSION</div>
          <div><span class="tv-val" id="t-comp">—</span><span class="tv-unit">ft</span></div>
        </div>
        <div class="tv" id="tv-wow">
          <div class="tv-label">WEIGHT-ON-WHEELS</div>
          <div><span class="tv-val" id="t-wow">—</span></div>
        </div>
      </div>
      <div class="phase-row">
        <span class="phase-label">FLIGHT PHASE</span>
        <span class="phase-val" id="t-phase">—</span>
      </div>
      <div style="font-size:.55rem;color:var(--dim);font-family:var(--mono);margin-top:.5rem;text-align:right;" id="tel-ts">—</div>
    </div>

    <div class="panel">
      <div class="ptitle">STRUT PARAMETERS
        <span class="ptitle-badge" style="background:var(--blue2);color:var(--blue)">MATLAB</span>
      </div>
      <table class="kb-table">
        <tr><td>K STIFFNESS (N/m)</td><td id="kb-k">—</td></tr>
        <tr><td>B DAMPING (Ns/m)</td><td id="kb-b">—</td></tr>
        <tr><td>K EFFECTIVE</td><td id="kb-ke">—</td></tr>
        <tr><td>B EFFECTIVE</td><td id="kb-be">—</td></tr>
        <tr><td>STIFFNESS DEGRAD</td><td id="kb-deg" style="color:var(--warn)">—</td></tr>
        <tr><td>COMPRESSION (ft)</td><td id="kb-cmp">—</td></tr>
      </table>
    </div>
  </div>

  <div class="col">
    <div class="panel">
      <div class="ptitle">AI · PHYSICS FUSION CLASSIFICATION</div>

      <div class="cls-block" id="cls-block" style="border-color:var(--muted)">
        <div class="cls-name" id="cls-name" style="color:var(--teal)">AWAITING DATA</div>
        <div class="cls-conf-row">
          <span class="cls-conf-val" id="cls-conf">0%</span>
          <div class="conf-track"><div class="conf-fill" id="conf-fill" style="width:0%"></div></div>
          <span class="cls-conf-val" id="cls-src" style="color:var(--teal)">—</span>
        </div>
        <div class="meta-row">
          <div class="mkv"><span class="k">PHYSICS CLASS</span><div class="v" id="m-pc">—</div></div>
          <div class="mkv"><span class="k">ATA</span><div class="v" id="m-ata">—</div></div>
          <div class="mkv"><span class="k">MAINT CAT</span><div class="v" id="m-maint" style="max-width:200px;font-size:.68rem">—</div></div>
          <div class="mkv"><span class="k">EST. COST</span><div class="v" id="m-cost" style="color:var(--warn)">—</div></div>
        </div>
      </div>

      <div class="cs23-panel clear" id="cs23-panel">
        <div class="cs23-top">
          <span class="cs23-title" id="cs23-title" style="color:var(--ok)">CS-23.473 CLEAR</span>
          <span class="dispatch" id="dispatch-tag" style="background:var(--ok2);color:var(--ok)">DISPATCH OK</span>
        </div>
        <div class="cs23-vals" id="cs23-vals">Sink — m/s · Accel — g</div>
      </div>

      <div class="ptitle" style="margin-top:.2rem">PRISMATIC JOINT PHYSICS</div>
      <div class="gauges-row">
        <div class="gauge">
          <div class="gauge-lbl">DEFLECTION</div>
          <div><span class="gauge-val" id="g-defl" style="color:var(--teal)">—</span><span class="gauge-unit">mm</span></div>
          <div class="gauge-track"><div class="gauge-fill" id="gb-d" style="width:0%;background:var(--teal)"></div></div>
        </div>
        <div class="gauge">
          <div class="gauge-lbl">STRUT FORCE</div>
          <div><span class="gauge-val" id="g-force" style="color:var(--teal)">—</span><span class="gauge-unit">kN</span></div>
          <div class="gauge-track"><div class="gauge-fill" id="gb-f" style="width:0%;background:var(--teal)"></div></div>
        </div>
        <div class="gauge">
          <div class="gauge-lbl">ACCEL</div>
          <div><span class="gauge-val" id="g-accel" style="color:var(--teal)">—</span><span class="gauge-unit">g</span></div>
          <div class="gauge-track"><div class="gauge-fill" id="gb-a" style="width:0%;background:var(--teal)"></div></div>
        </div>
        <div class="gauge">
          <div class="gauge-lbl">PRESSURE</div>
          <div><span class="gauge-val" id="g-pres" style="color:var(--teal)">—</span><span class="gauge-unit">kPa</span></div>
          <div class="gauge-track"><div class="gauge-fill" id="gb-p" style="width:0%;background:var(--teal)"></div></div>
        </div>
      </div>
    </div>

    <div class="panel">
      <div class="ptitle">HEALTH INDEX · PROGNOSTIC TREND</div>
      <div class="health-row">
        <svg class="big-ring" width="90" height="90" viewBox="0 0 90 90">
          <circle class="ring-track2" cx="45" cy="45" r="37"/>
          <circle class="ring-fill2" id="ring-arc" cx="45" cy="45" r="37" stroke="var(--ok)" stroke-dasharray="232.5" stroke-dashoffset="232.5"/>
          <text class="ring-center" x="45" y="43">
            <tspan class="ring-pct-text" id="ring-pct-txt" fill="var(--ok)">—</tspan>
          </text>
          <text class="ring-center ring-sub-text" x="45" y="55">HEALTH</text>
        </svg>
        <div>
          <div class="rul-big" id="rul-val" style="color:var(--ok)">—</div>
          <div class="rul-lbl">REMAINING LANDINGS (RUL)</div>
          <div class="rul-sub" id="rul-sub">—</div>
        </div>
      </div>
      <div class="sparkline-wrap">
        <div class="spark-label">
          <span>HEALTH TREND (last 24 readings)</span>
          <span id="spark-range">—</span>
        </div>
        <canvas id="spark" height="64"></canvas>
      </div>
    </div>
  </div>

  <div class="col">
    <div class="panel" style="flex:2">
      <div class="ptitle">LANDING EVENT LOG</div>
      <div class="event-list" id="event-list">
        <div style="font-size:.65rem;color:var(--dim);padding:.5rem 0;">Awaiting landing events…</div>
      </div>
    </div>
    <div class="panel">
      <div class="ptitle">NETWORK STATUS</div>
      <table class="kb-table">
        <tr><td>SUPERVISORS ONLINE</td><td id="ns-sup" style="color:var(--teal)">0</td></tr>
        <tr><td>DATA SOURCE</td><td id="ns-dsrc">—</td></tr>
        <tr><td>DECISION ENGINE</td><td id="ns-dec">—</td></tr>
        <tr><td>LAST DIAG UPDATE</td><td id="ns-diag-ts" style="font-size:.55rem">—</td></tr>
        <tr><td>LAST PHYS UPDATE</td><td id="ns-phys-ts" style="font-size:.55rem">—</td></tr>
        <tr><td>WS LATENCY</td><td id="ns-lat" style="color:var(--ok)">—</td></tr>
      </table>
    </div>

    <div class="panel">
      <div class="ptitle">FG BRIDGE LINK
        <span class="ptitle-badge" id="bridge-badge" style="background:var(--ink4);color:var(--dim)">—</span>
      </div>
      <div class="link-row">
        <span>UDP :15000</span>
        <span class="link-status" id="bridge-link-status">
          <span class="link-dot down" id="bridge-dot"></span><span id="bridge-link-txt">NO SIGNAL</span>
        </span>
      </div>
      <div class="link-row"><span>FLIGHT PHASE</span><span id="bridge-phase" style="color:var(--gold)">—</span></div>
      <div class="link-row"><span>GEAR COMMAND</span><span id="bridge-gear">—</span></div>
      <div class="link-row"><span>FRAMES RX</span><span id="bridge-frames">—</span></div>
      <div class="link-row"><span>DROPPED</span><span id="bridge-dropped">—</span></div>
      <div class="link-row"><span>DB WRITE</span><span id="bridge-db">—</span></div>
      <div class="link-row"><span>LAST TOUCHDOWN</span><span id="bridge-td" style="font-size:.58rem">—</span></div>
    </div>

    <div class="panel">
      <div class="ptitle">SUPERVISORS CONNECTED
        <span class="ptitle-badge" style="background:var(--teal2);color:var(--teal)" id="clients-count-badge">0</span>
      </div>
      <div class="client-list-wrap" id="client-list">
        <div class="empty-hint">No supervisors connected</div>
      </div>
    </div>
  </div>

</div>

<footer>
  <span>PHI-FLEET v1.0 · Do 228-212 · ATA 32 LANDING GEAR PHM · AFIT KADUNA</span>
  <span id="footer-ts">—</span>
  <span>CS-23 AMC 23.473(d) · PHI-Twin / PHI-CHAIN · SM BELLO</span>
</footer>

<script>
const SEV_COL  = {1:'var(--ok)',2:'var(--warn)',3:'var(--crit)',4:'var(--aog)'};
const SEV_LBL  = {1:'ROUTINE',2:'ADVISORY',3:'CRITICAL',4:'AOG'};
const SEV_BG   = {1:'var(--ok2)',2:'var(--warn2)',3:'var(--crit2)',4:'var(--aog2)'};
function clamp(v,mn,mx){return Math.min(mx,Math.max(mn,v));}
function pct(v,max){return clamp((v/max)*100,0,100);}
function barColor(p){return p>85?'var(--crit)':p>65?'var(--warn)':'var(--teal)';}
function setBar(id,val,max){
  const el=document.getElementById(id);if(!el)return;
  const p=pct(val,max);el.style.width=p+'%';el.style.background=barColor(p);
}

let trendData=[];
function drawSparkline(data){
  const canvas=document.getElementById('spark');if(!canvas)return;
  const W=canvas.offsetWidth||300, H=64; canvas.width=W; canvas.height=H;
  const ctx=canvas.getContext('2d'); ctx.clearRect(0,0,W,H);
  if(data.length<2)return;
  const min=Math.min(...data)-5, max=Math.max(...data)+5;
  const scX=i=>(i/(data.length-1))*W;
  const scY=v=>H-((v-min)/(max-min))*H*0.85-H*0.075;
  const grad=ctx.createLinearGradient(0,0,0,H);
  grad.addColorStop(0,'rgba(255,215,0,.25)'); grad.addColorStop(1,'rgba(255,215,0,.0)');
  ctx.beginPath(); ctx.moveTo(scX(0),H);
  data.forEach((v,i)=>ctx.lineTo(scX(i),scY(v))); ctx.lineTo(scX(data.length-1),H);
  ctx.closePath(); ctx.fillStyle=grad; ctx.fill();
  ctx.beginPath(); ctx.strokeStyle='rgba(255,215,0,.9)'; ctx.lineWidth=1.5; ctx.lineJoin='round';
  data.forEach((v,i)=>i===0?ctx.moveTo(scX(i),scY(v)):ctx.lineTo(scX(i),scY(v))); ctx.stroke();
  const y80=scY(80); ctx.beginPath(); ctx.strokeStyle='rgba(245,184,0,.3)'; ctx.setLineDash([4,4]); ctx.lineWidth=1; ctx.moveTo(0,y80); ctx.lineTo(W,y80); ctx.stroke(); ctx.setLineDash([]);
  const lx=scX(data.length-1), ly=scY(data[data.length-1]); ctx.beginPath(); ctx.arc(lx,ly,3.5,0,Math.PI*2); ctx.fillStyle='rgba(255,215,0,0.9)'; ctx.fill();
  document.getElementById('spark-range').textContent = `min ${Math.min(...data).toFixed(1)}%  ·  max ${Math.max(...data).toFixed(1)}%`;
}

function renderTelemetry(t){
  const set=(id,v)=>{const el=document.getElementById(id);if(el)el.textContent=v;};
  set('t-alt', t.alt_ft.toFixed(0)); set('t-gs', t.gs_kt.toFixed(1)); set('t-sink', t.sink_ms.toFixed(3));
  set('t-mass', t.mass_kg.toFixed(0)); set('t-torque',t.eng_torque.toFixed(1)); set('t-rpm', t.prop_rpm.toFixed(0));
  set('t-comp', t.comp_ft.toFixed(4)); set('t-phase', t.phase); set('tel-ts', t.ts.slice(0,19).replace('T',' ')+'Z');
  const wow=document.getElementById('tv-wow'); const wowVal=document.getElementById('t-wow');
  if(t.wow){wow.classList.add('wow-active'); wowVal.textContent='TOUCHDOWN';}else{wow.classList.remove('wow-active'); wowVal.textContent='AIRBORNE';}
  const sinkEl=document.getElementById('tv-sink'); sinkEl.className='tv'+(t.sink_ms>3.05?' crit':t.sink_ms>2.0?' warn':'');
  set('nv-phase', t.phase);
}

function renderDiag(g){
  const sev=g.sev||1; const col=SEV_COL[sev];
  document.getElementById('cls-name').textContent=g.cls; document.getElementById('cls-name').style.color=col;
  document.getElementById('cls-block').style.borderColor=col; document.getElementById('cls-block').style.background=SEV_BG[sev];
  document.getElementById('cls-conf').textContent=g.conf.toFixed(1)+'%'; document.getElementById('conf-fill').style.width=g.conf+'%';
  document.getElementById('cls-src').textContent=g.dec; document.getElementById('m-pc').textContent=g.phys_cls;
  document.getElementById('m-ata').textContent=g.ata; document.getElementById('m-maint').textContent=g.maint;
  document.getElementById('m-cost').textContent='$'+g.cost.toLocaleString();
  const panel=document.getElementById('cs23-panel'); const title=document.getElementById('cs23-title');
  const dtag=document.getElementById('dispatch-tag'); const cvals=document.getElementById('cs23-vals');
  cvals.textContent=`Sink ${g.sink_ms} m/s · Accel ${g.accel_g} g`;
  if(g.cs23){panel.className='cs23-panel triggered'; title.textContent='⚠ CS-23.473 TRIGGERED'; title.style.color='var(--crit)';}
  else{panel.className='cs23-panel clear'; title.textContent='✓ CS-23.473 CLEAR'; title.style.color='var(--ok)';}
  if(g.grounded){dtag.textContent='⛔  AOG — GROUNDED'; dtag.style.background='var(--aog2)'; dtag.style.color='var(--aog)';}
  else{dtag.textContent='✓  DISPATCH OK'; dtag.style.background='var(--ok2)'; dtag.style.color='var(--ok)';}
  document.getElementById('kb-k').textContent=g.k.toLocaleString(); document.getElementById('kb-b').textContent=g.b.toLocaleString();
  const deg=((1-g.k/150000)*100).toFixed(1); document.getElementById('kb-deg').textContent=deg+'%';
  document.getElementById('kb-deg').style.color=parseFloat(deg)>15?'var(--crit)':parseFloat(deg)>7?'var(--warn)':'var(--ok)';
  document.getElementById('kb-cmp').textContent=g.comp_ft.toFixed(4)+' ft';
  const hp=Math.min(100,Math.max(0,g.health)); const circ=232.5, off=circ*(1-hp/100);
  const arc=document.getElementById('ring-arc'); arc.style.strokeDashoffset=off;
  const hcol=hp>70?'var(--ok)':hp>40?'var(--warn)':'var(--crit)'; arc.style.stroke=hcol;
  const rpt=document.getElementById('ring-pct-txt'); rpt.textContent=hp.toFixed(1)+'%'; rpt.setAttribute('fill',hcol);
  const rul=Math.max(0,Math.round(hp*12-g.accel_g*150-parseFloat(deg)*15));
  const rulEl=document.getElementById('rul-val'); rulEl.textContent=rul; rulEl.style.color=rul>400?'var(--ok)':rul>150?'var(--warn)':'var(--crit)';
  document.getElementById('rul-sub').textContent=rul>400?'HEALTHY — CONTINUE OPS':rul>150?'MONITOR — PLAN INSPECTION':'⚠ INSPECT BEFORE NEXT FLIGHT';
  document.getElementById('nv-health').textContent=hp.toFixed(1)+'%'; document.getElementById('nv-hbar').style.width=hp+'%';
  document.getElementById('nv-hbar').style.background=hcol; document.getElementById('nv-g').textContent=g.accel_g.toFixed(2)+'g';
  document.getElementById('nv-sev').textContent=SEV_LBL[sev]; document.getElementById('nv-sev').style.color=col;
  document.getElementById('ns-dsrc').textContent=g.src; document.getElementById('ns-dec').textContent=g.dec;
  document.getElementById('ns-diag-ts').textContent=g.ts.slice(0,19).replace('T',' ')+'Z';
}

function renderPhysics(p){
  const set=(id,v,dec=2)=>{const el=document.getElementById(id);if(el)el.textContent=typeof v==='number'?v.toFixed(dec):v;};
  set('g-defl', p.defl_mm); set('g-force', p.force_kN); set('g-accel', p.accel_g, 3); set('g-pres', p.pres_kPa, 1);
  setBar('gb-d', p.defl_mm, 180); setBar('gb-f', p.force_kN, 30); setBar('gb-a', p.accel_g, 2.5); setBar('gb-p', p.pres_kPa, 10000);
  document.getElementById('kb-ke').textContent=p.k_eff.toLocaleString(); document.getElementById('kb-be').textContent=p.b_eff.toLocaleString();
  document.getElementById('ns-phys-ts').textContent=p.ts.slice(0,19).replace('T',' ')+'Z';
}

const SEV_ICON={1:'○',2:'◈',3:'◉',4:'⬣'};
function renderEvents(evts){
  const list=document.getElementById('event-list');
  if(!evts.length){ list.innerHTML='<div style="font-size:.65rem;color:var(--dim)">No events yet</div>'; return; }
  list.innerHTML=evts.map((e,i)=>{
    const col=SEV_COL[e.sev]; const ts=e.ts.slice(0,19).replace('T',' '); const aog=e.gnd?'<span class="aog-badge">AOG</span>':'';
    return `<div class="event${i===0?' slide-in':''}" style="border-left-color:${col}">
      <div class="event-top"><span class="event-cls" style="color:${col}">${SEV_ICON[e.sev]} ${e.cls}</span><span class="event-g">${e.g}g</span></div>
      <div class="event-bottom"><span class="dot-tag" style="background:${col}"></span>${e.phase} · CS23:${e.cs23?'⚠':'✓'} · $${Number(e.cost).toLocaleString()} · ${ts}${aog}</div>
    </div>`;
  }).join('');
}

function renderBridge(b){
  const dot=document.getElementById('bridge-dot'), txt=document.getElementById('bridge-link-txt'), badge=document.getElementById('bridge-badge');
  if(b.link){
    dot.className='link-dot up'; txt.textContent='LIVE — '+(b.wow?'ON GROUND':'AIRBORNE');
    badge.style.background='var(--ok2)'; badge.style.color='var(--ok)'; badge.textContent='LINKED';
  } else {
    dot.className='link-dot down'; txt.textContent='NO SIGNAL';
    badge.style.background='var(--crit2)'; badge.style.color='var(--crit)'; badge.textContent='NO LINK';
  }
  document.getElementById('bridge-phase').textContent=b.phase;
  document.getElementById('bridge-gear').textContent=b.gear;
  document.getElementById('bridge-frames').textContent=b.frames.toLocaleString();
  document.getElementById('bridge-dropped').textContent=b.dropped.toLocaleString();
  const dbEl=document.getElementById('bridge-db');
  dbEl.textContent=b.db_connected?'CONNECTED':'RECONNECTING…'; dbEl.style.color=b.db_connected?'var(--ok)':'var(--warn)';
  document.getElementById('bridge-td').textContent=(b.last_td && b.last_td!=='—')?b.last_td.slice(0,19).replace('T',' ')+'Z':'—';
}

function renderClients(clients){
  const el=document.getElementById('client-list'), badge=document.getElementById('clients-count-badge');
  badge.textContent=clients.length;
  if(!clients.length){ el.innerHTML='<div class="empty-hint">No supervisors connected</div>'; return; }
  el.innerHTML=clients.map(c=>{
    const m=Math.floor(c.duration_s/60), s=c.duration_s%60;
    return `<div class="client-row"><span class="client-ip">${c.ip}</span><span>${m}m ${s}s</span></div>`;
  }).join('');
}

let ws, pingStart, reconnTimer;
function setPill(live){
  const pill=document.getElementById('live-pill'); const txt=document.getElementById('live-txt');
  pill.className='live-pill'+(live?' on':''); txt.textContent=live?'LIVE':'RECONNECTING…';
}
function connect(){
  const proto=location.protocol==='https:'?'wss':'ws'; ws=new WebSocket(`${proto}://${location.host}/ws`);
  ws.onopen=()=>{setPill(true); clearTimeout(reconnTimer); pingStart=Date.now(); ws.send('ping');};
  ws.onmessage=e=>{
    if(e.data==='pong'){document.getElementById('ns-lat').textContent=(Date.now()-pingStart)+'ms'; return;}
    const d=JSON.parse(e.data); if(d.type==='waiting') return;
    if(d.type==='fleet'){
      document.getElementById('sup-num').textContent=d.supervisors||0; document.getElementById('ns-sup').textContent=d.supervisors||0;
      document.getElementById('footer-ts').textContent=new Date().toLocaleTimeString()+' UTC';
      if(d.tel) renderTelemetry(d.tel); if(d.diag) renderDiag(d.diag); if(d.phys) renderPhysics(d.phys);
      if(d.events) renderEvents(d.events); if(d.trend && d.trend.length){trendData=d.trend; drawSparkline(trendData);}
      if(d.bridge) renderBridge(d.bridge); if(d.clients) renderClients(d.clients);
    }
  };
  ws.onclose=()=>{setPill(false); reconnTimer=setTimeout(connect,3000);}; ws.onerror=()=>ws.close();
}
setInterval(()=>{if(ws&&ws.readyState===1){pingStart=Date.now();ws.send('ping');}},15000);
window.addEventListener('resize',()=>{if(trendData.length) drawSparkline(trendData);});
connect();
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
async def root(): return HTML

if __name__ == "__main__":
    uvicorn.run("fleet_dashboard:app", host="0.0.0.0", port=8765, reload=False, log_level="info")