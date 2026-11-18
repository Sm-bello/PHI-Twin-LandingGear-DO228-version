"""
AeroTwin Live Diagnostics & AI Co-Pilot
Dornier 228-212 | DNAA (Abuja) → DNMM (Lagos)

Architecture:
  Port 5000 - Python is SERVER. FlightGear connects and sends telemetry.
  Port 5001 - FG is SERVER. Python connects and sends autopilot commands.

What this script does:
  1. Receives live telemetry from FlightGear at 20Hz
  2. Logs every packet to PostgreSQL (Grafana + MATLAB read from here)
  3. YOU fly manually. Co-pilot AI engages at 500ft AGL.
  4. Co-pilot flies full route: climb → cruise → approach → ILS → land
  5. At touchdown: AI model classifies landing gear health and shows popup
"""

import os
import socket
import psycopg2
import tensorflow as tf
import numpy as np
import tkinter as tk
from tkinter import messagebox
import math
import time
import threading

# ─── CONFIG ──────────────────────────────────────────────────────────────────
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(_ROOT, "AeroTwin_V16_Model.h5")
DB_CONFIG  = {
    "dbname":   os.environ.get("AEROTWIN_DB_NAME", "project_db"),
    "user":     os.environ.get("AEROTWIN_DB_USER", "postgres"),
    "password": os.environ.get("AEROTWIN_DB_PASSWORD", ""),
    "host":     os.environ.get("AEROTWIN_DB_HOST", "localhost")
}

class_names = [
    'Normal', 'N2 Leak', 'Worn Seal', 'Early Degrad', 'Thermal',
    'Brake Fade', 'Tire Burst', 'Corrosion', 'Hard Landing',
    'Combined', 'Critical'
]

# Lagos DNMM Runway 18L
LAGOS_LAT  = 6.5774
LAGOS_LON  = 3.3212
ILS_FREQ   = 109.70

# Ports — must match BAT file
FG_TELEMETRY_PORT = 5000   # Python listens here (server)
FG_COMMAND_PORT   = 5001   # FG listens here, Python connects (client)

# AP mode codes → sent to /sim/aerotwin/ap-mode-code
# Nasal script in do228-set.xml translates these to actual AP lock properties
AP_OFF      = 0   # Autopilot off
AP_HDG_ALT  = 1   # dg-heading-hold + altitude-hold
AP_NAV1_ALT = 2   # nav1-hold + altitude-hold
AP_ILS      = 3   # nav1-hold + gs1-hold (ILS glideslope)


# ─── LOAD AI MODEL ───────────────────────────────────────────────────────────
print("🧠 Loading AI Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ AI Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ AI Model load FAILED: {e}")
    print("   Diagnostics will show 'Unknown' at touchdown.")
    model = None


# ─── GLOBAL FLIGHT STATE ─────────────────────────────────────────────────────
class FlightState:
    def __init__(self):
        self.fg_connected    = False
        self.last_packet     = time.time()
        self.alt             = 0.0
        self.wow             = 1.0
        self.groundspeed     = 0.0
        self.phase           = "PREFLIGHT"
        self.dist_nm         = 999.0
        self.co_pilot_active = False
        self.td_reported     = False  # touchdown diagnostic fired

state = FlightState()


# ─── HEARTBEAT THREAD ────────────────────────────────────────────────────────
def heartbeat_monitor():
    last_print = 0.0
    while True:
        time.sleep(2.0)
        now = time.time()

        if not state.fg_connected:
            if now - last_print > 10.0:
                print("💤 Waiting for FlightGear to connect on port 5000...")
                last_print = now
            continue

        silence = now - state.last_packet
        if silence > 5.0:
            if now - last_print > 10.0:
                print(f"⚠️  No telemetry for {silence:.0f}s — FG paused?")
                last_print = now
        elif now - last_print > 30.0:
            status = "ON" if state.co_pilot_active else "standby (you fly)"
            print(f"💓 Phase:{state.phase:8s} | Alt:{state.alt:6.0f}ft | "
                  f"GS:{state.groundspeed:5.0f}kt | "
                  f"Dist:{state.dist_nm:5.0f}NM | Co-Pilot:{status}")
            last_print = now


# ─── NAVIGATION ──────────────────────────────────────────────────────────────
def haversine_nm(lat1, lon1, lat2, lon2):
    """Great-circle distance in nautical miles."""
    R = 3440.065
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp     = math.radians(lat2 - lat1)
    dl     = math.radians(lon2 - lon1)
    a = math.sin(dp/2)**2 + math.cos(p1) * math.cos(p2) * math.sin(dl/2)**2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def bearing_to_lagos(lat, lon):
    """Initial bearing from current position to Lagos DNMM."""
    lat1, lon1 = math.radians(lat),       math.radians(lon)
    lat2, lon2 = math.radians(LAGOS_LAT), math.radians(LAGOS_LON)
    dl = lon2 - lon1
    x  = math.sin(dl) * math.cos(lat2)
    y  = math.cos(lat1)*math.sin(lat2) - math.sin(lat1)*math.cos(lat2)*math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360) % 360


# ─── SOCKET HELPERS ──────────────────────────────────────────────────────────
def connect_fg_command(max_tries=30, delay=2.0):
    """Connect to FlightGear's command input port (FG is server on 5001)."""
    print(f"🔌 Connecting to FG command port {FG_COMMAND_PORT}...")
    for i in range(1, max_tries + 1):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(5.0)
            s.connect(('127.0.0.1', FG_COMMAND_PORT))
            s.settimeout(None)
            print(f"✅ Command link established (attempt {i})")
            return s
        except (ConnectionRefusedError, socket.timeout, OSError):
            if i < max_tries:
                time.sleep(delay)
    print("❌ Could not connect to FG command port. Autopilot commands disabled.")
    return None

def send_command(sock, hdg, tgt_alt, ap_master, ap_mode,
                 nav_freq, thr0, thr1, flaps, gear):
    """
    Send 10-field command to FlightGear.
    Must match <input> section of aerotwin_proto.xml exactly.
    """
    if sock is None:
        return False
    try:
        # Clamp all values to safe ranges
        hdg      = max(0.0,   min(360.0,   float(hdg)))
        tgt_alt  = max(0.0,   min(50000.0, float(tgt_alt)))
        ap_master= max(0,     min(1,       int(ap_master)))
        ap_mode  = max(0,     min(3,       int(ap_mode)))
        nav_freq = max(108.0, min(118.0,   float(nav_freq)))
        thr0     = max(0.0,   min(1.0,     float(thr0)))
        thr1     = max(0.0,   min(1.0,     float(thr1)))
        flaps    = max(0.0,   min(1.0,     float(flaps)))
        gear     = 1 if gear > 0.5 else 0

        line = (f"{hdg:.1f},{tgt_alt:.1f},{ap_master},{ap_mode},"
                f"{nav_freq:.2f},{thr0:.3f},{thr1:.3f},"
                f"{flaps:.2f},{gear},0.000\n")
        sock.sendall(line.encode('utf-8'))
        return True
    except (BrokenPipeError, ConnectionResetError, OSError) as e:
        print(f"⚠️  Command send failed: {e}")
        return False


# ─── AI DIAGNOSTIC ───────────────────────────────────────────────────────────
def show_diagnostic_popup(cls, conf):
    def _popup():
        root = tk.Tk()
        root.withdraw()
        messagebox.showinfo(
            "🛰️ AEROTWIN — LANDING GEAR DIAGNOSTIC",
            f"LOCATION: LAGOS (DNMM) — TOUCHDOWN DETECTED\n\n"
            f"AI CLASSIFICATION : {cls}\n"
            f"CONFIDENCE        : {conf*100:.1f}%\n\n"
            f"STATUS: Telemetry logged to PostgreSQL / Grafana"
        )
        root.destroy()
    threading.Thread(target=_popup, daemon=True).start()

def run_ai_diagnostic(fields):
    """
    Run CNN+LSTM inference on landing telemetry.
    Model was trained on 12 features — uses f[0:12].
    If model expects different input shape, adjust the reshape below.
    """
    if model is None:
        return 'Unknown', 0.0
    try:
        x    = np.array(fields[:12], dtype=np.float32).reshape(1, 1, 12)
        pred = model.predict(x, verbose=0)
        idx  = int(np.argmax(pred[0]))
        conf = float(pred[0][idx])
        return class_names[idx], conf
    except Exception as e:
        print(f"⚠️  AI inference error: {e}")
        return 'Unknown', 0.0


# ─── MAIN MISSION LOOP ───────────────────────────────────────────────────────
def start_mission():

    # ── Database ─────────────────────────────────────────────────────────────
    try:
        conn   = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        print("✅ PostgreSQL connected.")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # ── Telemetry server (Python = server, FG connects to us) ────────────────
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(('127.0.0.1', FG_TELEMETRY_PORT))
    srv.listen(1)
    srv.settimeout(1.0)

    print(f"\n🧠 AeroTwin listening on port {FG_TELEMETRY_PORT}")
    print("   Waiting for FlightGear...")
    print("   ─────────────────────────────────────────────────")
    print("   YOU take off manually from DNAA runway 22.")
    print("   AI Co-Pilot engages automatically at 500ft AGL.")
    print("   ─────────────────────────────────────────────────\n")

    threading.Thread(target=heartbeat_monitor, daemon=True).start()

    # ── Outer loop: reconnects when FG restarts ───────────────────────────────
    while True:
        state.co_pilot_active = False
        state.td_reported     = False

        # Wait for FlightGear to connect
        try:
            fg_sock, addr = srv.accept()
            fg_sock.settimeout(5.0)
        except socket.timeout:
            continue
        except Exception as e:
            print(f"❌ Accept error: {e}")
            continue

        state.fg_connected = True
        state.last_packet  = time.time()
        print(f"\n📡 FlightGear connected from {addr}")
        print("   Telemetry streaming. Take off when ready.\n")

        # Now connect to FG's command port (FG opens this as server)
        cmd_sock = connect_fg_command()

        buf              = ""
        last_cmd_time    = 0.0
        last_phase       = ""
        copilot_hdg      = 235.0

        # ── Inner loop: per-packet processing ────────────────────────────────
        try:
            while True:
                try:
                    chunk = fg_sock.recv(4096).decode('utf-8', errors='ignore')
                except socket.timeout:
                    continue

                if not chunk:
                    print("\n🔌 FlightGear disconnected. Waiting for reconnect...")
                    break

                state.last_packet = time.time()
                buf  += chunk
                lines = buf.split('\n')
                buf   = lines[-1]

                for line in lines[:-1]:
                    line = line.strip()
                    if not line or ',' not in line:
                        continue
                    try:
                        f = [float(x) for x in line.split(',')]
                        if len(f) < 16:
                            continue
                    except ValueError:
                        continue

                    # ── Unpack 16-field telemetry (aerotwin_proto.xml output) ─
                    sink_fps      = f[0]
                    weight_lbs    = f[1]
                    nose_steer    = f[2]
                    alt           = f[3]
                    wow_nose      = f[4]
                    wow_left      = f[5]
                    wow_right     = f[6]
                    compression   = f[7]
                    gspeed_kt     = f[8]
                    prop_rpm      = f[9]
                    engine_torque = f[10]
                    airspeed_kt   = f[11]
                    lat           = f[12]
                    lon           = f[13]
                    vsi_fpm       = f[14] if len(f) > 14 else 0.0
                    # f[15]       = elevator_trim (available but not used here)

                    # Any gear touching ground = WOW
                    wow = 1.0 if max(wow_nose, wow_left, wow_right) > 0.5 else 0.0

                    dist_nm = haversine_nm(lat, lon, LAGOS_LAT, LAGOS_LON)

                    # ── Flight Phase ──────────────────────────────────────────
                    if   dist_nm < 5  and alt < 500: phase = "LANDING"
                    elif dist_nm < 40 and alt < 8000: phase = "APPROACH"
                    elif alt > 9000:                  phase = "CRUISE"
                    elif alt > 500:                   phase = "CLIMB"
                    else:                             phase = "TAKEOFF"

                    # Update state
                    state.alt         = alt
                    state.wow         = wow
                    state.groundspeed = gspeed_kt
                    state.phase       = phase
                    state.dist_nm     = dist_nm

                    # Announce phase changes
                    if phase != last_phase:
                        print(f"✈️  PHASE: {last_phase} → {phase} "
                              f"| Alt:{alt:.0f}ft | Dist:{dist_nm:.0f}NM")
                        last_phase = phase

                    # ── PostgreSQL Logging ────────────────────────────────────
                    # MATLAB reads latest row; Grafana reads all rows
                    try:
                        cursor.execute(
                            "INSERT INTO telemetry "
                            "(prop_rpm, engine_torque, groundspeed_kt, alt_agl_ft, "
                            " sink_rate_fps, compression_ft, wow_status, "
                            " mass_lbs, latitude, longitude, nose_steer_norm, flight_phase) "
                            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                            (prop_rpm, engine_torque, gspeed_kt, alt,
                             sink_fps, compression, bool(wow > 0.5),
                             weight_lbs, lat, lon, nose_steer, phase)
                        )
                        conn.commit()
                    except Exception as db_err:
                        print(f"⚠️  DB write error: {db_err}")
                        try: conn.rollback()
                        except: pass

                    # ── Co-Pilot Engagement (500ft AGL, wheels off ground) ────
                    if alt > 500 and not state.co_pilot_active and wow < 0.5:
                        copilot_hdg = bearing_to_lagos(lat, lon)
                        state.co_pilot_active = True
                        print(f"\n{'='*55}")
                        print(f"  ✈️  AI CO-PILOT ENGAGED")
                        print(f"  Altitude  : {alt:.0f}ft AGL")
                        print(f"  Distance  : {dist_nm:.0f}NM to Lagos DNMM")
                        print(f"  Heading   : {copilot_hdg:.0f}° to runway 18L")
                        print(f"  ILS Freq  : {ILS_FREQ} MHz")
                        print(f"{'='*55}\n")

                    # ── Co-Pilot Flight Commands ──────────────────────────────
                    if state.co_pilot_active:
                        now = time.time()
                        if now - last_cmd_time > 0.2:   # 5Hz command rate

                            # Defaults
                            hdg      = copilot_hdg
                            tgt_alt  = 10000.0
                            ap_mstr  = 1
                            ap_mode  = AP_HDG_ALT
                            nav_freq = ILS_FREQ
                            thr      = 0.0
                            flaps    = 0.0
                            gear     = 0     # up by default

                            if phase == "CLIMB":
                                # ── STALL PROTECTION: VSI-based throttle ─────
                                # Do228 Vy ≈ 130 KIAS. Target 800–1000 fpm.
                                # Pure altitude-hold pitches nose up too hard
                                # and bleeds airspeed into a stall. Instead we
                                # hold heading and govern throttle based on the
                                # actual vertical speed reported by the VSI.
                                tgt_alt = 10000.0
                                gear    = 0         # retract gear after takeoff
                                flaps   = 0.0
                                ap_mode = AP_HDG_ALT

                                if vsi_fpm < 400:
                                    # Too slow a climb — add power
                                    thr = 0.92
                                elif vsi_fpm > 1200:
                                    # Climbing too steeply — nose too high,
                                    # airspeed decaying. Back off a little.
                                    thr = 0.80
                                else:
                                    # 400–1200 fpm: normal climb band. Hold 90%.
                                    thr = 0.90

                                # Hard airspeed floor: if IAS drops below 100kt
                                # override target altitude to push the nose down
                                # and apply max power simultaneously.
                                if airspeed_kt < 100:
                                    tgt_alt = max(alt - 500, 1000.0)
                                    thr     = 0.95
                                    print(f"⚠️  AIRSPEED LOW ({airspeed_kt:.0f}kt) — "
                                          f"lowering nose, max power")

                            elif phase == "CRUISE":
                                tgt_alt = 10000.0
                                # gentle throttle management
                                if alt < 9500:    thr = 0.78
                                elif alt > 10500: thr = 0.65
                                else:             thr = 0.72
                                gear    = 0
                                flaps   = 0.0
                                ap_mode = AP_HDG_ALT

                            elif phase == "APPROACH":
                                if dist_nm > 30:
                                    tgt_alt = 5000.0
                                    thr     = 0.60
                                    flaps   = 0.0
                                    gear    = 0
                                else:
                                    tgt_alt = 2500.0
                                    thr     = 0.50
                                    flaps   = 0.25   # partial flaps
                                    gear    = 1      # lower gear
                                ap_mode = AP_NAV1_ALT

                            elif phase == "LANDING":
                                tgt_alt = 1500.0     # glideslope takes over
                                thr     = 0.35
                                flaps   = 0.50
                                gear    = 1
                                ap_mode = AP_ILS      # ILS: nav1 + glideslope

                                if alt < 200:  thr = 0.20
                                if alt < 50:
                                    thr     = 0.0
                                    ap_mstr = 0       # disengage AP for flare
                                    ap_mode = AP_OFF

                            ok = send_command(cmd_sock, hdg, tgt_alt, ap_mstr,
                                              ap_mode, nav_freq, thr, thr, flaps, gear)
                            if not ok:
                                print("🔄 Command link lost — reconnecting...")
                                try: cmd_sock.close()
                                except: pass
                                cmd_sock = connect_fg_command(max_tries=10, delay=1.0)

                            last_cmd_time = now

                    # ── AI Diagnostic at Touchdown ────────────────────────────
                    if wow > 0.5 and gspeed_kt > 40 and not state.td_reported:
                        cls, conf = run_ai_diagnostic(f)
                        print(f"\n🔬 TOUCHDOWN DIAGNOSTIC: {cls} ({conf*100:.1f}%)")
                        show_diagnostic_popup(cls, conf)

                        # Log to flight_diagnostics table
                        try:
                            cursor.execute(
                                "INSERT INTO flight_diagnostics "
                                "(ai_classification, confidence_score, "
                                " k_stiffness_val, b_damping_val) "
                                "VALUES (%s,%s,%s,%s)",
                                (cls, conf, 150000.0, 4000.0)
                            )
                            conn.commit()
                        except Exception as db_err:
                            print(f"⚠️  Diagnostic DB error: {db_err}")
                            try: conn.rollback()
                            except: pass

                        state.td_reported     = True
                        state.co_pilot_active = False
                        print("🏁 Mission complete. Co-pilot disengaged.\n")

        except Exception as e:
            print(f"⚠️  Link error: {e}")
        finally:
            try: fg_sock.close()
            except: pass
            try:
                if cmd_sock: cmd_sock.close()
            except: pass
            state.fg_connected    = False
            state.co_pilot_active = False
            print("🔄 Ready for next flight...\n")


if __name__ == "__main__":
    start_mission()