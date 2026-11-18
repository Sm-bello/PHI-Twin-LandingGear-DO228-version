"""
╔══════════════════════════════════════════════════════════════════════╗
║  fg_bridge.py — FlightGear Do228 ↔ PostgreSQL Bridge                ║
║  AeroTwin v2.1  |  Replaces aerotwin_robot_aircraft.py               ║
║                                                                      ║
║  DATA FLOW:                                                          ║
║    FlightGear UDP :15000 (20Hz, 16-field CSV)                        ║
║        → parse → PostgreSQL.telemetry                               ║
║        → gear logic → PostgreSQL.gear_commands                      ║
║        → gear feedback UDP → FlightGear :15001                      ║
║                                                                      ║
║  HOW TO LAUNCH FlightGear (copy-paste):                              ║
║    fgfs --aircraft=Do228 --airport=DNAA                              ║
║         --generic=socket,out,20,,15000,udp,aerotwin_proto            ║
║         --generic=socket,in,5,,15001,udp,aerotwin_proto              ║
║         --telnet=5501                                                ║
║    (aerotwin_proto.xml must be in FlightGear/data/Protocol/)         ║
║                                                                      ║
║  Run: python fg_bridge.py                                            ║
╚══════════════════════════════════════════════════════════════════════╝

Field index from aerotwin_proto.xml:
  f[0]  vertical_speed_fps   (negative = descending)
  f[1]  weight_lbs
  f[2]  nose_steer_norm
  f[3]  alt_agl_ft
  f[4]  wow_nose
  f[5]  wow_left
  f[6]  wow_right
  f[7]  compression_ft       ← MATLAB uses for gear spring model
  f[8]  groundspeed_kt
  f[9]  prop_rpm
  f[10] engine_torque_ftlb
  f[11] airspeed_kt
  f[12] latitude_deg
  f[13] longitude_deg
  f[14] vsi_fpm
  f[15] elevator_trim
"""

import socket
import threading
import time
import struct
import psycopg2
import psycopg2.extras
from datetime import datetime, timezone

from aerotwin_config import get_db_config, get_fg_config

# ── Configuration ──────────────────────────────────────────────────────────
_FG = get_fg_config()
FG_UDP_LISTEN_PORT  = _FG["listen_port"]   # Port Python listens on (FG sends here)
FG_UDP_SEND_PORT    = _FG["send_port"]     # Port FG listens on (Python sends gear cmd here)
FG_HOST             = _FG["host"]
DB_WRITE_HZ         = 5       # Write rate to PostgreSQL (don't spam at 20Hz)
GEAR_CMD_HZ         = 2       # How often to send gear command back to FG
PRINT_HZ            = 1       # Console heartbeat rate

DB = get_db_config()

# ── Gear logic thresholds (Do228-212) ─────────────────────────────────────
GEAR_UP_ALT_FT     = 1000    # Retract gear above this after takeoff
GEAR_DOWN_ALT_FT   = 1200    # Extend gear below this on approach
APPROACH_VS_FPM    = -200    # Negative = descending; trigger gear extend
TAKEOFF_GS_KT      = 40      # Ground speed above this = rolling / airborne


# ── Shared state (written by UDP thread, read by DB thread) ───────────────
class State:
    lock     = threading.Lock()
    raw      = None           # latest parsed dict
    prev_wow = False          # for touchdown edge detection
    gear_cmd = "DOWN"         # "UP" or "DOWN" — sent back to FG
    last_td  = 0.0            # time of last touchdown (for debounce)
    phase    = "GROUND"
    frames   = 0
    dropped  = 0

state = State()


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE DERIVATION
# ═══════════════════════════════════════════════════════════════════════════
def derive_phase(alt_agl: float, vs_fps: float, gs_kt: float,
                 wow: bool, gear_norm: float) -> str:
    """
    Derive a human-readable flight phase from raw FG telemetry.
    vs_fps: vertical speed in fps (negative = descending in FlightGear convention)
    """
    vs_fpm = vs_fps * 60.0

    if wow and gs_kt < TAKEOFF_GS_KT:
        return "GROUND"
    if wow and gs_kt >= TAKEOFF_GS_KT:
        return "LANDING ROLL"
    if not wow and alt_agl < 500:
        return "TAKEOFF"
    if not wow and alt_agl < 10000 and vs_fpm > 400:
        return "CLIMB"
    if not wow and alt_agl > 10000 and abs(vs_fpm) < 400:
        return "CRUISE"
    if not wow and alt_agl > 10000 and vs_fpm < -400:
        return "DESCENT"
    if not wow and alt_agl < 5000 and vs_fpm < APPROACH_VS_FPM:
        return "APPROACH"
    if not wow and alt_agl < 500 and vs_fpm < 0:
        return "FINAL"
    return "FLIGHT"


# ═══════════════════════════════════════════════════════════════════════════
#  GEAR COMMAND LOGIC
# ═══════════════════════════════════════════════════════════════════════════
def decide_gear(alt_agl: float, vs_fps: float, wow: bool, phase: str) -> str:
    """
    Autonomous gear command for Do228.
    Returns "UP" or "DOWN".
    """
    vs_fpm = vs_fps * 60.0

    # Always down when on ground or very low
    if wow or alt_agl < 200:
        return "DOWN"

    # Retract after confirmed airborne and climbing through 1000ft
    if alt_agl > GEAR_UP_ALT_FT and vs_fpm > 0 and phase in ("TAKEOFF", "CLIMB"):
        return "UP"

    # Extend on approach / descent below threshold
    if alt_agl < GEAR_DOWN_ALT_FT and vs_fpm < APPROACH_VS_FPM:
        return "DOWN"

    # During cruise / high altitude: keep up
    if phase in ("CRUISE", "DESCENT") and alt_agl > GEAR_DOWN_ALT_FT:
        return "UP"

    # Default: maintain current state (don't oscillate)
    return state.gear_cmd


# ═══════════════════════════════════════════════════════════════════════════
#  UDP LISTENER THREAD  (FlightGear → Python, 20 Hz)
# ═══════════════════════════════════════════════════════════════════════════
def udp_listener():
    """
    Listens for 16-field CSV UDP datagrams from FlightGear.
    Parses into a dict and stores in State.raw at up to 20Hz.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("0.0.0.0", FG_UDP_LISTEN_PORT))
    sock.settimeout(5.0)

    print(f"  [UDP-RX] Listening on UDP port {FG_UDP_LISTEN_PORT} ← FlightGear")

    while True:
        try:
            data, addr = sock.recvfrom(1024)
            raw = data.decode("utf-8", errors="replace").strip()
            parts = raw.split(",")

            if len(parts) < 16:
                state.dropped += 1
                continue

            f = [float(x) for x in parts[:16]]

            wow_any = (f[4] > 0.5) or (f[5] > 0.5) or (f[6] > 0.5)
            phase   = derive_phase(f[3], f[0], f[8], wow_any, f[4])
            gear    = decide_gear(f[3], f[0], wow_any, phase)

            # FlightGear convention: vertical_speed_fps is NEGATIVE when descending.
            # CS-23.473 and all downstream thresholds expect positive sink-rate magnitude.
            # Keep signed value for diagnostics; always publish positive magnitude for thresholds.
            raw_vs_fps = f[0]
            sink_mag_fps = abs(raw_vs_fps)
            mass_lbs_raw = f[1]
            # Guard: Do-228 typical operating mass ~4500–5700 kg. If UDP reports ~0,
            # use a realistic default so Simscape / AI never see mass=0.
            if mass_lbs_raw < 100:          # < ~45 kg is clearly bogus
                mass_lbs_raw = 12125.0      # ≈ 5500 kg
            parsed = {
                "sink_rate_fps":  raw_vs_fps,                 # keep signed for debug
                "sink_rate_ms":   sink_mag_fps * 0.3048,      # POSITIVE magnitude (m/s)
                "mass_lbs":       mass_lbs_raw,
                "mass_kg":        mass_lbs_raw * 0.453592,
                "nose_steer":     f[2],
                "alt_agl_ft":     f[3],
                "wow_nose":       f[4] > 0.5,
                "wow_left":       f[5] > 0.5,
                "wow_right":      f[6] > 0.5,
                "wow_any":        wow_any,
                "compression_ft": f[7],
                "groundspeed_kt": f[8],
                "prop_rpm":       f[9],
                "engine_torque":  f[10],
                "airspeed_kt":    f[11],
                "latitude":       f[12],
                "longitude":      f[13],
                "vsi_fpm":        f[14],
                "elevator_trim":  f[15],
                "flight_phase":   phase,
                "gear_cmd":       gear,
                "ts":             datetime.now(timezone.utc),
            }

            with state.lock:
                state.raw      = parsed
                state.gear_cmd = gear
                state.phase    = phase
                state.frames  += 1

        except socket.timeout:
            continue
        except ValueError:
            state.dropped += 1
        except Exception as e:
            print(f"  [UDP-RX] Error: {e}")
            time.sleep(0.1)


# ═══════════════════════════════════════════════════════════════════════════
#  DB WRITER THREAD  (parsed telemetry → PostgreSQL, 5 Hz)
# ═══════════════════════════════════════════════════════════════════════════
def db_writer():
    """
    Writes latest telemetry to PostgreSQL at DB_WRITE_HZ.
    Also handles touchdown detection and gear_commands table.
    """
    interval = 1.0 / DB_WRITE_HZ
    conn = _db_connect()
    _ensure_schema(conn)

    print(f"  [DB-WR] Writing to PostgreSQL at {DB_WRITE_HZ}Hz")

    while True:
        time.sleep(interval)

        with state.lock:
            d = state.raw
            gear = state.gear_cmd
            prev_wow = state.prev_wow

        if d is None:
            continue

        try:
            cur = conn.cursor()

            # ── 1. Write telemetry ─────────────────────────────────────
            cur.execute("""
                INSERT INTO telemetry (
                    timestamp, sink_rate_fps, compression_ft,
                    wow_status, groundspeed_kt, alt_agl_ft,
                    mass_lbs, engine_torque, prop_rpm,
                    flight_phase, airspeed_kt, latitude_deg,
                    longitude_deg, wow_nose, wow_left, wow_right,
                    vsi_fpm, mass_kg, sink_rate_ms
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
            """, (
                d["ts"],
                d["sink_rate_fps"],
                d["compression_ft"],
                d["wow_any"],
                d["groundspeed_kt"],
                d["alt_agl_ft"],
                d["mass_lbs"],
                d["engine_torque"],
                d["prop_rpm"],
                d["flight_phase"],
                d["airspeed_kt"],
                d["latitude"],
                d["longitude"],
                d["wow_nose"],
                d["wow_left"],
                d["wow_right"],
                d["vsi_fpm"],
                d["mass_kg"],
                d["sink_rate_ms"],
            ))

            # ── 2. Write gear command ──────────────────────────────────
            cur.execute("""
                INSERT INTO gear_commands (ts, command)
                VALUES (%s, %s)
            """, (d["ts"], gear))

            # ── 3. Touchdown detection → flag for MATLAB ───────────────
            now = time.time()
            wow_rising = d["wow_any"] and not prev_wow
            if wow_rising and (now - state.last_td > 5.0):
                state.last_td = now
                print(f"\n  [TOUCHDOWN] WOW detected | "
                      f"sink={d['sink_rate_ms']:.3f}m/s | "
                      f"mass={d['mass_kg']:.0f}kg | "
                      f"phase={d['flight_phase']}")

                # Insert a touchdown marker so MATLAB polling picks it up
                cur.execute("""
                    INSERT INTO touchdown_events (
                        ts, sink_rate_ms, compression_ft,
                        mass_kg, groundspeed_kt, flight_phase
                    ) VALUES (%s, %s, %s, %s, %s, %s)
                    ON CONFLICT DO NOTHING
                """, (
                    d["ts"], d["sink_rate_ms"], d["compression_ft"],
                    d["mass_kg"], d["groundspeed_kt"], d["flight_phase"],
                ))

            with state.lock:
                state.prev_wow = d["wow_any"]
            conn.commit()

        except psycopg2.OperationalError:
            print("  [DB-WR] Connection lost — reconnecting…")
            try: conn.close()
            except: pass
            conn = _db_connect()
            _ensure_schema(conn)

        except Exception as e:
            print(f"  [DB-WR] Error: {e}")
            try: conn.rollback()
            except: pass


# ═══════════════════════════════════════════════════════════════════════════
#  GEAR COMMAND SENDER  (Python → FlightGear UDP, 2 Hz)
# ═══════════════════════════════════════════════════════════════════════════
def gear_sender():
    """
    Sends 10-field CSV commands back to FlightGear via UDP (port 15001).
    Format matches the <input> section of aerotwin_proto.xml:
      [0] heading_bug_deg
      [1] target_altitude_ft
      [2] ap_master (0=off)
      [3] ap_mode_code
      [4] nav1_freq
      [5] throttle_eng0
      [6] throttle_eng1
      [7] flaps_norm
      [8] gear_down  ← 1=DOWN, 0=UP
      [9] elevator_trim
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    interval = 1.0 / GEAR_CMD_HZ
    print(f"  [UDP-TX] Sending gear commands to FlightGear port {FG_UDP_SEND_PORT}")

    while True:
        time.sleep(interval)

        with state.lock:
            gear   = state.gear_cmd
            d      = state.raw

        if d is None:
            continue

        gear_norm = 1.0 if gear == "DOWN" else 0.0

        # Build 10-field CSV — only overriding gear (index 8)
        # AP and throttle fields use 0 so FG ignores them
        # (FG only writes the property if the protocol says to)
        cmd_str = (
            f"0.0,"          # [0] heading_bug — not overriding
            f"0.0,"          # [1] target_alt  — not overriding
            f"0.0,"          # [2] ap_master   — not overriding
            f"0.0,"          # [3] ap_mode     — not overriding
            f"0.0,"          # [4] nav1_freq   — not overriding
            f"0.0,"          # [5] throttle 0  — not overriding
            f"0.0,"          # [6] throttle 1  — not overriding
            f"0.0,"          # [7] flaps       — not overriding
            f"{gear_norm:.1f},"  # [8] GEAR
            f"0.0\n"         # [9] elev trim   — not overriding
        )

        try:
            sock.sendto(cmd_str.encode("utf-8"), (FG_HOST, FG_UDP_SEND_PORT))
        except Exception as e:
            print(f"  [UDP-TX] Send error: {e}")


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLE HEARTBEAT
# ═══════════════════════════════════════════════════════════════════════════
def heartbeat():
    interval = 1.0 / PRINT_HZ
    last_frames = 0

    while True:
        time.sleep(interval)

        with state.lock:
            d      = state.raw
            fps    = state.frames - last_frames
            last_frames = state.frames
            gear   = state.gear_cmd
            phase  = state.phase

        if d is None:
            print(f"  [BRIDGE] Waiting for FlightGear UDP packets on port {FG_UDP_LISTEN_PORT}…")
            continue

        wow_str = "WOW" if d["wow_any"] else "AIR"
        print(
            f"  [{phase:<12}] "
            f"Alt:{d['alt_agl_ft']:7.0f}ft  "
            f"GS:{d['groundspeed_kt']:5.1f}kt  "
            f"Sink:{d['sink_rate_ms']:6.3f}m/s  "
            f"Comp:{d['compression_ft']:.4f}ft  "
            f"{wow_str}  GEAR:{gear}  "
            f"{fps}Hz  drop:{state.dropped}"
        )


# ═══════════════════════════════════════════════════════════════════════════
#  DATABASE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _db_connect():
    for attempt in range(1, 6):
        try:
            c = psycopg2.connect(**DB)
            print(f"  [DB] Connected to PostgreSQL (attempt {attempt})")
            return c
        except Exception as e:
            print(f"  [DB] Connection attempt {attempt}/5 failed: {e}")
            time.sleep(3)
    raise RuntimeError("Cannot connect to PostgreSQL after 5 attempts")


def _ensure_schema(conn):
    """
    Add any missing columns to telemetry and create helper tables.
    Safe to run on every startup — uses IF NOT EXISTS / IF NOT EXISTS.
    """
    cur = conn.cursor()

    # Extra columns FlightGear provides that robot aircraft didn't
    extra_cols = [
        ("airspeed_kt",   "FLOAT"),
        ("latitude_deg",  "FLOAT"),
        ("longitude_deg", "FLOAT"),
        ("wow_nose",      "BOOLEAN"),
        ("wow_left",      "BOOLEAN"),
        ("wow_right",     "BOOLEAN"),
        ("vsi_fpm",       "FLOAT"),
        ("mass_kg",       "FLOAT"),
        ("sink_rate_ms",  "FLOAT"),
        ("flight_phase",  "VARCHAR(20)"),
    ]
    for col, dtype in extra_cols:
        cur.execute(f"""
            ALTER TABLE telemetry
            ADD COLUMN IF NOT EXISTS {col} {dtype}
        """)

    # Touchdown events table (MATLAB polls this)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS touchdown_events (
            id             SERIAL PRIMARY KEY,
            ts             TIMESTAMPTZ DEFAULT NOW(),
            sink_rate_ms   FLOAT,
            compression_ft FLOAT,
            mass_kg        FLOAT,
            groundspeed_kt FLOAT,
            flight_phase   VARCHAR(20),
            matlab_seen    BOOLEAN DEFAULT FALSE
        )
    """)

    # Gear commands table (MATLAB reads, fg_bridge writes)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS gear_commands (
            id      SERIAL PRIMARY KEY,
            ts      TIMESTAMPTZ DEFAULT NOW(),
            command VARCHAR(10)
        )
    """)

    conn.commit()
    print("  [DB] Schema verified / updated ✓")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("""
╔══════════════════════════════════════════════════════════════════════╗
║  PHI-FLEET  ·  fg_bridge.py  ·  FlightGear → PostgreSQL             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Step 1: Start FlightGear with these extra flags:                   ║
║                                                                      ║
║   --generic=socket,out,20,,15000,udp,aerotwin_proto                 ║
║   --generic=socket,in,5,,15001,udp,aerotwin_proto                   ║
║   --telnet=5501                                                      ║
║                                                                      ║
║  Full launch command:                                                ║
║   fgfs --aircraft=Do228 --airport=DNAA                              ║
║        --generic=socket,out,20,,15000,udp,aerotwin_proto            ║
║        --generic=socket,in,5,,15001,udp,aerotwin_proto              ║
║        --telnet=5501                                                 ║
║                                                                      ║
║  Step 2: Run this bridge → python fg_bridge.py                      ║
║  Step 3: Run dashboard   → python phi_fleet_server.py               ║
║  Step 4: In MATLAB       → launch_twin.m                            ║
╚══════════════════════════════════════════════════════════════════════╝
    """)

    # Thread factory: maps name → target function (so we can rebuild dead ones)
    THREAD_TARGETS = {
        "UDP-RX":    udp_listener,
        "DB-WR":     db_writer,
        # "UDP-TX":    gear_sender,
        "HEARTBEAT": heartbeat,
    }

    threads = {
        name: threading.Thread(target=fn, daemon=True, name=name)
        for name, fn in THREAD_TARGETS.items()
    }

    for t in threads.values():
        t.start()
        time.sleep(0.1)

    print("  [BRIDGE] All threads running. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(2)
            for name, t in list(threads.items()):
                if not t.is_alive():
                    print(f"  [BRIDGE] ⚠ Thread {name} died — restarting…")
                    new_t = threading.Thread(
                        target=THREAD_TARGETS[name], daemon=True, name=name
                    )
                    new_t.start()
                    threads[name] = new_t
    except KeyboardInterrupt:
        print("\n  [BRIDGE] Shutting down cleanly.")


if __name__ == "__main__":
    main()