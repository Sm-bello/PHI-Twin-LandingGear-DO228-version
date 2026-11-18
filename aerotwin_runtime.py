"""Shared physics, AI-fusion, schema, and 3D visualization helpers for PHI-Twin."""

import math
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyvista as pv

CONFIDENCE_THRESHOLD = 0.65

CLASS_NAMES = [
    "Normal",
    "N2 Leak",
    "Worn Seal",
    "Early Degrad",
    "Thermal Degrad",
    "Brake Fade",
    "Tire Burst",
    "Corrosion",
    "Hard Landing",
    "Combined Fault",
    "Impending Failure",
]

SEV_ACCENT = {
    1: (0.0, 0.90, 0.46),
    2: (1.0, 0.92, 0.0),
    3: (1.0, 0.24, 0.0),
    4: (0.88, 0.25, 0.98),
}

AMM_META: Dict[int, Tuple[int, int, str, str]] = {
    0: (1, 450, "ATA 32-10", "Routine Line Check"),
    1: (2, 850, "ATA 32-10", "Nitrogen System Service"),
    2: (2, 1250, "ATA 32-10", "Hydraulic Damper Service"),
    3: (2, 1800, "ATA 32-10", "Degradation Trending Inspection"),
    4: (2, 1500, "ATA 32-10", "Thermal Stress Inspection"),
    5: (2, 2000, "ATA 32-10", "Asymmetric Load Inspection"),
    6: (3, 3500, "ATA 32-40", "Wheel Assembly Inspection"),
    7: (2, 2200, "ATA 32-10", "Corrosion Control Treatment"),
    8: (3, 4500, "ATA 32-00", "Hard Landing Inspection"),
    9: (4, 12000, "ATA 32-10", "Component Overhaul"),
    10: (4, 35000, "ATA 32-00", "Emergency AOG Maintenance"),
}


def compute_health_factor(
    sink_rate_ms: float, compression_ft: float, max_accel_g: float, mass_kg: float
) -> float:
    """Estimate landing gear strut health index in [0.0, 1.0]."""
    health = 1.0
    if sink_rate_ms > 0.5:
        health -= (sink_rate_ms - 0.5) * 0.15
    if compression_ft > 0.15:
        health -= (compression_ft - 0.15) * 2.0
    if max_accel_g > 1.5:
        health -= (max_accel_g - 1.5) * 0.2
    mass_factor = min(1.0, mass_kg / 5500.0)
    health *= 0.8 + 0.2 * mass_factor
    return max(0.0, min(1.0, health))


def estimate_k_b_parameters(compression_ft: float, sink_rate_ms: float) -> Tuple[float, float]:
    """Estimate reduced-order stiffness K (N/m) and damping B (N*s/m)."""
    k_baseline, b_baseline = 220000.0, 7500.0
    k_reduction = (compression_ft / 0.1) * 0.05 if compression_ft > 0 else 0.0
    b_reduction = (sink_rate_ms - 0.1) / 2.0 if sink_rate_ms > 0.1 else 0.0
    return (
        max(100000.0, k_baseline * (1.0 - k_reduction)),
        max(1000.0, b_baseline * (1.0 - b_reduction)),
    )


def synthesize_physics_arrays(
    k: float, b: float, mass_kg: float, sink_ms: float, comp_ft: float
) -> List[np.ndarray]:
    """Synthesize a 50-step 1-DOF oleo response when MATLAB Simscape is offline."""
    n = 50
    t = np.linspace(0.0, 0.70, n)
    om = math.sqrt(max(k / mass_kg, 1e-3))
    zd = min(b / (2.0 * math.sqrt(k * mass_kg)), 0.995)
    od = om * math.sqrt(max(1.0 - zd**2, 1e-4))
    comp0 = comp_ft * 0.3048
    defl_m = comp0 * np.exp(-zd * om * t) * np.abs(np.cos(od * t))
    force = k * defl_m + b * sink_ms * np.exp(-om * t * 0.8)
    vel = -zd * om * defl_m + od * comp0 * np.exp(-zd * om * t) * np.sin(od * t)
    accel = force / (mass_kg * 9.81)
    pres = force / (np.pi * 0.050**2) / 1e6
    return [defl_m, vel, force, accel, pres]


def build_ai_features(raw_arrays) -> np.ndarray:
    """Normalize 5 time-series arrays into shape (50, 5) float32 tensor."""
    sigs = []
    for arr in raw_arrays[:5]:
        a = np.array(arr, dtype=np.float32)
        if len(a) > 50:
            a = a[:50]
        elif len(a) < 50:
            a = np.pad(a, (0, 50 - len(a)), "edge")
        sigs.append(a)
    return np.stack(sigs, axis=1)


def classify_physics_fallback(
    sink_ms: float, comp_ft: float, accel_g: float, health: float
) -> int:
    """Deterministic plant-reference class index under CS-23.473 and strut limits."""
    if accel_g >= 1.80 or sink_ms >= 3.05:
        return 8  # Hard Landing
    if comp_ft > 0.25 or health < 0.25:
        return 10  # Impending Failure
    if health < 0.35:
        return 4  # Thermal Degrad
    if health < 0.55:
        return 2  # Worn Seal
    if health < 0.75:
        return 1  # N2 Leak
    return 0  # Normal


def get_amm_metadata(cls_idx: int) -> Tuple[int, int, str, str]:
    """Return (severity, cost_usd, ata_chapter, maintenance_category) for a fault index."""
    return AMM_META.get(cls_idx, AMM_META[0])


def poll_unread_matlab_physics(conn, wait_seconds: float = 2.0) -> Optional[dict]:
    """Check matlab_physics_output for an unread Simscape impact trace."""
    deadline = time.time() + max(0.0, wait_seconds)
    cur = conn.cursor()
    try:
        while True:
            cur.execute(
                """
                SELECT id, sim_deflection_raw, sim_velocity_raw, sim_force_raw,
                       sim_accel_raw, sim_pressure_raw, sim_accel_max_g,
                       k_eff, b_eff, mass_kg, sink_rate_ms
                FROM matlab_physics_output
                WHERE ready_for_ai = TRUE AND ai_consumed = FALSE
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if row:
                cur.execute(
                    "UPDATE matlab_physics_output SET ai_consumed = TRUE WHERE id = %s",
                    (row[0],),
                )
                conn.commit()
                return {
                    "raw_arrays": [row[1], row[2], row[3], row[4], row[5]],
                    "accel_g": float(row[6] or 0.0),
                    "k_eff": float(row[7] or 150000.0),
                    "b_eff": float(row[8] or 4000.0),
                    "mass_kg": float(row[9] or 5500.0),
                    "sink_ms": abs(float(row[10] or 0.0)),
                }
            if time.time() >= deadline:
                return None
            time.sleep(0.25)
    except Exception:
        return None
    finally:
        cur.close()


def ensure_telemetry_schema(conn) -> None:
    """Safely ensure telemetry columns exist on existing databases."""
    cur = conn.cursor()
    try:
        cols = [
            ("flight_phase", "VARCHAR(20)"),
            ("airspeed_kt", "FLOAT"),
            ("latitude_deg", "FLOAT"),
            ("longitude_deg", "FLOAT"),
            ("wow_nose", "BOOLEAN"),
            ("wow_left", "BOOLEAN"),
            ("wow_right", "BOOLEAN"),
            ("vsi_fpm", "FLOAT"),
            ("mass_kg", "FLOAT"),
            ("sink_rate_ms", "FLOAT"),
        ]
        for col, dtype in cols:
            cur.execute(f"ALTER TABLE telemetry ADD COLUMN IF NOT EXISTS {col} {dtype}")
        conn.commit()
    finally:
        cur.close()


def fetch_latest_phys_payload(cur, round_fn) -> Optional[dict]:
    """Build the `phys` WebSocket payload from the latest matlab_physics_output row."""
    try:
        cur.execute(
            """
            SELECT sim_deflection_max, sim_force_max, sim_accel_max_g,
                   sim_pressure_max, k_eff, b_eff, ts
            FROM matlab_physics_output
            ORDER BY ts DESC
            LIMIT 1
            """
        )
        phys = cur.fetchone()
        if not phys:
            return None
        return {
            "defl_mm": round_fn(float(phys[0] or 0.0) * 1000.0, 2),
            "force_kN": round_fn(float(phys[1] or 0.0) / 1000.0, 2),
            "accel_g": round_fn(phys[2], 3),
            "pres_kPa": round_fn(float(phys[3] or 0.0) / 1000.0, 1),
            "k_eff": round_fn(phys[4], 0),
            "b_eff": round_fn(phys[5], 0),
            "ts": str(phys[6]),
        }
    except Exception:
        return None


def fetch_fleet_snapshot(db_cfg: dict, trend_n: int, supervisors_count: int) -> Optional[dict]:
    """Query PostgreSQL for the latest telemetry, diagnosis, physics, and event history."""
    import psycopg2
    from datetime import datetime

    try:
        c = psycopg2.connect(**db_cfg)
        cur = c.cursor()
        cur.execute(
            "SELECT sink_rate_fps, compression_ft, wow_status, groundspeed_kt, alt_agl_ft, "
            "mass_lbs, engine_torque, prop_rpm, flight_phase, timestamp, sink_rate_ms, mass_kg "
            "FROM telemetry ORDER BY timestamp DESC LIMIT 1"
        )
        tel = cur.fetchone()
        cur.execute(
            "SELECT ai_classification, confidence_score, physics_class, k_stiffness_val, "
            "b_damping_val, sink_rate_ms, max_accel_g, compression_ft, health_factor, "
            "cs23_triggered, severity_level, ata_chapter, maintenance_cat, estimated_cost_usd, "
            "grounded, data_source, decision_source, ts FROM flight_diagnostics "
            "ORDER BY ts DESC LIMIT 1"
        )
        diag = cur.fetchone()
        cur.execute(
            "SELECT ai_classification, severity_level, max_accel_g, cs23_triggered, "
            "estimated_cost_usd, grounded, flight_phase, ts FROM flight_diagnostics "
            "ORDER BY ts DESC LIMIT 8"
        )
        events = cur.fetchall()
        cur.execute(f"SELECT health_factor FROM flight_diagnostics ORDER BY ts DESC LIMIT {int(trend_n)}")
        trend_rows = cur.fetchall()

        def f(v, n=3):
            return round(float(v or 0), n)

        phys = fetch_latest_phys_payload(cur, f)
        c.close()
    except Exception:
        return None

    out = {"supervisors": supervisors_count, "ts": str(datetime.utcnow())}
    if tel:
        sink_ms_val = tel[10] if tel[10] is not None else abs(float(tel[0] or 0.0)) * 0.3048
        mass_kg_val = tel[11] if tel[11] is not None else float(tel[5] or 0.0) * 0.453592
        out["tel"] = {
            "sink_fps": f(tel[0], 2),
            "sink_ms": f(abs(float(sink_ms_val or 0.0)), 3),
            "comp_ft": f(tel[1], 4),
            "wow": bool(tel[2]),
            "gs_kt": f(tel[3], 1),
            "alt_ft": f(tel[4], 0),
            "mass_lbs": f(tel[5], 0),
            "mass_kg": f(mass_kg_val, 1),
            "eng_torque": f(tel[6], 1),
            "prop_rpm": f(tel[7], 0),
            "phase": tel[8] or "—",
            "ts": str(tel[9]),
        }
    if diag:
        out["diag"] = {
            "cls": diag[0] or "—",
            "conf": f((diag[1] or 0) * 100, 1),
            "phys_cls": diag[2] or "—",
            "k": f(diag[3], 0),
            "b": f(diag[4], 0),
            "sink_ms": f(diag[5], 3),
            "accel_g": f(diag[6], 3),
            "comp_ft": f(diag[7], 4),
            "health": f((diag[8] or 0) * 100, 1),
            "cs23": bool(diag[9]),
            "sev": int(diag[10] or 1),
            "ata": diag[11] or "—",
            "maint": diag[12] or "—",
            "cost": f(diag[13], 0),
            "grounded": bool(diag[14]),
            "src": diag[15] or "—",
            "dec": diag[16] or "—",
            "ts": str(diag[17]),
        }
    if phys:
        out["phys"] = phys

    out["events"] = [
        {
            "cls": e[0] or "—",
            "sev": int(e[1] or 1),
            "g": f(e[2], 2),
            "cs23": bool(e[3]),
            "cost": f(e[4], 0),
            "gnd": bool(e[5]),
            "phase": e[6] or "—",
            "ts": str(e[7]),
        }
        for e in events
    ]
    out["trend"] = [f((r[0] or 0) * 100, 1) for r in reversed(trend_rows)]
    return out


def strut_geometry(compression_m: float, sev: int):
    """Build PyVista meshes for the 3D Dornier 228 oleo-pneumatic strut."""
    parts = []
    housing = pv.Cylinder(
        center=(0, 0, 0.30), direction=(0, 0, 1), radius=0.065, height=0.60, resolution=64
    )
    housing["stress"] = np.clip(np.linspace(0.1, 0.45, housing.n_points), 0, 1)
    parts.append(("housing", housing, "stress", "Blues_r", [0, 1], 0.55))

    piston = pv.Cylinder(
        center=(0, 0, -compression_m * 0.5 - 0.12),
        direction=(0, 0, 1),
        radius=0.048,
        height=0.36,
        resolution=64,
    )
    sv = min(1.0, compression_m / 0.180)
    piston["stress"] = np.full(piston.n_points, sv)
    parts.append(("piston", piston, "stress", "coolwarm", [0, 1], 1.0))

    for dx in (-0.09, 0.09):
        s = pv.Box(bounds=(dx - 0.012, dx + 0.012, -0.020, 0.020, -compression_m * 0.5 - 0.25, 0.05))
        s["stress"] = np.full(s.n_points, sv * 0.6)
        parts.append((f"sc{dx}", s, "stress", "coolwarm", [0, 1], 1.0))

    axle_z = -compression_m * 0.5 - 0.40
    axle = pv.Cylinder(
        center=(0, 0, axle_z), direction=(1, 0, 0), radius=0.022, height=0.34, resolution=32
    )
    parts.append(("axle", axle, None, (0.55, 0.55, 0.60), None, 1.0))

    tire_col = (0.85, 0.10, 0.05) if sev >= 3 else (0.18, 0.18, 0.18)
    tire = pv.Sphere(radius=0.130, center=(0, 0, axle_z), theta_resolution=40, phi_resolution=40)
    parts.append(("tire", tire, None, tire_col, None, 1.0))

    patch = pv.Disc(
        center=(0, 0, axle_z - 0.130), normal=(0, 0, 1), inner=0.0, outer=0.08, r_res=1, c_res=40
    )
    acc = SEV_ACCENT.get(sev, SEV_ACCENT[1])
    parts.append(("patch", patch, None, (acc[0] * 0.9, acc[1] * 0.9, acc[2] * 0.9), None, 0.75))
    return parts


def launch_strut_plotter(
    sink_ms: float,
    comp_ft: float,
    k_eff: float,
    mass_kg: float,
    health: float,
    sev: int,
    ai_cls: str,
    ai_conf: float,
    raw_arrays,
    active_plotter=None,
):
    """Open or refresh the 4-panel PyVista 3D structural state window."""
    try:
        from pyvistaqt import BackgroundPlotter
    except Exception:
        return active_plotter

    if active_plotter is not None:
        try:
            active_plotter.close()
        except Exception:
            pass

    bg = "#0f1419"
    comp_m = comp_ft * 0.3048
    degrad_k = (1.0 - k_eff / 150000.0) * 100.0
    accel_max = float(np.max(raw_arrays[3])) if raw_arrays else (sink_ms * k_eff) / (mass_kg * 9.81)
    rul = max(0, int(health * 1200 - accel_max * 150 - degrad_k * 15))
    title = f"NEURAL SENSE | {ai_cls} ({ai_conf * 100:.0f}%) | Sev {sev} | RUL: ~{rul}"

    pl = BackgroundPlotter(shape=(2, 2), title=title, window_size=(1480, 820), border=False)
    pl.set_background(bg)

    pl.subplot(0, 0)
    pl.set_background(bg)
    sbar = dict(
        title="Stress Index",
        color="white",
        title_font_size=10,
        label_font_size=9,
        n_labels=3,
        position_x=0.80,
        position_y=0.05,
        width=0.12,
        height=0.40,
    )
    for name, mesh, scalars, cm_or_col, clim, opacity in strut_geometry(comp_m, sev):
        if scalars:
            pl.add_mesh(
                mesh,
                scalars=scalars,
                cmap=cm_or_col,
                clim=clim,
                opacity=opacity,
                show_scalar_bar=(name == "piston"),
                scalar_bar_args=sbar if name == "piston" else {},
            )
        else:
            pl.add_mesh(mesh, color=cm_or_col or (0.55, 0.55, 0.60), opacity=opacity)
    pl.add_text("OLEO STRUT — STRUCTURAL STATE", font_size=9, color="cyan", position="upper_edge")

    bpts = np.array([[-0.30, -0.30, 0.82], [0.30, -0.30, 0.82], [0.30, 0.30, 0.82], [-0.30, 0.30, 0.82]])
    ring = pv.lines_from_points(np.vstack([bpts, bpts[0]]))
    pl.add_mesh(ring, color=SEV_ACCENT.get(sev, SEV_ACCENT[1]), line_width=4, render_lines_as_tubes=True)
    pl.camera_position = [(0.55, -0.80, 0.20), (0.00, 0.00, -0.10), (0.00, 0.00, 1.00)]

    t = np.linspace(0.0, 0.70, 50)
    defl_mm = np.asarray(raw_arrays[0]) * 1000.0
    force_kN = np.asarray(raw_arrays[2]) / 1000.0
    accel_g = np.asarray(raw_arrays[3])

    pl.subplot(0, 1)
    chart_f = pv.Chart2D(size=(1.0, 1.0), loc=(0.0, 0.0))
    chart_f.background_color = bg
    chart_f.line(t, force_kN, color=(1.0, 0.70, 0.0), width=2.5, label="Force (kN)")
    chart_f.x_label = "Time (s)"
    chart_f.y_label = "Force (kN)"
    pl.add_chart(chart_f)
    pl.add_text("FORCE PROFILE", font_size=9, color="cyan", position="upper_edge")

    pl.subplot(1, 0)
    chart_d = pv.Chart2D(size=(1.0, 1.0), loc=(0.0, 0.0))
    chart_d.background_color = bg
    chart_d.line(t, defl_mm, color=(1.0, 0.85, 0.0), width=2.5, label="Deflection (mm)")
    chart_d.x_label = "Time (s)"
    chart_d.y_label = "Deflection (mm)"
    pl.add_chart(chart_d)
    pl.add_text("COMPRESSION PROFILE", font_size=9, color="cyan", position="upper_edge")

    pl.subplot(1, 1)
    chart_g = pv.Chart2D(size=(1.0, 1.0), loc=(0.0, 0.0))
    chart_g.background_color = bg
    chart_g.line(t, accel_g, color=(0.20, 0.95, 0.30), width=2.5, label="G-Load (g)")
    chart_g.line(t, np.full_like(t, 1.80), color=(1.0, 0.90, 0.0), width=1.8, style="--")
    chart_g.x_label = "Time (s)"
    chart_g.y_label = "Acceleration (g)"
    pl.add_chart(chart_g)
    pl.add_text("G-LOAD / REGULATORY CHECK", font_size=9, color="cyan", position="upper_edge")
    return pl


def run_touchdown_diagnostics(
    conn,
    telemetry_row,
    ai_model,
    s_seq,
    s_scl,
    active_plotter=None,
    matlab_wait_s: float = 2.0,
):
    """Run AI-physics fusion on touchdown telemetry, log to DB, and update PyVista."""
    sink_ms = abs(float(telemetry_row[1] or 0.0))
    comp_ft = float(telemetry_row[2] or 0.0)
    mass_kg = float(telemetry_row[3] or 5500.0)
    if mass_kg < 100.0:
        mass_kg = 5500.0
    accel_est = abs(float(telemetry_row[4] or 0.0)) / 300.0

    matlab_phys = poll_unread_matlab_physics(conn, wait_seconds=matlab_wait_s)
    if matlab_phys:
        raw_arrays = matlab_phys["raw_arrays"]
        k_eff = matlab_phys["k_eff"]
        b_eff = matlab_phys["b_eff"]
        accel_est = max(accel_est, matlab_phys["accel_g"])
        sink_ms = max(sink_ms, matlab_phys["sink_ms"])
        mass_kg = matlab_phys["mass_kg"] if matlab_phys["mass_kg"] >= 100.0 else mass_kg
        data_source = "MATLAB_Prismatic_Joint"
    else:
        k_eff, b_eff = estimate_k_b_parameters(comp_ft, sink_ms)
        raw_arrays = synthesize_physics_arrays(k_eff, b_eff, mass_kg, sink_ms, comp_ft)
        data_source = "Real_FG_Inference"

    health = compute_health_factor(sink_ms, comp_ft, accel_est, mass_kg)
    phys_idx = classify_physics_fallback(sink_ms, comp_ft, accel_est, health)
    phys_cls = CLASS_NAMES[phys_idx]

    final_idx = phys_idx
    conf = 0.85
    decision_source = "Physics_Fallback"

    if ai_model is not None and s_seq is not None and s_scl is not None:
        x_seq = build_ai_features(raw_arrays).reshape(1, 50, 5)
        x_seq_s = s_seq.transform(x_seq.reshape(-1, 5)).reshape(1, 50, 5)
        x_ctx = np.array([[mass_kg, sink_ms, 27.0, 0.8]])
        x_ctx_s = s_scl.transform(x_ctx)

        pred = ai_model.predict([x_seq_s, x_ctx_s], verbose=0)[0]
        ai_idx = int(np.argmax(pred))
        conf = float(pred[ai_idx])

        if conf >= CONFIDENCE_THRESHOLD:
            final_idx = ai_idx
            decision_source = "AI_Fusion"
        else:
            final_idx = phys_idx
            decision_source = "Physics_Fallback"

    ai_cls = CLASS_NAMES[final_idx]
    sev, cost, chapter, cat = get_amm_metadata(final_idx)
    cs23 = (accel_est >= 1.80) or (sink_ms >= 3.05)
    grounded = (sev == 4) or cs23

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO flight_diagnostics (
            ai_classification, confidence_score, physics_class, health_factor,
            sink_rate_ms, compression_ft, max_accel_g, k_stiffness_val, b_damping_val,
            cs23_triggered, severity_level, ata_chapter, maintenance_cat,
            estimated_cost_usd, grounded, data_source, decision_source, flight_phase
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """,
        (
            ai_cls,
            conf,
            phys_cls,
            health,
            sink_ms,
            comp_ft,
            accel_est,
            k_eff,
            b_eff,
            cs23,
            sev,
            chapter,
            cat,
            cost,
            grounded,
            data_source,
            decision_source,
            "LANDING",
        ),
    )
    conn.commit()
    cur.close()

    print(
        f"  [AI] Classified: {ai_cls} ({conf * 100:.1f}%) | "
        f"Physics: {phys_cls} | Source: {decision_source} ({data_source}) | Health: {health * 100:.1f}%"
    )

    try:
        active_plotter = launch_strut_plotter(
            sink_ms, comp_ft, k_eff, mass_kg, health, sev, ai_cls, conf, raw_arrays, active_plotter
        )
    except Exception as plot_err:
        print(f"  [3D] PyVista viewer skipped: {plot_err}")

    return active_plotter
