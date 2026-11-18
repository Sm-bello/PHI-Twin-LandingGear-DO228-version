"""
╔════════════════════════════════════════════════════════════════════════════╗
║  AEROTWIN AI INFERENCE WORKER v2.1                                         ║
║  Real FlightGear → Telemetry Table → AI Diagnostics                        ║
║                                                                            ║
║  Purpose:                                                                  ║
║    • Reads real FG telemetry from telemetry table (20 Hz)                  ║
║    • Detects landings (wow_status 0→1)                                     ║
║    • Computes AI diagnostics (health, classification, forces)              ║
║    • Writes to flight_diagnostics table (per landing)                      ║
║    • Streams "neural sense" outputs to dashboard                           ║
║                                                                            ║
║  Run:  python aerotwin_inference_worker.py                                 ║
║  Logs: Check console for landing detection + AI inference events           ║
╚════════════════════════════════════════════════════════════════════════════╝
"""

import os
import psycopg2
import time
import math
import numpy as np
from datetime import datetime
import json

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
DB_CONFIG = {
    'dbname': os.environ.get('AEROTWIN_DB_NAME', 'project_db'),
    'user': os.environ.get('AEROTWIN_DB_USER', 'postgres'),
    'password': os.environ.get('AEROTWIN_DB_PASSWORD', ''),
    'host': os.environ.get('AEROTWIN_DB_HOST', 'localhost'),
    'port': int(os.environ.get('AEROTWIN_DB_PORT', '5432'))
}

# AI Model paths (stub for now - replace with actual model loading if needed)
MODEL_LOADED = False  # Set to True if you load TensorFlow models

# Landing detection state
LAST_WOW_STATUS = False
LANDING_THRESHOLD_G = 0.3  # g-forces needed to trigger "landing"

# Fault class mappings (from your robot_aircraft.py)
CLASS_NAMES = [
    'Normal',
    'N2 Leak',
    'Worn Seal',
    'Early Degrad',
    'Thermal Degrad',
    'Brake Fade',
    'Tire Burst',
    'Corrosion',
    'Hard Landing',
    'Combined Fault',
    'Impending Failure'
]

# ATA/Maintenance mappings
MAINTENANCE_MAP = {
    0: {'chapter': 'ATA 32-10', 'cat': 'Routine Line Check', 'cost': 450, 'sev': 1},
    1: {'chapter': 'ATA 32-10', 'cat': 'Nitrogen System Service', 'cost': 1200, 'sev': 2},
    2: {'chapter': 'ATA 32-11', 'cat': 'Seal Replacement', 'cost': 2500, 'sev': 2},
    3: {'chapter': 'ATA 32-12', 'cat': 'Structural Inspection', 'cost': 5000, 'sev': 2},
    4: {'chapter': 'ATA 32-12', 'cat': 'Component Overhaul', 'cost': 8000, 'sev': 3},
    5: {'chapter': 'ATA 32-40', 'cat': 'Brake Service', 'cost': 3000, 'sev': 2},
    6: {'chapter': 'ATA 32-50', 'cat': 'Tire Replacement', 'cost': 800, 'sev': 2},
    7: {'chapter': 'ATA 32-10', 'cat': 'Corrosion Treatment', 'cost': 2000, 'sev': 2},
    8: {'chapter': 'ATA 32-12', 'cat': 'Damage Assessment', 'cost': 6000, 'sev': 3},
    9: {'chapter': 'ATA 32-20', 'cat': 'Major Overhaul', 'cost': 15000, 'sev': 4},
    10: {'chapter': 'ATA 32-30', 'cat': 'Immediate Replacement', 'cost': 25000, 'sev': 4},
}

# ─────────────────────────────────────────────────────────────────────────────
# TERMINAL COLORS
# ─────────────────────────────────────────────────────────────────────────────
import os
if os.name == 'nt':
    os.system('color')
    
RS = '\033[0m'
BD = '\033[1m'
GN = '\033[92m'
YL = '\033[93m'
RD = '\033[91m'
BL = '\033[94m'
CY = '\033[96m'

def log(msg, color=CY):
    """Print timestamped log message."""
    ts = datetime.now().strftime('%H:%M:%S.%f')[:-3]
    print(f'{color}[{ts}]{RS} {msg}')

# ─────────────────────────────────────────────────────────────────────────────
# AI INFERENCE FUNCTIONS (simplified for real FG data)
# ─────────────────────────────────────────────────────────────────────────────

def compute_health_factor(sink_rate_ms, compression_ft, max_accel_g, mass_kg):
    """
    Estimate health factor (0.0 = failed, 1.0 = new) based on landing metrics.
    Uses physics-based heuristics (no ML model needed for real-time).
    """
    # Baseline: 1.0 (new condition)
    health = 1.0
    
    # Penalty for sink rate (harder landings = more wear)
    if sink_rate_ms > 0.5:
        health -= (sink_rate_ms - 0.5) * 0.15  # -0.075 per 0.5 m/s
    
    # Penalty for compression (deeper compression = more stress)
    if compression_ft > 0.15:
        health -= (compression_ft - 0.15) * 2.0  # -2.0 per 0.1 ft
    
    # Penalty for acceleration (high-G landings = more damage)
    if max_accel_g > 1.5:
        health -= (max_accel_g - 1.5) * 0.2  # -0.2 per 1G
    
    # Scale penalty by aircraft mass (heavier = worse)
    mass_factor = min(1.0, mass_kg / 5500.0)  # DO-228 MTOW ~5500 kg
    health *= (0.8 + 0.2 * mass_factor)
    
    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, health))


def estimate_k_b_parameters(compression_ft, sink_rate_ms, mass_kg):
    """
    Estimate effective stiffness (K) and damping (B) from landing metrics.
    Based on Dornier 228-212 oleo strut characteristics.
    
    DO-228 typical values:
      K ~ 200,000-250,000 N/m (new)
      B ~ 5,000-10,000 N⋅s/m (new)
    """
    # Baseline (new condition)
    k_baseline = 220000  # N/m
    b_baseline = 7500    # N⋅s/m
    
    # Compression reduces stiffness (oleo gas pressure decreases)
    # Each 0.1 ft compression → ~5% K reduction
    k_reduction = (compression_ft / 0.1) * 0.05 if compression_ft > 0 else 0
    k_eff = k_baseline * (1.0 - k_reduction)
    
    # Damping decreases with age/seal wear
    # Estimate from sink rate: slower dissipation = lower B
    if sink_rate_ms > 0.1:
        b_reduction = (sink_rate_ms - 0.1) / 2.0  # linear reduction
    else:
        b_reduction = 0
    b_eff = b_baseline * (1.0 - b_reduction)
    
    return max(100000, k_eff), max(1000, b_eff)


def classify_fault(sink_rate_ms, compression_ft, max_accel_g, health):
    """
    Classify landing as one of 11 fault classes.
    Uses physics-informed heuristics based on landing signature.
    """
    # Default: normal
    fault_idx = 0
    confidence = 0.95
    
    # Hard landing detection
    if max_accel_g > 2.0:
        fault_idx = 8  # Hard Landing
        confidence = 0.92
    
    # High-sink-rate detection (control loss or approach error)
    elif sink_rate_ms > 1.5:
        fault_idx = 9  # Combined Fault
        confidence = 0.85
    
    # Extreme compression (bottoming out)
    elif compression_ft > 0.25:
        fault_idx = 10  # Impending Failure
        confidence = 0.88
    
    # Low health score indicates degradation
    elif health < 0.3:
        fault_idx = 4  # Thermal Degrad / major wear
        confidence = 0.80
    elif health < 0.5:
        fault_idx = 2  # Worn Seal
        confidence = 0.75
    elif health < 0.7:
        fault_idx = 1  # N2 Leak
        confidence = 0.70
    
    # Add small noise to confidence for realism
    confidence = max(0.5, min(1.0, confidence + np.random.normal(0, 0.03)))
    
    return fault_idx, confidence


def check_cs23_473(sink_rate_ms, compression_ft, max_accel_g, mass_kg):
    """
    CS-23.473: Check landing gear loads for forward CG scenario.
    Typical CS-23.473 limits:
      - Max vertical load: 2.67g (or structural limit)
      - Max forward load: 0.3g
    """
    # Simplified check
    if max_accel_g > 2.5:
        return True  # Exceeds typical CS-23.473 limit
    return False


def get_maintenance_action(fault_idx, health):
    """
    Determine maintenance action based on fault class and health.
    Returns ATA chapter, maintenance category, cost estimate, and severity.
    """
    # Map fault to maintenance
    maint = MAINTENANCE_MAP.get(fault_idx, MAINTENANCE_MAP[0])
    
    # Adjust cost/severity based on health
    if health < 0.2:
        # Critical: double cost, force replacement
        maint['cost'] *= 1.5
        maint['sev'] = 4
        maint['grounded'] = True
    elif health < 0.4:
        maint['sev'] = 3
        maint['grounded'] = False
    elif health < 0.6:
        maint['sev'] = 2
        maint['grounded'] = False
    else:
        maint['sev'] = 1
        maint['grounded'] = False
    
    return maint


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def get_latest_telemetry():
    """Fetch latest telemetry from real FG stream."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute("""
            SELECT sink_rate_fps, compression_ft, wow_status,
                   groundspeed_kt, alt_agl_ft, mass_lbs,
                   engine_torque, prop_rpm, flight_phase, timestamp,
                   airspeed_kt, vsi_fpm, mass_kg
            FROM telemetry
            ORDER BY timestamp DESC LIMIT 1
        """)
        row = cur.fetchone()
        cur.close()
        conn.close()
        return row
    except Exception as e:
        log(f"Telemetry fetch error: {e}", RD)
        return None


def write_diagnosis(sink_rate_ms, compression_ft, max_accel_g, mass_kg, 
                   ai_class, confidence, health, k_eff, b_eff, cs23_triggered, maint):
    """Write AI diagnosis to flight_diagnostics table."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        
        cur.execute("""
            INSERT INTO flight_diagnostics (
                ai_classification, confidence_score, health_factor,
                sink_rate_ms, compression_ft, max_accel_g,
                k_stiffness_val, b_damping_val, cs23_triggered,
                severity_level, ata_chapter, maintenance_cat,
                estimated_cost_usd, grounded,
                data_source, decision_source
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """, (
            ai_class, confidence, health,
            sink_rate_ms, compression_ft, max_accel_g,
            k_eff, b_eff, cs23_triggered,
            maint['sev'], maint['chapter'], maint['cat'],
            maint['cost'], maint.get('grounded', False),
            'real_fg_inference', 'physics_heuristic'
        ))
        
        cur.execute("SELECT lastval()")
        diagnosis_id = cur.fetchone()[0]
        
        conn.commit()
        cur.close()
        conn.close()
        
        return diagnosis_id
    except Exception as e:
        log(f"Diagnosis write error: {e}", RD)
        return None


def detect_touchdown():
    """
    Detect landing (WOW status 0→1).
    Returns telemetry data if landing detected, else None.
    """
    global LAST_WOW_STATUS
    
    tel = get_latest_telemetry()
    if not tel:
        return None
    
    sink_rate_fps, compression_ft, wow_status = tel[0], tel[1], tel[2]
    
    # Landing: WOW transitions from False to True
    if wow_status and not LAST_WOW_STATUS:
        LAST_WOW_STATUS = True
        return tel
    
    LAST_WOW_STATUS = wow_status
    return None


# ─────────────────────────────────────────────────────────────────────────────
# MAIN INFERENCE LOOP
# ─────────────────────────────────────────────────────────────────────────────

def main():
    """Main inference worker loop."""
    log("╔═══════════════════════════════════════════════════════════════╗", CY)
    log("║  AEROTWIN AI INFERENCE WORKER v2.1                            ║", CY)
    log("║  Reading Real FG Telemetry → Computing Diagnostics            ║", CY)
    log("╚═══════════════════════════════════════════════════════════════╝", CY)
    log("Waiting for FG landings...", BL)
    
    landing_count = 0
    
    try:
        while True:
            # Check for landing
            tel = detect_touchdown()
            
            if tel:
                landing_count += 1
                log(f"\n{'='*70}", YL)
                log(f"🛬 LANDING #{landing_count} DETECTED!", GN)
                log(f"{'='*70}", YL)
                
                # Extract telemetry
                sink_rate_fps = float(tel[0] or 0)
                compression_ft = float(tel[1] or 0)
                groundspeed_kt = float(tel[3] or 0)
                mass_lbs = float(tel[5] or 5500 * 2.2)  # default to DO-228 MTOW
                vsi_fpm = float(tel[11] or 0)
                mass_kg = float(tel[12] or mass_lbs / 2.2)
                
                # Convert units
                sink_rate_ms = sink_rate_fps * 0.3048  # ft/s to m/s
                max_accel_g = abs(vsi_fpm) / 300.0  # rough estimate of G-forces
                
                log(f"  Sink Rate: {sink_rate_ms:.3f} m/s ({sink_rate_fps:.2f} ft/s)", CY)
                log(f"  Compression: {compression_ft:.4f} ft", CY)
                log(f"  Accel (est): {max_accel_g:.3f} G", CY)
                log(f"  Mass: {mass_kg:.0f} kg ({mass_lbs:.0f} lbs)", CY)
                
                # AI INFERENCE
                log(f"\n  {BL}Computing diagnostics...{RS}", BL)
                
                # 1. Health factor
                health = compute_health_factor(sink_rate_ms, compression_ft, max_accel_g, mass_kg)
                log(f"  ✓ Health Factor: {health:.3f} ({health*100:.1f}%)", 
                    GN if health > 0.7 else YL if health > 0.4 else RD)
                
                # 2. K/B parameters
                k_eff, b_eff = estimate_k_b_parameters(compression_ft, sink_rate_ms, mass_kg)
                log(f"  ✓ Stiffness (K): {k_eff:.0f} N/m", CY)
                log(f"  ✓ Damping (B): {b_eff:.0f} N⋅s/m", CY)
                
                # 3. Fault classification
                fault_idx, confidence = classify_fault(sink_rate_ms, compression_ft, max_accel_g, health)
                ai_class = CLASS_NAMES[fault_idx]
                log(f"  ✓ Classification: {ai_class} (confidence: {confidence*100:.1f}%)", BL)
                
                # 4. CS-23.473 check
                cs23 = check_cs23_473(sink_rate_ms, compression_ft, max_accel_g, mass_kg)
                log(f"  ✓ CS-23.473: {'⚠️  EXCEEDED' if cs23 else '✓ OK'}", YL if cs23 else GN)
                
                # 5. Maintenance action
                maint = get_maintenance_action(fault_idx, health)
                log(f"  ✓ Maintenance: {maint['cat']} ({maint['chapter']})", CY)
                log(f"  ✓ Est. Cost: ${maint['cost']:.0f} | Severity: {maint['sev']}/4", 
                    YL if maint['sev'] >= 2 else GN)
                
                # WRITE TO DATABASE
                diag_id = write_diagnosis(
                    sink_rate_ms, compression_ft, max_accel_g, mass_kg,
                    ai_class, confidence, health, k_eff, b_eff, cs23, maint
                )
                
                if diag_id:
                    log(f"\n  ✅ Diagnosis written to DB (ID: {diag_id})", GN)
                else:
                    log(f"\n  ❌ Failed to write diagnosis", RD)
                
                log(f"{'='*70}\n", YL)
            
            # Poll interval (don't hammer DB)
            time.sleep(0.5)
    
    except KeyboardInterrupt:
        log("\n\nShutdown requested. Goodbye!", GN)
    except Exception as e:
        log(f"Fatal error: {e}", RD)


if __name__ == "__main__":
    main()
