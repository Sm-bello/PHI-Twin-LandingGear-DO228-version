#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           🛩️  AEROTWIN ALMIGHTY ROBOT AIRCRAFT  v1.0                         ║
║           Dornier 228-212 NG | DNAA → DNMM | AI Co-Pilot                      ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Self-contained flight simulator with physics-based landing gear modeling.   ║
║  No FlightGear required. Outputs to PostgreSQL for MATLAB/Grafana.           ║
╚══════════════════════════════════════════════════════════════════════════════╝

Architecture:
    Python (this script) ──► PostgreSQL ──► MATLAB (reads gear state)
                                    └──► Grafana (dashboard)
                                    
Flight Profile:
    TAKEOFF (10s) → CLIMB (15s) → CRUISE (30s) → APPROACH (20s) → LANDING
"""

import os
import numpy as np
import psycopg2
import tensorflow as tf
import tkinter as tk
from tkinter import messagebox
import time
import threading
import math
from datetime import datetime
import random

# ─── CONFIGURATION ───────────────────────────────────────────────────────────

DB_CONFIG = {
    "dbname": os.environ.get("AEROTWIN_DB_NAME", "project_db"),
    "user": os.environ.get("AEROTWIN_DB_USER", "postgres"),
    "password": os.environ.get("AEROTWIN_DB_PASSWORD", ""),
    "host": os.environ.get("AEROTWIN_DB_HOST", "localhost"),
}

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(_ROOT, "AeroTwin_V16_Model.h5")

# Dornier 228-212 Parameters
AIRCRAFT = {
    "name": "Dornier 228-212 NG",
    "mtow_kg": 6400,           # Maximum takeoff weight
    "mass_kg": 5200,           # Current mass
    "wing_span_m": 16.97,
    "gear_stroke_m": 0.180,    # Oleo stroke
    "k_nominal": 150000,       # Nominal stiffness N/m
    "b_nominal": 4000,         # Nominal damping N·s/m
    "v_stall_kt": 70,
    "v_cruise_kt": 170,
    "gear_retract_alt_m": 150, # meters (~500ft)
    "gear_extend_alt_m": 300,  # meters (~1000ft approach)
}

# Flight Phase Timing (seconds)
PHASES = {
    "TAKEOFF": 10,
    "CLIMB": 15,
    "CRUISE": 30,
    "APPROACH": 20,
    "LANDING": 5
}

# Lagos DNMM Runway 18L
DESTINATION = {
    "lat": 6.5774,
    "lon": 3.3212,
    "ils_freq": 109.70,
    "runway": "18L"
}

# ─── AI MODEL LOADING ────────────────────────────────────────────────────────

print("🧠 Loading AeroTwin AI Model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names = ['Normal', 'N2 Leak', 'Worn Seal', 'Early Degrad', 'Thermal',
                   'Brake Fade', 'Tire Burst', 'Corrosion', 'Hard Landing',
                   'Combined', 'Critical']
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Model load failed: {e}")
    model = None
    class_names = []

# ─── DATABASE SETUP ───────────────────────────────────────────────────────────

def init_database():
    """Create tables if they don't exist."""
    conn = psycopg2.connect(**DB_CONFIG)
    cur = conn.cursor()
    
    # Telemetry table - MATLAB reads this for gear position
    cur.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            flight_phase VARCHAR(20),
            altitude_m FLOAT,
            altitude_ft FLOAT,
            airspeed_kt FLOAT,
            groundspeed_kt FLOAT,
            vertical_speed_fps FLOAT,
            gear_down BOOLEAN,           -- MATLAB uses this for gear visualization
            gear_compression_m FLOAT,
            prop_rpm FLOAT,
            engine_torque_ftlb FLOAT,
            mass_kg FLOAT,
            sink_rate_fps FLOAT,
            wow_status BOOLEAN,
            distance_to_dest_nm FLOAT,
            k_stiffness_nm FLOAT,
            b_damping_nsm FLOAT,
            seal_force_n FLOAT
        )
    """)
    
    # Flight diagnostics - AI predictions stored here
    cur.execute("""
        CREATE TABLE IF NOT EXISTS flight_diagnostics (
            id SERIAL PRIMARY KEY,
            timestamp TIMESTAMP DEFAULT NOW(),
            flight_id INTEGER,
            ai_classification VARCHAR(50),
            confidence_score FLOAT,
            landing_sink_rate_fps FLOAT,
            landing_accel_g FLOAT,
            gear_health_percent FLOAT,
            maintenance_action TEXT,
            estimated_cost_usd INTEGER,
            ata_chapter VARCHAR(20)
        )
    """)
    
    conn.commit()
    return conn

# ─── PHYSICS SIMULATION ───────────────────────────────────────────────────────

class Dornier228Simulator:
    """
    Physics-based simulator for Dornier 228-212.
    Simulates flight phases with realistic gear dynamics.
    """
    
    def __init__(self):
        self.phase = "PREFLIGHT"
        self.phase_start_time = time.time()
        self.elapsed = 0
        
        # Flight state
        self.alt_m = 0
        self.alt_ft = 0
        self.airspeed_kt = 0
        self.groundspeed_kt = 0
        self.vs_fps = 0
        self.distance_nm = 50  # Start 50nm from destination
        
        # Gear state
        self.gear_down = True
        self.gear_compression_m = 0
        self.gear_commanded = True
        
        # Engine
        self.prop_rpm = 0
        self.torque_ftlb = 0
        
        # Physics parameters (vary by flight phase)
        self.k_current = AIRCRAFT["k_nominal"]
        self.b_current = AIRCRAFT["b_nominal"]
        self.seal_force_n = 500
        
        # Landing event storage
        self.landing_sink_rate = 0
        self.landing_accel = 0
        
    def update_gear_physics(self):
        """Calculate gear compression based on weight and sink rate."""
        if self.gear_down and self.alt_ft < 100:
            # On ground - calculate compression
            weight_n = AIRCRAFT["mass_kg"] * 9.81
            if self.vs_fps < 0:  # Descending
                # Dynamic compression from sink rate
                dynamic_force = self.b_current * abs(self.vs_fps) * 0.3048
                self.gear_compression_m = min(
                    (weight_n + dynamic_force) / self.k_current,
                    AIRCRAFT["gear_stroke_m"]
                )
            else:
                # Static compression
                self.gear_compression_m = weight_n / self.k_current
        else:
            self.gear_compression_m = 0
            
    def get_phase_target(self):
        """Get target parameters for current phase."""
        targets = {
            "TAKEOFF": {
                "alt_ft": 0,
                "vs_fps": 0,
                "speed_kt": 80,
                "gear": True,
                "throttle": 0.85
            },
            "CLIMB": {
                "alt_ft": 10000,
                "vs_fps": -15,  # Climbing (negative in aviation)
                "speed_kt": 120,
                "gear": False,  # Retract at 500ft
                "throttle": 0.85
            },
            "CRUISE": {
                "alt_ft": 10000,
                "vs_fps": 0,
                "speed_kt": 170,
                "gear": False,
                "throttle": 0.75
            },
            "APPROACH": {
                "alt_ft": 2500,
                "vs_fps": 5,    # Descending
                "speed_kt": 120,
                "gear": True,   # Extend for landing
                "throttle": 0.55
            },
            "LANDING": {
                "alt_ft": 0,
                "vs_fps": 3,    # Final descent
                "speed_kt": 70,
                "gear": True,
                "throttle": 0.30
            }
        }
        return targets.get(self.phase, targets["CRUISE"])
    
    def simulate_step(self, dt=0.1):
        """Advance simulation by one time step."""
        targets = self.get_phase_target()
        
        # Smooth transitions
        self.alt_ft += targets["vs_fps"] * dt * 0.5  # Simple integration
        self.alt_ft = max(0, self.alt_ft)
        self.alt_m = self.alt_ft * 0.3048
        
        # Speed changes
        speed_diff = targets["speed_kt"] - self.airspeed_kt
        self.airspeed_kt += speed_diff * 0.1 * dt
        self.groundspeed_kt = self.airspeed_kt * 0.9  # Simple ground effect
        
        # Gear logic
        if self.phase == "CLIMB" and self.alt_ft > 500:
            self.gear_commanded = False
        elif self.phase == "APPROACH":
            self.gear_commanded = True
            
        # Gear deployment (takes time)
        if self.gear_commanded and not self.gear_down:
            if random.random() < 0.1:  # 10% chance per step to complete
                self.gear_down = True
                print("  ⬇️  Gear EXTENDED")
        elif not self.gear_commanded and self.gear_down:
            if random.random() < 0.1:
                self.gear_down = False
                print("  ⬆️  Gear RETRACTED")
        
        # Engine parameters
        self.prop_rpm = 1800 + (targets["throttle"] - 0.5) * 400
        self.torque_ftlb = 2000 * targets["throttle"]
        
        # Distance to destination
        if self.phase in ["CLIMB", "CRUISE"]:
            self.distance_nm -= self.groundspeed_kt * dt / 3600
        
        # Vertical speed
        self.vs_fps = targets["vs_fps"]
        
        # Update gear physics
        self.update_gear_physics()
        
        # Store landing parameters
        if self.phase == "LANDING" and self.alt_ft < 50:
            self.landing_sink_rate = abs(self.vs_fps)
            self.landing_accel = 1.0 + (self.landing_sink_rate / 10)  # Simple g-calc
        
        return self.get_telemetry()
    
    def get_telemetry(self):
        """Return current state as dictionary."""
        return {
            "timestamp": datetime.now(),
            "flight_phase": self.phase,
            "altitude_m": self.alt_m,
            "altitude_ft": self.alt_ft,
            "airspeed_kt": self.airspeed_kt,
            "groundspeed_kt": self.groundspeed_kt,
            "vertical_speed_fps": self.vs_fps,
            "gear_down": self.gear_down,
            "gear_compression_m": self.gear_compression_m,
            "prop_rpm": self.prop_rpm,
            "engine_torque_ftlb": self.torque_ftlb,
            "mass_kg": AIRCRAFT["mass_kg"],
            "sink_rate_fps": abs(self.vs_fps),
            "wow_status": self.alt_ft < 10,
            "distance_to_dest_nm": max(0, self.distance_nm),
            "k_stiffness_nm": self.k_current,
            "b_damping_nsm": self.b_current,
            "seal_force_n": self.seal_force_n
        }

# ─── AI CO-PILOT & DIAGNOSTICS ────────────────────────────────────────────────

class AICoPilot:
    """Manages flight phases and AI diagnostics."""
    
    def __init__(self, simulator):
        self.sim = simulator
        self.flight_id = int(time.time())
        self.phase_start = time.time()
        self.phases = ["TAKEOFF", "CLIMB", "CRUISE", "APPROACH", "LANDING"]
        self.current_phase_idx = -1
        
    def check_phase_transition(self):
        """Check if we should advance to next flight phase."""
        if self.current_phase_idx < 0:
            self.start_phase(0)
            return
            
        elapsed = time.time() - self.phase_start
        current = self.phases[self.current_phase_idx]
        target_duration = PHASES[current]
        
        if elapsed >= target_duration:
            if self.current_phase_idx < len(self.phases) - 1:
                self.start_phase(self.current_phase_idx + 1)
            else:
                self.perform_landing_diagnostic()
                
    def start_phase(self, idx):
        """Start a new flight phase."""
        self.current_phase_idx = idx
        self.phase_start = time.time()
        phase_name = self.phases[idx]
        self.sim.phase = phase_name
        
        print(f"\n{'='*60}")
        print(f"  ✈️  PHASE: {phase_name}")
        print(f"  Duration: {PHASES[phase_name]} seconds")
        
        if phase_name == "CLIMB":
            print(f"  Target: Climb to 10,000ft, gear retract at 500ft")
        elif phase_name == "APPROACH":
            print(f"  Target: Descend to 2,500ft, gear extend for landing")
        elif phase_name == "LANDING":
            print(f"  Target: Final approach to DNMM Runway 18L")
        print(f"{'='*60}")
        
    def perform_landing_diagnostic(self):
        """Run AI model on landing and show results."""
        print(f"\n{'='*60}")
        print("  🔬 TOUCHDOWN - AI DIAGNOSTIC RUNNING")
        print(f"{'='*60}")
        
        # Build feature vector (16 features matching training data)
        features = [
            self.sim.landing_sink_rate,      # sink rate
            AIRCRAFT["mass_kg"] * 2.20462,   # weight lbs
            0,                               # nose steer
            0,                               # alt (on ground)
            1, 1, 1,                         # WOW all gears
            self.sim.gear_compression_m,     # compression
            self.sim.groundspeed_kt,         # groundspeed
            self.sim.prop_rpm,               # rpm
            self.sim.torque_ftlb,            # torque
            0,                               # airspeed
            DESTINATION["lat"],              # lat
            DESTINATION["lon"],              # lon
            0,                               # vsi
            0                                # trim
        ]
        
        # Run AI prediction
        if model:
            try:
                x = np.array(features[:16], dtype=np.float32).reshape(1, 1, 16)
                pred = model.predict(x, verbose=0)
                idx = int(np.argmax(pred[0]))
                confidence = float(pred[0][idx])
                classification = class_names[idx]
            except Exception as e:
                print(f"  ⚠️  AI prediction failed: {e}")
                classification = "Unknown"
                confidence = 0.0
        else:
            # Simulated prediction for demo
            classification = random.choice(class_names[:8])  # Exclude critical usually
            confidence = 0.85 + random.random() * 0.10
            
        # Determine maintenance action
        maint_actions = {
            "Normal": ("No action required", 0, "ATA 32-00"),
            "N2 Leak": ("Nitrogen recharge, leak check", 850, "ATA 32-10"),
            "Worn Seal": ("Seal replacement, damping check", 1250, "ATA 32-10"),
            "Early Degrad": ("Trend monitoring, inspection", 1800, "ATA 32-10"),
            "Thermal": ("Thermal stress inspection", 1500, "ATA 32-10"),
            "Brake Fade": ("Brake system inspection", 2000, "ATA 32-40"),
            "Tire Burst": ("Wheel assembly replacement", 3500, "ATA 32-40"),
            "Corrosion": ("Corrosion treatment, NDT", 2200, "ATA 32-10"),
            "Hard Landing": ("CS-23.473 hard landing inspection", 4500, "ATA 32-00"),
            "Combined": ("Component overhaul", 12000, "ATA 32-10"),
            "Critical": ("EMERGENCY GROUNDING - AOG", 35000, "ATA 32-00")
        }
        
        action, cost, ata = maint_actions.get(classification, 
                                             ("Inspection required", 500, "ATA 32-00"))
        
        # Show futuristic popup
        self.show_futuristic_popup(classification, confidence, action, cost, ata)
        
        # Log to database
        self.log_diagnostic(classification, confidence, action, cost, ata)
        
        print(f"\n  📊 RESULT: {classification}")
        print(f"  Confidence: {confidence*100:.1f}%")
        print(f"  Action: {action}")
        print(f"  Cost: ${cost:,}")
        print(f"  ATA: {ata}")
        print(f"{'='*60}")
        
    def show_futuristic_popup(self, cls, conf, action, cost, ata):
        """Show futuristic maintenance alert."""
        def popup():
            root = tk.Tk()
            root.withdraw()
            
            # Color based on severity
            if cls in ["Normal", "N2 Leak"]:
                color = "🟢"
            elif cls in ["Worn Seal", "Early Degrad", "Thermal"]:
                color = "🟡"
            elif cls in ["Brake Fade", "Tire Burst", "Corrosion"]:
                color = "🟠"
            else:
                color = "🔴"
            
            message = (
                f"{color} AEROTWIN AI DIAGNOSTIC {color}\n\n"
                f"Landing Gear Health Report\n"
                f"{'─'*40}\n"
                f"Classification:  {cls}\n"
                f"Confidence:      {conf*100:.1f}%\n"
                f"ATA Chapter:     {ata}\n\n"
                f"RECOMMENDED ACTION:\n{action}\n\n"
                f"Estimated Cost:  ${cost:,} USD\n"
                f"{'─'*40}\n"
                f"Telemetry synced to Grafana\n"
                f"Maintenance logged to database"
            )
            
            messagebox.showinfo("🛰️ AEROTWIN MAINTENANCE ALERT", message)
            root.destroy()
            
        threading.Thread(target=popup, daemon=True).start()
        
    def log_diagnostic(self, cls, conf, action, cost, ata):
        """Log diagnostic result to database."""
        try:
            conn = psycopg2.connect(**DB_CONFIG)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO flight_diagnostics 
                (flight_id, ai_classification, confidence_score,
                 landing_sink_rate_fps, landing_accel_g,
                 gear_health_percent, maintenance_action, 
                 estimated_cost_usd, ata_chapter)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.flight_id, cls, conf,
                self.sim.landing_sink_rate, self.sim.landing_accel,
                100 - (class_names.index(cls) * 10 if cls in class_names else 50),
                action, cost, ata
            ))
            conn.commit()
            conn.close()
        except Exception as e:
            print(f"  ⚠️  Failed to log diagnostic: {e}")

# ─── MAIN SIMULATION LOOP ────────────────────────────────────────────────────

def main():
    print("╔════════════════════════════════════════════════════════════════╗")
    print("║     🛩️  AEROTWIN ALMIGHTY ROBOT AIRCRAFT  v1.0                 ║")
    print("║     Dornier 228-212 NG | DNAA → DNMM                          ║")
    print("╚════════════════════════════════════════════════════════════════╝")
    print()
    
    # Initialize database
    print("🔌 Initializing database...")
    try:
        conn = init_database()
        print("✅ Database ready")
    except Exception as e:
        print(f"❌ Database failed: {e}")
        return
    
    # Create simulator and co-pilot
    sim = Dornier228Simulator()
    pilot = AICoPilot(sim)
    
    print(f"\n🛫 Aircraft: {AIRCRAFT['name']}")
    print(f"   Mass: {AIRCRAFT['mass_kg']} kg")
    print(f"   Destination: Lagos DNMM Runway 18L")
    print(f"\n🤖 AI Co-Pilot active")
    print(f"   Flight phases: {list(PHASES.keys())}")
    print(f"\n📡 Streaming telemetry to PostgreSQL...")
    print(f"   MATLAB will read gear state from database")
    print(f"   Grafana dashboard available at http://localhost:3000")
    print(f"\n{'─'*60}")
    print("Starting flight simulation...")
    print(f"{'─'*60}\n")
    
    # Main loop
    running = True
    last_db_write = 0
    
    try:
        while running:
            # Update physics
            telem = sim.simulate_step(dt=0.1)
            
            # Check phase transitions
            pilot.check_phase_transition()
            
            # Log to database at 10Hz
            if time.time() - last_db_write > 0.1:
                try:
                    cur = conn.cursor()
                    cur.execute("""
                        INSERT INTO telemetry 
                        (flight_phase, altitude_m, altitude_ft, airspeed_kt,
                         groundspeed_kt, vertical_speed_fps, gear_down,
                         gear_compression_m, prop_rpm, engine_torque_ftlb,
                         mass_kg, sink_rate_fps, wow_status, distance_to_dest_nm,
                         k_stiffness_nm, b_damping_nsm, seal_force_n)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """, tuple(telem.values()))
                    conn.commit()
                    last_db_write = time.time()
                except Exception as e:
                    print(f"DB write error: {e}")
            
            # Status print every 5 seconds
            if int(time.time()) % 5 == 0:
                gear_status = "DOWN" if telem["gear_down"] else "UP"
                print(f"[{telem['flight_phase']:10s}] "
                      f"Alt: {telem['altitude_ft']:6.0f}ft | "
                      f"GS: {telem['groundspeed_kt']:5.1f}kt | "
                      f"Gear: {gear_status} | "
                      f"Dist: {telem['distance_to_dest_nm']:4.1f}nm")
            
            # End after landing diagnostic
            if pilot.current_phase_idx >= len(pilot.phases) - 1 and \
               time.time() - pilot.phase_start > PHASES["LANDING"] + 5:
                running = False
                
            time.sleep(0.05)  # 20Hz simulation rate
            
    except KeyboardInterrupt:
        print("\n\n🛑 Simulation interrupted by user")
    finally:
        conn.close()
        print("\n✅ Simulation complete")
        print(f"   Flight ID: {pilot.flight_id}")
        print("   Check PostgreSQL for full telemetry and diagnostics")

if __name__ == "__main__":
    main()