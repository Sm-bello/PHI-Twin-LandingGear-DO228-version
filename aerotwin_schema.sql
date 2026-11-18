-- ═══════════════════════════════════════════════════════════════════════════
--  AEROTWIN v2.1.1 — pgAdmin-SAFE DATABASE SCHEMA (UPDATED)
--  Pure SQL, no transactions, statement-by-statement
--  
--  HOW TO USE IN pgADMIN:
--  1. Open pgAdmin → Connect to project_db
--  2. Click Tools → Query Tool
--  3. Copy-paste THIS ENTIRE SCRIPT into the Query editor
--  4. Click ▶️ "Execute Query"
--  5. Check "Messages" tab for ✓ success
--  
--  Safe to run multiple times (idempotent)
-- ═══════════════════════════════════════════════════════════════════════════

-- TABLE 1: TELEMETRY (from FG via fg_connect.py, 20Hz)
DROP TABLE IF EXISTS telemetry CASCADE;
CREATE TABLE telemetry (
    id                  SERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ DEFAULT NOW(),
    sink_rate_fps       FLOAT,
    sink_rate_ms        FLOAT,
    compression_ft      FLOAT,
    wow_status          BOOLEAN,
    wow_nose            BOOLEAN,
    wow_left            BOOLEAN,
    wow_right           BOOLEAN,
    groundspeed_kt      FLOAT,
    alt_agl_ft          FLOAT,
    mass_lbs            FLOAT,
    mass_kg             FLOAT,
    engine_torque       FLOAT,
    prop_rpm            FLOAT,
    airspeed_kt         FLOAT,
    latitude_deg        FLOAT,
    longitude_deg       FLOAT,
    vsi_fpm             FLOAT,
    elevator_trim       FLOAT,
    nose_steer_norm     FLOAT,
    flight_phase        VARCHAR(20)
);
CREATE INDEX idx_telemetry_ts ON telemetry (timestamp DESC);
CREATE INDEX idx_telemetry_wow ON telemetry (wow_status);

-- TABLE 2: GEAR_COMMANDS (autonomous UP/DOWN, 2Hz)
DROP TABLE IF EXISTS gear_commands CASCADE;
CREATE TABLE gear_commands (
    id              SERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ DEFAULT NOW(),
    command         VARCHAR(10)
);
CREATE INDEX idx_gear_commands_ts ON gear_commands (ts DESC);

-- TABLE 3: TOUCHDOWN_EVENTS (landing markers)
DROP TABLE IF EXISTS touchdown_events CASCADE;
CREATE TABLE touchdown_events (
    id              SERIAL PRIMARY KEY,
    ts              TIMESTAMPTZ DEFAULT NOW(),
    sink_rate_ms    FLOAT,
    compression_ft  FLOAT,
    mass_kg         FLOAT,
    groundspeed_kt  FLOAT,
    flight_phase    VARCHAR(20),
    matlab_seen     BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_touchdown_ts ON touchdown_events (ts DESC);

-- TABLE 4: FLIGHT_DIAGNOSTICS (AI health + classification)
DROP TABLE IF EXISTS flight_diagnostics CASCADE;
CREATE TABLE flight_diagnostics (
    id                      SERIAL PRIMARY KEY,
    ts                      TIMESTAMPTZ DEFAULT NOW(),
    ai_classification       VARCHAR(50),
    confidence_score        FLOAT,
    physics_class           VARCHAR(50),
    k_stiffness_val         FLOAT,
    b_damping_val           FLOAT,
    sink_rate_ms            FLOAT,
    max_accel_g             FLOAT,
    compression_ft          FLOAT,
    health_factor           FLOAT,
    estimated_rul_landings  INT,
    cs23_triggered          BOOLEAN,
    severity_level          INT,
    ata_chapter             VARCHAR(10),
    maintenance_cat         VARCHAR(50),
    estimated_cost_usd      FLOAT,
    grounded                BOOLEAN,
    data_source             VARCHAR(50),
    decision_source         VARCHAR(50),
    flight_phase            VARCHAR(20)   -- FIXED: Added to silence server errors
);
CREATE INDEX idx_flight_diag_ts ON flight_diagnostics (ts DESC);
CREATE INDEX idx_flight_diag_severity ON flight_diagnostics (severity_level DESC);

-- TABLE 5: MATLAB_PHYSICS_OUTPUT (Simscape Prismatic Joint)
DROP TABLE IF EXISTS matlab_physics_output CASCADE;
CREATE TABLE matlab_physics_output (
    id                      SERIAL PRIMARY KEY,
    ts                      TIMESTAMPTZ DEFAULT NOW(),
    sim_deflection_max      FLOAT,
    sim_deflection_raw      FLOAT[],
    sim_velocity_max        FLOAT,
    sim_velocity_raw        FLOAT[],
    sim_force_max           FLOAT,
    sim_force_raw           FLOAT[],
    sim_accel_max_g         FLOAT,
    sim_accel_raw           FLOAT[],
    sim_pressure_max        FLOAT,
    sim_pressure_raw        FLOAT[],
    k_eff                   FLOAT,
    b_eff                   FLOAT,
    mass_kg                 FLOAT,
    sink_rate_ms            FLOAT,
    compression_ft          FLOAT,
    wow_status              BOOLEAN,
    flight_phase            VARCHAR(20),
    ready_for_ai            BOOLEAN DEFAULT FALSE,
    ai_consumed             BOOLEAN DEFAULT FALSE
);
CREATE INDEX idx_matlab_physics_ts ON matlab_physics_output (ts DESC);

-- TABLE 6: VISUALIZATION_STATE (3D mesh snapshots)
DROP TABLE IF EXISTS visualization_state CASCADE;
CREATE TABLE visualization_state (
    id                      SERIAL PRIMARY KEY,
    ts                      TIMESTAMPTZ DEFAULT NOW(),
    gear_mesh_json          TEXT,
    force_vectors_json      TEXT,
    deflection_ft           FLOAT,
    max_force_n             FLOAT,
    color_severity          VARCHAR(20),
    phase                   VARCHAR(20),
    ai_class                VARCHAR(50)
);
CREATE INDEX idx_viz_ts ON visualization_state (ts DESC);

-- ═══════════════════════════════════════════════════════════════════════════
--  VERIFICATION QUERIES (run these after to confirm)
-- ═══════════════════════════════════════════════════════════════════════════

-- Check 1: Count all tables
SELECT 
    table_name,
    COUNT(*) as column_count
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('telemetry', 'flight_diagnostics', 'matlab_physics_output', 
                     'touchdown_events', 'gear_commands', 'visualization_state')
GROUP BY table_name
ORDER BY table_name;

-- Check 2: Verify flight_diagnostics has all AI columns (including flight_phase)
SELECT column_name, data_type
FROM information_schema.columns
WHERE table_name = 'flight_diagnostics'
ORDER BY ordinal_position;

-- Check 3: Current data in telemetry (FG stream)
SELECT COUNT(*) as telemetry_rows, 
       MAX(timestamp) as latest_timestamp
FROM telemetry;

-- Check 4: Current data in flight_diagnostics (AI diagnostics)
SELECT COUNT(*) as diagnostics_rows,
       MAX(ts) as latest_diagnosis
FROM flight_diagnostics;

-- ═══════════════════════════════════════════════════════════════════════════
--  DONE! All 6 tables created and ready for data ingestion.
-- ═══════════════════════════════════════════════════════════════════════════