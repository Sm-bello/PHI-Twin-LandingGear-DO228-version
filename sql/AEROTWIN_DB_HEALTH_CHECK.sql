-- ═══════════════════════════════════════════════════════════════════════════
--  AEROTWIN v2.1 — DATABASE HEALTH CHECK (READ-ONLY)
--  Safe to run anytime. Does NOT drop or modify any tables.
--
--  HOW TO USE IN pgAdmin:
--  1. Connect to project_db
--  2. Tools → Query Tool
--  3. Paste this entire script
--  4. Click Execute (▶)
--  5. Read the Data Output tabs / Messages from top to bottom
-- ═══════════════════════════════════════════════════════════════════════════

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 1: TABLE EXISTENCE
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 1 — TABLES PRESENT' AS section;
SELECT table_name
FROM information_schema.tables
WHERE table_schema = 'public'
  AND table_name IN (
      'telemetry',
      'gear_commands',
      'touchdown_events',
      'flight_diagnostics',
      'matlab_physics_output',
      'visualization_state'
  )
ORDER BY table_name;

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 2: TELEMETRY (FlightGear → fg_connect.py → PostgreSQL)
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 2 — TELEMETRY STREAM' AS section;
SELECT
    COUNT(*)                         AS total_rows,
    MAX(timestamp)                   AS latest_timestamp,
    MAX(sink_rate_ms)                AS max_sink_ms,
    MIN(sink_rate_ms)                AS min_sink_ms,
    MAX(mass_kg)                     AS max_mass_kg,
    SUM(CASE WHEN wow_status THEN 1 ELSE 0 END) AS wow_true_count
FROM telemetry;

SELECT 'SECTION 2b — LAST 5 TELEMETRY FRAMES' AS section;
SELECT
    timestamp,
    ROUND(sink_rate_ms::numeric, 3)  AS sink_ms,
    ROUND(mass_kg::numeric, 0)       AS mass_kg,
    wow_status,
    flight_phase,
    ROUND(alt_agl_ft::numeric, 0)    AS alt_ft,
    ROUND(groundspeed_kt::numeric, 1) AS gs_kt
FROM telemetry
ORDER BY timestamp DESC
LIMIT 5;

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 3: TOUCHDOWN EVENTS
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 3 — TOUCHDOWN EVENTS' AS section;
SELECT
    COUNT(*)                         AS total_touchdowns,
    MAX(ts)                          AS latest_touchdown,
    MAX(sink_rate_ms)                AS hardest_sink_ms,
    MAX(mass_kg)                     AS max_mass_kg
FROM touchdown_events;

SELECT 'SECTION 3b — LAST 5 TOUCHDOWNS' AS section;
SELECT
    ts,
    ROUND(sink_rate_ms::numeric, 3)  AS sink_ms,
    ROUND(mass_kg::numeric, 0)       AS mass_kg,
    ROUND(compression_ft::numeric, 3) AS comp_ft,
    flight_phase,
    matlab_seen
FROM touchdown_events
ORDER BY ts DESC
LIMIT 5;

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 4: MATLAB PHYSICS OUTPUT (Simscape → DB)
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 4 — MATLAB PHYSICS OUTPUT' AS section;
SELECT
    COUNT(*)                         AS total_sims,
    MAX(ts)                          AS latest_sim,
    MAX(sink_rate_ms)                AS max_sink_ms,
    MAX(sim_accel_max_g)             AS max_accel_g,
    MAX(sim_force_max)               AS max_force_n,
    SUM(CASE WHEN ready_for_ai THEN 1 ELSE 0 END) AS ready_for_ai_count
FROM matlab_physics_output;

SELECT 'SECTION 4b — LAST 5 PHYSICS ROWS' AS section;
SELECT
    ts,
    ROUND(sink_rate_ms::numeric, 3)      AS sink_ms,
    ROUND(mass_kg::numeric, 0)           AS mass_kg,
    ROUND(sim_deflection_max::numeric, 4) AS defl_m,
    ROUND(sim_force_max::numeric, 0)     AS force_n,
    ROUND(sim_accel_max_g::numeric, 3)   AS accel_g,
    ready_for_ai,
    ai_consumed
FROM matlab_physics_output
ORDER BY ts DESC
LIMIT 5;

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 5: AI FLIGHT DIAGNOSTICS (phi_fleet_server → DB)
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 5 — AI DIAGNOSTICS' AS section;
SELECT
    COUNT(*)                         AS total_diagnoses,
    MAX(ts)                          AS latest_diagnosis,
    COUNT(DISTINCT ai_classification) AS unique_classes
FROM flight_diagnostics;

SELECT 'SECTION 5b — LAST 5 DIAGNOSES' AS section;
SELECT
    ts,
    ai_classification,
    ROUND(confidence_score::numeric, 3) AS conf,
    physics_class,
    ROUND(sink_rate_ms::numeric, 3)     AS sink_ms,
    ROUND(max_accel_g::numeric, 3)      AS accel_g,
    cs23_triggered,
    decision_source,
    data_source
FROM flight_diagnostics
ORDER BY ts DESC
LIMIT 5;

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 6: GEAR COMMANDS
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 6 — GEAR COMMANDS' AS section;
SELECT
    COUNT(*) AS total_commands,
    MAX(ts)  AS latest_command
FROM gear_commands;

SELECT 'SECTION 6b — LAST 5 GEAR COMMANDS' AS section;
SELECT ts, command
FROM gear_commands
ORDER BY ts DESC
LIMIT 5;

-- ───────────────────────────────────────────────────────────────────────────
-- SECTION 7: PIPELINE HEALTH SUMMARY (one glance)
-- ───────────────────────────────────────────────────────────────────────────
SELECT 'SECTION 7 — PIPELINE HEALTH SUMMARY' AS section;
SELECT
    (SELECT COUNT(*) FROM telemetry)              AS telemetry_rows,
    (SELECT COUNT(*) FROM touchdown_events)       AS touchdown_rows,
    (SELECT COUNT(*) FROM matlab_physics_output)  AS physics_rows,
    (SELECT COUNT(*) FROM flight_diagnostics)     AS diagnosis_rows,
    (SELECT COUNT(*) FROM gear_commands)          AS gear_cmd_rows,
    (SELECT MAX(timestamp) FROM telemetry)        AS last_telemetry,
    (SELECT MAX(ts) FROM matlab_physics_output)   AS last_physics,
    (SELECT MAX(ts) FROM flight_diagnostics)      AS last_diagnosis;

-- ═══════════════════════════════════════════════════════════════════════════
--  HOW TO READ THE RESULTS
-- ═══════════════════════════════════════════════════════════════════════════
--  telemetry_rows > 0 and last_telemetry is recent
--      → FlightGear + fg_connect.py are writing
--  physics_rows increases after each landing
--      → launch_twin.m / Simscape is writing
--  diagnosis_rows increases after each landing
--      → phi_fleet_server AI is writing
--  If any section shows 0 after a landing, that link in the chain is broken.
-- ═══════════════════════════════════════════════════════════════════════════
