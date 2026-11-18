-- ═══════════════════════════════════════════════════════════════════════════
--  AEROTWIN DATA FLOW VERIFICATION
--  
--  Run these queries in pgAdmin to check that data is flowing correctly:
--  Tools → Query Tool → paste these → Execute (▶️)
--  
--  Expected flow:
--    FG Telemetry (UDP) → fg_connect.py → telemetry table
--    Telemetry table → aerotwin_inference_worker.py → flight_diagnostics table
--    Both tables → phi_fleet_server_v2.2.py → browser dashboard
-- ═══════════════════════════════════════════════════════════════════════════

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #1: All 6 tables exist
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    table_name,
    COUNT(*) as column_count
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name IN ('telemetry', 'flight_diagnostics', 'matlab_physics_output', 
                     'touchdown_events', 'gear_commands', 'visualization_state')
GROUP BY table_name
ORDER BY table_name;

-- Expected output:
-- ┌─────────────────────────┬──────────────┐
-- │ table_name              │ column_count │
-- ├─────────────────────────┼──────────────┤
-- │ flight_diagnostics      │         20   │
-- │ gear_commands           │          3   │
-- │ matlab_physics_output   │         21   │
-- │ telemetry               │         21   │
-- │ touchdown_events        │          8   │
-- │ visualization_state     │          9   │
-- └─────────────────────────┴──────────────┘

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #2: FG Telemetry is being received
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    COUNT(*) as total_telemetry_records,
    MAX(timestamp) as latest_telemetry_time,
    COUNT(CASE WHEN timestamp > NOW() - INTERVAL '10 seconds' THEN 1 END) as records_last_10sec,
    MAX(alt_agl_ft) as max_altitude_ft,
    MAX(mass_kg) as aircraft_mass_kg,
    MAX(groundspeed_kt) as max_groundspeed
FROM telemetry;

-- Expected output (assuming fg_connect is running):
-- ┌──────────────────────┬───────────────────────────┬──────────────────┬────────────────┬───────────────────┬────────────────┐
-- │ total_telemetry_     │ latest_telemetry_time     │ records_last_    │ max_altitude   │ aircraft_mass_    │ max_ground     │
-- │ records              │                           │ 10sec            │ _ft            │ kg                │ speed          │
-- ├──────────────────────┼───────────────────────────┼──────────────────┼────────────────┼───────────────────┼────────────────┤
-- │       1200           │ 2026-05-12 14:23:45.123+00│       20         │     1500       │      5200         │     45         │
-- └──────────────────────┴───────────────────────────┴──────────────────┴────────────────┴───────────────────┴────────────────┘

-- If records_last_10sec is 0 or 20:
--   • 0 → fg_connect is NOT running or FlightGear isn't sending data
--   • 20 → Data IS flowing (20 Hz = 20 records per second)

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #3: Inference worker is computing diagnostics
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    COUNT(*) as total_landings,
    MAX(ts) as latest_diagnosis_time,
    AVG(health_factor) as avg_health,
    AVG(confidence_score) as avg_confidence,
    MAX(severity_level) as max_severity_seen
FROM flight_diagnostics;

-- Expected output (after landing the aircraft at least once):
-- ┌────────────────┬──────────────────────────────┬──────────────┬──────────────────┬─────────────────────┐
-- │ total_landings │ latest_diagnosis_time        │ avg_health   │ avg_confidence   │ max_severity_seen   │
-- ├────────────────┼──────────────────────────────┼──────────────┼──────────────────┼─────────────────────┤
-- │       1        │ 2026-05-12 14:23:52.456+00   │    0.88      │      0.92        │        2            │
-- └────────────────┴──────────────────────────────┴──────────────┴──────────────────┴─────────────────────┘

-- If total_landings is 0:
--   • aerotwin_inference_worker is NOT running or no landings detected
--   • Make sure to land the aircraft in FlightGear (wheels touch ground)

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #4: Latest diagnosis detail
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    id,
    ts,
    ai_classification,
    confidence_score,
    health_factor,
    sink_rate_ms,
    compression_ft,
    max_accel_g,
    k_stiffness_val,
    b_damping_val,
    severity_level,
    ata_chapter,
    maintenance_cat,
    estimated_cost_usd,
    grounded,
    data_source,
    decision_source
FROM flight_diagnostics
ORDER BY ts DESC
LIMIT 1;

-- This is your "neural sense" output!
-- You'll see:
--   • Classification: Normal, N2 Leak, Worn Seal, etc.
--   • Health: 0.0 (failed) to 1.0 (new)
--   • Forces: K (stiffness), B (damping), accel
--   • Maintenance: ATA code, cost estimate, severity

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #5: Check data timestamps are fresh
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    'telemetry' as table_name,
    COUNT(*) as row_count,
    MAX(timestamp) as latest_data,
    EXTRACT(EPOCH FROM (NOW() - MAX(timestamp)))::INT as seconds_ago
FROM telemetry

UNION ALL

SELECT 
    'flight_diagnostics' as table_name,
    COUNT(*) as row_count,
    MAX(ts) as latest_data,
    EXTRACT(EPOCH FROM (NOW() - MAX(ts)))::INT as seconds_ago
FROM flight_diagnostics;

-- Expected output:
-- ┌──────────────────┬───────────┬──────────────────────────────┬──────────────┐
-- │ table_name       │ row_count │ latest_data                  │ seconds_ago  │
-- ├──────────────────┼───────────┼──────────────────────────────┼──────────────┤
-- │ telemetry        │    1200   │ 2026-05-12 14:23:45.123+00   │      2       │
-- │ flight_diag...   │       1   │ 2026-05-12 14:23:52.456+00   │     95       │
-- └──────────────────┴───────────┴──────────────────────────────┴──────────────┘

-- If seconds_ago > 60 for telemetry:
--   → fg_connect is stalled or FlightGear stopped sending
-- If seconds_ago is huge for flight_diagnostics:
--   → No recent landings detected

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #6: Check for WOW status (landing detection)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    wow_status,
    COUNT(*) as record_count,
    MAX(timestamp) as latest_with_this_status
FROM telemetry
WHERE timestamp > NOW() - INTERVAL '1 minute'
GROUP BY wow_status
ORDER BY wow_status DESC;

-- Expected while flying:
-- ┌────────────┬───────────────┬───────────────────────────────┐
-- │ wow_status │ record_count  │ latest_with_this_status       │
-- ├────────────┼───────────────┼───────────────────────────────┤
-- │ false      │     1100      │ 2026-05-12 14:23:40.000+00    │ (in air)
-- │ true       │        0      │ NULL                          │ (not on ground)
-- └────────────┴───────────────┴───────────────────────────────┘

-- Expected right after landing:
-- ┌────────────┬───────────────┬───────────────────────────────┐
-- │ wow_status │ record_count  │ latest_with_this_status       │
-- ├────────────┼───────────────┼───────────────────────────────┤
-- │ true       │     150       │ 2026-05-12 14:23:50.500+00    │ (on ground)
-- │ false      │      50       │ 2026-05-12 14:23:45.000+00    │ (was flying)
-- └────────────┴───────────────┴───────────────────────────────┘

-- The inference worker detects the transition: false → true (touchdown)
-- This is what triggers the diagnostic write

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  VERIFICATION #7: Detailed breakdown of landings
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SELECT 
    ROW_NUMBER() OVER (ORDER BY ts) as landing_number,
    ts,
    ai_classification,
    health_factor,
    sink_rate_ms,
    compression_ft,
    max_accel_g,
    severity_level,
    maintenance_cat,
    grounded
FROM flight_diagnostics
ORDER BY ts DESC;

-- This shows your landing history with all "neural sense" outputs:
-- Landing #1, #2, #3, etc.
-- Each row = one landing event with full diagnostics

-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
--  QUICK STATUS CHECK (all in one query)
-- ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

SELECT 'OVERALL STATUS' as check_name,
       CASE 
           WHEN (SELECT COUNT(*) FROM telemetry WHERE timestamp > NOW() - INTERVAL '10 seconds') > 0
           THEN '✅ FG Telemetry FLOWING'
           ELSE '❌ FG Telemetry STALLED'
       END as status,
       (SELECT COUNT(*) FROM telemetry) as telemetry_rows,
       (SELECT COUNT(*) FROM flight_diagnostics) as diagnosis_rows,
       CASE 
           WHEN (SELECT COUNT(*) FROM flight_diagnostics) > 0
           THEN '✅ Landing Inference ACTIVE'
           ELSE '⚠️  No landings detected (fly and land)'
       END as inference_status;

-- This gives you a quick health check in one view!

-- ═══════════════════════════════════════════════════════════════════════════
--  INTERPRETATION GUIDE
-- ═══════════════════════════════════════════════════════════════════════════

-- ✅ HEALTHY STATE:
--   • telemetry: 1000+ rows, latest < 2 seconds old
--   • flight_diagnostics: has rows (at least 1 landing)
--   • health_factor: 0.5 - 1.0 (smooth landings)
--   • severity_level: 1-2 (normal to caution)
--   • Browser dashboard: shows 3D gear + diagnostics

-- ⚠️  WARNING SIGNS:
--   • telemetry count stuck (not increasing) → fg_connect not running
--   • flight_diagnostics empty after landing → inference_worker not running
--   • health_factor < 0.3 → hard landing or aircraft damaged
--   • severity_level = 4 → aircraft grounded (needs immediate inspection)

-- ❌ ERROR STATES:
--   • Tables don't exist → SQL script didn't run
--   • Database connection fails → PostgreSQL down
--   • Dashboard says "DB Status: ◯" → no fresh data in last 10 seconds

-- ═══════════════════════════════════════════════════════════════════════════
--  RUNNING THESE QUERIES:
--
--  In pgAdmin:
--    1. Tools → Query Tool
--    2. Copy any query above
--    3. Click ▶️ Execute
--    4. Check "Data Output" tab for results
--
--  Repeat as you fly to track data flow in real-time!
-- ═══════════════════════════════════════════════════════════════════════════
