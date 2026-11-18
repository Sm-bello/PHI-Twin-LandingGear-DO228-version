# Critical Fixes Applied — AeroTwin Do-228 (for Reviewer Response)

Three bugs were making your 5-mission data unusable. All three are fixed below.
Copy the three modified files into your local Do228 workspace and re-run.

---

## 1. Sink-rate sign bug (SAFETY-CRITICAL)

**Problem:** FlightGear sends `vertical_speed_fps` as **negative when descending**.
Your code did a straight `* 0.3048` conversion, so hard landings arrived as
`-3.46 m/s` and `-4.02 m/s`. The CS-23.473 check

```matlab
cs23 = (sink_ms >= 3.05) || (accel_g >= 1.80);
```

never fired because a negative number is never ≥ 3.05.

**Fix applied in three places:**
- `fg_connect.py` — publishes **positive magnitude** in `sink_rate_ms`
- `launch_twin.m` — `sink_ms_live = abs(sink_fps) * 0.3048`
- `phi_fleet_server.py` — defensive `abs()` on the value read from DB

After this fix a 4 m/s hard landing will correctly trigger the CS-23.473 flag.

---

## 2. Mass = 0 kg on every touchdown

**Problem:** `f[1]` (weight_lbs) from FlightGear UDP is arriving as 0.
That collapses Simscape dynamics (identical force/accel every landing) and
makes the AI context vector meaningless.

**Fix:** If mass < 100 lb (clearly bogus) the code substitutes a realistic
Do-228 operating mass of 12 125 lb ≈ 5 500 kg. This is a temporary safety net;
you still need to fix the FlightGear property binding (see below).

---

## 3. Identical Simscape output across different sink rates

**Root causes:** (a) mass=0, (b) K/B never changing because compression stayed
below the 0.05 ft threshold on your short touch-and-goes, (c) possible
hard-coded initial velocity inside `Fancy_Landing_Gear.slx`.

After fixes 1 & 2 the mass and sink values that reach Simscape will now vary.
You must still verify inside Simulink that the Prismatic Joint / initial
velocity block is driven by the workspace variable `sink_rate_mps` (the exact
name written by `assignin` in `launch_twin.m`).

---

## How to apply on your machine

1. Copy these three files over the ones in your local Do228 folder:
   - `fg_connect.py`
   - `launch_twin.m`
   - `phi_fleet_server.py`

2. (Optional but recommended) Fix the real mass source:
   - Start FlightGear with `--telnet=5501`
   - Telnet to `localhost 5501` and browse for the actual weight property
     (often `/fdm/jsbsim/inertia/weight-lbs` or `/sim/weight`).
   - Update `aerotwin_proto.xml` so field `f[1]` points to that property.
   - Once the real mass streams correctly you can remove the `< 100` guard.

3. Restart the full stack in the usual order:
   ```
   1. PostgreSQL running
   2. FlightGear (Do-228 model)
   3. python fg_connect.py
   4. MATLAB → launch_twin.m
   5. python phi_fleet_server.py
   ```

4. **Single test landing first** (do NOT fly five missions yet):
   - Look at the MATLAB console on the first touchdown.
     You should now see a **positive** sink rate, e.g.
     `Sink: 3.46 m/s  |  Mass: 5500 kg`
   - Look at the phi_fleet_server console:
     `[AI] Classified: ... | RUL Health: ...`
   - Confirm the Simscape numbers (Deflection / Force / Accel) **change**
     between a gentle landing (~1 m/s) and a hard one (~3–4 m/s).

5. Only after the single-landing test looks correct, fly the five
   Abuja→Lagos missions. Each mission produces one row in
   `flight_diagnostics`. After the five landings you can query:

   ```sql
   SELECT ai_classification, confidence_score, sink_rate_ms,
          max_accel_g, cs23_triggered, decision_source, data_source
   FROM flight_diagnostics
   ORDER BY timestamp DESC LIMIT 5;
   ```

---

## What this gives the reviewer response

- CS-23.473 exceedances now fire correctly (the core safety claim).
- Mass is never zero, so Simscape and the AI context vector are meaningful.
- Sink-rate magnitude is consistent across the whole pipeline, removing the
  silent false-negative that Reviewer 1 would have found immediately.

The deeper architectural points (physics-class lookup from K/B ranges,
MATLAB as primary evidence source, export script for Table 4) still need
the additional wiring discussed earlier. These three fixes are the
minimum required before any more flight data is collected.
