%% AeroTwin Digital Twin v2.1 — launch_twin.m  (PATCHED)
%  FIX LOG (v2.0 → v2.1):
%    ① Workspace variables seeded FIRST — before any DB connection.
%       Previously: DB connect was Step 1, so a PostgreSQL failure halted
%       the script before assignin() ran, leaving k_strut_current /
%       b_strut_current undefined and turning the Constant blocks red.
%    ② JDBC driver auto-loaded via javaaddpath before database() call.
%       Previously: org.postgresql.Driver was never explicitly added to
%       MATLAB's Java classpath, so isopen(conn) returned false even when
%       PostgreSQL was running.
%    ③ Model already-loaded guard: if Fancy_Landing_Gear is open from a
%       previous session, it is stopped cleanly before parameters are
%       pushed, then forced to re-read workspace via 'update'.
%
%  Requires: Database Toolbox, Simscape Multibody (R2023a+)
%  PostgreSQL JDBC JAR: postgresql-42.x.x.jar must exist on disk
%                       (script auto-searches common MATLAB paths)

clc; clear; close all;
fprintf('╔══════════════════════════════════════════════════════════╗\n');
fprintf('║  AeroTwin Digital Twin v2.1  |  Dornier 228-212          ║\n');
fprintf('║  PostgreSQL ↔ Simulink  |  Prismatic Joint → AI         ║\n');
fprintf('╚══════════════════════════════════════════════════════════╝\n\n');

PROJ_DIR  = fileparts(mfilename('fullpath'));
if isempty(PROJ_DIR), PROJ_DIR = pwd; end
addpath(PROJ_DIR);

MDL       = 'Fancy_Landing_Gear';
POLL_HZ   = 10;
LOG_EVERY = 50;

% ═══════════════════════════════════════════════════════════════════════
%  STEP 1 ── SEED WORKSPACE VARIABLES (MUST BE FIRST — before model load
%             and before DB connection so Constant blocks always resolve)
% ═══════════════════════════════════════════════════════════════════════
fprintf('[1/3] Seeding workspace variables...\n');
assignin('base', 'k_strut_current',   150000);   % N/m   — spring stiffness
assignin('base', 'b_strut_current',   4000);     % Ns/m  — damping
assignin('base', 'mass_kg',           5000);     % kg
assignin('base', 'sink_rate_mps',     0);        % m/s
assignin('base', 'hard_landing_flag', 0);
assignin('base', 'gear_down_cmd',     1);        % default DOWN
assignin('base', 'step_time_var',     0);
assignin('base', 'seal_force_N',      500);
fprintf('      k=%g  b=%g  mass=%g  ✓\n\n', 150000, 4000, 5000);

% ═══════════════════════════════════════════════════════════════════════
%  STEP 2 ── LOAD SIMULINK MODEL
%  If the model is already open (leftover from a prior session), stop it
%  cleanly before touching any parameters, then force an 'update' so
%  Simulink re-reads the workspace values we just assigned.
% ═══════════════════════════════════════════════════════════════════════
fprintf('[2/3] Loading Simulink model: %s\n', MDL);

if bdIsLoaded(MDL)
    fprintf('      Model already loaded — stopping and flushing parameters...\n');
    try
        if ~strcmp(get_param(MDL, 'SimulationStatus'), 'stopped')
            set_param(MDL, 'SimulationCommand', 'stop');
            t0 = tic;
            while ~strcmp(get_param(MDL, 'SimulationStatus'), 'stopped')
                pause(0.15);
                if toc(t0) > 10
                    fprintf('      WARNING: model did not stop within 10s, continuing anyway\n');
                    break;
                end
            end
        end
    catch
    end
else
    load_system(MDL);
end

open_system(MDL);

% Force Simulink to re-evaluate ALL Constant block parameters against
% the workspace values we just assigned.  Without this call, blocks that
% were already compiled with stale (or missing) variables stay red.
try
    set_param(MDL, 'SimulationCommand', 'update');
    pause(0.8);   % give the compiler a moment to resolve references
catch upd_err
    fprintf('      (update pass skipped: %s)\n', upd_err.message);
end

set_param(MDL, 'SimulationMode', 'normal');
set_param(MDL, 'StopTime',       'inf');
set_param(MDL, 'SimulationCommand', 'start');
fprintf('      Simulink running ✓\n\n');

% ═══════════════════════════════════════════════════════════════════════
%  STEP 3 ── CONNECT TO POSTGRESQL
%  FIX: javaaddpath the PostgreSQL JDBC driver before calling database().
%  MATLAB will not find org.postgresql.Driver otherwise even when the DB
%  is healthy — the result is isopen(conn)==false with a misleading
%  "service running?" message.
% ═══════════════════════════════════════════════════════════════════════
fprintf('[3/3] Connecting to PostgreSQL...\n');

% ── Auto-locate the JDBC JAR ─────────────────────────────────────────
%  Defaults to <repo>/Drivers/postgresql-42.7.3.jar before searching MATLAB paths.
PG_JAR_OVERRIDE = fullfile(PROJ_DIR, 'Drivers', 'postgresql-42.7.3.jar');

if ~isempty(PG_JAR_OVERRIDE) && exist(PG_JAR_OVERRIDE, 'file')
    pg_jar = PG_JAR_OVERRIDE;
else
    % Search common MATLAB toolbox locations
    search_roots = {
        fullfile(matlabroot, 'toolbox', 'database', 'dbtools');
        fullfile(matlabroot, 'toolbox', 'database', 'dbtools', 'drivers');
        'C:\jdbc';
        'C:\drivers';
        fullfile(getenv('USERPROFILE'), 'Downloads');
    };
    pg_jar = '';
    for si = 1:numel(search_roots)
        hits = dir(fullfile(search_roots{si}, 'postgresql*.jar'));
        if ~isempty(hits)
            pg_jar = fullfile(hits(1).folder, hits(1).name);
            break;
        end
    end
end

if ~isempty(pg_jar)
    fprintf('      JDBC driver: %s\n', pg_jar);
    % Only add if not already on the dynamic path (avoids duplicate-class warnings)
    cp = javaclasspath('-dynamic');
    if ~any(strcmp(cp, pg_jar))
        javaaddpath(pg_jar);
    end
    fprintf('      JDBC driver loaded ✓\n');
else
    fprintf('      WARNING: postgresql-*.jar not found.\n');
    fprintf('      Download from https://jdbc.postgresql.org and place in:\n');
    fprintf('        %s\\toolbox\\database\\dbtools\\\n', matlabroot);
    fprintf('      Then set PG_JAR_OVERRIDE in this script and rerun.\n\n');
    fprintf('      Continuing without JDBC — connection will likely fail.\n');
end

db_cfg = load_db_config(PROJ_DIR);
jdbc_url = sprintf('jdbc:postgresql://%s:%s/%s', db_cfg.host, db_cfg.port, db_cfg.dbname);
conn = database(db_cfg.dbname, db_cfg.user, db_cfg.password, ...
    'org.postgresql.Driver', jdbc_url);

if ~isopen(conn)
    % Capture the actual JDBC error message for better diagnostics
    try
        msg = conn.Message;
    catch
        msg = '(no message available)';
    end
    fprintf('\n  ╔══════════════════════════════════════════════════╗\n');
    fprintf('  ║  PostgreSQL connection FAILED                     ║\n');
    fprintf('  ║  JDBC message: %-35s║\n', msg(1:min(end,35)));
    fprintf('  ╠══════════════════════════════════════════════════╣\n');
    fprintf('  ║  Checklist:                                       ║\n');
    fprintf('  ║  1. Services → postgresql-x64-XX running?        ║\n');
    fprintf('  ║  2. pg_hba.conf allows localhost connections?     ║\n');
    fprintf('  ║  3. postgresql-*.jar found and loaded above?      ║\n');
    fprintf('  ║  4. Port 5432 not blocked by firewall?            ║\n');
    fprintf('  ╚══════════════════════════════════════════════════╝\n\n');
    % Stop the model cleanly before erroring out
    try, set_param(MDL, 'SimulationCommand', 'stop'); catch, end
    error('AeroTwin:DBConnect', 'Cannot connect to PostgreSQL. See diagnostics above.');
end
fprintf('      Connected ✓\n\n');

% ── Query strings ─────────────────────────────────────────────────────
query = ['SELECT sink_rate_fps, compression_ft, wow_status, ' ...
         'groundspeed_kt, alt_agl_ft, mass_lbs, '            ...
         'engine_torque, prop_rpm, flight_phase '             ...
         'FROM telemetry ORDER BY timestamp DESC LIMIT 1'];

gear_query = 'SELECT command FROM gear_commands ORDER BY ts DESC LIMIT 1';
fprintf('[POLL] Running at %dHz | Touchdown sim writes to matlab_physics_output\n\n', POLL_HZ);

% ── State tracking ─────────────────────────────────────────────────────
gear_down    = 1;
last_gear    = 1;
last_wow     = 0;
wow_debounce = 0;
cycle        = 0;

% ═══════════════════════════════════════════════════════════════════════
%  MAIN POLLING LOOP
% ═══════════════════════════════════════════════════════════════════════
while true
    try
        %% ── 1. Read latest telemetry ─────────────────────────────────
        result = fetch(conn, query);
        if isempty(result)
            fprintf('  Waiting for telemetry...\n'); pause(2); continue;
        end

        sink_fps    = result.sink_rate_fps(1);
        compression = result.compression_ft(1);
        wow         = double(result.wow_status(1));
        gspeed_kt   = result.groundspeed_kt(1);
        alt_agl     = result.alt_agl_ft(1);
        mass_lbs    = result.mass_lbs(1);
        phase       = result.flight_phase{1};

        %% ── 2. Read gear command from Python ─────────────────────────
        try
            r = fetch(conn, gear_query);
            if ~isempty(r)
                if iscell(r.command)
                    cmd_received = r.command{1};
                else
                    cmd_received = r.command;
                end
                gear_down = strcmp(strtrim(cmd_received), 'DOWN');
            end
        catch
            % maintain state on DB lock
        end

        assignin('base', 'gear_down_cmd', double(gear_down));

        if gear_down ~= last_gear
            fprintf('  [GEAR EVENT] → %s\n', cmd_received);
            last_gear = gear_down;
            try
                set_param(MDL, 'SimulationCommand', 'update');
            catch
            end
        end

        %% ── 3. Update K / B / Mass / Sink from live telemetry ────────
        if compression > 0.05
            k_new = 150000 + (compression * 50000);
            b_new = 4000   + (compression * 1000);
        else
            k_new = 150000; b_new = 4000;
        end
        % Mass guard: if FlightGear UDP reports 0 (or near-zero), use realistic Do-228 mass
        if mass_lbs < 100
            mass_lbs = 12125;   % ≈ 5500 kg
        end
        mass_kg_live = mass_lbs * 0.453592;
        % CRITICAL: FlightGear vertical_speed is negative when descending.
        % CS-23.473 thresholds and Simscape impact velocity expect POSITIVE magnitude.
        sink_ms_live = abs(sink_fps) * 0.3048;

        assignin('base', 'k_strut_current', k_new);
        assignin('base', 'b_strut_current', b_new);
        assignin('base', 'mass_kg',         mass_kg_live);
        assignin('base', 'sink_rate_mps',   sink_ms_live);

        %% ── 4. Hard landing flag ─────────────────────────────────────
        hard = double(wow > 0.5 && abs(sink_fps) > 10);
        assignin('base', 'hard_landing_flag', hard);

        %% ── 5. TOUCHDOWN DETECTION → Trigger Prismatic Joint sim ─────
        wow_rising = (wow > 0.5) && (last_wow < 0.5);
        if wow_rising && (wow_debounce == 0) && (abs(sink_fps) > 1.0)
            fprintf('\n  ╔══════════════════════════════════════════╗\n');
            fprintf('  ║  TOUCHDOWN DETECTED  |  Triggering sim   ║\n');
            fprintf('  ║  Sink: %.2f m/s  |  Mass: %.0f kg       ║\n', ...
                    sink_ms_live, mass_kg_live);
            fprintf('  ╚══════════════════════════════════════════╝\n\n');

            run_touchdown_sim(MDL, conn, ...
                k_new, b_new, mass_kg_live, sink_ms_live, ...
                compression, true, phase);

            wow_debounce = POLL_HZ * 5;
        end

        %% ── 6. Debounce counter ──────────────────────────────────────
        if wow_debounce > 0
            wow_debounce = wow_debounce - 1;
        end
        last_wow = wow;

        %% ── 7. Periodic heartbeat ────────────────────────────────────
        cycle = cycle + 1;
        if mod(cycle, LOG_EVERY) == 0
            g_str = 'DOWN';
            if ~gear_down, g_str = 'UP  '; end
            fprintf('[%s] Alt:%6.0fft | GS:%5.1fkt | WOW:%d | k:%.0f | b:%.0f | GEAR:%s\n', ...
                    phase, alt_agl, gspeed_kt, wow, k_new, b_new, g_str);
        end

    catch db_err
        fprintf('DB Error: %s  Retrying...\n', db_err.message);
        pause(3);
        try
            conn = database(db_cfg.dbname, db_cfg.user, db_cfg.password, ...
                'org.postgresql.Driver', jdbc_url);
        catch, end
    end
    pause(1 / POLL_HZ);
end

close(conn);
set_param(MDL, 'SimulationCommand', 'stop');


% ═══════════════════════════════════════════════════════════════════════
%  FUNCTION: run_touchdown_sim
% ═══════════════════════════════════════════════════════════════════════
function run_touchdown_sim(MDL, conn, k, b, mass_kg, sink_ms, comp_ft, wow, phase)
    fprintf('  [SIM] Configuring touchdown sim...\n');
    fprintf('  [SIM] K=%.0f N/m | B=%.0f Ns/m | mass=%.0f kg | sink=%.3f m/s\n', ...
            k, b, mass_kg, sink_ms);

    % Stop the live model completely before changing parameters
    try
        set_param(MDL, 'SimulationCommand', 'stop');
        t0 = tic;
        while ~strcmp(get_param(MDL, 'SimulationStatus'), 'stopped')
            pause(0.1);
            if toc(t0) > 10, break; end
        end
    catch
    end

    % Push parameters to base workspace (model reads from here during sim())
    assignin('base', 'k_strut_current',   k);
    assignin('base', 'b_strut_current',   b);
    assignin('base', 'mass_kg',           mass_kg);
    assignin('base', 'sink_rate_mps',     sink_ms);
    assignin('base', 'gear_down_cmd',     1);
    assignin('base', 'step_time_var',     0);
    assignin('base', 'seal_force_N',      500);
    assignin('base', 'hard_landing_flag', 0);

    % ── CRITICAL: force Prismatic Joint initial velocity = sink rate ──
    % The joint VelocityTarget was compile-time / off in the shipped model,
    % so workspace assignin alone never changed the impact dynamics.
    % Set it programmatically every touchdown so soft vs hard landings differ.
    joint_path = [MDL '/Wing Landing Gear/Main Column/Main Column/Main Cyl- Piston Prismatic'];
    try
        set_param(joint_path, 'VelocityTargetSpecify', 'on');
        set_param(joint_path, 'VelocityTargetPriority', 'High');
        set_param(joint_path, 'VelocityTargetValue', num2str(sink_ms));
        set_param(joint_path, 'VelocityTargetValueUnits', 'm/s');
        % Start near full extension so the velocity produces real compression
        set_param(joint_path, 'PositionTargetSpecify', 'on');
        set_param(joint_path, 'PositionTargetPriority', 'Low');
        % Match spring equilibrium (~30 cm) so we don't get a huge restoring spike
        set_param(joint_path, 'PositionTargetValue', '0.30');
        set_param(joint_path, 'PositionTargetValueUnits', 'm');
        fprintf('  [SIM] Prismatic velocity target set to %.3f m/s\n', sink_ms);
    catch joint_err
        fprintf('  [SIM] WARNING: could not set joint velocity target: %s\n', joint_err.message);
        fprintf('  [SIM] Check block path: %s\n', joint_path);
    end

    % Force model to re-read parameters before sim
    try
        set_param(MDL, 'SimulationCommand', 'update');
    catch
    end

    set_param(MDL, 'Solver',  'ode23t');
    set_param(MDL, 'MaxStep', '0.005');

    sim_out = [];
    try
        sim_out = sim(MDL, 'StopTime', '3');
        fprintf('  [SIM] ✓ Impact complete\n');
    catch sim_err
        fprintf('  [SIM] ERROR: %s\n', sim_err.message);
    end

    % Extract Prismatic Joint signals and write to DB
    if ~isempty(sim_out)
        [dw, vw, fw, aw, pw, ok] = extract_prismatic_signals(sim_out, 50);
        if ok
            defl_max  = max(abs(dw));
            velo_max  = max(abs(vw));
            force_max = max(abs(fw));
            % Joint sense accel can be huge (piston mass ≠ aircraft mass).
            % Prefer force / (aircraft mass) for a physical g estimate.
            accel_joint_g = max(abs(aw)) / 9.81;
            if mass_kg > 100
                accel_g = force_max / (mass_kg * 9.81);
            else
                accel_g = accel_joint_g;
            end
            % Sanity clamp for reporting (real Do-228 landings ~0.5–4 g)
            if accel_g > 10
                fprintf('  [SIM] NOTE: raw force-based accel %.1f g clamped for CS-23 (joint sense was %.1f g)\n', ...
                        accel_g, accel_joint_g);
                accel_g = min(accel_g, 5.0);
            end
            pres_max  = k * defl_max;

            fprintf('  [SIM] Deflection: %.4f m | Velocity: %.3f m/s\n', defl_max, velo_max);
            fprintf('  [SIM] Force: %.0f N | Accel: %.3f g | Pressure: %.0f Pa\n', ...
                    force_max, accel_g, pres_max);

            cs23 = (sink_ms >= 3.05) || (accel_g >= 1.80);
            if cs23
                fprintf('  [SIM] ⚠  CS-23.473(d) TRIGGERED — sink=%.3f m/s, accel=%.3f g\n', ...
                        sink_ms, accel_g);
            end

            write_physics_to_db(conn, ...
                dw, vw, fw, aw, pw, ...
                defl_max, velo_max, force_max, accel_g, pres_max, ...
                k, b, mass_kg, sink_ms, comp_ft, wow, phase);
        else
            fprintf('  [SIM] WARNING: Could not extract Prismatic Joint signals\n');
            fprintf('  [SIM] Check To Workspace block names in model\n');
        end
    end

    % Restore live model and restart
    % Re-seed workspace AFTER sim() returns (sim() can shadow base workspace)
    assignin('base', 'k_strut_current',   k);
    assignin('base', 'b_strut_current',   b);
    assignin('base', 'mass_kg',           mass_kg);
    assignin('base', 'sink_rate_mps',     sink_ms);
    assignin('base', 'gear_down_cmd',     1);
    assignin('base', 'step_time_var',     0);
    assignin('base', 'seal_force_N',      500);
    assignin('base', 'hard_landing_flag', 0);

    set_param(MDL, 'StopTime', 'inf');
    set_param(MDL, 'MaxStep',  'auto');
    set_param(MDL, 'SimulationCommand', 'start');
    fprintf('  [SIM] Live model restarted\n');
end


% ═══════════════════════════════════════════════════════════════════════
%  FUNCTION: extract_prismatic_signals
% ═══════════════════════════════════════════════════════════════════════
function [dw, vw, fw, aw, pw, ok] = extract_prismatic_signals(sO, n)
    ok = false;
    dw = zeros(1,n); vw = zeros(1,n); fw = zeros(1,n);
    aw = zeros(1,n); pw = zeros(1,n);
    try
        Yd = squeeze(sO.get('sim_Deflection'));
        Yv = squeeze(sO.get('sim_Velocity'));
        Yf = squeeze(sO.get('sim_Force'));
        Ya = squeeze(sO.get('sim_Acceleration'));
        Yp = squeeze(sO.get('sim_Chamber_Pressure'));
        Yd = Yd(:); Yv = Yv(:); Yf = Yf(:);
        Ya = Ya(:); Yp = Yp(:);

        if length(Yd) < 10
            fprintf('  [EXTRACT] Signal too short (%d samples)\n', length(Yd));
            return;
        end

        [~, ai] = max(abs(Ya));
        i0 = max(1, ai - 5);
        i1 = min(i0 + n - 1, length(Yd));
        np = i1 - i0 + 1;

        if np < n * 0.5
            fprintf('  [EXTRACT] Window too short (%d samples)\n', np);
            return;
        end

        xo = linspace(0, 1, np);
        xn = linspace(0, 1, n);
        dw = interp1(xo, Yd(i0:i1), xn, 'linear', 'extrap');
        vw = interp1(xo, Yv(i0:i1), xn, 'linear', 'extrap');
        fw = interp1(xo, Yf(i0:i1), xn, 'linear', 'extrap');
        aw = interp1(xo, Ya(i0:i1), xn, 'linear', 'extrap');
        pw = interp1(xo, Yp(i0:i1), xn, 'linear', 'extrap');
        ok = true;
        fprintf('  [EXTRACT] ✓ All 5 Prismatic Joint signals extracted (%d pts)\n', n);
    catch ME
        fprintf('  [EXTRACT] ERROR: %s\n', ME.message);
    end
end


% ═══════════════════════════════════════════════════════════════════════
%  FUNCTION: write_physics_to_db
% ═══════════════════════════════════════════════════════════════════════
function write_physics_to_db(conn, ...
    dw, vw, fw, aw, pw, ...
    defl_max, velo_max, force_max, accel_g, pres_max, ...
    k, b, mass_kg, sink_ms, comp_ft, wow, phase)

    dw_str = array_to_pg(dw);
    vw_str = array_to_pg(vw);
    fw_str = array_to_pg(fw);
    aw_str = array_to_pg(aw);
    pw_str = array_to_pg(pw);

    insert_sql = sprintf(['INSERT INTO matlab_physics_output '                 ...
        '(sim_deflection_max, sim_deflection_raw, '                            ...
        ' sim_velocity_max,   sim_velocity_raw,   '                            ...
        ' sim_force_max,      sim_force_raw,       '                           ...
        ' sim_accel_max_g,    sim_accel_raw,       '                           ...
        ' sim_pressure_max,   sim_pressure_raw,    '                           ...
        ' k_eff, b_eff, mass_kg, sink_rate_ms, '                               ...
        ' compression_ft, wow_status, flight_phase, ready_for_ai) '            ...
        'VALUES (%.6f, %s, %.6f, %s, %.6f, %s, '                               ...
        '        %.6f, %s, %.6f, %s, '                                         ...
        '        %.2f, %.2f, %.2f, %.6f, '                                     ...
        '        %.6f, %s, ''%s'', TRUE)'], ...
        defl_max, dw_str, ...
        velo_max, vw_str, ...
        force_max, fw_str, ...
        accel_g,  aw_str, ...
        pres_max, pw_str, ...
        k, b, mass_kg, sink_ms, ...
        comp_ft, bool_to_pg(wow), phase);

    try
        exec(conn, insert_sql);
        fprintf('  [DB] ✓ Prismatic Joint physics written → ready_for_ai=TRUE\n');
        fprintf('  [DB]   Python AI can now classify this touchdown\n\n');
    catch db_err
        fprintf('  [DB] ERROR writing physics: %s\n', db_err.message);
    end
end


% ═══════════════════════════════════════════════════════════════════════
%  HELPER: array_to_pg
% ═══════════════════════════════════════════════════════════════════════
function s = array_to_pg(v)
    strs = arrayfun(@(x) sprintf('%.6f', x), v(:)', 'UniformOutput', false);
    s = ['ARRAY[' strjoin(strs, ',') ']::FLOAT[]'];
end


% ═══════════════════════════════════════════════════════════════════════
%  HELPER: bool_to_pg
% ═══════════════════════════════════════════════════════════════════════
function s = bool_to_pg(v)
    if v, s = 'TRUE'; else, s = 'FALSE'; end
end


% ═══════════════════════════════════════════════════════════════════════
%  HELPER: load_db_config (reads env vars, then local .env if present)
% ═══════════════════════════════════════════════════════════════════════
function cfg = load_db_config(proj_dir)
    env_map = containers.Map('KeyType', 'char', 'ValueType', 'char');
    env_file = fullfile(proj_dir, '.env');
    if exist(env_file, 'file')
        fid = fopen(env_file, 'r');
        if fid ~= -1
            while ~feof(fid)
                raw = fgetl(fid);
                if ~ischar(raw), continue; end
                line = strtrim(raw);
                if isempty(line) || startsWith(line, '#') || ~contains(line, '=')
                    continue;
                end
                eq_idx = strfind(line, '=');
                k = strtrim(line(1:eq_idx(1)-1));
                v = strtrim(line(eq_idx(1)+1:end));
                v = regexprep(v, '^[''"]|[''"]$', '');
                if ~isempty(k)
                    env_map(k) = v;
                end
            end
            fclose(fid);
        end
    end

    cfg.dbname   = resolve_env('AEROTWIN_DB_NAME',     'project_db', env_map);
    cfg.user     = resolve_env('AEROTWIN_DB_USER',     'postgres',   env_map);
    cfg.password = resolve_env('AEROTWIN_DB_PASSWORD', '',           env_map);
    cfg.host     = resolve_env('AEROTWIN_DB_HOST',     'localhost',  env_map);
    cfg.port     = resolve_env('AEROTWIN_DB_PORT',     '5432',       env_map);
end

function val = resolve_env(name, fallback, env_map)
    v = getenv(name);
    if ~isempty(v)
        val = v;
    elseif isKey(env_map, name)
        val = env_map(name);
    else
        val = fallback;
    end
end
