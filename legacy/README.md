# Legacy / Superseded Scripts

These files represent earlier prototypes or single-purpose helpers kept for historical traceability. They have been superseded by the active pipeline in the repository root:

| Legacy File | Superseded By | Notes |
| --- | --- | --- |
| `DO228-ROBOT-PILOT.py` | `aerotwin_robot_aircraft.py` | Early standalone Python mission demo (v1.0) |
| `live_diagnostics.py` | `fg_connect.py` + `phi_fleet_server.py` | Early TCP co-pilot listener (`:5000` / `:5001`) |
| `aerotwin_inference_worker.py` | `phi_fleet_server.py` / `fleet_dashboard.py` | Early standalone heuristic inference worker |
| `AeroTwin_Launch.bat` | `docs/AEROTWIN_STARTUP_GUIDE.txt` | Windows launcher for legacy TCP port layout |
| `Postgresql_Database_Script.sql` | `aerotwin_schema.sql` | Partial v16.3 `matlab_physics_output` migration |
| `generate_confusion_matrix.py` | `aerotwin_manuscript_validation.py` | Early all-dataset confusion matrix helper |
| `generate_rul_scatter.py` | `aerotwin_manuscript_validation.py` | Early RUL scatter helper |
| `generate_agreement_bars.py` | `aerotwin_manuscript_validation.py` | Early agreement bar chart helper |
