"""
╔════════════════════════════════════════════════════════════════════════╗
║  AeroTwin Schema Migration — Add missing columns to telemetry table    ║
║  Run this once if you get "column does not exist" errors              ║
║                                                                        ║
║  Usage: python migrate_schema.py                                      ║
╚════════════════════════════════════════════════════════════════════════╝
"""

import psycopg2
import sys

from aerotwin_config import get_db_config

DB = get_db_config()

def migrate():
    try:
        c = psycopg2.connect(**DB)
        cur = c.cursor()
        
        print("  [MIGRATE] Connecting to PostgreSQL…")
        
        # List of missing columns that might need to be added
        columns = [
            ("flight_phase",  "VARCHAR(20)"),
            ("airspeed_kt",   "FLOAT"),
            ("latitude_deg",  "FLOAT"),
            ("longitude_deg", "FLOAT"),
            ("wow_nose",      "BOOLEAN"),
            ("wow_left",      "BOOLEAN"),
            ("wow_right",     "BOOLEAN"),
            ("vsi_fpm",       "FLOAT"),
            ("mass_kg",       "FLOAT"),
            ("sink_rate_ms",  "FLOAT"),
        ]
        
        added_count = 0
        for col_name, col_type in columns:
            try:
                cur.execute(f"""
                    ALTER TABLE telemetry
                    ADD COLUMN IF NOT EXISTS {col_name} {col_type}
                """)
                c.commit()
                print(f"  [✓] {col_name:<20} {col_type}")
                added_count += 1
            except Exception as e:
                print(f"  [!] {col_name:<20} — {e}")
        
        print(f"\n  [MIGRATE] Schema migration complete — {added_count} columns added/verified")
        c.close()
        
    except psycopg2.OperationalError as e:
        print(f"  [ERROR] Cannot connect to PostgreSQL: {e}")
        print(f"  [HELP]  Check that postgres is running and credentials are correct in the script")
        sys.exit(1)
    except Exception as e:
        print(f"  [ERROR] {e}")
        sys.exit(1)

if __name__ == "__main__":
    migrate()
