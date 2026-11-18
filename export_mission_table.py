"""
export_mission_table.py
Run this AFTER your 5 (or N) FlightGear missions are done.

Pulls the last N landing rows from flight_diagnostics, prints a
Table-4-style summary to the console, and writes a CSV you can drop
straight into the manuscript / response letter.

Usage:
    python export_mission_table.py            # last 5 landings
    python export_mission_table.py --n 8       # last 8 landings
    python export_mission_table.py --since "2026-09-14 08:00"
"""
import argparse
import csv
import psycopg2

from aerotwin_config import get_db_config

DB = get_db_config()


def _available_columns(cur) -> set:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = 'flight_diagnostics'
        """
    )
    return {row[0] for row in cur.fetchall()}


def _build_query(cols: set, use_since: bool) -> str:
    raw_cls_expr = "ai_raw_class" if "ai_raw_class" in cols else "ai_classification"
    raw_conf_expr = "ai_raw_confidence" if "ai_raw_confidence" in cols else "confidence_score"
    agree_expr = "agree" if "agree" in cols else "(ai_classification = physics_class)"
    where_clause = "WHERE flight_phase = 'LANDING'" + (" AND ts >= %s" if use_since else "")
    order_clause = "ORDER BY ts ASC" if use_since else "ORDER BY ts DESC LIMIT %s"
    return f"""
        SELECT ts, {raw_cls_expr}, {raw_conf_expr}, physics_class, ai_classification,
               decision_source, data_source, {agree_expr}, k_stiffness_val, b_damping_val,
               sink_rate_ms, cs23_triggered, grounded
        FROM flight_diagnostics
        {where_clause}
        {order_clause}
    """


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="Number of most recent landings to export")
    ap.add_argument("--since", type=str, default=None,
                     help="Instead of --n, export every landing since this timestamp "
                          "(e.g. '2026-09-14 08:00')")
    ap.add_argument("--out", type=str, default="mission_table.csv")
    args = ap.parse_args()

    conn = psycopg2.connect(**DB)
    cur = conn.cursor()
    cols = _available_columns(cur)

    if args.since:
        cur.execute(_build_query(cols, use_since=True), (args.since,))
        rows = cur.fetchall()
    else:
        cur.execute(_build_query(cols, use_since=False), (args.n,))
        rows = list(reversed(cur.fetchall()))  # chronological order, run 1 first

    cur.close()
    conn.close()

    if not rows:
        print("No landing rows found. Did you fly any missions yet?")
        return

    header = ["Run", "AI Class", "AI Conf.", "Physics Class", "Fused Decision",
              "Decision Source", "Physics Source", "Agree", "K_eff", "B_eff",
              "Sink (m/s)", "CS-23.473", "Grounded", "Timestamp"]

    print("\nTable 4 — Real-Time Mission Results (regenerated from flight_diagnostics)\n")
    print(f"{'Run':<4}{'AI Class':<16}{'Conf.':<8}{'Physics Class':<16}"
          f"{'Decision':<10}{'Src':<18}{'Agree':<7}{'CS23':<6}")
    print("-" * 90)

    csv_rows = []
    for i, r in enumerate(rows, start=1):
        (ts, ai_raw_class, ai_raw_conf, physics_class, fused_class,
         decision_source, data_source, agree, k_eff, b_eff, sink_ms, cs23, grounded) = r

        conf_pct = f"{(ai_raw_conf or 0)*100:.0f}%"
        agree_mark = "Y" if agree else "N"
        cs23_mark = "EXCEEDED" if cs23 else "OK"

        print(f"{i:<4}{(ai_raw_class or '-'):<16}{conf_pct:<8}{(physics_class or '-'):<16}"
              f"{(decision_source or '-'):<10}{(data_source or '-'):<18}{agree_mark:<7}{cs23_mark:<6}")

        csv_rows.append([
            i, ai_raw_class, conf_pct, physics_class, fused_class,
            decision_source, data_source, agree_mark, round(k_eff or 0, 0),
            round(b_eff or 0, 0), round(sink_ms or 0, 3), cs23_mark,
            "GROUNDED" if grounded else "OK", ts
        ])

    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(csv_rows)

    matlab_count = sum(1 for r in csv_rows if r[6] == "MATLAB_Simscape")
    print(f"\n{matlab_count}/{len(csv_rows)} landings used MATLAB_Simscape "
          f"(the rest fell back to the Python reduced-order model).")
    print(f"CSV written to: {args.out}")


if __name__ == "__main__":
    main()
