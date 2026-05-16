"""
Exp2 — Compute cumulative metrics from xyz_metrics.csv.
Same logic as tools_exp3/compute_cumulative_metrics.py.

Input:  exp2(generation_blind)/analysis_results/xyz_metrics.csv
Output: exp2(generation_blind)/analysis_results/xyz_cumulative_metrics.csv

Usage:
  python tools_exp2/compute_cumulative_metrics_xyz.py
"""
import os
import csv
from collections import defaultdict

EXP2_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)')
RESULT_ROOT = os.path.join(EXP2_ROOT, 'analysis_results')


def to_float(x):
    try: return float(x)
    except Exception: return None


def running_mean(vals):
    out, s, n = [], 0.0, 0
    for v in vals:
        if v is None:
            out.append(None)
            continue
        s += v; n += 1
        out.append(s / n)
    return out


def running_strict(vals):
    out, ok = [], True
    for v in vals:
        if v is None:
            out.append(None)
            continue
        if v != 1.0:
            ok = False
        out.append(1.0 if ok else 0.0)
    return out


def main():
    in_path  = os.path.join(RESULT_ROOT, 'xyz_metrics.csv')
    out_path = os.path.join(RESULT_ROOT, 'xyz_cumulative_metrics.csv')

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run run_analysis.py first.")
        return

    with open(in_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    groups = defaultdict(list)
    for r in rows:
        key = (r['model'], r['file_idx'], r['player1_id'], r['player2_id'], r['pred_p1_id'], r['pred_p2_id'])
        groups[key].append(r)

    out_rows = []
    for key, grp in groups.items():
        grp_sorted = sorted(grp, key=lambda r: int(r.get('window_idx', 0)))

        ce_vals      = [to_float(r['ce'])           for r in grp_sorted]
        mse_vals     = [to_float(r['mse'])          for r in grp_sorted]
        overlap_vals = [to_float(r['overlap_rate']) for r in grp_sorted]
        strict_vals  = [to_float(r['strict_exact']) for r in grp_sorted]

        ce_cum      = running_mean(ce_vals)
        mse_cum     = running_mean(mse_vals)
        overlap_cum = running_mean(overlap_vals)
        strict_cum  = running_strict(strict_vals)

        for i, r in enumerate(grp_sorted):
            out_rows.append({
                **r,
                'ce_cum':      ce_cum[i],
                'mse_cum':     mse_cum[i],
                'overlap_cum': overlap_cum[i],
                'strict_cum':  strict_cum[i],
            })

    if not out_rows:
        print("No data."); return

    os.makedirs(RESULT_ROOT, exist_ok=True)
    fields = list(out_rows[0].keys())
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows -> {out_path}")


if __name__ == '__main__':
    main()
