"""
Exp3 — Compute cumulative (running) metrics from per-window metrics CSV.
For each file, computes running mean of CE, MSE, overlap_rate, strict_exact
up to each window index.

Input:  exp3(complex_markov)/analysis_results/{markov_set}_metrics.csv
Output: exp3(complex_markov)/analysis_results/{markov_set}_cumulative_metrics.csv

Usage:
  python tools_exp3/compute_cumulative_metrics.py --markov-set qrs
"""
import os
import csv
import argparse
from collections import defaultdict

EXP3_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
RESULT_ROOT = os.path.join(EXP3_ROOT, 'analysis_results')


def to_float(x):
    try: return float(x)
    except Exception: return None


def running_mean(vals):
    """Running mean over observed values only; missing windows remain None."""
    out, s, n = [], 0.0, 0
    for v in vals:
        if v is None:
            out.append(None)
            continue
        s += v; n += 1
        out.append(s / n)
    return out


def running_strict(vals):
    """Cumulative strict: 1 only if ALL windows so far are 1, drops to 0 permanently once any window fails."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--markov-set', required=True, choices=['xyz', 'qrs', 'tuv', 'xyz_opp'])
    args = parser.parse_args()
    ms = args.markov_set

    in_path  = os.path.join(RESULT_ROOT, f'{ms}_metrics.csv')
    out_path = os.path.join(RESULT_ROOT, f'{ms}_cumulative_metrics.csv')

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run run_analysis.py first.")
        return

    with open(in_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Group rows by file-level key, preserve window order
    groups = defaultdict(list)
    for r in rows:
        key = (r['model'], r['player1_id'], r['player2_id'], r['pred_p1_id'], r['pred_p2_id'])
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

    print(f"Wrote {len(out_rows)} rows → {out_path}")


if __name__ == '__main__':
    main()
