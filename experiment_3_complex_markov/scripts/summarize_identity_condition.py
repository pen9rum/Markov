"""
Exp3 — Summarize generation metrics by identity correctness condition.
Mirrors exp2/summarize_identity_condition.py.

Input:  exp3(complex_markov)/analysis_results/{markov_set}_metrics.csv
Output: exp3(complex_markov)/analysis_results/{markov_set}_identity_condition_summary.csv

Usage:
  python tools_exp3/summarize_identity_condition.py --markov-set qrs
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


def avg(vals):
    vs = [v for v in vals if v is not None]
    return sum(vs) / len(vs) if vs else None


def fmt(x):
    return f'{x:.6f}' if x is not None else ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markov-set', required=True, choices=['xyz', 'qrs', 'tuv', 'xyz_opp'])
    args = parser.parse_args()
    ms = args.markov_set

    in_path  = os.path.join(RESULT_ROOT, f'{ms}_metrics.csv')
    out_path = os.path.join(RESULT_ROOT, f'{ms}_identity_condition_summary.csv')

    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found."); return

    with open(in_path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))

    # Group by (model, combo_type, both_correct)
    groups = defaultdict(lambda: defaultdict(list))

    for r in rows:
        model  = r.get('model', '')
        ct     = r.get('combo_type', '')
        bc     = r.get('both_correct', '')
        mc     = r.get('markov_id_match', '')
        key    = (model, ct)
        cond   = bc  # '1' correct, '0' wrong, '' unknown

        for metric in ('ce', 'mse', 'overlap_rate', 'strict_exact'):
            v = to_float(r.get(metric))
            if v is not None:
                groups[(model, ct, cond)][metric].append(v)
                # also by markov_id_match
                if mc in ('0', '1'):
                    groups[(model, ct, f'markov_{mc}')][metric].append(v)

    out_rows = []
    for (model, combo_type, condition), metrics in sorted(groups.items()):
        out_rows.append({
            'model':        model,
            'combo_type':   combo_type,
            'condition':    condition,
            'n_windows':    len(metrics.get('ce', [])),
            'ce_avg':       fmt(avg(metrics.get('ce', []))),
            'mse_avg':      fmt(avg(metrics.get('mse', []))),
            'overlap_avg':  fmt(avg(metrics.get('overlap_rate', []))),
            'strict_avg':   fmt(avg(metrics.get('strict_exact', []))),
        })

    if not out_rows:
        print("No data."); return

    os.makedirs(RESULT_ROOT, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)

    print(f"Wrote {len(out_rows)} rows → {out_path}")

    # Console summary
    print(f"\n{'='*60}")
    print(f"Identity Condition Summary — {ms.upper()}")
    print(f"{'='*60}")
    for r in out_rows:
        if r['condition'] in ('0', '1'):
            label = 'Correct' if r['condition'] == '1' else 'Wrong'
            print(f"  {r['model']:20s}  combo={r['combo_type']}  {label:8s}  "
                  f"overlap={r['overlap_avg'][:6]}  strict={r['strict_avg'][:6]}  "
                  f"ce={r['ce_avg'][:6]}  n={r['n_windows']}")


if __name__ == '__main__':
    main()
