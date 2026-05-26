import os
import csv
from collections import defaultdict


def safe_int(v):
    try:
        return int(float(v))
    except Exception:
        return 0


def safe_float_or_none(v):
    if v in (None, ''):
        return None
    try:
        return float(v)
    except Exception:
        return None


def avg(nums):
    return sum(nums) / len(nums) if nums else None


def fmt(x):
    if x is None:
        return ''
    return f"{x:.6f}"


def main():
    root = os.getcwd()
    src = os.path.join(root, 'exp2(generation_blind)', 'analysis_results', 'slump_metrics.csv')
    out_csv = os.path.join(root, 'exp2(generation_blind)', 'analysis_results', 'identity_condition_summary.csv')

    if not os.path.exists(src):
        print('ERROR: slump_metrics.csv not found')
        return

    # file-level dedup (slump_metrics has many window rows per file)
    file_rows = {}
    with open(src, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fpath = row.get('file')
            if not fpath:
                continue
            file_rows.setdefault(fpath, row)

    rows = list(file_rows.values())
    total = len(rows)

    both_correct = [r for r in rows if safe_int(r.get('id_match_both')) == 1]
    both_wrong = [r for r in rows if safe_int(r.get('id_match_both')) == 0]

    def strict_rate(items):
        vals = [safe_float_or_none(r.get('strict_exact_match_all')) for r in items]
        vals = [v for v in vals if v is not None]
        return avg(vals)

    def markov_strict_rate(items):
        vals = []
        for r in items:
            side = (r.get('markov_side') or '').strip()
            if side == 'p1':
                v = safe_float_or_none(r.get('strict_exact_match_p1'))
            elif side == 'p2':
                v = safe_float_or_none(r.get('strict_exact_match_p2'))
            else:
                v = None
            if v is not None:
                vals.append(v)
        return avg(vals)

    def markov_exact_rate(items):
        vals = [safe_float_or_none(r.get('exact_match')) for r in items]
        vals = [v for v in vals if v is not None]
        return avg(vals)

    summary = [
        {
            'group': 'all_files',
            'count': total,
            'ratio': 1.0,
            'strict_exact_match_all_rate': strict_rate(rows),
            'markov_strict_rate': markov_strict_rate(rows),
            'markov_window_exact_avg': markov_exact_rate(rows),
        },
        {
            'group': 'id_both_correct',
            'count': len(both_correct),
            'ratio': (len(both_correct) / total) if total else 0.0,
            'strict_exact_match_all_rate': strict_rate(both_correct),
            'markov_strict_rate': markov_strict_rate(both_correct),
            'markov_window_exact_avg': markov_exact_rate(both_correct),
        },
        {
            'group': 'id_both_wrong',
            'count': len(both_wrong),
            'ratio': (len(both_wrong) / total) if total else 0.0,
            'strict_exact_match_all_rate': strict_rate(both_wrong),
            'markov_strict_rate': markov_strict_rate(both_wrong),
            'markov_window_exact_avg': markov_exact_rate(both_wrong),
        },
    ]

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        fields = [
            'group', 'count', 'ratio',
            'strict_exact_match_all_rate',
            'markov_strict_rate',
            'markov_window_exact_avg',
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in summary:
            w.writerow({
                'group': r['group'],
                'count': r['count'],
                'ratio': fmt(r['ratio']),
                'strict_exact_match_all_rate': fmt(r['strict_exact_match_all_rate']),
                'markov_strict_rate': fmt(r['markov_strict_rate']),
                'markov_window_exact_avg': fmt(r['markov_window_exact_avg']),
            })

    print('WROTE:', out_csv)
    for r in summary:
        print(
            f"{r['group']}: count={r['count']}, ratio={r['ratio']:.4f}, "
            f"strict_all={fmt(r['strict_exact_match_all_rate'])}, "
            f"markov_strict={fmt(r['markov_strict_rate'])}, "
            f"markov_exact_avg={fmt(r['markov_window_exact_avg'])}"
        )


if __name__ == '__main__':
    main()
