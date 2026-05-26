"""
Exp3 — Export per-file simulation summary to CSV.
Usage:
  python tools_exp3/export_sim_results.py --markov-set qrs
"""
import os
import sys
import json
import csv
import argparse

EXP3_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
GEN_ROOT    = os.path.join(EXP3_ROOT, 'generation')
RESULT_ROOT = os.path.join(EXP3_ROOT, 'analysis_results')


def export(markov_set, model_filter=None):
    root = os.path.join(GEN_ROOT, markov_set)
    if not os.path.isdir(root):
        print(f"ERROR: {root} not found"); return

    rows = []
    for model_dir in sorted(os.listdir(root)):
        if model_filter and model_dir != model_filter:
            continue
        model_path = os.path.join(root, model_dir)
        if not os.path.isdir(model_path): continue
        for ctx_dir in os.listdir(model_path):
            ctx_path = os.path.join(model_path, ctx_dir)
            if not os.path.isdir(ctx_path): continue
            for type_dir in os.listdir(ctx_path):
                type_path = os.path.join(ctx_path, type_dir)
                if not os.path.isdir(type_path): continue
                combo_type = (2 if 'markov_p1' in type_dir
                              else 3 if 'markov_p2' in type_dir else 1)
                for fname in sorted(os.listdir(type_path)):
                    if not fname.endswith('.json'): continue
                    fpath = os.path.join(type_path, fname)
                    try:
                        with open(fpath, encoding='utf-8') as f:
                            d = json.load(f)
                    except Exception as e:
                        rows.append({'file': fname, 'model': model_dir, 'status': 'load_error', 'error': str(e)})
                        continue

                    llm = d.get('llm_simulation', {})
                    rows.append({
                        'file':           fname,
                        'model':          d.get('model', model_dir),
                        'markov_set':     d.get('markov_set', markov_set),
                        'combo_type':     combo_type,
                        'player1_id':     d.get('player1_id', ''),
                        'player2_id':     d.get('player2_id', ''),
                        'context_rounds': d.get('context_rounds', ''),
                        'simulate_rounds':d.get('simulate_rounds', ''),
                        'capture_rounds': d.get('capture_rounds', ''),
                        'success':        d.get('success', False),
                        'parsed_rounds':  llm.get('parsed_rounds', 0),
                        'complete':       llm.get('complete', False),
                        'pred_p1_id':     (llm.get('p1_identity') or '').strip().upper(),
                        'pred_p2_id':     (llm.get('p2_identity') or '').strip().upper(),
                        'p1_correct':     int((llm.get('p1_identity') or '').strip().upper() == d.get('player1_id', '')),
                        'p2_correct':     int((llm.get('p2_identity') or '').strip().upper() == d.get('player2_id', '')),
                        'error':          d.get('error', ''),
                    })

    if not rows:
        print("No files found."); return

    os.makedirs(RESULT_ROOT, exist_ok=True)
    out = os.path.join(RESULT_ROOT, f'{markov_set}_simulation_summary.csv')
    with open(out, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    ok = sum(1 for r in rows if r.get('success') and r.get('parsed_rounds', 0) >= int(r.get('capture_rounds') or 1500))
    print(f"Exported {len(rows)} files ({ok} complete) → {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markov-set', required=True, choices=['xyz', 'qrs', 'tuv', 'xyz_opp'])
    parser.add_argument('--model', default=None)
    args = parser.parse_args()
    export(args.markov_set, args.model)


if __name__ == '__main__':
    main()
