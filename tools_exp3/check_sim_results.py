"""
Exp3 — Check generation results completeness.
Usage:
  python tools_exp3/check_sim_results.py --markov-set qrs
  python tools_exp3/check_sim_results.py --markov-set tuv --model deepseek-reasoner
"""
import os
import sys
import json
import argparse
from collections import defaultdict

EXP3_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
GEN_ROOT  = os.path.join(EXP3_ROOT, 'generation')

CAPTURE_ROUNDS = 1500


def check(markov_set, model_filter=None, capture=CAPTURE_ROUNDS):
    root = os.path.join(GEN_ROOT, markov_set)
    if not os.path.isdir(root):
        print(f"ERROR: {root} not found")
        return

    stats = defaultdict(lambda: {'ok': 0, 'failed': 0, 'partial': 0, 'files': []})

    for model_dir in sorted(os.listdir(root)):
        if model_filter and model_dir != model_filter:
            continue
        model_path = os.path.join(root, model_dir)
        if not os.path.isdir(model_path):
            continue
        for ctx_dir in os.listdir(model_path):
            ctx_path = os.path.join(model_path, ctx_dir)
            if not os.path.isdir(ctx_path):
                continue
            for type_dir in os.listdir(ctx_path):
                type_path = os.path.join(ctx_path, type_dir)
                if not os.path.isdir(type_path):
                    continue
                key = f"{model_dir}/{ctx_dir}/{type_dir}"
                for fname in sorted(os.listdir(type_path)):
                    if not fname.endswith('.json'):
                        continue
                    fpath = os.path.join(type_path, fname)
                    try:
                        with open(fpath, encoding='utf-8') as f:
                            d = json.load(f)
                    except Exception as e:
                        stats[key]['failed'] += 1
                        stats[key]['files'].append(('LOAD_ERR', fname, str(e)))
                        continue

                    if not d.get('success', True):
                        stats[key]['failed'] += 1
                        stats[key]['files'].append(('FAILED', fname, d.get('error', '')))
                        continue

                    llm = d.get('llm_simulation', {})
                    parsed = llm.get('parsed_rounds', 0)
                    if parsed < capture:
                        stats[key]['partial'] += 1
                        stats[key]['files'].append(('PARTIAL', fname, f"parsed={parsed}/{capture}"))
                    else:
                        stats[key]['ok'] += 1

    print(f"\n{'='*70}")
    print(f"Exp3 Generation Check — {markov_set.upper()}")
    print(f"{'='*70}")
    total_ok = total_fail = total_partial = 0
    for key in sorted(stats):
        s = stats[key]
        total_ok      += s['ok']
        total_fail    += s['failed']
        total_partial += s['partial']
        print(f"\n  {key}")
        print(f"    OK={s['ok']}  PARTIAL={s['partial']}  FAILED={s['failed']}")
        for status, fname, detail in s['files']:
            if status != 'OK':
                print(f"    [{status}] {fname}  — {detail}")

    print(f"\n{'='*70}")
    print(f"  TOTAL: OK={total_ok}  PARTIAL={total_partial}  FAILED={total_fail}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--markov-set', required=True, choices=['xyz', 'qrs', 'tuv', 'xyz_opp'])
    parser.add_argument('--model', default=None)
    parser.add_argument('--capture-rounds', type=int, default=CAPTURE_ROUNDS)
    args = parser.parse_args()
    check(args.markov_set, args.model, args.capture_rounds)


if __name__ == '__main__':
    main()
