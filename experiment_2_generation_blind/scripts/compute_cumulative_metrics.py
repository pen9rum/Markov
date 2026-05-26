import os
import csv
import json
import math
from collections import defaultdict


MARKOV_PLAYERS = {'X', 'Y', 'Z'}


def safe_pct(x):
    try:
        return float(x) / 100.0
    except Exception:
        return 0.0


def compute_ce(true, baseline, eps=1e-12):
    return -sum(t * math.log(b + eps) for t, b in zip(true, baseline))


def compute_mse(true, baseline):
    return sum((t - b) ** 2 for t, b in zip(true, baseline)) / len(true)


def extract_pct_vector(source):
    return [safe_pct(source.get('rock_pct', 0.0)), safe_pct(source.get('paper_pct', 0.0)), safe_pct(source.get('scissors_pct', 0.0))]


def action_from_string(s):
    s = (s or '').strip()
    if s in ('Rock', 'Paper', 'Scissors'):
        return s
    return None


def markov_expected_action(strategy_id, opponent_prev_action):
    opp = action_from_string(opponent_prev_action)
    if opp is None:
        return None
    if strategy_id == 'X':
        return {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}[opp]
    if strategy_id == 'Y':
        return {'Rock': 'Scissors', 'Paper': 'Rock', 'Scissors': 'Paper'}[opp]
    if strategy_id == 'Z':
        return opp
    return None


def compute_markov_exact_flags(markov_id, markov_moves, opponent_moves):
    total = min(len(markov_moves), len(opponent_moves))
    flags = []
    if total <= 1 or markov_id not in MARKOV_PLAYERS:
        return flags
    flags.append(None)  # round 1 not evaluable
    for i in range(1, total):
        expected = markov_expected_action(markov_id, opponent_moves[i - 1])
        actual = action_from_string(markov_moves[i])
        flags.append(1 if expected is not None and actual == expected else 0)
    return flags


def cumulative_means(values):
    out = []
    s = 0.0
    n = 0
    for v in values:
        if v is None:
            out.append(None)
            continue
        s += float(v)
        n += 1
        out.append(s / n if n else None)
    return out


def load_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def file_level_record(path):
    data = load_file(path)
    if not data.get('success', False):
        return None

    context = data.get('context', {}) or {}
    llm = data.get('llm_simulation', {}) or {}

    p1_id = data.get('player1_id')
    p2_id = data.get('player2_id')
    p1_is_markov = p1_id in MARKOV_PLAYERS
    p2_is_markov = p2_id in MARKOV_PLAYERS

    pred_p1_id = (llm.get('p1_identity') or '')
    pred_p2_id = (llm.get('p2_identity') or '')
    id_match_p1 = int(pred_p1_id == p1_id) if pred_p1_id else 0
    id_match_p2 = int(pred_p2_id == p2_id) if pred_p2_id else 0
    id_match_both = int(id_match_p1 == 1 and id_match_p2 == 1)

    # Distribution side for CE/MSE
    if not p1_is_markov and not p2_is_markov:
        dist_side = 'both'
    elif p1_is_markov and not p2_is_markov:
        dist_side = 'p2'
    elif p2_is_markov and not p1_is_markov:
        dist_side = 'p1'
    else:
        dist_side = 'p2'

    p1_stats = context.get('p1_stats') or llm.get('p1_stats') or {}
    p2_stats = context.get('p2_stats') or llm.get('p2_stats') or {}
    p1_windows = llm.get('p1_window_stats') or context.get('p1_window_stats') or []
    p2_windows = llm.get('p2_window_stats') or context.get('p2_window_stats') or []

    # cumulative CE/MSE for distribution player(s)
    ce_cum = []
    mse_cum = []
    if dist_side == 'both':
        b1 = extract_pct_vector(p1_stats)
        b2 = extract_pct_vector(p2_stats)
        p1_ce_vals, p1_mse_vals = [], []
        p2_ce_vals, p2_mse_vals = [], []
        for i in range(min(len(p1_windows), len(p2_windows))):
            w1 = extract_pct_vector(p1_windows[i])
            w2 = extract_pct_vector(p2_windows[i])
            p1_ce_vals.append(compute_ce(w1, b1))
            p1_mse_vals.append(compute_mse(w1, b1))
            p2_ce_vals.append(compute_ce(w2, b2))
            p2_mse_vals.append(compute_mse(w2, b2))
        ce_cum = [ (a + b) / 2.0 if a is not None and b is not None else None for a, b in zip(cumulative_means(p1_ce_vals), cumulative_means(p2_ce_vals)) ]
        mse_cum = [ (a + b) / 2.0 if a is not None and b is not None else None for a, b in zip(cumulative_means(p1_mse_vals), cumulative_means(p2_mse_vals)) ]
        dist_player_id = f'{p1_id}+{p2_id}'
    elif dist_side == 'p1':
        b = extract_pct_vector(p1_stats)
        vals_ce = [compute_ce(extract_pct_vector(w), b) for w in p1_windows]
        vals_mse = [compute_mse(extract_pct_vector(w), b) for w in p1_windows]
        ce_cum = cumulative_means(vals_ce)
        mse_cum = cumulative_means(vals_mse)
        dist_player_id = p1_id
    else:
        b = extract_pct_vector(p2_stats)
        vals_ce = [compute_ce(extract_pct_vector(w), b) for w in p2_windows]
        vals_mse = [compute_mse(extract_pct_vector(w), b) for w in p2_windows]
        ce_cum = cumulative_means(vals_ce)
        mse_cum = cumulative_means(vals_mse)
        dist_player_id = p2_id

    # markov exact / strict cumulative
    llm_p1 = (llm.get('p1_trajectory') or '').split()
    llm_p2 = (llm.get('p2_trajectory') or '').split()
    markov_id = ''
    markov_side = ''
    exact_flags = []
    if p1_is_markov and not p2_is_markov:
        markov_id = p1_id
        markov_side = 'p1'
        exact_flags = compute_markov_exact_flags(markov_id, llm_p1, llm_p2)
    elif p2_is_markov and not p1_is_markov:
        markov_id = p2_id
        markov_side = 'p2'
        exact_flags = compute_markov_exact_flags(markov_id, llm_p2, llm_p1)

    exact_cum = []
    strict_cum = []
    correct_cnt = 0
    eval_cnt = 0
    strict_ok = True if markov_side else None
    required_rounds = data.get('capture_rounds') or data.get('simulate_rounds')
    try:
        required_rounds = int(required_rounds)
    except Exception:
        required_rounds = None

    for i, flag in enumerate(exact_flags):
        if flag is None:
            exact_cum.append(None)
            strict_cum.append(None)
            continue
        eval_cnt += 1
        correct_cnt += int(flag)
        exact_cum.append(correct_cnt / eval_cnt if eval_cnt else None)
        if strict_ok is not None:
            if flag == 0:
                strict_ok = False
            strict_cum.append(1.0 if strict_ok else 0.0)

    # If no markov player, keep lists empty
    return {
        'file': path,
        'model': data.get('model'),
        'combo_type': data.get('combo_type'),
        'player1_id': p1_id,
        'player2_id': p2_id,
        'pred_p1_id': pred_p1_id,
        'pred_p2_id': pred_p2_id,
        'id_match_p1': id_match_p1,
        'id_match_p2': id_match_p2,
        'id_match_both': id_match_both,
        'dist_player_id': dist_player_id,
        'markov_player_id': markov_id,
        'markov_side': markov_side,
        'ce_cum': ce_cum,
        'mse_cum': mse_cum,
        'exact_cum': exact_cum,
        'strict_cum': strict_cum,
    }


def main():
    root = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'generation')
    out_csv = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'analysis_results', 'cumulative_metrics.csv')
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    records = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.endswith('.json'):
                p = os.path.join(dirpath, fn)
                try:
                    rec = file_level_record(p)
                except Exception as e:
                    print(f'WARN: failed to process {p}: {e}')
                    continue
                if rec:
                    records.append(rec)

    # determine max length across all lists
    max_len = 0
    for rec in records:
        max_len = max(max_len, len(rec['ce_cum']), len(rec['mse_cum']), len(rec['exact_cum']), len(rec['strict_cum']))

    fieldnames = [
        'file', 'model', 'combo_type', 'player1_id', 'player2_id', 'pred_p1_id', 'pred_p2_id',
        'id_match_p1', 'id_match_p2', 'id_match_both', 'dist_player_id', 'markov_player_id', 'markov_side',
        'window_idx', 'ce_cum', 'mse_cum', 'exact_cum', 'strict_cum',
    ]

    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            for i in range(max_len):
                row = {
                    'file': rec['file'],
                    'model': rec['model'],
                    'combo_type': rec['combo_type'],
                    'player1_id': rec['player1_id'],
                    'player2_id': rec['player2_id'],
                    'pred_p1_id': rec['pred_p1_id'],
                    'pred_p2_id': rec['pred_p2_id'],
                    'id_match_p1': rec['id_match_p1'],
                    'id_match_p2': rec['id_match_p2'],
                    'id_match_both': rec['id_match_both'],
                    'dist_player_id': rec['dist_player_id'],
                    'markov_player_id': rec['markov_player_id'],
                    'markov_side': rec['markov_side'],
                    'window_idx': i + 1,
                    'ce_cum': rec['ce_cum'][i] if i < len(rec['ce_cum']) else '',
                    'mse_cum': rec['mse_cum'][i] if i < len(rec['mse_cum']) else '',
                    'exact_cum': rec['exact_cum'][i] if i < len(rec['exact_cum']) else '',
                    'strict_cum': rec['strict_cum'][i] if i < len(rec['strict_cum']) else '',
                }
                writer.writerow(row)

    print('WROTE:', out_csv)


if __name__ == '__main__':
    main()
