import os
import json
import csv
import math


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
    # source expected to have rock_pct, paper_pct, scissors_pct
    return [safe_pct(source.get('rock_pct', 0.0)), safe_pct(source.get('paper_pct', 0.0)), safe_pct(source.get('scissors_pct', 0.0))]


def action_from_string(s):
    s = (s or '').strip()
    if s == 'Rock':
        return 'Rock'
    if s == 'Paper':
        return 'Paper'
    if s == 'Scissors':
        return 'Scissors'
    return None


def markov_expected_action(strategy_id, opponent_prev_action):
    opp = action_from_string(opponent_prev_action)
    if opp is None:
        return None

    # X = Win-Last, Y = Lose-Last, Z = Copy-Last
    if strategy_id == 'X':
        if opp == 'Rock':
            return 'Paper'
        if opp == 'Paper':
            return 'Scissors'
        return 'Rock'
    if strategy_id == 'Y':
        if opp == 'Rock':
            return 'Scissors'
        if opp == 'Paper':
            return 'Rock'
        return 'Paper'
    if strategy_id == 'Z':
        return opp
    return None


def compute_markov_exact_match_by_window(markov_id, markov_moves, opponent_moves, window_size=100):
    """
    Return list of exact-match rates (0~1) by window for a Markov player.
    Round 1 has no previous-opponent action, so it is excluded from matching.
    """
    total = min(len(markov_moves), len(opponent_moves))
    if total <= 1 or markov_id not in MARKOV_PLAYERS:
        return []

    # Per-round match flag, index aligned with rounds (0-based)
    # round 1 => None (not evaluable)
    flags = [None]
    for i in range(1, total):
        expected = markov_expected_action(markov_id, opponent_moves[i - 1])
        actual = action_from_string(markov_moves[i])
        if expected is None or actual is None:
            flags.append(None)
        else:
            flags.append(1 if actual == expected else 0)

    out = []
    for start in range(0, total, window_size):
        chunk = flags[start:start + window_size]
        evals = [x for x in chunk if x is not None]
        if not evals:
            out.append(None)
        else:
            out.append(sum(evals) / len(evals))
    return out


def compute_markov_strict_match_by_window(markov_id, markov_moves, opponent_moves, window_size=100):
    """
    Return list of strict (all-or-nothing) exact-match values by window.
    For each window:
      1 -> all evaluable rounds in that window are correct
      0 -> at least one evaluable round is wrong
      None -> no evaluable rounds in that window
    Round 1 has no previous-opponent action, so it is excluded from matching.
    """
    total = min(len(markov_moves), len(opponent_moves))
    if total <= 1 or markov_id not in MARKOV_PLAYERS:
        return []

    flags = [None]
    for i in range(1, total):
        expected = markov_expected_action(markov_id, opponent_moves[i - 1])
        actual = action_from_string(markov_moves[i])
        if expected is None or actual is None:
            flags.append(None)
        else:
            flags.append(1 if actual == expected else 0)

    out = []
    for start in range(0, total, window_size):
        chunk = flags[start:start + window_size]
        evals = [x for x in chunk if x is not None]
        if not evals:
            out.append(None)
        else:
            out.append(1 if all(x == 1 for x in evals) else 0)
    return out


def compute_markov_strict_match(markov_id, markov_moves, opponent_moves, required_rounds=None):
    """
    All-or-nothing exact match for Markov sequence.
    Returns:
      1 -> all evaluable rounds are correct
      0 -> at least one round is wrong (or insufficient rounds when required_rounds is set)
      None -> no Markov player
    """
    if markov_id not in MARKOV_PLAYERS:
        return None

    total = min(len(markov_moves), len(opponent_moves))
    if required_rounds is not None:
        if total < required_rounds:
            return 0
        total = required_rounds

    # Need at least round 2 to evaluate Markov rule
    if total <= 1:
        return 0

    for i in range(1, total):
        expected = markov_expected_action(markov_id, opponent_moves[i - 1])
        actual = action_from_string(markov_moves[i])
        if expected is None or actual is None:
            return 0
        if actual != expected:
            return 0
    return 1


def process_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if not data.get('success', False):
        return []

    # Decide which side is non-Markov (for CE/MSE) and which side is Markov (for exact match)
    context = data.get('context', {}) or {}
    llm = data.get('llm_simulation', {}) or {}

    p1_id = data.get('player1_id')
    p2_id = data.get('player2_id')
    pred_p1_id = (llm.get('p1_identity') or '')
    pred_p2_id = (llm.get('p2_identity') or '')

    id_match_p1 = int(pred_p1_id == p1_id) if pred_p1_id else 0
    id_match_p2 = int(pred_p2_id == p2_id) if pred_p2_id else 0
    id_match_both = int(id_match_p1 == 1 and id_match_p2 == 1)

    p1_is_markov = p1_id in MARKOV_PLAYERS
    p2_is_markov = p2_id in MARKOV_PLAYERS

    def mean_pct_vector(a, b):
        return [
            (a[0] + b[0]) / 2.0,
            (a[1] + b[1]) / 2.0,
            (a[2] + b[2]) / 2.0,
        ]

    def avg_sidewise_metrics(window_idx):
        p1_stats = context.get('p1_stats') or llm.get('p1_stats') or {}
        p2_stats = context.get('p2_stats') or llm.get('p2_stats') or {}
        p1_windows = llm.get('p1_window_stats') or context.get('p1_window_stats') or []
        p2_windows = llm.get('p2_window_stats') or context.get('p2_window_stats') or []

        if window_idx >= len(p1_windows) or window_idx >= len(p2_windows):
            return None

        b1 = extract_pct_vector(p1_stats)
        b2 = extract_pct_vector(p2_stats)
        w1 = extract_pct_vector(p1_windows[window_idx])
        w2 = extract_pct_vector(p2_windows[window_idx])

        ce1 = compute_ce(w1, b1)
        ce2 = compute_ce(w2, b2)
        mse1 = compute_mse(w1, b1)
        mse2 = compute_mse(w2, b2)

        return {
            'ce': (ce1 + ce2) / 2.0,
            'mse': (mse1 + mse2) / 2.0,
            'baseline': mean_pct_vector(b1, b2),
            'window': mean_pct_vector(w1, w2),
        }

    # For CE/MSE: track the distribution player.
    # Type1 (both non-Markov) uses P1/P2 averaged metrics.
    if not p1_is_markov and not p2_is_markov:
        dist_side = 'both'
    elif p1_is_markov and not p2_is_markov:
        dist_side = 'p2'
    elif p2_is_markov and not p1_is_markov:
        dist_side = 'p1'
    else:
        # both non-Markov or both Markov (fallback) -> keep p2 for continuity
        dist_side = 'p2'

    if dist_side == 'both':
        p1_stats = context.get('p1_stats') or llm.get('p1_stats') or {}
        p2_stats = context.get('p2_stats') or llm.get('p2_stats') or {}
        dist_player_id = f'{p1_id}+{p2_id}'
        p1_windows = llm.get('p1_window_stats') or context.get('p1_window_stats') or []
        p2_windows = llm.get('p2_window_stats') or context.get('p2_window_stats') or []
        max_windows = min(len(p1_windows), len(p2_windows), 10)
        baseline = mean_pct_vector(extract_pct_vector(p1_stats), extract_pct_vector(p2_stats))
        dist_window_stats = []
        dist_ce_mse = []
        for idx in range(max_windows):
            w = avg_sidewise_metrics(idx)
            if w is None:
                break
            dist_window_stats.append({'rock_pct': w['window'][0] * 100.0, 'paper_pct': w['window'][1] * 100.0, 'scissors_pct': w['window'][2] * 100.0})
            dist_ce_mse.append((w['ce'], w['mse']))
    elif dist_side == 'p1':
        dist_stats = context.get('p1_stats') or llm.get('p1_stats') or {}
        dist_window_stats = llm.get('p1_window_stats') or context.get('p1_window_stats') or []
        dist_player_id = p1_id
    else:
        dist_stats = context.get('p2_stats') or llm.get('p2_stats') or {}
        dist_window_stats = llm.get('p2_window_stats') or context.get('p2_window_stats') or []
        dist_player_id = p2_id

    if dist_side != 'both':
        baseline = extract_pct_vector(dist_stats)

    # For EXACT MATCH: track the Markov player if present
    markov_player_id = None
    markov_side = ''
    exact_by_window = []
    strict_by_window = []
    llm_p1 = (llm.get('p1_trajectory') or '').split()
    llm_p2 = (llm.get('p2_trajectory') or '').split()
    required_rounds = data.get('capture_rounds') or data.get('simulate_rounds')
    try:
        required_rounds = int(required_rounds)
    except Exception:
        required_rounds = None

    if p1_is_markov and not p2_is_markov:
        markov_player_id = p1_id
        markov_side = 'p1'
        exact_by_window = compute_markov_exact_match_by_window(markov_player_id, llm_p1, llm_p2)
        strict_by_window = compute_markov_strict_match_by_window(markov_player_id, llm_p1, llm_p2)
    elif p2_is_markov and not p1_is_markov:
        markov_player_id = p2_id
        markov_side = 'p2'
        exact_by_window = compute_markov_exact_match_by_window(markov_player_id, llm_p2, llm_p1)
        strict_by_window = compute_markov_strict_match_by_window(markov_player_id, llm_p2, llm_p1)

    rows = []
    max_windows = min(len(dist_window_stats), 10)
    for i in range(max_windows):
        w = dist_window_stats[i]
        true = extract_pct_vector(w)
        if dist_side == 'both':
            ce, mse = dist_ce_mse[i]
        else:
            ce = compute_ce(true, baseline)
            mse = compute_mse(true, baseline)
        exact_match = None
        strict_exact_match = None
        if i < len(exact_by_window):
            exact_match = exact_by_window[i]
        if i < len(strict_by_window):
            strict_exact_match = strict_by_window[i]

        rows.append({
            'file': path,
            'model': data.get('model'),
            'player1_id': data.get('player1_id'),
            'player2_id': data.get('player2_id'),
            'combo_type': data.get('combo_type'),
            'pred_p1_id': pred_p1_id,
            'pred_p2_id': pred_p2_id,
            'id_match_p1': id_match_p1,
            'id_match_p2': id_match_p2,
            'id_match_both': id_match_both,
            'dist_player_id': dist_player_id,
            'markov_player_id': markov_player_id or '',
            'markov_side': markov_side,
            'window_idx': i + 1,
            'window_end': (i + 1) * 100,
            'parsed_rounds': data.get('llm_simulation', {}).get('parsed_rounds', data.get('parsed_rounds')),
            'ce': ce,
            'mse': mse,
            'exact_match': exact_match,
            'strict_exact_match_all': strict_exact_match,
            'strict_exact_match_p1': strict_exact_match if markov_side == 'p1' else '',
            'strict_exact_match_p2': strict_exact_match if markov_side == 'p2' else '',
            'markov_id_match': int((pred_p1_id == p1_id) if markov_side == 'p1' else ((pred_p2_id == p2_id) if markov_side == 'p2' else 0)),
            'baseline_rock_pct': baseline[0] * 100.0,
            'baseline_paper_pct': baseline[1] * 100.0,
            'baseline_scissors_pct': baseline[2] * 100.0,
            'window_rock_pct': true[0] * 100.0,
            'window_paper_pct': true[1] * 100.0,
            'window_scissors_pct': true[2] * 100.0,
        })

    return rows


def main():
    root = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'generation')
    out_path = os.path.join(os.getcwd(), 'exp2(generation_blind)', 'analysis_results', 'slump_metrics.csv')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fieldnames = [
        'file', 'model', 'player1_id', 'player2_id', 'combo_type', 'pred_p1_id', 'pred_p2_id', 'id_match_p1', 'id_match_p2', 'id_match_both', 'dist_player_id', 'markov_player_id', 'markov_side', 'window_idx', 'window_end', 'parsed_rounds',
        'ce', 'mse',
        'exact_match',
        'strict_exact_match_all', 'strict_exact_match_p1', 'strict_exact_match_p2', 'markov_id_match',
        'baseline_rock_pct', 'baseline_paper_pct', 'baseline_scissors_pct',
        'window_rock_pct', 'window_paper_pct', 'window_scissors_pct',
    ]

    rows_written = 0
    with open(out_path, 'w', newline='', encoding='utf-8') as csvf:
        writer = csv.DictWriter(csvf, fieldnames=fieldnames)
        writer.writeheader()

        for dirpath, dirnames, filenames in os.walk(root):
            for fn in filenames:
                if not fn.endswith('.json'):
                    continue
                p = os.path.join(dirpath, fn)
                try:
                    rows = process_file(p)
                except Exception as e:
                    print(f"WARN: failed to process {p}: {e}")
                    continue

                for r in rows:
                    writer.writerow(r)
                    rows_written += 1

    print(f"WROTE: {out_path} ({rows_written} rows)")


if __name__ == '__main__':
    main()
