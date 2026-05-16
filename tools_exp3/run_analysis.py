"""
Exp3 Analysis Pipeline
Reads all generation JSONs for a given markov-set and produces:
  1. Identification metrics  — overall / Markov / non-Markov accuracy, confusion matrix
  2. Generation metrics      — overlap rate, strict exact match, CE, MSE
     (consistent with exp2/compute_slump_metrics.py definitions)

CE/MSE:  compare each LLM window distribution vs the context baseline distribution
         (measures whether LLM maintains the correct distribution over time)
Overlap: % of deterministic Markov rounds where LLM correctly applied the rule,
         using LLM's own simulation as state (internal consistency check)
Strict:  1 if ALL deterministic rounds in a window follow the rule, else 0

Usage:
  python tools_exp3/run_analysis.py --markov-set qrs
  python tools_exp3/run_analysis.py --markov-set tuv --model deepseek-reasoner
"""
import os
import sys
import json
import math
import argparse
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

EXP3_ROOT   = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
GEN_ROOT    = os.path.join(EXP3_ROOT, 'generation')
RESULT_ROOT = os.path.join(EXP3_ROOT, 'analysis_results')
PLOT_ROOT   = os.path.join(EXP3_ROOT, 'plots')

MARKOV_SETS = {
    'xyz': {'X', 'Y', 'Z'},
    'qrs': {'Q', 'R', 'S'},
    'tuv': {'T', 'U', 'V'},
    'xyz_opp': {'x', 'y', 'z'},
}

VALID_IDS = {
    'xyz': list('ABCDEFGHIJKLMNOPXYZ'),
    'qrs': list('ABCDEFGHIJKLMNOPQRS'),
    'tuv': list('ABCDEFGHIJKLMNOPTUV'),
    'xyz_opp': list('ABCDEFGHIJKLMNOPxyz'),
}

# ---------------------------------------------------------------------------
# Markov rule checker
# ---------------------------------------------------------------------------

_BEAT = {'Rock': 'Paper', 'Paper': 'Scissors', 'Scissors': 'Rock'}
_LOSE = {'Rock': 'Scissors', 'Paper': 'Rock', 'Scissors': 'Paper'}
_ALL  = ('Rock', 'Paper', 'Scissors')


def _expected_xyz(strategy, opp_prev):
    """Expected action for X/Y/Z given opponent's previous move. Always deterministic."""
    if opp_prev not in _ALL:
        return None
    if strategy == 'X':
        return _BEAT[opp_prev]
    if strategy == 'Y':
        return _LOSE[opp_prev]
    if strategy == 'Z':
        return opp_prev
    return None


def _expected_qrs(strategy, self_prev, opp_prev):
    """Expected action for Q/R/S.
    
    Returns:
    - 'set' tuple: valid actions (for draw → any is correct)
    - single action: deterministic choice (non-draw)
    - None: if inputs invalid
    """
    if self_prev not in _ALL or opp_prev not in _ALL:
        return None
    
    # draw case: any action is valid
    if self_prev == opp_prev:
        return _ALL
    
    missing = [a for a in _ALL if a != self_prev and a != opp_prev]
    if len(missing) != 1:
        return None
    
    m = missing[0]
    if strategy == 'Q':
        return m
    if strategy == 'R':
        return _BEAT[m]
    if strategy == 'S':
        return _LOSE[m]
    return None


def _expected_tuv(strategy, s_t2, s_t1, o_t2, o_t1):
    """Expected action for T/U/V.
    
    Returns:
    - 'set' tuple: valid actions (for 0 or 2 missing → any in set is correct)
    - single action: deterministic choice (for 1 missing)
    - None: if inputs invalid
    """
    for a in (s_t2, s_t1, o_t2, o_t1):
        if a not in _ALL:
            return None
    appeared = {s_t2, s_t1, o_t2, o_t1}
    missing  = [a for a in _ALL if a not in appeared]
    
    # len(missing) == 0: all 3 actions appeared → any choice is correct
    if len(missing) == 0:
        return _ALL  # any action is valid
    
    # len(missing) == 1: deterministic
    if len(missing) == 1:
        m = missing[0]
        if strategy == 'T':
            return m
        if strategy == 'U':
            return _BEAT[m]
        if strategy == 'V':
            return _LOSE[m]
    
    # len(missing) == 2: 50/50 choice
    if len(missing) == 2:
        if strategy == 'T':
            return tuple(missing)  # either missing[0] or missing[1]
        if strategy == 'U':
            return tuple(_BEAT[m] for m in missing)  # either beat(missing[0]) or beat(missing[1])
        if strategy == 'V':
            return tuple(_LOSE[m] for m in missing)  # either lose(missing[0]) or lose(missing[1])
    
    return None


def _expected_xyz_opp(strategy, opp_t2, opp_t1):
    """Expected action for x/y/z. Returns None when opp played same action twice (stochastic)."""
    if opp_t2 not in _ALL or opp_t1 not in _ALL:
        return None
    appeared = {opp_t2, opp_t1}
    missing = [a for a in _ALL if a not in appeared]
    if len(missing) != 1:   # opp_t2 == opp_t1 → 2 missing → stochastic
        return None
    m = missing[0]
    if strategy == 'x':
        return m
    if strategy == 'y':
        return _BEAT[m]
    if strategy == 'z':
        return _LOSE[m]


def compute_markov_overlap_by_window(markov_id, markov_moves, opponent_moves, window=100):
    """
    Returns (overlap_by_window, strict_by_window).
    overlap: fraction of deterministic rounds correctly predicted per window.
    strict:  1 if ALL deterministic rounds in a window are correct, else 0.
    Rounds with no deterministic expected action are excluded.
    """
    total = min(len(markov_moves), len(opponent_moves))

    flags = []   # None = skip; 1 = correct; 0 = wrong

    if markov_id in ('X', 'Y', 'Z'):
        flags.append(None)   # round 1 has no history
        for i in range(1, total):
            exp = _expected_xyz(markov_id, opponent_moves[i - 1])
            flags.append(None if exp is None else int(markov_moves[i] == exp))

    elif markov_id in ('Q', 'R', 'S'):
        flags.append(None)   # round 1
        for i in range(1, total):
            exp = _expected_qrs(markov_id, markov_moves[i - 1], opponent_moves[i - 1])
            if exp is None:
                flags.append(None)
            elif isinstance(exp, tuple):
                # Stochastic case: any action in allowed set is correct
                flags.append(1 if markov_moves[i] in exp else 0)
            else:
                # Deterministic case: exact match required
                flags.append(1 if markov_moves[i] == exp else 0)

    elif markov_id in ('T', 'U', 'V'):
        flags.append(None)   # round 1
        flags.append(None)   # round 2
        for i in range(2, total):
            exp = _expected_tuv(markov_id,
                                markov_moves[i - 2], markov_moves[i - 1],
                                opponent_moves[i - 2], opponent_moves[i - 1])
            if exp is None:
                flags.append(None)
            elif isinstance(exp, tuple):
                flags.append(1 if markov_moves[i] in exp else 0)
            else:
                flags.append(1 if markov_moves[i] == exp else 0)

    elif markov_id in ('x', 'y', 'z'):
        flags.append(None)   # round 1
        flags.append(None)   # round 2
        for i in range(2, total):
            exp = _expected_xyz_opp(markov_id, opponent_moves[i - 2], opponent_moves[i - 1])
            flags.append(None if exp is None else int(markov_moves[i] == exp))

    overlap_by_win = []
    strict_by_win  = []
    for start in range(0, total, window):
        chunk = flags[start:start + window]
        evals = [x for x in chunk if x is not None]
        if not evals:
            overlap_by_win.append(None)
            strict_by_win.append(None)
        else:
            overlap_by_win.append(sum(evals) / len(evals))
            strict_by_win.append(1 if all(x == 1 for x in evals) else 0)

    return overlap_by_win, strict_by_win


# ---------------------------------------------------------------------------
# CE / MSE helpers
# ---------------------------------------------------------------------------

def _pct_vec(stats: dict):
    """Extract [rock_pct, paper_pct, scissors_pct] as fractions (0-1)."""
    def _f(k):
        try:
            return float(stats.get(k, 0)) / 100.0
        except Exception:
            return 0.0
    return [_f('rock_pct'), _f('paper_pct'), _f('scissors_pct')]


def compute_ce(window_dist, baseline_dist, eps=1e-12):
    """H(window, baseline) = -Σ window_i * log(baseline_i)"""
    return -sum(w * math.log(max(b, eps)) for w, b in zip(window_dist, baseline_dist))


def compute_mse(window_dist, baseline_dist):
    return sum((w - b) ** 2 for w, b in zip(window_dist, baseline_dist)) / 3


def ce_mse_by_window(baseline_dist, llm_window_stats, max_windows=10):
    """Compute CE and MSE for each of the first max_windows LLM-generated windows."""
    results = []
    for ws in llm_window_stats[:max_windows]:
        w_dist = _pct_vec(ws)
        results.append((compute_ce(w_dist, baseline_dist),
                         compute_mse(w_dist, baseline_dist)))
    return results


# ---------------------------------------------------------------------------
# Load JSON files
# ---------------------------------------------------------------------------

def load_all_jsons(markov_set: str, model_filter: str = None) -> list:
    root = os.path.join(GEN_ROOT, markov_set)
    records = []
    if not os.path.isdir(root):
        print(f"ERROR: {root} not found")
        return records
    for model_dir in os.listdir(root):
        if model_filter and model_dir != model_filter:
            continue
        model_path = os.path.join(root, model_dir)
        if not os.path.isdir(model_path):
            continue
        for ctx_dir in os.listdir(model_path):
            if ctx_dir != 'ctx1000_sim1500':
                continue
            ctx_path = os.path.join(model_path, ctx_dir)
            if not os.path.isdir(ctx_path):
                continue
            for type_dir in os.listdir(ctx_path):
                type_path = os.path.join(ctx_path, type_dir)
                if not os.path.isdir(type_path):
                    continue
                combo_type = (1 if 'non_markov' in type_dir
                              else 2 if 'markov_p1' in type_dir
                              else 3)
                for fname in os.listdir(type_path):
                    if not fname.endswith('.json'):
                        continue
                    fpath = os.path.join(type_path, fname)
                    try:
                        with open(fpath, encoding='utf-8') as f:
                            data = json.load(f)
                        data['_combo_type'] = combo_type
                        data['_model_dir']  = model_dir
                        data['_file_idx']   = len(records)
                        records.append(data)
                    except Exception as e:
                        print(f"  ⚠️  {fpath}: {e}")
    print(f"Loaded {len(records)} JSON files.")
    return records


# ---------------------------------------------------------------------------
# Compute metrics per record  (mirrors exp2/compute_slump_metrics.py logic)
# ---------------------------------------------------------------------------

def compute_record_metrics(rec: dict, markov_set: str) -> list:
    """
    Returns a list of window-level row dicts (one per 100-round window).
    Empty list if the record has no usable data.
    """
    markov = MARKOV_SETS[markov_set]
    combo_type = rec.get('_combo_type', 0)
    p1_id      = rec.get('player1_id', '')
    p2_id      = rec.get('player2_id', '')
    model      = rec.get('model', rec.get('_model_dir', ''))
    file_idx   = rec.get('_file_idx', 0)

    llm = rec.get('llm_simulation', {})
    if not llm:
        return []

    pred_p1 = (llm.get('p1_identity') or '').strip()
    pred_p2 = (llm.get('p2_identity') or '').strip()

    # --- Identification flags ---
    p1_correct    = int(pred_p1 == p1_id) if pred_p1 else None
    p2_correct    = int(pred_p2 == p2_id) if pred_p2 else None
    both_correct  = int(p1_correct == 1 and p2_correct == 1) if (p1_correct is not None and p2_correct is not None) else None

    p1_is_markov  = p1_id in markov
    p2_is_markov  = p2_id in markov

    if combo_type == 2:         # p1 is Markov
        markov_correct    = p1_correct
        nonmarkov_correct = p2_correct
    elif combo_type == 3:       # p2 is Markov
        markov_correct    = p2_correct
        nonmarkov_correct = p1_correct
    else:                       # type1: no Markov
        markov_correct    = None
        nonmarkov_correct = both_correct

    # --- CE / MSE: baseline from context, per-window from LLM ---
    ctx = rec.get('context', {})
    p1_ctx_stats = ctx.get('p1_stats', {})
    p2_ctx_stats = ctx.get('p2_stats', {})

    llm_p1_wins = llm.get('p1_window_stats', [])
    llm_p2_wins = llm.get('p2_window_stats', [])

    if p1_is_markov and not p2_is_markov:
        # CE/MSE on the non-Markov side (p2)
        baseline     = _pct_vec(p2_ctx_stats)
        ce_mse_wins  = ce_mse_by_window(baseline, llm_p2_wins)
        dist_side    = 'p2'
    elif p2_is_markov and not p1_is_markov:
        baseline     = _pct_vec(p1_ctx_stats)
        ce_mse_wins  = ce_mse_by_window(baseline, llm_p1_wins)
        dist_side    = 'p1'
    else:
        # type1: average p1 and p2
        b1 = _pct_vec(p1_ctx_stats)
        b2 = _pct_vec(p2_ctx_stats)
        baseline     = [(b1[i] + b2[i]) / 2 for i in range(3)]
        w1 = ce_mse_by_window(b1, llm_p1_wins)
        w2 = ce_mse_by_window(b2, llm_p2_wins)
        ce_mse_wins  = [((w1[i][0] + w2[i][0]) / 2,
                          (w1[i][1] + w2[i][1]) / 2)
                         for i in range(min(len(w1), len(w2)))]
        dist_side    = 'both'

    # --- Overlap / Strict: internal Markov rule consistency ---
    llm_p1_moves = llm.get('p1_trajectory', '').split()
    llm_p2_moves = llm.get('p2_trajectory', '').split()

    overlap_by_win = []
    strict_by_win  = []
    markov_side    = ''
    markov_player_id = ''

    if p1_is_markov and not p2_is_markov:
        markov_player_id = p1_id
        markov_side      = 'p1'
        overlap_by_win, strict_by_win = compute_markov_overlap_by_window(
            p1_id, llm_p1_moves, llm_p2_moves)
    elif p2_is_markov and not p1_is_markov:
        markov_player_id = p2_id
        markov_side      = 'p2'
        overlap_by_win, strict_by_win = compute_markov_overlap_by_window(
            p2_id, llm_p2_moves, llm_p1_moves)

    # --- Build window rows ---
    n_windows = min(len(ce_mse_wins), 10)
    rows = []
    for w_idx in range(n_windows):
        ce_v, mse_v = ce_mse_wins[w_idx]
        overlap_v = overlap_by_win[w_idx] if w_idx < len(overlap_by_win) else None
        strict_v  = strict_by_win[w_idx]  if w_idx < len(strict_by_win)  else None

        # markov_id_match: did the LLM correctly identify the Markov player?
        if markov_side == 'p1':
            markov_id_match = int(pred_p1 == p1_id) if pred_p1 else None
        elif markov_side == 'p2':
            markov_id_match = int(pred_p2 == p2_id) if pred_p2 else None
        else:
            markov_id_match = None

        rows.append({
            'model':             model,
            'file_idx':          file_idx,
            'markov_set':        markov_set,
            'combo_type':        combo_type,
            'player1_id':        p1_id,
            'player2_id':        p2_id,
            'pred_p1_id':        pred_p1,
            'pred_p2_id':        pred_p2,
            'p1_correct':        p1_correct,
            'p2_correct':        p2_correct,
            'both_correct':      both_correct,
            'markov_correct':    markov_correct,
            'nonmarkov_correct': nonmarkov_correct,
            'markov_side':       markov_side,
            'markov_player_id':  markov_player_id,
            'markov_id_match':   markov_id_match,
            'dist_side':         dist_side,
            'window_idx':        w_idx + 1,
            'window_end':        (w_idx + 1) * 100,
            'ce':                ce_v,
            'mse':               mse_v,
            'overlap_rate':      overlap_v,     # exact_match in exp2 terminology
            'strict_exact':      strict_v,      # strict_exact_match in exp2 terminology
        })
    return rows


# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------

def save_csv(rows: list, path: str):
    if not rows:
        return
    fields = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  CSV → {path}")


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _avg(vals):
    vs = [v for v in vals if v is not None]
    return sum(vs) / len(vs) if vs else None


def _rate(rows, key):
    vs = [r[key] for r in rows if r.get(key) is not None]
    return (sum(vs) / len(vs), len(vs)) if vs else (None, 0)


def _by_model(rows, key, model_list):
    d = defaultdict(list)
    for r in rows:
        v = r.get(key)
        if v is not None:
            d[r['model']].append(v)
    return {m: (_avg(d[m]), len(d[m])) for m in model_list}


# ---------------------------------------------------------------------------
# Plot 1 — Identification accuracy
# ---------------------------------------------------------------------------

def plot_identification(rows, outdir, markov_set):
    # de-duplicate to one row per file (take first window)
    seen, file_rows = set(), []
    for r in rows:
        key = (r['model'], r['file_idx'])
        if key not in seen:
            seen.add(key)
            file_rows.append(r)

    models = sorted({r['model'] for r in file_rows})
    if not models:
        return

    markov_rows    = [r for r in file_rows if r['combo_type'] in (2, 3)]
    nonmarkov_rows = file_rows   # nonmarkov_correct is defined for all types

    x = np.arange(len(models))
    w = 0.25

    fig, ax = plt.subplots(figsize=(max(10, len(models) * 2.5), 6))

    def get_vals(subset, key):
        rates, ns = [], []
        for m in models:
            vs = [r[key] for r in subset if r['model'] == m and r.get(key) is not None]
            rates.append(sum(vs) / len(vs) if vs else 0)
            ns.append(len(vs))
        return rates, ns

    ov_r, ov_n   = get_vals(file_rows,    'both_correct')
    mk_r, mk_n   = get_vals(markov_rows,  'markov_correct')
    nm_r, nm_n   = get_vals(nonmarkov_rows, 'nonmarkov_correct')

    b0 = ax.bar(x - w, ov_r, w, label='Overall (both correct)', color='#2E86DE')
    b1 = ax.bar(x,     mk_r, w, label=f'{markov_set.upper()} (Markov) correct', color='#E74C3C')
    b2 = ax.bar(x + w, nm_r, w, label='Non-Markov correct', color='#27AE60')

    for bars, vals, ns in [(b0, ov_r, ov_n), (b1, mk_r, mk_n), (b2, nm_r, nm_n)]:
        for bar, v, n in zip(bars, vals, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{v:.2f}\nn={n}', ha='center', va='bottom', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.25)
    ax.set_ylabel('Accuracy')
    ax.set_title(f'Identification Accuracy — {markov_set.upper()}')
    ax.axhline(1/3, color='purple', linestyle='--', linewidth=1.2, label='Random (1/3)')
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'identification_accuracy.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 2 — Confusion matrix
# ---------------------------------------------------------------------------

def plot_confusion(rows, outdir, markov_set):
    ids    = VALID_IDS[markov_set]
    id2idx = {v: i for i, v in enumerate(ids)}
    models = sorted({r['model'] for r in rows})
    if not models:
        return

    # de-duplicate to file level
    seen, file_rows = set(), []
    for r in rows:
        key = (r['model'], r['file_idx'])
        if key not in seen:
            seen.add(key)
            file_rows.append(r)

    ncols = min(len(models), 2)
    nrows = math.ceil(len(models) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 9, nrows * 8))
    axes = np.array(axes).flatten()

    for ax_idx, model in enumerate(models):
        ax  = axes[ax_idx]
        mat = np.zeros((len(ids), len(ids)), dtype=int)
        for r in file_rows:
            if r['model'] != model:
                continue
            for true_id, pred_id in [(r['player1_id'], r['pred_p1_id']),
                                      (r['player2_id'], r['pred_p2_id'])]:
                ti = id2idx.get(true_id)
                pi = id2idx.get(pred_id)
                if ti is not None and pi is not None:
                    mat[ti][pi] += 1

        im = ax.imshow(mat, cmap='Blues')
        ax.set_xticks(range(len(ids)))
        ax.set_yticks(range(len(ids)))
        ax.set_xticklabels(ids, fontsize=8)
        ax.set_yticklabels(ids, fontsize=8)
        ax.set_xlabel('Predicted', fontsize=9)
        ax.set_ylabel('True', fontsize=9)
        ax.set_title(f'Confusion Matrix — {model}', fontsize=10)
        fig.colorbar(im, ax=ax)
        mx = mat.max() if mat.max() else 1
        for i in range(len(ids)):
            for j in range(len(ids)):
                if mat[i, j] > 0:
                    ax.text(j, i, str(mat[i, j]), ha='center', va='center', fontsize=6,
                            color='white' if mat[i, j] > mx * 0.6 else 'black')

    for ax in axes[len(models):]:
        ax.set_visible(False)

    plt.suptitle(f'Prediction Confusion — {markov_set.upper()}', fontsize=12)
    plt.tight_layout()
    path = os.path.join(outdir, 'confusion_matrix.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 2b — Binary confusion matrices  (Markov/NonMarkov class + identity)
# ---------------------------------------------------------------------------

def _draw_2x2(ax, tp, fp, fn, tn, title=''):
    mat = np.array([[tn, fp], [fn, tp]], dtype=float)
    total = mat.sum()
    ax.imshow(mat, cmap='Blues', vmin=0, vmax=max(total, 1))
    for i, (row_lbl, col_lbl) in enumerate([('Non-Markov', 'Non-Markov'), ('Markov', 'Markov')]):
        for j in range(2):
            v = int(mat[i, j])
            pct = v / total * 100 if total > 0 else 0
            color = 'white' if (mat[i, j] / max(total, 1)) > 0.55 else 'black'
            lbl = [['TN', 'FP'], ['FN', 'TP']][i][j]
            ax.text(j, i, f"{lbl}\n{v}\n({pct:.0f}%)",
                    ha='center', va='center', fontsize=10, fontweight='bold', color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Non-Markov', 'Markov'])
    ax.set_yticklabels(['Non-Markov', 'Markov'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    if title:
        ax.set_title(title, fontsize=10, fontweight='bold')


def plot_confusion_binary(file_rows, outdir, markov_set):
    """
    confusion_markov.png   — binary Markov class detection (TP/FP/FN/TN)
    confusion_identity.png — same but TP requires exact player ID match
    """
    markov  = MARKOV_SETS[markov_set]
    models  = sorted({r['model'] for r in file_rows})
    if not models:
        return

    for fname, title_prefix, need_exact in [
        ('confusion_markov.png',   'Markov Class Detection',           False),
        ('confusion_identity.png', 'Markov Detection + Exact ID (TP)', True),
    ]:
        n   = len(models)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        for col, model in enumerate(models):
            tp = fp = fn = tn = 0
            for r in file_rows:
                if r['model'] != model:
                    continue
                for true_id, pred_id in [(r['player1_id'], r['pred_p1_id']),
                                          (r['player2_id'], r['pred_p2_id'])]:
                    if not pred_id:
                        continue
                    t_m   = true_id in markov
                    p_m   = pred_id in markov
                    exact = (pred_id == true_id)
                    if need_exact:
                        tp += int(t_m and exact)
                        fp += int((not t_m) and p_m)
                        fn += int(t_m and not exact)
                        tn += int((not t_m) and not p_m)
                    else:
                        tp += int(t_m and p_m)
                        fp += int((not t_m) and p_m)
                        fn += int(t_m and not p_m)
                        tn += int((not t_m) and not p_m)
            _draw_2x2(axes[0][col], tp, fp, fn, tn, title=model)
        fig.suptitle(f'{title_prefix} — {markov_set.upper()}', fontsize=13)
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 2c — acc_vs_rounds_overall.png  +  accuracy_summary.csv
# ---------------------------------------------------------------------------

def plot_acc_overall(file_rows, outdir, markov_set):
    """Mirrors exp1 acc_vs_rounds_overall.png — overall/markov/nonmarkov acc per model."""
    models       = sorted({r['model'] for r in file_rows})
    markov_rows  = [r for r in file_rows if r['combo_type'] in (2, 3)]
    x = np.arange(len(models))
    w = 0.25

    def get_vals(subset, key):
        rates, ns = [], []
        for m in models:
            vs = [r[key] for r in subset if r['model'] == m and r.get(key) is not None]
            rates.append(sum(vs) / len(vs) if vs else 0)
            ns.append(len(vs))
        return rates, ns

    ov_r, ov_n = get_vals(file_rows,   'both_correct')
    mk_r, mk_n = get_vals(markov_rows, 'markov_correct')
    nm_r, nm_n = get_vals(file_rows,   'nonmarkov_correct')

    fig, ax = plt.subplots(figsize=(max(8, len(models) * 2.5), 5))
    b0 = ax.bar(x - w, ov_r, w, label='Overall',    color='#2E86DE')
    b1 = ax.bar(x,     mk_r, w, label=f'{markov_set.upper()} (Markov)', color='#E74C3C')
    b2 = ax.bar(x + w, nm_r, w, label='Non-Markov', color='#27AE60')
    for bars, vals, ns in [(b0, ov_r, ov_n), (b1, mk_r, mk_n), (b2, nm_r, nm_n)]:
        for bar, v, n in zip(bars, vals, ns):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{v:.2f}\nn={n}', ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 1.3); ax.set_ylabel('Accuracy')
    ax.set_title(f'Overall Identification Accuracy — {markov_set.upper()}')
    ax.axhline(1/3, color='purple', linestyle='--', linewidth=1.2, label='Random (1/3)')
    ax.legend(fontsize=8); ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(outdir, 'acc_vs_rounds_overall.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot → {path}')


def save_accuracy_csv(file_rows, result_root, markov_set):
    models      = sorted({r['model'] for r in file_rows})
    markov_rows = [r for r in file_rows if r['combo_type'] in (2, 3)]
    out_rows    = []

    def acc_row(label, subset, mk_sub):
        ov = [r['both_correct']     for r in subset if r.get('both_correct')     is not None]
        mk = [r['markov_correct']   for r in mk_sub  if r.get('markov_correct')   is not None]
        nm = [r['nonmarkov_correct'] for r in subset if r.get('nonmarkov_correct') is not None]
        return {
            'model':         label,
            'n_files':       len(subset),
            'overall_acc':   f'{sum(ov)/len(ov):.4f}' if ov else '',
            'overall_n':     len(ov),
            'markov_acc':    f'{sum(mk)/len(mk):.4f}' if mk else '',
            'markov_n':      len(mk),
            'nonmarkov_acc': f'{sum(nm)/len(nm):.4f}' if nm else '',
            'nonmarkov_n':   len(nm),
        }

    for m in models:
        out_rows.append(acc_row(m,
                                [r for r in file_rows   if r['model'] == m],
                                [r for r in markov_rows if r['model'] == m]))
    out_rows.append(acc_row('ALL', file_rows, markov_rows))

    os.makedirs(result_root, exist_ok=True)
    path = os.path.join(result_root, f'{markov_set}_accuracy_summary.csv')
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader(); writer.writerows(out_rows)
    print(f'  CSV  → {path}')


# ---------------------------------------------------------------------------
# Plot 4b — Identity condition detail  (mirrors exp2/plot_identity_condition.py)
# ---------------------------------------------------------------------------

def _annotate_bars(ax, bars, vals, ns, y_top=1.0):
    offset = max(y_top * 0.03, 0.01)
    for bar, v, n in zip(bars, vals, ns):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=8)


def plot_identity_condition_detail(all_rows, file_rows, outdir, markov_set):
    """
    identity_condition_distribution.png        — CE/MSE split by dist-player ID correct/wrong
    identity_condition_distribution_by_model.png
    identity_condition_markov.png              — overlap/strict split by Markov ID correct/wrong
    identity_condition_markov_by_model.png
    """
    models = sorted({r['model'] for r in all_rows})

    # ── Distribution side (CE / MSE) ──────────────────────────────────────
    mstats_d = {m: {'ce_c': [], 'ce_w': [], 'mse_c': [], 'mse_w': []} for m in models}
    ce_c, ce_w, mse_c, mse_w = [], [], [], []

    for row in all_rows:
        m   = row.get('model', '')
        cid = row.get('nonmarkov_correct')    # dist player correctly identified?
        ce  = row.get('ce');  mse = row.get('mse')
        if cid is None or m not in mstats_d:
            continue
        ce_v = None if ce is None else float(ce)
        mse_v = None if mse is None else float(mse)
        bucket = 'c' if cid == 1 else 'w'
        if ce_v is not None:
            mstats_d[m][f'ce_{bucket}'].append(ce_v)
            (ce_c if bucket == 'c' else ce_w).append(ce_v)
        if mse_v is not None:
            mstats_d[m][f'mse_{bucket}'].append(mse_v)
            (mse_c if bucket == 'c' else mse_w).append(mse_v)

    def _avg_l(lst): return sum(lst) / len(lst) if lst else 0.0

    for fname, overall_fig_data, by_model_fig_data, title1, title2, ylabel1, ylabel2, ylim in [
        (
            'identity_condition_distribution.png',
            ([_avg_l(ce_c), _avg_l(ce_w)], [len(ce_c), len(ce_w)],
             [_avg_l(mse_c), _avg_l(mse_w)], [len(mse_c), len(mse_w)]),
            (mstats_d, 'ce_c', 'ce_w', 'mse_c', 'mse_w'),
            'Distribution Player — CE',  'Distribution Player — MSE',
            'CE (lower = better)', 'MSE (lower = better)', None,
        ),
    ]:
        groups = ['Identity Correct', 'Identity Wrong']
        ce_vals, ce_ns, mse_vals, mse_ns = overall_fig_data
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        b0 = axes[0].bar(groups, ce_vals,  color=['#2E86DE', '#E74C3C'])
        b1 = axes[1].bar(groups, mse_vals, color=['#2E86DE', '#E74C3C'])
        axes[0].set_title(title1); axes[0].set_ylabel(ylabel1); axes[0].grid(axis='y', alpha=0.3)
        axes[1].set_title(title2); axes[1].set_ylabel(ylabel2); axes[1].grid(axis='y', alpha=0.3)
        _annotate_bars(axes[0], b0, ce_vals,  ce_ns)
        _annotate_bars(axes[1], b1, mse_vals, mse_ns)
        plt.tight_layout()
        path = os.path.join(outdir, fname)
        fig.savefig(path, dpi=150); plt.close(fig)
        print(f'  Plot → {path}')

    x    = np.arange(len(models)); bw = 0.35
    ce_c_m  = [_avg_l(mstats_d[m]['ce_c'])  for m in models]
    ce_w_m  = [_avg_l(mstats_d[m]['ce_w'])  for m in models]
    mse_c_m = [_avg_l(mstats_d[m]['mse_c']) for m in models]
    mse_w_m = [_avg_l(mstats_d[m]['mse_w']) for m in models]
    ce_cn_m  = [len(mstats_d[m]['ce_c'])  for m in models]
    ce_wn_m  = [len(mstats_d[m]['ce_w'])  for m in models]
    mse_cn_m = [len(mstats_d[m]['mse_c']) for m in models]
    mse_wn_m = [len(mstats_d[m]['mse_w']) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    b00 = axes[0].bar(x - bw/2, ce_c_m,  bw, label='Identity Correct', color='#2E86DE')
    b01 = axes[0].bar(x + bw/2, ce_w_m,  bw, label='Identity Wrong',   color='#E74C3C')
    b10 = axes[1].bar(x - bw/2, mse_c_m, bw, label='Identity Correct', color='#2E86DE')
    b11 = axes[1].bar(x + bw/2, mse_w_m, bw, label='Identity Wrong',   color='#E74C3C')
    for ax, t, yl in [(axes[0], f'Distribution CE by Model — {markov_set.upper()}', 'CE'),
                      (axes[1], f'Distribution MSE by Model — {markov_set.upper()}', 'MSE')]:
        ax.set_title(t); ax.set_ylabel(yl); ax.grid(axis='y', alpha=0.3)
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right'); ax.legend(fontsize=8)
    _annotate_bars(axes[0], b00, ce_c_m,  ce_cn_m);  _annotate_bars(axes[0], b01, ce_w_m,  ce_wn_m)
    _annotate_bars(axes[1], b10, mse_c_m, mse_cn_m); _annotate_bars(axes[1], b11, mse_w_m, mse_wn_m)
    plt.tight_layout()
    path = os.path.join(outdir, 'identity_condition_distribution_by_model.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'  Plot → {path}')

    # ── Markov side (overlap / strict) ────────────────────────────────────
    mstats_mk = {m: {'ov_c': [], 'ov_w': [], 'st_c': [], 'st_w': []} for m in models}
    ov_c, ov_w, st_c, st_w = [], [], [], []

    for row in all_rows:
        m   = row.get('model', '')
        cid = row.get('markov_correct')
        ov  = row.get('overlap_rate'); st = row.get('strict_exact')
        if cid is None or m not in mstats_mk:
            continue
        bucket = 'c' if cid == 1 else 'w'
        if ov is not None:
            mstats_mk[m][f'ov_{bucket}'].append(float(ov))
            (ov_c if bucket == 'c' else ov_w).append(float(ov))
        if st is not None:
            mstats_mk[m][f'st_{bucket}'].append(float(st))
            (st_c if bucket == 'c' else st_w).append(float(st))

    groups   = ['Identity Correct', 'Identity Wrong']
    ov_vals  = [_avg_l(ov_c), _avg_l(ov_w)]
    st_vals  = [_avg_l(st_c), _avg_l(st_w)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    b0 = axes[0].bar(groups, ov_vals, color=['#2E86DE', '#E74C3C'])
    b1 = axes[1].bar(groups, st_vals, color=['#2E86DE', '#E74C3C'])
    axes[0].set_title(f'Markov Overlap Rate — {markov_set.upper()}')
    axes[0].set_ylabel('Overlap Rate'); axes[0].set_ylim(0, 1.1)
    axes[0].axhline(1/3, color='purple', linestyle='--', linewidth=1.2, label='Random (1/3)')
    axes[0].grid(axis='y', alpha=0.3); axes[0].legend(fontsize=8)
    axes[1].set_title(f'Markov Strict Exact — {markov_set.upper()}')
    axes[1].set_ylabel('Strict Exact Rate'); axes[1].set_ylim(0, 1.1)
    axes[1].grid(axis='y', alpha=0.3)
    _annotate_bars(axes[0], b0, ov_vals, [len(ov_c), len(ov_w)], y_top=1.0)
    _annotate_bars(axes[1], b1, st_vals, [len(st_c), len(st_w)], y_top=1.0)
    plt.tight_layout()
    path = os.path.join(outdir, 'identity_condition_markov.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'  Plot → {path}')

    ov_c_m = [_avg_l(mstats_mk[m]['ov_c']) for m in models]
    ov_w_m = [_avg_l(mstats_mk[m]['ov_w']) for m in models]
    st_c_m = [_avg_l(mstats_mk[m]['st_c']) for m in models]
    st_w_m = [_avg_l(mstats_mk[m]['st_w']) for m in models]
    ov_cn_m = [len(mstats_mk[m]['ov_c']) for m in models]
    ov_wn_m = [len(mstats_mk[m]['ov_w']) for m in models]
    st_cn_m = [len(mstats_mk[m]['st_c']) for m in models]
    st_wn_m = [len(mstats_mk[m]['st_w']) for m in models]

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    b00 = axes[0].bar(x - bw/2, ov_c_m, bw, label='Identity Correct', color='#2E86DE')
    b01 = axes[0].bar(x + bw/2, ov_w_m, bw, label='Identity Wrong',   color='#E74C3C')
    b10 = axes[1].bar(x - bw/2, st_c_m, bw, label='Identity Correct', color='#2E86DE')
    b11 = axes[1].bar(x + bw/2, st_w_m, bw, label='Identity Wrong',   color='#E74C3C')
    axes[0].set_title(f'Markov Overlap Rate by Model — {markov_set.upper()}')
    axes[0].set_ylabel('Rate'); axes[0].set_ylim(0, 1.1)
    axes[0].axhline(1/3, color='purple', linestyle='--', linewidth=1.2)
    axes[1].set_title(f'Markov Strict Exact by Model — {markov_set.upper()}')
    axes[1].set_ylabel('Rate'); axes[1].set_ylim(0, 1.1)
    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right')
        ax.grid(axis='y', alpha=0.3); ax.legend(fontsize=8)
    _annotate_bars(axes[0], b00, ov_c_m, ov_cn_m, y_top=1.0)
    _annotate_bars(axes[0], b01, ov_w_m, ov_wn_m, y_top=1.0)
    _annotate_bars(axes[1], b10, st_c_m, st_cn_m, y_top=1.0)
    _annotate_bars(axes[1], b11, st_w_m, st_wn_m, y_top=1.0)
    plt.tight_layout()
    path = os.path.join(outdir, 'identity_condition_markov_by_model.png')
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 3 — Generation metrics (CE, MSE, overlap) by model
# ---------------------------------------------------------------------------

def plot_generation(rows, outdir, markov_set):
    gen_rows = [r for r in rows if r.get('ce') is not None]
    if not gen_rows:
        print('  No generation metrics data.')
        return

    models = sorted({r['model'] for r in gen_rows})
    x  = np.arange(len(models))
    bw = 0.35

    # CE and MSE: all rows
    ce_avg  = {m: _avg([r['ce']  for r in gen_rows if r['model'] == m]) for m in models}
    mse_avg = {m: _avg([r['mse'] for r in gen_rows if r['model'] == m]) for m in models}
    ce_n    = {m: len([r for r in gen_rows if r['model'] == m and r.get('ce') is not None]) for m in models}

    # Overlap rate and strict: Markov rows only
    mk_rows = [r for r in gen_rows if r.get('overlap_rate') is not None]
    ov_avg  = {m: _avg([r['overlap_rate'] for r in mk_rows if r['model'] == m]) for m in models}
    st_avg  = {m: _avg([r['strict_exact'] for r in mk_rows if r['model'] == m and r.get('strict_exact') is not None]) for m in models}
    ov_n    = {m: len([r for r in mk_rows if r['model'] == m]) for m in models}

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    baseline_ce = -math.log(1/3)   # CE when LLM outputs uniform

    def _auto_ylim(vals, pad=0.25):
        top = max(vals) if vals else 1.0
        return (0, max(top * (1 + pad), 0.05))

    def bar_single(ax, avg_dict, n_dict, title, ylabel, ylim=None, baseline=None):
        vals = [avg_dict.get(m) or 0 for m in models]
        ns   = [n_dict.get(m, 0) for m in models]
        bars = ax.bar(x, vals, color='#2E86DE')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(*(ylim if ylim else _auto_ylim(vals + ([baseline] if baseline else []))))
        if baseline is not None:
            ax.axhline(baseline, color='purple', linestyle='--', linewidth=1.2, label=f'Baseline={baseline:.3f}')
            ax.legend(fontsize=8)
        for bar, v, n in zip(bars, vals, ns):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ax.get_ylim()[1]*0.01,
                    f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=7)

    def bar_pair(ax, avg1, avg2, n1, n2, title, ylabel, lbl1, lbl2, ylim=None, baseline=None):
        v1 = [avg1.get(m) or 0 for m in models]
        v2 = [avg2.get(m) or 0 for m in models]
        b0 = ax.bar(x - bw/2, v1, bw, label=lbl1, color='#2E86DE')
        b1 = ax.bar(x + bw/2, v2, bw, label=lbl2, color='#E74C3C')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(*(ylim if ylim else _auto_ylim(v1 + v2)))
        if baseline is not None:
            ax.axhline(baseline, color='purple', linestyle='--', linewidth=1.2)
        ax.legend(fontsize=8)
        for bars, vals, ns in [(b0, v1, [n1.get(m,0) for m in models]),
                                (b1, v2, [n2.get(m,0) for m in models])]:
            for bar, v, n in zip(bars, vals, ns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + ax.get_ylim()[1]*0.01,
                        f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=7)

    bar_single(axes[0], ce_avg, ce_n, f'CE — {markov_set.upper()}', 'CE (lower = better)', baseline=baseline_ce)
    bar_single(axes[1], mse_avg, ce_n, f'MSE — {markov_set.upper()}', 'MSE (lower = better)')
    bar_pair(axes[2], ov_avg, st_avg, ov_n, ov_n,
             f'Markov Rule Overlap — {markov_set.upper()}', 'Rate',
             'Overlap Rate', 'Strict Exact', ylim=(0, 1.25), baseline=1/3)

    plt.tight_layout()
    path = os.path.join(outdir, 'generation_metrics.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 4 — Generation metrics conditioned on identity correctness
# ---------------------------------------------------------------------------

def plot_identity_condition(rows, outdir, markov_set):
    gen_rows = [r for r in rows if r.get('ce') is not None and r.get('both_correct') is not None]
    if not gen_rows:
        return

    models = sorted({r['model'] for r in gen_rows})

    def split(key, cond_key='both_correct'):
        c = defaultdict(list)
        w = defaultdict(list)
        for r in gen_rows:
            v = r.get(key)
            b = r.get(cond_key)
            if v is None or b is None:
                continue
            (c if b else w)[r['model']].append(v)
        c_avg = {m: _avg(c[m]) for m in models}
        w_avg = {m: _avg(w[m]) for m in models}
        c_n   = {m: len(c[m]) for m in models}
        w_n   = {m: len(w[m]) for m in models}
        return c_avg, w_avg, c_n, w_n

    ce_c, ce_w, ce_cn, ce_wn   = split('ce')
    mse_c, mse_w, mse_cn, mse_wn = split('mse')
    ov_c, ov_w, ov_cn, ov_wn   = split('overlap_rate')

    x  = np.arange(len(models))
    bw = 0.35
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    def cond_bar(ax, ca, wa, cn, wn, title, ylabel, ylim=None, baseline=None):
        cv = [ca.get(m) or 0 for m in models]
        wv = [wa.get(m) or 0 for m in models]
        b0 = ax.bar(x - bw/2, cv, bw, label='Identity Correct', color='#2E86DE')
        b1 = ax.bar(x + bw/2, wv, bw, label='Identity Wrong',   color='#E74C3C')
        ax.set_xticks(x); ax.set_xticklabels(models, rotation=15, ha='right')
        ax.set_title(title); ax.set_ylabel(ylabel); ax.grid(axis='y', alpha=0.3)
        if ylim: ax.set_ylim(*ylim)
        if baseline is not None:
            ax.axhline(baseline, color='purple', linestyle='--', linewidth=1.2)
        ax.legend(fontsize=8)
        for bars, vals, ns in [(b0, cv, [cn.get(m,0) for m in models]),
                                (b1, wv, [wn.get(m,0) for m in models])]:
            for bar, v, n in zip(bars, vals, ns):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                        f'{v:.3f}\nn={n}', ha='center', va='bottom', fontsize=7)

    cond_bar(axes[0], ce_c, ce_w, ce_cn, ce_wn,
             f'CE by Identity — {markov_set.upper()}', 'CE')
    cond_bar(axes[1], mse_c, mse_w, mse_cn, mse_wn,
             f'MSE by Identity — {markov_set.upper()}', 'MSE')
    cond_bar(axes[2], ov_c, ov_w, ov_cn, ov_wn,
             f'Overlap Rate by Identity — {markov_set.upper()}', 'Rate',
             ylim=(0, 1.1), baseline=1/3)

    plt.tight_layout()
    path = os.path.join(outdir, 'identity_condition.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 5 — Per-model per-window line charts (mirrors plot_slump_metrics.py)
# ---------------------------------------------------------------------------

COMBO_COLORS = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}
COMBO_NAMES  = {1: 'Type1: NonMarkov vs NonMarkov',
                2: 'Type2: Markov_P1 vs NonMarkov',
                3: 'Type3: NonMarkov vs Markov_P2'}


def plot_per_model_by_window(rows, outdir, markov_set):
    """One figure per model: 2x2 line charts (CE, MSE, Overlap, Strict) vs window, by combo type."""
    gen_rows = [r for r in rows if r.get('ce') is not None]
    if not gen_rows:
        return

    models = sorted({r['model'] for r in gen_rows})
    windows = sorted({r['window_idx'] for r in gen_rows})

    for model in models:
        m_rows = [r for r in gen_rows if r['model'] == model]
        combos = sorted({r['combo_type'] for r in m_rows})

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()

        for c in combos:
            c_rows = [r for r in m_rows if r['combo_type'] == c]
            color  = COMBO_COLORS.get(c)
            label  = COMBO_NAMES.get(c, f'combo_{c}')

            ce_x, ce_y   = [], []
            mse_x, mse_y = [], []
            ov_x,  ov_y  = [], []
            st_x,  st_y  = [], []

            for w in windows:
                w_rows = [r for r in c_rows if r['window_idx'] == w]
                if not w_rows:
                    continue
                ce_vals  = [r['ce']          for r in w_rows if r.get('ce')          is not None]
                mse_vals = [r['mse']         for r in w_rows if r.get('mse')         is not None]
                ov_vals  = [r['overlap_rate'] for r in w_rows if r.get('overlap_rate') is not None]
                st_vals  = [r['strict_exact'] for r in w_rows if r.get('strict_exact') is not None]
                if ce_vals:
                    ce_x.append(w);  ce_y.append(sum(ce_vals)  / len(ce_vals))
                if mse_vals:
                    mse_x.append(w); mse_y.append(sum(mse_vals) / len(mse_vals))
                if ov_vals:
                    ov_x.append(w);  ov_y.append(sum(ov_vals)  / len(ov_vals))
                if st_vals:
                    st_x.append(w);  st_y.append(sum(st_vals)  / len(st_vals))

            if ce_x:  axes[0].plot(ce_x,  ce_y,  marker='o', linewidth=2, color=color, label=label)
            if mse_x: axes[1].plot(mse_x, mse_y, marker='o', linewidth=2, color=color, label=label)
            if ov_x:  axes[2].plot(ov_x,  ov_y,  marker='o', linewidth=2, color=color, label=label)
            if st_x:  axes[3].plot(st_x,  st_y,  marker='o', linewidth=2, color=color, label=label)

        for ax, title, ylabel, ylim, baseline in [
            (axes[0], f'{model} — CE vs Window',            'Cross-Entropy',  None,       None),
            (axes[1], f'{model} — MSE vs Window',           'MSE',            None,       None),
            (axes[2], f'{model} — Overlap Rate vs Window',  'Overlap Rate',   (0, 1.05),  1/3),
            (axes[3], f'{model} — Exact Match vs Window',   'Exact Match',    (0, 1.05),  None),
        ]:
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel('Window Index (100 rounds per window)')
            ax.set_ylabel(ylabel)
            ax.set_xticks(windows)
            ax.grid(True, alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
            if baseline is not None:
                ax.axhline(baseline, color='purple', linestyle='--', linewidth=1.5, label='Random (1/3)')
            handles, _ = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=9)

        plt.tight_layout()
        clean_model = model.replace('/', '_')
        path = os.path.join(outdir, f'{clean_model}_slump.png')
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Plot 6 — Cross-model per-window line charts (mirrors plot_model_comparison.py)
# ---------------------------------------------------------------------------

MODEL_COLORS = {
    'deepseek-chat':     '#1f77b4',
    'deepseek-reasoner': '#ff7f0e',
    'gpt-5':             '#2ca02c',
    'gpt-5-mini':        '#d62728',
}


def plot_model_comparison_by_window(rows, outdir, markov_set):
    """All models in one 2x2 figure: CE, MSE, Overlap, Strict vs window index."""
    gen_rows = [r for r in rows if r.get('ce') is not None]
    if not gen_rows:
        return

    models  = sorted({r['model'] for r in gen_rows})
    windows = sorted({r['window_idx'] for r in gen_rows})

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    for model in models:
        color   = MODEL_COLORS.get(model)
        m_rows  = [r for r in gen_rows if r['model'] == model]

        ce_x, ce_y   = [], []
        mse_x, mse_y = [], []
        ov_x,  ov_y  = [], []
        st_x,  st_y  = [], []

        for w in windows:
            w_rows   = [r for r in m_rows if r['window_idx'] == w]
            ce_vals  = [r['ce']           for r in w_rows if r.get('ce')           is not None]
            mse_vals = [r['mse']          for r in w_rows if r.get('mse')          is not None]
            ov_vals  = [r['overlap_rate']  for r in w_rows if r.get('overlap_rate') is not None]
            st_vals  = [r['strict_exact']  for r in w_rows if r.get('strict_exact') is not None]
            if ce_vals:
                ce_x.append(w);  ce_y.append(sum(ce_vals)  / len(ce_vals))
            if mse_vals:
                mse_x.append(w); mse_y.append(sum(mse_vals) / len(mse_vals))
            if ov_vals:
                ov_x.append(w);  ov_y.append(sum(ov_vals)  / len(ov_vals))
            if st_vals:
                st_x.append(w);  st_y.append(sum(st_vals)  / len(st_vals))

        if ce_x:  axes[0].plot(ce_x,  ce_y,  marker='o', linewidth=2, color=color, label=model)
        if mse_x: axes[1].plot(mse_x, mse_y, marker='o', linewidth=2, color=color, label=model)
        if ov_x:  axes[2].plot(ov_x,  ov_y,  marker='o', linewidth=2, color=color, label=model)
        if st_x:  axes[3].plot(st_x,  st_y,  marker='o', linewidth=2, color=color, label=model)

    titles = [
        (f'Model Comparison — CE ({markov_set.upper()})',           'Cross-Entropy',  None,      None),
        (f'Model Comparison — MSE ({markov_set.upper()})',          'MSE',            None,      None),
        (f'Model Comparison — Overlap Rate ({markov_set.upper()})', 'Overlap Rate',   (0, 1.05), 1/3),
        (f'Model Comparison — Exact Match ({markov_set.upper()})',  'Exact Match',    (0, 1.05), None),
    ]
    for ax, (title, ylabel, ylim, baseline) in zip(axes, titles):
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Window Index (100 rounds per window)')
        ax.set_ylabel(ylabel)
        ax.set_xticks(windows)
        ax.grid(True, alpha=0.3)
        if ylim:
            ax.set_ylim(*ylim)
        if baseline is not None:
            ax.axhline(baseline, color='purple', linestyle='--', linewidth=1.5, label='Random (1/3)')
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend(fontsize=9)

    plt.tight_layout()
    path = os.path.join(outdir, 'model_comparison_by_window.png')
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'  Plot → {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Exp3 Analysis Pipeline')
    parser.add_argument('--markov-set', required=True, choices=['xyz', 'qrs', 'tuv', 'xyz_opp'])
    parser.add_argument('--model', type=str, default=None)
    args = parser.parse_args()

    markov_set   = args.markov_set
    model_filter = args.model.replace('/', '_') if args.model else None

    print(f"\n{'='*70}")
    print(f"Exp3 Analysis — markov-set={markov_set.upper()}")
    print(f"{'='*70}")

    records = load_all_jsons(markov_set, model_filter)
    if not records:
        print('No data found. Run run_generation.py first.')
        return

    all_rows = []
    for rec in records:
        all_rows.extend(compute_record_metrics(rec, markov_set))
    print(f"Computed {len(all_rows)} window-level rows.")

    if not all_rows:
        print('No metrics computed.')
        return

    # ---- Console summary ----
    # De-dup to file level for identification metrics
    seen, file_rows = set(), []
    for r in all_rows:
        key = (r['model'], r['file_idx'])
        if key not in seen:
            seen.add(key)
            file_rows.append(r)

    markov_file_rows = [r for r in file_rows if r['combo_type'] in (2, 3)]
    ov_r, ov_n   = _rate(file_rows,        'both_correct')
    mk_r, mk_n   = _rate(markov_file_rows,  'markov_correct')
    nm_r, nm_n   = _rate(file_rows,         'nonmarkov_correct')
    gen_rows     = [r for r in all_rows if r.get('overlap_rate') is not None]
    ov_gen, ov_gn = _rate(gen_rows, 'overlap_rate')

    print(f"\n  Overall accuracy   : {ov_r*100:.1f}% (n={ov_n})" if ov_r is not None else "\n  Overall accuracy   : N/A")
    print(f"  Markov accuracy    : {mk_r*100:.1f}% (n={mk_n})" if mk_r is not None else "  Markov accuracy    : N/A")
    print(f"  Non-Markov accuracy: {nm_r*100:.1f}% (n={nm_n})" if nm_r is not None else "  Non-Markov accuracy: N/A")
    print(f"  Overlap rate (avg) : {ov_gen*100:.1f}% (n={ov_gn})" if ov_gen is not None else "  Overlap rate       : N/A")

    # ---- Save CSV ----
    os.makedirs(RESULT_ROOT, exist_ok=True)
    save_csv(all_rows, os.path.join(RESULT_ROOT, f'{markov_set}_metrics.csv'))

    # ---- Plots ----
    outdir = os.path.join(PLOT_ROOT, markov_set)
    os.makedirs(outdir, exist_ok=True)
    print('\n  Generating plots ...')
    plot_identification(file_rows, outdir, markov_set)
    plot_confusion(file_rows, outdir, markov_set)
    plot_confusion_binary(file_rows, outdir, markov_set)
    plot_acc_overall(file_rows, outdir, markov_set)
    save_accuracy_csv(file_rows, RESULT_ROOT, markov_set)
    plot_generation(all_rows, outdir, markov_set)
    plot_identity_condition(all_rows, outdir, markov_set)
    plot_identity_condition_detail(all_rows, file_rows, outdir, markov_set)
    plot_per_model_by_window(all_rows, outdir, markov_set)
    plot_model_comparison_by_window(all_rows, outdir, markov_set)

    print(f'\nDone. Plots → {outdir}')


if __name__ == '__main__':
    main()
