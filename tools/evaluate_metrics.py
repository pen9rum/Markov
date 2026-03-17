import argparse
import json
import random
from pathlib import Path


# =========================
# Player Definitions
# =========================

PLAYER_DISTS = {
    "A": (0.0, 0.0, 1.0),
    "B": (1.0, 0.0, 0.0),
    "C": (0.0, 1.0, 0.0),
    "D": (0.333, 0.333, 0.334),
    "E": (0.5, 0.5, 0.0),
    "F": (0.5, 0.0, 0.5),
    "G": (0.0, 0.5, 0.5),
    "H": (0.5, 0.25, 0.25),
    "I": (0.25, 0.5, 0.25),
    "J": (0.25, 0.25, 0.5),
    "K": (0.5, 0.333, 0.167),
    "L": (0.5, 0.167, 0.333),
    "M": (0.333, 0.5, 0.167),
    "N": (0.167, 0.5, 0.333),
    "O": (0.333, 0.167, 0.5),
    "P": (0.167, 0.333, 0.5),
}

MARKOV_PLAYERS = {"X", "Y", "Z"}


# =========================
# Utils
# =========================

def mean(values):
    return sum(values) / len(values) if values else None


def precision_recall_f1(tp, fp, fn):
    """Compute precision / recall / F1 safely."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    if precision is None or recall is None or (precision + recall) == 0:
        f1 = None
    else:
        f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def safe_min(values):
    """Return min of non-None values, or None if all None"""
    filtered = [v for v in values if v is not None]
    return min(filtered) if filtered else None


def safe_max(values):
    """Return max of non-None values, or None if all None"""
    filtered = [v for v in values if v is not None]
    return max(filtered) if filtered else None


def normalize_counts(counts):
    total = sum(counts)
    if total <= 0:
        return [1 / 3, 1 / 3, 1 / 3]
    return [x / total for x in counts]


def tv_distance(q, p):
    """
    Total Variation distance
    TV(q, p) = 1/2 * sum_i |q_i - p_i|
    range: [0, 1]
    lower is better
    """
    return 0.5 * sum(abs(q[i] - p[i]) for i in range(3))


def cross_entropy(p_star, p_hat, epsilon=1e-10):
    """
    Cross Entropy
    CE(p*, p^) = -sum_c p*_c * log(p^_c + epsilon)
    
    Args:
        p_star: ground truth distribution (win, draw, loss)
        p_hat: predicted distribution (win, draw, loss)
        epsilon: small constant to avoid log(0)
    
    Returns:
        cross entropy value
    """
    import math
    return -sum(p_star[i] * math.log(p_hat[i] + epsilon) for i in range(3))


def brier_score(p_star, p_hat):
    """
    Brier Score
    Brier(p*, p^) = sum_c (p^_c - p*_c)^2
    
    Args:
        p_star: ground truth distribution (win, draw, loss)
        p_hat: predicted distribution (win, draw, loss)
    
    Returns:
        brier score, range [0, 1]
    """
    return sum((p_hat[i] - p_star[i]) ** 2 for i in range(3))


def ev_loss(p_star, p_hat):
    """
    Expected Value Loss
    EV(p) = p_win - p_loss
    EVLoss = (EV(p*) - EV(p^))^2
    
    Args:
        p_star: ground truth distribution (win, draw, loss)
        p_hat: predicted distribution (win, draw, loss)
    
    Returns:
        ev loss, range [0, 4]
    """
    ev_star = p_star[0] - p_star[2]  # win - loss
    ev_hat = p_hat[0] - p_hat[2]     # win - loss
    return (ev_star - ev_hat) ** 2


def win_rate(p1, p2):
    """
    P1 win probability for two static/distribution players
    p1, p2 are tuples: (rock, paper, scissors)
    """
    r1, p1_, s1 = p1
    r2, p2_, s2 = p2
    return r1 * s2 + p1_ * r2 + s1 * p2_


# =========================
# RPS Rules
# =========================

def beat(action):
    return {
        "rock": "paper",
        "paper": "scissors",
        "scissors": "rock",
    }[action]


def lose(action):
    return {
        "rock": "scissors",
        "paper": "rock",
        "scissors": "paper",
    }[action]


def sample(dist):
    r = random.random()
    if r < dist[0]:
        return "rock"
    if r < dist[0] + dist[1]:
        return "paper"
    return "scissors"


# =========================
# Simulation for Markov Players
# =========================

def simulate(p1_id, p2_id, rounds=50000, seed=42):
    """
    Estimate P1 win rate for pairs involving Markov players.
    """
    random.seed(seed)

    prev1 = "rock"
    prev2 = "paper"
    wins = 0

    for _ in range(rounds):
        # Player 1
        if p1_id in PLAYER_DISTS:
            a1 = sample(PLAYER_DISTS[p1_id])
        elif p1_id == "X":
            a1 = beat(prev2)
        elif p1_id == "Y":
            a1 = lose(prev2)
        else:  # Z
            a1 = prev2

        # Player 2
        if p2_id in PLAYER_DISTS:
            a2 = sample(PLAYER_DISTS[p2_id])
        elif p2_id == "X":
            a2 = beat(prev1)
        elif p2_id == "Y":
            a2 = lose(prev1)
        else:  # Z
            a2 = prev1

        if (
            (a1 == "rock" and a2 == "scissors")
            or (a1 == "paper" and a2 == "rock")
            or (a1 == "scissors" and a2 == "paper")
        ):
            wins += 1

        prev1 = a1
        prev2 = a2

    return wins / rounds


# =========================
# Summary Helpers
# =========================

def summarize(rows):
    acc = [r["ACC"] for r in rows]
    mda = [r["MDA"] for r in rows]
    markov_exact = [r.get("MarkovExact", r["MDA"]) for r in rows]
    tv = [r["TV"] for r in rows if r["TV"] is not None]
    wr_gap = [r["WR_gap"] for r in rows]
    tp_total = sum(r.get("MarkovTP", 0) for r in rows)
    fp_total = sum(r.get("MarkovFP", 0) for r in rows)
    fn_total = sum(r.get("MarkovFN", 0) for r in rows)
    tn_total = sum(r.get("MarkovTN", 0) for r in rows)
    markov_precision, markov_recall, markov_f1 = precision_recall_f1(tp_total, fp_total, fn_total)

    tp_strict_total = sum(r.get("MarkovTP_strict", 0) for r in rows)
    fp_strict_total = sum(r.get("MarkovFP_strict", 0) for r in rows)
    fn_strict_total = sum(r.get("MarkovFN_strict", 0) for r in rows)
    tn_strict_total = sum(r.get("MarkovTN_strict", 0) for r in rows)
    markov_precision_strict, markov_recall_strict, markov_f1_strict = precision_recall_f1(tp_strict_total, fp_strict_total, fn_strict_total)
    # 收集所有样本的 CE, Brier, EVLoss（保留 None）
    ce_raw = [r["CE"] for r in rows]
    brier_raw = [r["Brier"] for r in rows]
    evloss_raw = [r["EVLoss"] for r in rows]

    # 只取非 None 的值用于归一化计算
    ce_valid = [c for c in ce_raw if c is not None]
    brier_valid = [b for b in brier_raw if b is not None]
    evloss_valid = [e for e in evloss_raw if e is not None]

    # Calculate Union Loss with normalization
    # CE normalization: min-max
    ce_min = safe_min(ce_valid)
    ce_max = safe_max(ce_valid)
    
    # Brier normalization: already in [0, 1], no change needed
    
    # EVLoss max value is 4
    
    # Calculate Union Loss for each sample that has all three metrics
    union_values = []
    for i in range(len(rows)):
        ce_val = ce_raw[i]
        brier_val = brier_raw[i]
        evloss_val = evloss_raw[i]
        
        # Only calculate Union if all three metrics are available
        if ce_val is not None and brier_val is not None and evloss_val is not None:
            # Normalize CE
            if ce_min is not None and ce_max is not None and ce_max > ce_min:
                ce_norm = (ce_val - ce_min) / (ce_max - ce_min)
            else:
                ce_norm = 0.5  # If max == min, set to 0.5
            
            # Brier is already normalized
            brier_norm = brier_val
            
            # Normalize EVLoss
            evloss_norm = evloss_val / 4.0
            
            # Calculate Union
            union = (ce_norm + brier_norm + evloss_norm) / 3.0
            union_values.append(union)
    
    return {
        "samples": len(rows),
        "ACC": mean(acc),
        "MDA": mean(mda),
        "MarkovExact": mean(markov_exact),
        "MarkovPrecision": markov_precision,
        "MarkovRecall": markov_recall,
        "MarkovF1": markov_f1,
        "MarkovTP": tp_total,
        "MarkovFP": fp_total,
        "MarkovFN": fn_total,
        "MarkovTN": tn_total,
        "MarkovTP_strict": tp_strict_total,
        "MarkovFP_strict": fp_strict_total,
        "MarkovFN_strict": fn_strict_total,
        "MarkovTN_strict": tn_strict_total,
        "TV": mean(tv),
        "WR_gap": mean(wr_gap),
        "CE": mean(ce_valid),
        "Brier": mean(brier_valid),
        "EVLoss": mean(evloss_valid),
        "Union": mean(union_values) if union_values else None,
    }


def print_summary(title, summary):
    print(f"\n=== {title} ===")
    print(summary)


# =========================
# Core Evaluation
# =========================

def evaluate_file(data, file_path):
    if not data.get("parse_success"):
        return None

    gt = data.get("ground_truth")
    pr = data.get("predictions")

    if not gt or not pr:
        return None

    gt1 = gt.get("player1_identity")
    gt2 = gt.get("player2_identity")

    pred1 = pr.get("player1", {}).get("identity")
    pred2 = pr.get("player2", {}).get("identity")

    if not gt1 or not gt2 or not pred1 or not pred2:
        return None

    # =========================
    # ACC
    # =========================
    acc = int(gt1 == pred1 and gt2 == pred2)

    # =========================
    # MDA
    # =========================
    gt_markov = (gt1 in MARKOV_PLAYERS) or (gt2 in MARKOV_PLAYERS)
    pred_markov = (pred1 in MARKOV_PLAYERS) or (pred2 in MARKOV_PLAYERS)
    mda = int(gt_markov == pred_markov)

    # =========================
    # Player-level Markov detection metrics
    # =========================
    gt_labels = [int(gt1 in MARKOV_PLAYERS), int(gt2 in MARKOV_PLAYERS)]
    pred_labels = [int(pred1 in MARKOV_PLAYERS), int(pred2 in MARKOV_PLAYERS)]

    markov_tp = sum(1 for gt_l, pr_l in zip(gt_labels, pred_labels) if gt_l == 1 and pr_l == 1)
    markov_fp = sum(1 for gt_l, pr_l in zip(gt_labels, pred_labels) if gt_l == 0 and pr_l == 1)
    markov_fn = sum(1 for gt_l, pr_l in zip(gt_labels, pred_labels) if gt_l == 1 and pr_l == 0)
    markov_tn = sum(1 for gt_l, pr_l in zip(gt_labels, pred_labels) if gt_l == 0 and pr_l == 0)

    # Strict version: TP/TN require exact identity match for ALL players
    # TP_strict = gt Markov  AND pred == gt  (e.g. X→X)
    # FN_strict = gt Markov  AND pred != gt  (wrong class or wrong Markov identity)
    # TN_strict = gt Non-Markov AND pred == gt  (e.g. D→D)
    # FP_strict = gt Non-Markov AND pred != gt  (predicted as Markov OR wrong Non-Markov, e.g. D→E)
    markov_tp_strict = sum(1 for gt_id, pr_id in zip([gt1, gt2], [pred1, pred2])
                           if gt_id in MARKOV_PLAYERS and pr_id == gt_id)
    markov_fn_strict = sum(1 for gt_id, pr_id in zip([gt1, gt2], [pred1, pred2])
                           if gt_id in MARKOV_PLAYERS and pr_id != gt_id)
    markov_tn_strict = sum(1 for gt_id, pr_id in zip([gt1, gt2], [pred1, pred2])
                           if gt_id not in MARKOV_PLAYERS and pr_id == gt_id)
    markov_fp_strict = sum(1 for gt_id, pr_id in zip([gt1, gt2], [pred1, pred2])
                           if gt_id not in MARKOV_PLAYERS and pr_id != gt_id)

    # =========================
    # TV distance
    # =========================
    tv_values = []

    for gt_id, key in [(gt1, "player1"), (gt2, "player2")]:
        if gt_id in PLAYER_DISTS:
            q = PLAYER_DISTS[gt_id]
            counts = pr[key]["counts"]
            p = normalize_counts([
                counts["rock"],
                counts["paper"],
                counts["scissors"],
            ])
            tv_values.append(tv_distance(q, p))

    tv = mean(tv_values)

    # =========================
    # WR gap
    # =========================
    if gt1 in PLAYER_DISTS and gt2 in PLAYER_DISTS:
        wr_gt = win_rate(PLAYER_DISTS[gt1], PLAYER_DISTS[gt2])
    else:
        wr_gt = simulate(gt1, gt2, seed=123)

    if pred1 in PLAYER_DISTS and pred2 in PLAYER_DISTS:
        wr_pred = win_rate(PLAYER_DISTS[pred1], PLAYER_DISTS[pred2])
    else:
        wr_pred = simulate(pred1, pred2, seed=456)

    wr_gap = abs(wr_gt - wr_pred)

    # =========================
    # New metrics: CE, Brier, EVLoss
    # =========================
    # Calculate outcome probabilities (win, draw, loss) for both players
    ce_values = []
    brier_values = []
    evloss_values = []

    for gt_id, key in [(gt1, "player1"), (gt2, "player2")]:
        if gt_id in PLAYER_DISTS:
            # Ground truth distribution
            q = PLAYER_DISTS[gt_id]
            # Predicted distribution from counts
            counts = pr[key]["counts"]
            p = normalize_counts([
                counts["rock"],
                counts["paper"],
                counts["scissors"],
            ])
            
            # Get opponent distribution
            if key == "player1":
                if gt2 in PLAYER_DISTS:
                    opp_dist = PLAYER_DISTS[gt2]
                else:
                    # For Markov players, use predicted distribution
                    opp_counts = pr["player2"]["counts"]
                    opp_dist = normalize_counts([
                        opp_counts["rock"],
                        opp_counts["paper"],
                        opp_counts["scissors"],
                    ])
            else:
                if gt1 in PLAYER_DISTS:
                    opp_dist = PLAYER_DISTS[gt1]
                else:
                    # For Markov players, use predicted distribution
                    opp_counts = pr["player1"]["counts"]
                    opp_dist = normalize_counts([
                        opp_counts["rock"],
                        opp_counts["paper"],
                        opp_counts["scissors"],
                    ])
            
            # Calculate outcome probabilities (win, draw, loss)
            # p = (r, p, s), opp = (r', p', s')
            # win: r*s' + p*r' + s*p'
            # draw: r*r' + p*p' + s*s'
            # loss: r*p' + p*s' + s*r'
            p_win_gt = q[0]*opp_dist[2] + q[1]*opp_dist[0] + q[2]*opp_dist[1]
            p_draw_gt = q[0]*opp_dist[0] + q[1]*opp_dist[1] + q[2]*opp_dist[2]
            p_loss_gt = q[0]*opp_dist[1] + q[1]*opp_dist[2] + q[2]*opp_dist[0]
            
            p_win_pred = p[0]*opp_dist[2] + p[1]*opp_dist[0] + p[2]*opp_dist[1]
            p_draw_pred = p[0]*opp_dist[0] + p[1]*opp_dist[1] + p[2]*opp_dist[2]
            p_loss_pred = p[0]*opp_dist[1] + p[1]*opp_dist[2] + p[2]*opp_dist[0]
            
            p_star = (p_win_gt, p_draw_gt, p_loss_gt)
            p_hat = (p_win_pred, p_draw_pred, p_loss_pred)
            
            ce_values.append(cross_entropy(p_star, p_hat))
            brier_values.append(brier_score(p_star, p_hat))
            evloss_values.append(ev_loss(p_star, p_hat))

    ce = mean(ce_values) if ce_values else None
    brier = mean(brier_values) if brier_values else None
    evloss = mean(evloss_values) if evloss_values else None

    return {
        "file": str(file_path),
        "gt_player1": gt1,
        "gt_player2": gt2,
        "pred_player1": pred1,
        "pred_player2": pred2,
        "ACC": acc,
        "MDA": mda,
        "MarkovExact": mda,
        "MarkovTP": markov_tp,
        "MarkovFP": markov_fp,
        "MarkovFN": markov_fn,
        "MarkovTN": markov_tn,
        "MarkovTP_strict": markov_tp_strict,
        "MarkovFP_strict": markov_fp_strict,
        "MarkovFN_strict": markov_fn_strict,
        "MarkovTN_strict": markov_tn_strict,
        "TV": tv,
        "WR_gap": wr_gap,
        "CE": ce,
        "Brier": brier,
        "EVLoss": evloss,
    }


def evaluate_folder(folder):
    folder_path = Path(folder)
    files = list(folder_path.rglob("*.json"))

    all_rows = []
    non_markov_rows = []
    with_markov_rows = []
    skipped = []

    for f in files:
        if f.name in {"evaluation_summary.json", "evaluation_detail.json"}:
            continue

        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)

            row = evaluate_file(data, f)
            if row is None:
                skipped.append(str(f))
                continue

            all_rows.append(row)

            lower_path = str(f).lower()
            if "with_markov" in lower_path or "type2_with_markov" in lower_path:
                with_markov_rows.append(row)
            else:
                non_markov_rows.append(row)

        except Exception as e:
            skipped.append(f"{f} | error: {e}")

    overall_summary = summarize(all_rows)
    non_markov_summary = summarize(non_markov_rows)
    with_markov_summary = summarize(with_markov_rows)

    result = {
        "overall": overall_summary,
        "non_markov": non_markov_summary,
        "with_markov": with_markov_summary,
        "skipped_count": len(skipped),
    }

    detail = {
        "rows": all_rows,
        "skipped_files": skipped,
    }

    summary_path = folder_path / "evaluation_summary.json"
    detail_path = folder_path / "evaluation_detail.json"

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(detail, f, ensure_ascii=False, indent=2)

    print_summary("OVERALL", overall_summary)
    print_summary("NON MARKOV", non_markov_summary)
    print_summary("WITH MARKOV", with_markov_summary)

    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved detail  to: {detail_path}")

    if skipped:
        print(f"\nSkipped files: {len(skipped)}")


# =========================
# CLI
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder",
        type=str,
        required=True,
        help="Path to parsed_output model folder"
    )

    args = parser.parse_args()
    evaluate_folder(args.folder)