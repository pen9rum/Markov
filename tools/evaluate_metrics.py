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
    tv = [r["TV"] for r in rows if r["TV"] is not None]
    wr_gap = [r["WR_gap"] for r in rows]

    return {
        "samples": len(rows),
        "ACC": mean(acc),
        "MDA": mean(mda),
        "TV": mean(tv),
        "WR_gap": mean(wr_gap),
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

    return {
        "file": str(file_path),
        "gt_player1": gt1,
        "gt_player2": gt2,
        "pred_player1": pred1,
        "pred_player2": pred2,
        "ACC": acc,
        "MDA": mda,
        "TV": tv,
        "WR_gap": wr_gap,
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