import argparse
import csv
import math
import re
from pathlib import Path


ACTIONS = ("rock", "paper", "scissors")
ACTION_INDEX = {a: i for i, a in enumerate(ACTIONS)}

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

MARKOV_RULES = {
    "X": "win_last",
    "Y": "lose_last",
    "Z": "copy_last",
}

VALID_CANDIDATES = tuple(PLAYER_DISTS) + tuple(MARKOV_RULES)
MAIN_SOURCE_MODELS = ("deepseek-chat", "deepseek-reasoner", "gpt-5", "gpt-5-mini")


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


def expected_markov_action(rule, previous_opponent_action):
    if rule == "win_last":
        return beat(previous_opponent_action)
    if rule == "lose_last":
        return lose(previous_opponent_action)
    if rule == "copy_last":
        return previous_opponent_action
    raise ValueError(f"Unknown rule: {rule}")


def parse_actions(text):
    return [m.group(0).lower() for m in re.finditer(r"\bRock\b|\bPaper\b|\bScissors\b", text, re.I)]


def extract_trajectory(text, player_num):
    pattern = (
        rf"Player{player_num}\s*\([^)]+\)\s*Trajectory:\s*\n"
        rf"(.*?)\n\s*\n"
    )
    match = re.search(pattern, text, flags=re.S | re.I)
    if not match:
        raise ValueError(f"Could not parse Player{player_num} trajectory")
    actions = parse_actions(match.group(1))
    if not actions:
        raise ValueError(f"Empty Player{player_num} trajectory")
    return actions


def parse_analysis_txt(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"Match:\s*([A-Z])\s+vs\s+([A-Z])", text)
    if not match:
        raise ValueError("Could not parse ground-truth identities")
    rounds_match = re.search(r"Total Rounds:\s*(\d+)", text)
    p1 = extract_trajectory(text, 1)
    p2 = extract_trajectory(text, 2)
    if len(p1) != len(p2):
        raise ValueError(f"Trajectory length mismatch: {len(p1)} vs {len(p2)}")
    return {
        "gt_player1": match.group(1),
        "gt_player2": match.group(2),
        "rounds": int(rounds_match.group(1)) if rounds_match else len(p1),
        "p1_actions": p1,
        "p2_actions": p2,
    }


def distribution_loglik(actions, dist, epsilon):
    # Keep the intended marginal model, with a tiny numerical floor so log(0)
    # does not crash diagnostics.
    total = 0.0
    for action in actions:
        p = dist[ACTION_INDEX[action]]
        total += math.log(max(p, epsilon))
    return total


def markov_loglik(actions, opponent_actions, rule, epsilon):
    match_prob = 1.0 - epsilon
    mismatch_prob = epsilon / 2.0
    total = math.log(1.0 / 3.0)  # first round has no previous opponent move
    for t in range(1, len(actions)):
        expected = expected_markov_action(rule, opponent_actions[t - 1])
        total += math.log(match_prob if actions[t] == expected else mismatch_prob)
    return total


def score_candidate(candidate, actions, opponent_actions, epsilon):
    if candidate in PLAYER_DISTS:
        return distribution_loglik(actions, PLAYER_DISTS[candidate], epsilon)
    return markov_loglik(actions, opponent_actions, MARKOV_RULES[candidate], epsilon)


def predict_identity(actions, opponent_actions, epsilon):
    scored = [
        (score_candidate(candidate, actions, opponent_actions, epsilon), candidate)
        for candidate in VALID_CANDIDATES
    ]
    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return scored[0][1], scored[0][0]


def candidate_probabilities(actions, opponent_actions, epsilon):
    scores = {
        candidate: score_candidate(candidate, actions, opponent_actions, epsilon)
        for candidate in VALID_CANDIDATES
    }
    max_score = max(scores.values())
    weights = {candidate: math.exp(score - max_score) for candidate, score in scores.items()}
    total = sum(weights.values())
    probs = {candidate: weights[candidate] / total for candidate in VALID_CANDIDATES}
    ranked = sorted(scores, key=lambda candidate: (scores[candidate], candidate), reverse=True)
    return scores, probs, ranked


def top_k_summary(scores, probs, ranked, k=5):
    top = ranked[:k]
    return {
        "identities": "|".join(top),
        "loglik": "|".join(f"{scores[c]:.6g}" for c in top),
        "prob": "|".join(f"{probs[c]:.6g}" for c in top),
    }


def classify_type(path):
    lower = str(path).lower()
    if "type2_with_markov" in lower:
        return "type2_with_markov"
    if "type1_non_markov" in lower:
        return "type1_non_markov"
    return "unknown"


def evaluate_file(path, source_model, epsilon):
    item = parse_analysis_txt(path)
    p1_scores, p1_probs, p1_ranked = candidate_probabilities(item["p1_actions"], item["p2_actions"], epsilon)
    p2_scores, p2_probs, p2_ranked = candidate_probabilities(item["p2_actions"], item["p1_actions"], epsilon)
    pred1 = p1_ranked[0]
    pred2 = p2_ranked[0]
    score1 = p1_scores[pred1]
    score2 = p2_scores[pred2]
    p1_top5 = top_k_summary(p1_scores, p1_probs, p1_ranked)
    p2_top5 = top_k_summary(p2_scores, p2_probs, p2_ranked)
    gt1 = item["gt_player1"]
    gt2 = item["gt_player2"]

    p1_correct = int(pred1 == gt1)
    p2_correct = int(pred2 == gt2)
    exact = int(p1_correct and p2_correct)
    player_acc = (p1_correct + p2_correct) / 2.0

    markov_correct = []
    nonmarkov_correct = []
    for gt, pred in ((gt1, pred1), (gt2, pred2)):
        if gt in MARKOV_RULES:
            markov_correct.append(int(pred == gt))
        else:
            nonmarkov_correct.append(int(pred == gt))

    row = {
        "source_model": source_model,
        "rounds": item["rounds"],
        "type": classify_type(path),
        "file": str(path),
        "gt_player1": gt1,
        "gt_player2": gt2,
        "pred_player1": pred1,
        "pred_player2": pred2,
        "p1_correct": p1_correct,
        "p2_correct": p2_correct,
        "exact_match": exact,
        "player_level_accuracy": player_acc,
        "markov_player_accuracy": mean(markov_correct),
        "nonmarkov_player_accuracy": mean(nonmarkov_correct),
        "p1_loglik": score1,
        "p2_loglik": score2,
        "p1_top5_identities": p1_top5["identities"],
        "p1_top5_loglik": p1_top5["loglik"],
        "p1_top5_prob": p1_top5["prob"],
        "p1_loglik_gap": score1 - p1_scores[p1_ranked[1]],
        "p2_top5_identities": p2_top5["identities"],
        "p2_top5_loglik": p2_top5["loglik"],
        "p2_top5_prob": p2_top5["prob"],
        "p2_loglik_gap": score2 - p2_scores[p2_ranked[1]],
    }
    for candidate in VALID_CANDIDATES:
        row[f"p1_loglik_{candidate}"] = p1_scores[candidate]
    for candidate in VALID_CANDIDATES:
        row[f"p1_prob_{candidate}"] = p1_probs[candidate]
    for candidate in VALID_CANDIDATES:
        row[f"p2_loglik_{candidate}"] = p2_scores[candidate]
    for candidate in VALID_CANDIDATES:
        row[f"p2_prob_{candidate}"] = p2_probs[candidate]
    return row


def mean(values):
    vals = [v for v in values if v is not None and v != ""]
    return sum(vals) / len(vals) if vals else None


def summarize(rows, source_model, rounds, row_type):
    return {
        "source_model": source_model,
        "rounds": rounds,
        "type": row_type,
        "samples": len(rows),
        "exact_match_accuracy": mean([r["exact_match"] for r in rows]),
        "player_level_accuracy": mean([r["player_level_accuracy"] for r in rows]),
        "markov_player_accuracy": mean([r["markov_player_accuracy"] for r in rows]),
        "nonmarkov_player_accuracy": mean([r["nonmarkov_player_accuracy"] for r in rows]),
    }


def build_summaries(detail_rows):
    summaries = []
    keys = sorted({(r["source_model"], r["rounds"]) for r in detail_rows})
    for source_model, rounds in keys:
        base = [r for r in detail_rows if r["source_model"] == source_model and r["rounds"] == rounds]
        for row_type in ("overall", "type1_non_markov", "type2_with_markov"):
            rows = base if row_type == "overall" else [r for r in base if r["type"] == row_type]
            if rows:
                summaries.append(summarize(rows, source_model, rounds, row_type))

    for rounds in sorted({r["rounds"] for r in detail_rows}):
        base = [r for r in detail_rows if r["rounds"] == rounds]
        for row_type in ("overall", "type1_non_markov", "type2_with_markov"):
            rows = base if row_type == "overall" else [r for r in base if r["type"] == row_type]
            if rows:
                summaries.append(summarize(rows, "all_main_sources", rounds, row_type))
    return summaries


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collect_files(batch_root, source_models):
    files = []
    for model in source_models:
        model_root = batch_root / model
        if not model_root.exists():
            continue
        for path in sorted(model_root.rglob("analysis_*.txt")):
            if "type1_non_markov" in str(path) or "type2_with_markov" in str(path):
                files.append((model, path))
    return files


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-root", default=str(repo / "exp1(strategy)" / "batch_results"))
    parser.add_argument("--outdir", default=str(repo / "exp1(strategy)" / "analysis_results"))
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument(
        "--source-models",
        nargs="+",
        default=list(MAIN_SOURCE_MODELS),
        help="Batch-result model folders to use as the exact LLM test trajectories.",
    )
    args = parser.parse_args()

    batch_root = Path(args.batch_root)
    outdir = Path(args.outdir)
    files = collect_files(batch_root, args.source_models)
    if not files:
        raise SystemExit(f"No exp1 analysis txt files found under {batch_root}")

    detail_rows = []
    skipped = []
    for source_model, path in files:
        try:
            detail_rows.append(evaluate_file(path, source_model, args.epsilon))
        except Exception as exc:
            skipped.append({"source_model": source_model, "file": str(path), "error": str(exc)})

    summary_rows = build_summaries(detail_rows)

    detail_fields = [
        "source_model", "rounds", "type", "file",
        "gt_player1", "gt_player2", "pred_player1", "pred_player2",
        "p1_correct", "p2_correct", "exact_match", "player_level_accuracy",
        "markov_player_accuracy", "nonmarkov_player_accuracy",
        "p1_loglik", "p2_loglik",
        "p1_top5_identities", "p1_top5_loglik", "p1_top5_prob", "p1_loglik_gap",
        "p2_top5_identities", "p2_top5_loglik", "p2_top5_prob", "p2_loglik_gap",
    ]
    detail_fields += [f"p1_loglik_{candidate}" for candidate in VALID_CANDIDATES]
    detail_fields += [f"p1_prob_{candidate}" for candidate in VALID_CANDIDATES]
    detail_fields += [f"p2_loglik_{candidate}" for candidate in VALID_CANDIDATES]
    detail_fields += [f"p2_prob_{candidate}" for candidate in VALID_CANDIDATES]
    summary_fields = [
        "source_model", "rounds", "type", "samples",
        "exact_match_accuracy", "player_level_accuracy",
        "markov_player_accuracy", "nonmarkov_player_accuracy",
    ]
    skipped_fields = ["source_model", "file", "error"]

    write_csv(outdir / "mle_identification_baseline_detail.csv", detail_rows, detail_fields)
    write_csv(outdir / "mle_identification_baseline_summary.csv", summary_rows, summary_fields)
    write_csv(outdir / "mle_identification_baseline_skipped.csv", skipped, skipped_fields)

    print(f"Evaluated files: {len(detail_rows)}")
    print(f"Skipped files: {len(skipped)}")
    print(outdir / "mle_identification_baseline_detail.csv")
    print(outdir / "mle_identification_baseline_summary.csv")
    print(outdir / "mle_identification_baseline_skipped.csv")


if __name__ == "__main__":
    main()
