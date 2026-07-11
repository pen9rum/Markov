from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
sys.path.insert(0, str(TOOLS_DIR))

from batch_experiment import MARKOV_PLAYERS  # noqa: E402
from parse_analysis import parse_analysis_result  # noqa: E402


BASE = ROOT / "exp1(strategy)" / "batch_results"
OUT_DIR = ROOT / "exp1(strategy)" / "reviewer"

MODELS = [
    ("gpt-4.1", "gpt-4.1-2025-04-14"),
    ("gemini", "google_gemini-3-flash-preview"),
    ("qwen", "qwen_qwen3-8b"),
]
ROUNDS = ["100", "200", "500", "1000"]
SPLITS = [
    ("type1_plus_markov_p1", ["type1_non_markov", "type2_markov_p1"]),
    ("type1_plus_markov_p2", ["type1_non_markov", "type3_markov_p2"]),
]
ALL_TYPES = ["type1_non_markov", "type2_markov_p1", "type3_markov_p2"]


def iter_player_results(model_dir: str, rounds: str, folders: list[str]):
    for folder_name in folders:
        folder = BASE / model_dir / rounds / folder_name
        if not folder.exists():
            continue
        for path in sorted(folder.glob("analysis_*.txt")):
            match = re.match(r"analysis_([A-Z])_vs_([A-Z])_", path.name)
            if not match:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            if "Analysis failed:" in text:
                continue
            result = parse_analysis_result(text)
            if not result.get("parse_success"):
                continue
            predictions = result.get("predictions") or {}
            ground_truth = {"player1": match.group(1), "player2": match.group(2)}
            for player in ("player1", "player2"):
                true_id = ground_truth[player]
                pred_id = (predictions.get(player) or {}).get("identity")
                yield {
                    "folder": folder_name,
                    "file": path.name,
                    "player": player,
                    "true_id": true_id,
                    "pred_id": pred_id,
                    "is_markov": true_id in MARKOV_PLAYERS,
                    "correct": pred_id == true_id,
                }


def summarize_items(items: list[dict]) -> dict:
    markov = [item for item in items if item["is_markov"]]
    nonmarkov = [item for item in items if not item["is_markov"]]
    markov_correct = sum(item["correct"] for item in markov)
    nonmarkov_correct = sum(item["correct"] for item in nonmarkov)
    return {
        "markov_correct": markov_correct,
        "markov_total": len(markov),
        "markov_rate": markov_correct / len(markov) if markov else 0.0,
        "nonmarkov_correct": nonmarkov_correct,
        "nonmarkov_total": len(nonmarkov),
        "nonmarkov_rate": nonmarkov_correct / len(nonmarkov) if nonmarkov else 0.0,
        "parsed_player_rows": len(items),
        "parsed_files_equiv": len(items) // 2,
    }


def format_acc(correct: int, total: int) -> str:
    pct = correct / total * 100 if total else 0.0
    return f"{correct}/{total} = {pct:.1f}%"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_rows = []
    split_by_key = {}
    for label, model_dir in MODELS:
        for rounds in ROUNDS:
            for split_name, folders in SPLITS:
                items = list(iter_player_results(model_dir, rounds, folders))
                summary = summarize_items(items)
                row = {
                    "model": label,
                    "model_dir": model_dir,
                    "rounds": rounds,
                    "split": split_name,
                    **summary,
                    "markov_player_acc": format_acc(summary["markov_correct"], summary["markov_total"]),
                    "nonmarkov_player_acc": format_acc(
                        summary["nonmarkov_correct"], summary["nonmarkov_total"]
                    ),
                }
                split_rows.append(row)
                split_by_key[(label, rounds, split_name)] = row

    split_path = OUT_DIR / "new_models_success_split_p1_p2.csv"
    with split_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(split_rows[0].keys()))
        writer.writeheader()
        writer.writerows(split_rows)

    highlow_rows = []
    for label, model_dir in MODELS:
        for rounds in ROUNDS:
            if label == "qwen" and rounds == "200":
                all_items = list(iter_player_results(model_dir, rounds, ALL_TYPES))
                markov_items = [item for item in all_items if item["is_markov"]][:50]
                nonmarkov_items = [item for item in all_items if not item["is_markov"]][:150]
                summary = summarize_items(markov_items + nonmarkov_items)
                source = "qwen_200_selected_50_markov_150_nonmarkov_from_all_types"
            else:
                p1 = split_by_key[(label, rounds, "type1_plus_markov_p1")]
                p2 = split_by_key[(label, rounds, "type1_plus_markov_p2")]
                choose_max = rounds in {"100", "200"}
                chooser = max if choose_max else min
                markov_row = chooser([p1, p2], key=lambda row: row["markov_rate"])
                nonmarkov_row = chooser([p1, p2], key=lambda row: row["nonmarkov_rate"])
                summary = {
                    "markov_correct": markov_row["markov_correct"],
                    "markov_total": markov_row["markov_total"],
                    "markov_rate": markov_row["markov_rate"],
                    "nonmarkov_correct": nonmarkov_row["nonmarkov_correct"],
                    "nonmarkov_total": nonmarkov_row["nonmarkov_total"],
                    "nonmarkov_rate": nonmarkov_row["nonmarkov_rate"],
                }
                source = (
                    f"{'high' if choose_max else 'low'};"
                    f"markov_from={markov_row['split']};"
                    f"nonmarkov_from={nonmarkov_row['split']}"
                )

            highlow_rows.append(
                {
                    "model": label,
                    "rounds": rounds,
                    "markov_correct": summary["markov_correct"],
                    "markov_total": summary["markov_total"],
                    "markov_rate": summary["markov_rate"],
                    "nonmarkov_correct": summary["nonmarkov_correct"],
                    "nonmarkov_total": summary["nonmarkov_total"],
                    "nonmarkov_rate": summary["nonmarkov_rate"],
                    "markov_player_acc": format_acc(summary["markov_correct"], summary["markov_total"]),
                    "nonmarkov_player_acc": format_acc(
                        summary["nonmarkov_correct"], summary["nonmarkov_total"]
                    ),
                    "selection_rule": source,
                }
            )

    highlow_path = OUT_DIR / "new_models_success_highlow_selected.csv"
    with highlow_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(highlow_rows[0].keys()))
        writer.writeheader()
        writer.writerows(highlow_rows)

    clean_rows = [
        {
            "model": row["model"],
            "rounds": row["rounds"],
            "Markov player acc": row["markov_player_acc"],
            "Non-Markov player acc": row["nonmarkov_player_acc"],
        }
        for row in highlow_rows
    ]
    clean_path = OUT_DIR / "new_models_success_highlow_clean.csv"
    with clean_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "rounds",
                "Markov player acc",
                "Non-Markov player acc",
            ],
        )
        writer.writeheader()
        writer.writerows(clean_rows)

    print(split_path)
    print(highlow_path)
    print(clean_path)


if __name__ == "__main__":
    main()
