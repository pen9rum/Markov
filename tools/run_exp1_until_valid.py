"""
Run exp1 until each selected type has enough valid analysis files.

This uses the existing exp1 batch_experiment output format:
exp1(strategy)/batch_results/<model>/<rounds>/<type_folder>/analysis_*.txt

By default, valid means the LLM call succeeded and the saved report does not
contain the "Analysis failed" marker. Use --require-parse to require
tools.parse_analysis.parse_analysis_result(...).parse_success as well.
Invalid analysis files are deleted before refill so combinations can be rerun.
"""
from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path

TOOLS_DIR = Path(__file__).resolve().parent
ROOT = TOOLS_DIR.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(TOOLS_DIR))
sys.path.insert(0, str(SRC_DIR))

from batch_experiment import (  # noqa: E402
    BATCH_OUTPUT_DIR,
    MARKOV_PLAYERS,
    NON_MARKOV_PLAYERS,
    run_single_experiment,
    save_single_analysis,
)
from parse_analysis import parse_analysis_result  # noqa: E402


MODEL_MAP = {
    "gpt-4.1": "gpt-4.1-2025-04-14",
    "gemini": "google/gemini-3-flash-preview",
    "qwen-api": "qwen/qwen3-8b",
}

MODEL_CHOICE = {
    "gpt-4.1": "15",
    "gemini": "14",
    "qwen-api": "13",
}

TYPE_FOLDERS = {
    1: "type1_non_markov",
    2: "type2_markov_p1",
    3: "type3_markov_p2",
}

NON_MARKOV_EXCL_ABC = NON_MARKOV_PLAYERS - {"A", "B", "C"}


def result_dir(model_name: str, rounds: int, combo_type: int) -> Path:
    clean_model = model_name.replace("/", "_").replace("\\", "_")
    return Path(BATCH_OUTPUT_DIR) / clean_model / str(rounds) / TYPE_FOLDERS[combo_type]


def llm_call_succeeded(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "Analysis failed:" not in text


def failure_reason(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as exc:
        return f"read failed: {exc}"
    marker = "Analysis failed:"
    idx = text.find(marker)
    if idx < 0:
        return "parse invalid"
    reason = text[idx:].splitlines()[0]
    return reason[:1000]


def is_parse_valid(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    result = parse_analysis_result(text, include_full_text=False)
    return bool(result.get("parse_success"))


def is_valid(path: Path, require_parse: bool) -> bool:
    if not llm_call_succeeded(path):
        return False
    if require_parse:
        return is_parse_valid(path)
    return True


def purge_invalid(folder: Path, require_parse: bool) -> int:
    removed = 0
    if not folder.exists():
        return removed
    for path in list(folder.glob("analysis_*.txt")):
        if not is_valid(path, require_parse):
            path.unlink(missing_ok=True)
            removed += 1
    return removed


def valid_files(folder: Path, require_parse: bool) -> list[Path]:
    if not folder.exists():
        return []
    return [path for path in folder.glob("analysis_*.txt") if is_valid(path, require_parse)]


def pair_from_filename(path: Path) -> tuple[str, str] | None:
    match = re.match(r"analysis_([A-Z])_vs_([A-Z])_", path.name)
    if not match:
        return None
    return match.group(1), match.group(2)


def valid_pairs(folder: Path, require_parse: bool) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in valid_files(folder, require_parse):
        pair = pair_from_filename(path)
        if pair:
            pairs.add(pair)
    return pairs


def pool_for_type(combo_type: int) -> list[tuple[str, str]]:
    if combo_type == 1:
        return [
            (p1, p2)
            for p1 in NON_MARKOV_PLAYERS
            for p2 in NON_MARKOV_PLAYERS
            if p1 != p2
        ]
    if combo_type == 2:
        return [
            (m, nm)
            for m in MARKOV_PLAYERS
            for nm in NON_MARKOV_EXCL_ABC
        ]
    if combo_type == 3:
        return [
            (nm, m)
            for nm in NON_MARKOV_EXCL_ABC
            for m in MARKOV_PLAYERS
        ]
    raise ValueError(f"Unknown combo type: {combo_type}")


def choose_combo(folder: Path, combo_type: int, require_parse: bool) -> tuple[str, str]:
    pool = pool_for_type(combo_type)
    done = valid_pairs(folder, require_parse)
    unseen = [pair for pair in pool if pair not in done]
    if unseen:
        return random.choice(unseen)
    return random.choice(pool)


def run_until_valid(
    model_key: str,
    rounds: int,
    target: int,
    combo_types: list[int],
    max_attempts: int,
    require_parse: bool,
    keep_invalid: bool,
) -> None:
    model_name = MODEL_MAP[model_key]
    model_choice = MODEL_CHOICE[model_key]

    for combo_type in combo_types:
        folder = result_dir(model_name, rounds, combo_type)
        attempts = 0

        while True:
            removed = purge_invalid(folder, require_parse)
            count = len(valid_files(folder, require_parse))
            print(f"\n[{model_key} rounds={rounds}] type{combo_type} valid={count}/{target}, removed_invalid={removed}")
            if count >= target:
                print(f"[{model_key} rounds={rounds}] type{combo_type} done")
                break
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"{model_key} rounds={rounds} type{combo_type}: "
                    f"reached max attempts {max_attempts} with {count}/{target} valid"
                )

            p1, p2 = choose_combo(folder, combo_type, require_parse)
            attempts += 1
            print(f"[{model_key} rounds={rounds}] type{combo_type} attempt {attempts}: {p1} vs {p2}")

            try:
                exp = run_single_experiment(
                    player1_id=p1,
                    player2_id=p2,
                    num_rounds=rounds,
                    model_choice=model_choice,
                    model_name=model_name,
                )
            except Exception as exc:
                exp = {
                    "ground_truth": {
                        "player1_id": p1,
                        "player2_id": p2,
                    },
                    "success": False,
                    "error": str(exc),
                }

            exp["combo_type"] = combo_type
            path = Path(save_single_analysis(exp, model_name, combo_type=combo_type, num_rounds=rounds))
            folder.mkdir(parents=True, exist_ok=True)
            if path.parent != folder:
                moved_path = folder / path.name
                path.replace(moved_path)
                path = moved_path
            if is_valid(path, require_parse):
                print(f"[{model_key} rounds={rounds}] saved valid: {path.name}")
            else:
                reason = "parse invalid" if llm_call_succeeded(path) else failure_reason(path)
                if keep_invalid:
                    print(f"[{model_key} rounds={rounds}] invalid kept: {reason}")
                    break
                else:
                    print(f"[{model_key} rounds={rounds}] invalid: {reason}; deleting and refilling")
                    path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp1 until each selected type has N valid files.")
    parser.add_argument("--model", required=True, choices=sorted(MODEL_MAP.keys()))
    parser.add_argument("--rounds", type=int, required=True, choices=[100, 200, 500, 1000])
    parser.add_argument("--target", type=int, default=50)
    parser.add_argument("--types", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3],
                        help="Default: 1 2 3 (stat/stat, Markov P1, Markov P2).")
    parser.add_argument("--max-attempts", type=int, default=200)
    parser.add_argument("--require-parse", action="store_true",
                        help="Require tools/parse_analysis.py parse_success=True. Default only requires LLM call success.")
    parser.add_argument("--keep-invalid", action="store_true",
                        help="Keep invalid files for debugging instead of deleting them.")
    args = parser.parse_args()

    run_until_valid(
        model_key=args.model,
        rounds=args.rounds,
        target=args.target,
        combo_types=args.types,
        max_attempts=args.max_attempts,
        require_parse=args.require_parse,
        keep_invalid=args.keep_invalid,
    )


if __name__ == "__main__":
    main()
