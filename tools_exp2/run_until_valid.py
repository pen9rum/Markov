"""
Run exp2 simulations until each model has enough valid saved results per type.

Valid means:
- JSON is readable
- success is true
- llm_simulation.parsed_rounds >= --min-parsed

Invalid JSON/TXT pairs for the selected config are removed before each refill
pass so simulate_trajectory.py's "already seen" logic will not skip them.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from simulate_trajectory import (  # noqa: E402
    MODEL_MAP,
    OUTPUT_DIR,
    TYPE_FOLDERS,
    generate_valid_combinations,
    run_simulation_experiment,
    save_result,
)


def clean_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


def parsed_rounds(data: dict) -> int:
    sim = data.get("llm_simulation") or {}
    return int(sim.get("parsed_rounds") or data.get("parsed_rounds") or 0)


def is_valid_json(path: Path, min_parsed: int) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return bool(data.get("success")) and parsed_rounds(data) >= min_parsed


def result_root(model_name: str, context: int, simulate: int) -> Path:
    return Path(OUTPUT_DIR) / clean_model_name(model_name) / f"ctx{context}_sim{simulate}"


def iter_json_files(root: Path):
    if root.exists():
        yield from root.rglob("*.json")


def delete_pair(json_path: Path) -> None:
    txt_path = json_path.with_suffix(".txt")
    json_path.unlink(missing_ok=True)
    txt_path.unlink(missing_ok=True)


def purge_invalid(root: Path, min_parsed: int) -> int:
    removed = 0
    for path in list(iter_json_files(root)):
        if not is_valid_json(path, min_parsed):
            delete_pair(path)
            removed += 1
    return removed


def valid_count(root: Path, min_parsed: int) -> int:
    return sum(1 for path in iter_json_files(root) if is_valid_json(path, min_parsed))


def existing_valid_pairs(root: Path, min_parsed: int) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for path in iter_json_files(root):
        if not is_valid_json(path, min_parsed):
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        p1 = data.get("player1_id")
        p2 = data.get("player2_id")
        if p1 and p2:
            pairs.add((p1, p2))
    return pairs


def all_combo_pool() -> list[tuple[int, str, str]]:
    pools = generate_valid_combinations()
    combos: list[tuple[int, str, str]] = []
    for combo_type, pool in enumerate(pools, start=1):
        combos.extend((combo_type, p1, p2) for p1, p2 in pool)
    return combos


def combo_pool_for_type(combo_type: int) -> list[tuple[int, str, str]]:
    pools = generate_valid_combinations()
    return [(combo_type, p1, p2) for p1, p2 in pools[combo_type - 1]]


def choose_next_combo(root: Path, min_parsed: int, combo_type: int) -> tuple[int, str, str]:
    pool = combo_pool_for_type(combo_type)
    done_pairs = existing_valid_pairs(root, min_parsed)
    unseen = [(ct, p1, p2) for ct, p1, p2 in pool if (p1, p2) not in done_pairs]
    if unseen:
        return random.choice(unseen)
    return random.choice(pool)


def valid_count_for_type(root: Path, min_parsed: int, combo_type: int) -> int:
    type_root = root / TYPE_FOLDERS[combo_type]
    return valid_count(type_root, min_parsed)


def run_until_valid(
    model_key: str,
    target: int,
    min_parsed: int,
    context: int,
    simulate: int,
    capture_rounds: int,
    include_kb: bool,
    max_attempts: int,
    combo_types: list[int],
) -> None:
    model_name, api_type = MODEL_MAP[model_key]
    root = result_root(model_name, context, simulate)

    for combo_type in combo_types:
        attempts = 0
        while True:
            removed = purge_invalid(root / TYPE_FOLDERS[combo_type], min_parsed)
            count = valid_count_for_type(root, min_parsed, combo_type)
            print(f"\n[{model_key}] type{combo_type} valid={count}/{target}, removed_invalid={removed}")
            if count >= target:
                print(f"[{model_key}] type{combo_type} done")
                break
            if attempts >= max_attempts:
                raise RuntimeError(
                    f"{model_key} type{combo_type}: reached max attempts ({max_attempts}) "
                    f"with {count}/{target} valid"
                )

            combo_type, p1, p2 = choose_next_combo(root, min_parsed, combo_type)
            attempts += 1
            print(f"[{model_key}] type{combo_type} attempt {attempts}: {p1} vs {p2}")

            try:
                exp = run_simulation_experiment(
                    p1,
                    p2,
                    context,
                    simulate,
                    capture_rounds,
                    api_type,
                    model_name,
                    include_kb,
                )
            except Exception as exc:
                exp = {
                    "success": False,
                    "error": str(exc),
                    "player1_id": p1,
                    "player2_id": p2,
                }

            exp["combo_type"] = combo_type
            out_dir = root / TYPE_FOLDERS[combo_type]
            path = Path(save_result(exp, str(out_dir)))

            if is_valid_json(path, min_parsed):
                print(f"[{model_key}] saved valid: {path.name}")
            else:
                got = parsed_rounds(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else 0
                print(f"[{model_key}] invalid parsed_rounds={got}; deleting and will refill")
                delete_pair(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run exp2 until each selected model has N valid results per type.")
    parser.add_argument("--models", nargs="+", default=["gpt-4.1", "qwen-api"],
                        choices=list(MODEL_MAP.keys()))
    parser.add_argument("--target", type=int, default=60)
    parser.add_argument("--types", nargs="+", type=int, default=[2, 3], choices=[1, 2, 3],
                        help="Combo types to fill. Default: 2 3 (Markov at P1 and Markov at P2).")
    parser.add_argument("--min-parsed", type=int, default=1000)
    parser.add_argument("--context", type=int, default=1000)
    parser.add_argument("--simulate", type=int, default=1500)
    parser.add_argument("--capture-rounds", type=int, default=None)
    parser.add_argument("--max-attempts", type=int, default=240)
    parser.add_argument("--no-kb", action="store_true")
    args = parser.parse_args()

    capture_rounds = args.capture_rounds if args.capture_rounds is not None else args.simulate
    for model_key in args.models:
        run_until_valid(
            model_key=model_key,
            target=args.target,
            min_parsed=args.min_parsed,
            context=args.context,
            simulate=args.simulate,
            capture_rounds=capture_rounds,
            include_kb=not args.no_kb,
            max_attempts=args.max_attempts,
            combo_types=args.types,
        )


if __name__ == "__main__":
    main()
