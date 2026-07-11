"""Run the isolated Exp1 trajectory-representation ablation.

This tool never reads from or writes to the main ``batch_results`` tree.  A
manifest is generated once and reused by every model/representation cell, so
the only experimental difference within a game is the trajectory block.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "tools")]

from analysis.llm import (  # noqa: E402
    build_game_analysis_prompt,
    get_response_openai,
    get_response_openrouter,
)
from core.game import Game  # noqa: E402
from parse_analysis import parse_final_answer  # noqa: E402

MARKOV = tuple("XYZ")
ORDINARY = tuple("DEFGHIJKLMNOP")  # matches Exp1's Markov-game exclusion of A/B/C
MODELS = {
    "gpt5mini": ("openai", "gpt-5-mini"),
    "gpt4.1": ("openai", "gpt-4.1-2025-04-14"),
    "gemini": ("openrouter", "google/gemini-3-flash-preview"),
}
REPRESENTATIONS = ("structured_table",)
PROMPT_VERSION = "canonical-exp1-tsv-only-v3"


def moves(text: str) -> list[str]:
    symbol = {"Rock": "R", "Paper": "P", "Scissors": "S", "R": "R", "P": "P", "S": "S"}
    values = [symbol.get(x, x) for x in text.split()]
    if any(x not in {"R", "P", "S"} for x in values):
        raise ValueError("Trajectory contains a non-R/P/S symbol")
    return values


def structured_table(p1: list[str], p2: list[str]) -> str:
    rows = ["Observed trajectory (all rounds):", "", "Round\tP1\tP2"]
    rows.extend(f"{i}\t{a}\t{b}" for i, (a, b) in enumerate(zip(p1, p2), 1))
    return "\n".join(rows)


def build_prompt(game: dict, representation: str) -> str:
    p1, p2 = moves(game["p1_trajectory"]), moves(game["p2_trajectory"])
    if representation != "structured_table":
        raise ValueError(f"Unsupported representation: {representation}")
    block = structured_table(p1, p2)
    return build_game_analysis_prompt(
        game["p1_trajectory"], game["p2_trajectory"], game["p1_wins"],
        game["p2_wins"], game["draws"], game["rounds"],
        trajectory_block=block,
    )


def make_manifest(path: Path, per_position: int, rounds: int, seed: int) -> list[dict]:
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["settings"] != {"per_position": per_position, "rounds": rounds, "seed": seed}:
            raise ValueError("Existing manifest settings differ; choose another --output-dir")
        return data["games"]
    random.seed(seed)
    games = []
    pairs = [(random.choice(MARKOV), random.choice(ORDINARY)) for _ in range(per_position)]
    pairs += [(random.choice(ORDINARY), random.choice(MARKOV)) for _ in range(per_position)]
    for i, (a, b) in enumerate(pairs, 1):
        result = Game.simulate(a, b, rounds)
        games.append({"game_id": i, "p1_id": a, "p2_id": b, "rounds": rounds,
                      "p1_wins": result.player1_wins, "p2_wins": result.player2_wins,
                      "draws": result.draws, "p1_trajectory": result.get_trajectory_string(1),
                      "p2_trajectory": result.get_trajectory_string(2)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"settings": {"per_position": per_position, "rounds": rounds, "seed": seed}, "games": games}, indent=2), encoding="utf-8")
    return games


def call(model_key: str, prompt: str) -> tuple[dict, str]:
    provider, name = MODELS[model_key]
    # These are the same provider defaults used by the main experiment.
    if provider == "openai":
        return get_response_openai(prompt, model_name=name)
    if provider == "openrouter":
        return get_response_openrouter(prompt, model_name=name, max_tokens=32768)
    raise ValueError(f"Unsupported provider: {provider}")


def valid_predictions(parsed: dict | None, rounds: int) -> bool:
    if not parsed or set(parsed) < {"player1", "player2"}:
        return False
    for key in ("player1", "player2"):
        player = parsed[key]
        if player.get("identity") not in set("ABCDEFGHIJKLMNOPXYZ"):
            return False
        counts = player.get("counts", {})
        if set(counts) < {"rock", "paper", "scissors"}:
            return False
        if sum(counts[x] for x in ("rock", "paper", "scissors")) != rounds:
            return False
    return True


def summarize(out: Path) -> None:
    rows = [r for p in out.glob("results/*/*/*/*/game_*.json")
            if (r := json.loads(p.read_text(encoding="utf-8"))).get("prompt_version") == PROMPT_VERSION]
    detail, grouped = [], {}
    for r in rows:
        pred = r.get("predictions") or {}
        for i in (1, 2):
            truth, guess = r[f"p{i}_id"], (pred.get(f"player{i}") or {}).get("identity")
            kind = "markov" if truth in MARKOV else "ordinary"
            item = {"model": r["model"], "rounds": r["rounds"], "markov_position": r["markov_position"], "representation": r["representation"], "game_id": r["game_id"],
                    "player": i, "player_type": kind, "truth": truth, "prediction": guess or "", "correct": int(guess == truth)}
            detail.append(item); grouped.setdefault((r["model"], r["rounds"], r["markov_position"], r["representation"], kind), []).append(item["correct"])
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("detail.csv", detail), ("accuracy.csv", [
        {"model": k[0], "rounds": k[1], "markov_position": k[2], "representation": k[3], "player_type": k[4], "correct": sum(v), "total": len(v), "accuracy": sum(v)/len(v)}
        for k, v in sorted(grouped.items())])):
        if data:
            with (out / name).open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=data[0].keys()); w.writeheader(); w.writerows(data)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    ap.add_argument("--representations", nargs="+", choices=REPRESENTATIONS, default=list(REPRESENTATIONS))
    ap.add_argument("--markov-position", choices=("p1", "p2", "both"), default="both")
    ap.add_argument("--per-position", type=int, default=50, help="games for each Markov position")
    ap.add_argument("--rounds", nargs="+", type=int, default=[100, 1000])
    ap.add_argument("--max-attempts", type=int, default=0, help="attempt cap per game; 0 retries until valid")
    ap.add_argument("--seed", type=int, default=20260711)
    ap.add_argument("--output-dir", type=Path, default=ROOT / "exp1(strategy)" / "prompting_ablation")
    ap.add_argument("--dry-run", action="store_true", help="build manifest/prompts without API calls")
    ap.add_argument("--no-summarize", action="store_true", help="do not rewrite shared CSV files (use for parallel cells)")
    ap.add_argument("--summarize-only", action="store_true", help="only combine completed cells into CSV files")
    args = ap.parse_args()
    if args.summarize_only:
        summarize(args.output_dir)
        print(f"Summary written: {args.output_dir / 'detail.csv'} and {args.output_dir / 'accuracy.csv'}", flush=True)
        return
    for rounds in args.rounds:
        games = make_manifest(args.output_dir / "manifests" / f"rounds_{rounds}.json", args.per_position, rounds, args.seed + rounds)
        if args.markov_position != "both":
            position_index = 1 if args.markov_position == "p1" else 2
            games = [g for g in games if g[f"p{position_index}_id"] in MARKOV]
        for model in args.models:
            for rep in args.representations:
                position = args.markov_position
                folder = args.output_dir / "results" / model / str(rounds) / position / rep; folder.mkdir(parents=True, exist_ok=True)
                failures = folder / "failed_attempts"; failures.mkdir(exist_ok=True)
                total = len(games)
                already_valid = 0
                for game in games:
                    existing = folder / f"game_{game['game_id']:03d}.json"
                    if existing.exists():
                        old = json.loads(existing.read_text(encoding="utf-8"))
                        already_valid += int(old.get("prompt_version") == PROMPT_VERSION and valid_predictions(old.get("predictions"), rounds))
                print(
                    f"\n[CELL] model={model} rounds={rounds} Markov-as={position.upper()} representation={rep} "
                    f"valid={already_valid}/{total}", flush=True
                )
                completed = already_valid
                for slot, game in enumerate(games, 1):
                    target = folder / f"game_{game['game_id']:03d}.json"
                    if target.exists():
                        old = json.loads(target.read_text(encoding="utf-8"))
                        if old.get("prompt_version") == PROMPT_VERSION and valid_predictions(old.get("predictions"), rounds):
                            continue
                    prompt = build_prompt(game, rep)
                    if args.dry_run:
                        (folder / f"game_{game['game_id']:03d}.prompt.txt").write_text(prompt, encoding="utf-8"); continue
                    for attempt in itertools.count(1):
                        output = ""
                        print(
                            f"[REQUEST] sample={slot:03d}/{total:03d} game_id={game['game_id']:03d} "
                            f"Markov-as-{'P1' if game['p1_id'] in MARKOV else 'P2'} attempt={attempt}",
                            flush=True,
                        )
                        try:
                            response, output = call(model, prompt); parsed = parse_final_answer(output)
                            if not valid_predictions(parsed, rounds):
                                raise ValueError("Final Answer parse failed or action counts do not sum to total rounds")
                            record = {**game, "model": model, "model_id": MODELS[model][1], "representation": rep,
                                      "prompt_version": PROMPT_VERSION, "markov_position": position,
                                      "attempt": attempt, "created_at": datetime.now(timezone.utc).isoformat(),
                                      "predictions": parsed, "raw_output": output,
                                      "usage": response.get("usage", response.get("usageMetadata", {}))}
                            target.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
                            completed += 1
                            print(f"[SUCCESS] game={game['game_id']:03d} valid={completed}/{total}", flush=True)
                            break
                        except Exception as exc:
                            print(f"[RETRY] game={game['game_id']:03d} attempt={attempt} reason={exc}", flush=True)
                            failed = {"game_id": game["game_id"], "attempt": attempt, "error": str(exc),
                                      "raw_output": output}
                            (failures / f"game_{game['game_id']:03d}_attempt_{attempt:02d}.json").write_text(
                                json.dumps(failed, ensure_ascii=False, indent=2), encoding="utf-8")
                            if args.max_attempts and attempt >= args.max_attempts:
                                raise RuntimeError(
                                    f"Could not obtain a parse-valid result: {model}/{rounds}/{rep}/game {game['game_id']}"
                                ) from exc
                print(f"[DONE] model={model} rounds={rounds} representation={rep} valid={completed}/{total}", flush=True)
    if not args.no_summarize:
        summarize(args.output_dir)


if __name__ == "__main__":
    main()
