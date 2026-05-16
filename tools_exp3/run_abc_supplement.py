import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP3_ROOT = ROOT / "exp3(complex_markov)"

SUPPLEMENT = {
    "qrs": "Q",
    "tuv": "T",
    "xyz_opp": "x",
}
ABC = ["A", "B", "C"]
TYPE_FOLDERS = {
    2: "type2_markov_p1",
    3: "type3_markov_p2",
}
BAD_ROOT = EXP3_ROOT / "bad_runs" / "abc_supplement"


def clean_model_name(model: str) -> str:
    return model.replace("/", "_")


def pair_files(markov_set: str, model: str, context: int, simulate: int, combo_type: int, p1: str, p2: str) -> list[Path]:
    folder = (
        EXP3_ROOT
        / "generation"
        / markov_set
        / clean_model_name(model)
        / f"ctx{context}_sim{simulate}"
        / TYPE_FOLDERS[combo_type]
    )
    return sorted(folder.glob(f"sim_{p1}_vs_{p2}_*.json"))


def has_complete_success(path: Path, expected_rounds: int) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        llm = data.get("llm_simulation", {})
        return (
            data.get("success") is True
            and llm.get("complete") is True
            and llm.get("parsed_rounds") == expected_rounds
        )
    except Exception:
        return False


def quarantine_bad_files(files: list[Path], expected_rounds: int) -> None:
    for path in files:
        if has_complete_success(path, expected_rounds):
            continue
        rel = path.relative_to(EXP3_ROOT / "generation")
        dst = BAD_ROOT / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            dst = dst.with_name(f"{dst.stem}_old{dst.suffix}")
        shutil.move(str(path), str(dst))
        print(f"QUARANTINE bad file: {path} -> {dst}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the 18-game ABC supplement for Exp3.")
    parser.add_argument("--model", default="deepseek-reasoner")
    parser.add_argument("--context", type=int, default=1000)
    parser.add_argument("--simulate", type=int, default=1500)
    parser.add_argument("--capture-rounds", type=int, default=None)
    parser.add_argument("--markov-set", choices=list(SUPPLEMENT.keys()), default=None)
    parser.add_argument("--force", action="store_true", help="Run even if a matching pair already exists.")
    args = parser.parse_args()

    capture = args.capture_rounds if args.capture_rounds is not None else args.simulate
    jobs = []
    selected = {args.markov_set: SUPPLEMENT[args.markov_set]} if args.markov_set else SUPPLEMENT
    for markov_set, markov_id in selected.items():
        for nm in ABC:
            jobs.append((markov_set, markov_id, nm, 2, markov_id, nm))
            jobs.append((markov_set, markov_id, nm, 3, nm, markov_id))

    runnable = []
    for markov_set, _markov_id, _nm, combo_type, p1, p2 in jobs:
        files = pair_files(markov_set, args.model, args.context, args.simulate, combo_type, p1, p2)
        complete = any(has_complete_success(path, capture) for path in files)
        if files:
            quarantine_bad_files(files, capture)
        if complete and not args.force:
            print(f"SKIP complete: {markov_set} {p1} vs {p2}")
            continue
        if files and not complete:
            print(f"RERUN incomplete/failed: {markov_set} {p1} vs {p2}")
        runnable.append((markov_set, p1, p2))

    print(f"ABC supplement jobs to run: {len(runnable)}")
    for idx, (markov_set, p1, p2) in enumerate(runnable, 1):
        cmd = [
            sys.executable,
            str(ROOT / "tools_exp3" / "run_generation.py"),
            "--markov-set",
            markov_set,
            "--p1",
            p1,
            "--p2",
            p2,
            "--context",
            str(args.context),
            "--simulate",
            str(args.simulate),
            "--capture-rounds",
            str(capture),
            "--model",
            args.model,
        ]
        print(f"\n[{idx}/{len(runnable)}] {markov_set}: {p1} vs {p2}")
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
