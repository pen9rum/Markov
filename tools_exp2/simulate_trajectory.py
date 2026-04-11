"""
LLM Trajectory Simulation Experiment

Given a previous trajectory of two anonymous players, ask the LLM to simulate
the next N rounds. Compare LLM output against the real game continuation.

Experiment types (3 kinds, non-overlapping combinations):
  Type 1 — both non-Markov      : A-P vs A-P  (p1 != p2)
  Type 2 — Markov is Player 1   : X/Y/Z vs A-P  (excluding A/B/C as non-Markov)
  Type 3 — Markov is Player 2   : A-P vs X/Y/Z  (excluding A/B/C as non-Markov)
"""
import os
import sys
import json
import argparse
import re
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import Game
from core.players import PLAYER_CONFIGS, Action


OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'exp2(generation_blind)', 'generation')

ACTIONS = {"Rock", "Paper", "Scissors"}
MARKOV_PLAYERS = {'X', 'Y', 'Z'}
NON_MARKOV_PLAYERS = {k for k in PLAYER_CONFIGS if k not in MARKOV_PLAYERS}
# Exclude pure-static trivial players when paired with Markov (same as batch_experiment.py)
NON_MARKOV_EXCL_ABC = NON_MARKOV_PLAYERS - {'A', 'B', 'C'}

TYPE_FOLDERS = {
    1: "type1_non_markov",
    2: "type2_markov_p1",
    3: "type3_markov_p2",
}


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_valid_combinations() -> tuple[list, list, list]:
    """
    Return (type1_pool, type2_pool, type3_pool).

    Type 1: both non-Markov  — all ordered pairs (p1, p2) with p1 != p2
    Type 2: Markov is P1     — (markov, non_markov) with non_markov not in {A,B,C}
    Type 3: Markov is P2     — (non_markov, markov) with non_markov not in {A,B,C}
    """
    type1 = [
        (p1, p2)
        for p1 in NON_MARKOV_PLAYERS
        for p2 in NON_MARKOV_PLAYERS
        if p1 != p2
    ]
    type2 = [
        (m, nm)
        for m in MARKOV_PLAYERS
        for nm in NON_MARKOV_EXCL_ABC
    ]
    type3 = [
        (nm, m)
        for nm in NON_MARKOV_EXCL_ABC
        for m in MARKOV_PLAYERS
    ]
    return type1, type2, type3


def get_existing_combinations(model_name: str, context: int, simulate: int,
                               combo_type: int) -> set:
    """Return the set of (p1, p2) pairs already saved for this config."""
    clean = model_name.replace('/', '_').replace('\\', '_')
    folder = os.path.join(
        OUTPUT_DIR, clean,
        f"ctx{context}_sim{simulate}",
        TYPE_FOLDERS[combo_type]
    )
    existing = set()
    if not os.path.isdir(folder):
        return existing
    for fname in os.listdir(folder):
        if fname.startswith("sim_") and fname.endswith(".json"):
            # sim_<p1>_vs_<p2>_<timestamp>.json
            m = re.match(r"sim_([A-Z])_vs_([A-Z])_", fname)
            if m:
                existing.add((m.group(1), m.group(2)))
    return existing


def select_combinations(combo_type: int, count: int,
                         model_name: str, context: int, simulate: int) -> list:
    """Sample `count` unique combinations of the given type, excluding already-run ones."""
    import random
    t1, t2, t3 = generate_valid_combinations()
    pool = {1: t1, 2: t2, 3: t3}[combo_type]

    existing = get_existing_combinations(model_name, context, simulate, combo_type)
    full_pool = list(pool)  # keep original for refill

    if existing:
        print(f"  Already have {len(existing)} results for type {combo_type}, excluding them first.")
        remaining = [c for c in full_pool if c not in existing]
    else:
        remaining = full_pool

    print(f"  Type {combo_type}: {len(remaining)} unseen, {len(full_pool)} total")

    if not full_pool:
        print("  Warning: pool is empty.")
        return []

    # Fill up to count: use unseen first, then re-sample from full pool for remainder
    if len(remaining) >= count:
        return random.sample(remaining, count)
    else:
        result = list(remaining)  # take all unseen
        still_need = count - len(result)
        refill = random.choices(full_pool, k=still_need)  # allow repeats from full pool
        print(f"  All unseen combos used ({len(remaining)}), re-sampling {still_need} from full pool.")
        result.extend(refill)
        return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_simulation_prompt(
    player1_trajectory: str,
    player2_trajectory: str,
    context_rounds: int,
    simulate_rounds: int,
    include_knowledge_base: bool = True,
) -> str:
    """
    Build the English prompt for the simulation task.

    Key design:
    - Include the strategy pool knowledge base (so LLM knows what strategies exist)
    - Do NOT reveal which strategy each player uses — LLM must infer from trajectory
    - Tell LLM: both players are from the pool; at most one is Markov (X/Y/Z)
    - Distribution constraint: each 100-round window should match the inferred pattern
    - Final output: SIMULATION block with P1: / P2: flat space-separated lists
    """
    knowledge_section = ""
    if include_knowledge_base:
        from analysis.llm import get_player_knowledge_base
        knowledge_section = get_player_knowledge_base() + "\n\n"

    p1_moves = player1_trajectory.split()
    p2_moves = player2_trajectory.split()
    assert len(p1_moves) == len(p2_moves) == context_rounds

    history_lines = "\n".join(
        f"Round {i+1}: P1={p1_moves[i]}, P2={p2_moves[i]}"
        for i in range(context_rounds)
    )

    prompt = f"""{knowledge_section}## Rock-Paper-Scissors Trajectory Simulation Task

You are observing a Rock-Paper-Scissors game between two **anonymous** players (referred to as P1 and P2).

**Player pool rules:**
- Both players are drawn from the strategy pool described above (strategies A through Z).
- At most ONE of them is a Markov/reactive player (X, Y, or Z). The other is a distribution player (A–P).
- It is also possible that BOTH are distribution players (neither is Markov).
- You do NOT know which specific strategy each player uses — you must infer it from the trajectory below.

### Previous Game History ({context_rounds} rounds):
{history_lines}

---

### Your Task
Simulate the next **{simulate_rounds} rounds** of this game (rounds {context_rounds + 1} to {context_rounds + simulate_rounds}).

**MANDATORY: You MUST attempt to generate all {simulate_rounds} rounds no matter what.**
Refusing, asking for clarification, or generating fewer rounds is not acceptable.
If perfect randomness is difficult, approximate it — a best-effort simulation is required.

**Step 1 — Strategy Inference:**
Analyze the history carefully to identify each player's most likely strategy.
Follow the same reasoning steps used in game analysis:
(1) Check whether either player's moves are reactive to the other's previous move (Markov pattern).
(2) If a Markov player is present, identify the other player's distribution strategy.
(3) If neither is Markov, identify both players' distribution strategies.

**Step 2 — Simulation:**
Generate exactly {simulate_rounds} rounds of play consistent with the strategies you inferred.

**Distribution Consistency Constraint (CRITICAL):**
- Do NOT batch actions by type (e.g., do NOT output 500 Rocks followed by 300 Papers then 200 Scissors).
- In every consecutive 100-round window of your simulation, each player's Rock/Paper/Scissors
  distribution must be approximately consistent with the overall distribution you inferred from
  the history. Actions must be realistically interleaved throughout all {simulate_rounds} rounds.

**Avoid Repetitive Patterns (best effort):**
- Prefer varied sequences over mechanical cycles like "Rock Paper Rock Paper Rock Paper ...".
- The sequence should resemble genuine random draws from the inferred probabilities.
- If exact randomness is hard to maintain over 1000 rounds, do your best — some variation is better than none.

**Output Format (REQUIRED — place this block at the very end of your response):**
After your analysis, output the simulation round by round.
Each round is ONE line containing BOTH players' moves simultaneously.

SIMULATION:
P1_identity: <single letter from A-Z>
P2_identity: <single letter from A-Z>
Round 1: <P1 action> <P2 action>
Round 2: <P1 action> <P2 action>
Round 3: <P1 action> <P2 action>
...
Round {simulate_rounds}: <P1 action> <P2 action>

Rules:
- P1_identity and P2_identity must each be a single uppercase letter from the valid strategy list (A–Z).
- Each <action> must be exactly one of: Rock, Paper, Scissors (capitalize first letter only).
- Every round line must have exactly two actions: P1's action first, then P2's action, separated by a space.
- There must be exactly {simulate_rounds} Round lines (Round 1 through Round {simulate_rounds}).
- No extra text inside the SIMULATION block other than the identity lines and round lines.
"""
    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_simulation_output(
    raw_output: str,
    capture_rounds: int,
) -> dict | None:
    """
    Parse the chunked SIMULATION block from LLM output.

    Expected format:
        SIMULATION:
        P1_identity: H
        P2_identity: X
        CHUNK_1 (rounds 1-100):
        P1: Rock Paper ... (100 moves)
        P2: Scissors Rock ... (100 moves)
        CHUNK_2 (rounds 101-200):
        P1: ...
        P2: ...
        ...

    Collects all P1 and P2 lines across all chunks and concatenates them.
    If the model returns fewer than capture_rounds rounds, keeps the partial
    output and marks it as incomplete instead of discarding it.
    """
    # Find the last SIMULATION: block; if missing, treat the whole output as the block
    sim_matches = list(re.finditer(r"SIMULATION\s*:(.*?)(?=SIMULATION\s*:|$)", raw_output, re.DOTALL | re.IGNORECASE))
    if sim_matches:
        sim_block = sim_matches[-1].group(1)
    else:
        # LLM skipped the header — use full output as fallback
        sim_block = raw_output

    # Parse identities
    p1_id_match = re.search(r"P1_identity\s*:\s*([A-Z])", sim_block, re.IGNORECASE)
    p2_id_match = re.search(r"P2_identity\s*:\s*([A-Z])", sim_block, re.IGNORECASE)
    p1_identity = p1_id_match.group(1).upper() if p1_id_match else None
    p2_identity = p2_id_match.group(1).upper() if p2_id_match else None
    if not p1_identity or not p2_identity:
        print("  ⚠️  Could not find P1_identity or P2_identity.")

    # Parse round-by-round lines: "Round N: <action> <action>"
    round_pattern = re.compile(
        r"Round\s+\d+\s*:\s*(Rock|Paper|Scissors)\s+(Rock|Paper|Scissors)",
        re.IGNORECASE,
    )

    p1_moves: list[str] = []
    p2_moves: list[str] = []

    for m in round_pattern.finditer(sim_block):
        p1_moves.append(m.group(1).capitalize())
        p2_moves.append(m.group(2).capitalize())

    if not p1_moves:
        print("  ⚠️  No round lines found in SIMULATION block.")
        return None

    # P1 and P2 are always aligned (same line), so lengths are always equal.
    # Accept if >= capture_rounds (trim excess); keep partial output if shorter.
    got = len(p1_moves)
    complete = got >= capture_rounds
    if got < capture_rounds:
        print(f"  ⚠️  Incomplete simulation: wanted {capture_rounds} rounds, got {got}. Keeping partial parse.")
    if got > capture_rounds:
        print(f"  ⚠️  Over-generated ({got} rounds), trimming to {capture_rounds}.")
    p1_moves = p1_moves[:capture_rounds]
    p2_moves = p2_moves[:capture_rounds]

    return {
        "p1_identity": p1_identity,
        "p2_identity": p2_identity,
        "p1_moves": p1_moves,
        "p2_moves": p2_moves,
        "parsed_rounds": len(p1_moves),
        "complete": complete,
    }


# ---------------------------------------------------------------------------
# Stats and accuracy
# ---------------------------------------------------------------------------

def compute_trajectory_stats(moves: list[str]) -> dict:
    total = len(moves)
    rock = moves.count("Rock")
    paper = moves.count("Paper")
    scissors = moves.count("Scissors")
    return {
        "total": total,
        "rock": rock,
        "paper": paper,
        "scissors": scissors,
        "rock_pct": rock / total * 100 if total else 0,
        "paper_pct": paper / total * 100 if total else 0,
        "scissors_pct": scissors / total * 100 if total else 0,
    }


def compute_window_stats(moves: list[str], window: int = 100) -> list[dict]:
    """Break moves into windows and compute per-window distribution."""
    result = []
    for start in range(0, len(moves), window):
        chunk = moves[start:start + window]
        if chunk:
            result.append(compute_trajectory_stats(chunk))
    return result


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_simulation_experiment(
    player1_id: str,
    player2_id: str,
    context_rounds: int,
    simulate_rounds: int,
    capture_rounds: int,
    api_type: str,
    model_name: str,
    include_knowledge_base: bool = True,
) -> dict:
    """
    Run one simulation experiment:
    1. Simulate the real game for context_rounds only (no future ground truth needed —
       distribution players have many valid continuations, so per-round comparison is meaningless).
    2. Give the context_rounds trajectory to the LLM.
    3. LLM infers player strategies and simulates simulate_rounds.
    4. Save LLM's inferred identities + simulated trajectory + distribution stats.
    """
    print(f"  Running real game ({context_rounds} rounds for context)...")
    real_result = Game.simulate(player1_id, player2_id, context_rounds)

    context_p1 = [a.value for a in real_result.player1_trajectory]
    context_p2 = [a.value for a in real_result.player2_trajectory]

    prompt = build_simulation_prompt(
        " ".join(context_p1),
        " ".join(context_p2),
        context_rounds,
        simulate_rounds,
        include_knowledge_base,
    )

    print(f"  Calling {api_type.upper()} API ({model_name})...")

    from analysis.llm import (
        get_response,
        get_response_gemini,
        get_response_openai,
        get_response_deepseek,
        get_response_jamba,
    )

    try:
        if api_type == "gemini":
            _, raw_output = get_response_gemini(prompt, model_name=model_name, max_tokens=16384)
        elif api_type == "openai":
            _, raw_output = get_response_openai(prompt, model_name=model_name, max_tokens=16384)
        elif api_type == "deepseek":
            # Let DeepSeek client decide model-specific defaults:
            # - deepseek-chat: max_tokens <= 8192
            # - deepseek-reasoner: max_tokens = 65536
            _, raw_output = get_response_deepseek(prompt, model_name=model_name)
        elif api_type == "jamba":
            _, raw_output = get_response_jamba(prompt, model_name=model_name, max_tokens=4096)
        else:  # qwen
            _, raw_output = get_response(prompt, model_name=model_name, max_tokens=16384)
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "player1_id": player1_id,
            "player2_id": player2_id,
        }

    parsed = parse_simulation_output(raw_output, capture_rounds)

    result = {
        "success": parsed is not None,
        "capture_rounds": capture_rounds,
        "player1_id": player1_id,
        "player2_id": player2_id,
        "player1_name": PLAYER_CONFIGS[player1_id][0],
        "player2_name": PLAYER_CONFIGS[player2_id][0],
        "context_rounds": context_rounds,
        "simulate_rounds": simulate_rounds,
        "model": model_name,
        "api_type": api_type,
        "context": {
            "p1_trajectory": " ".join(context_p1),
            "p2_trajectory": " ".join(context_p2),
            "p1_stats": compute_trajectory_stats(context_p1),
            "p2_stats": compute_trajectory_stats(context_p2),
        },
    }

    if parsed:
        llm_p1 = parsed["p1_moves"]
        llm_p2 = parsed["p2_moves"]
        result["llm_simulation"] = {
            "raw_output": raw_output,
            "p1_identity": parsed["p1_identity"],
            "p2_identity": parsed["p2_identity"],
            "parsed_rounds": parsed["parsed_rounds"],
            "complete": parsed["complete"],
            "p1_trajectory": " ".join(llm_p1),
            "p2_trajectory": " ".join(llm_p2),
            "p1_stats": compute_trajectory_stats(llm_p1),
            "p2_stats": compute_trajectory_stats(llm_p2),
            "p1_window_stats": compute_window_stats(llm_p1),
            "p2_window_stats": compute_window_stats(llm_p2),
        }
        print(f"  ✓ Parsed {len(llm_p1)} rounds (capture target: {capture_rounds})")
        if not parsed["complete"]:
            print(f"    ⚠️  Incomplete: only {len(llm_p1)} rounds captured")
        print(f"    LLM guessed: P1={parsed['p1_identity']}, P2={parsed['p2_identity']}")
        print(f"    P1 dist: R={result['llm_simulation']['p1_stats']['rock_pct']:.1f}% "
              f"P={result['llm_simulation']['p1_stats']['paper_pct']:.1f}% "
              f"S={result['llm_simulation']['p1_stats']['scissors_pct']:.1f}%")
        print(f"    P2 dist: R={result['llm_simulation']['p2_stats']['rock_pct']:.1f}% "
              f"P={result['llm_simulation']['p2_stats']['paper_pct']:.1f}% "
              f"S={result['llm_simulation']['p2_stats']['scissors_pct']:.1f}%")
    else:
        result["llm_simulation"] = {
            "raw_output": raw_output,
            "parsed_rounds": 0,
            "complete": False,
        }
        result["error"] = "Incomplete or unparseable simulation output"
        print("  ✗ Parsing failed (raw output saved for debugging)")

    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_result(exp: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p1 = exp.get("player1_id", "?")
    p2 = exp.get("player2_id", "?")
    base = f"sim_{p1}_vs_{p2}_{timestamp}"
    json_path = os.path.join(output_dir, base + ".json")
    txt_path  = os.path.join(output_dir, base + ".txt")

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("LLM Trajectory Simulation Report\n")
        f.write("=" * 80 + "\n\n")

        if not exp.get("success"):
            f.write(f"FAILED: {exp.get('error', 'Unknown')}\n\n")
            f.write(f"Match    : {p1} ({exp.get('player1_name','')}) vs {p2} ({exp.get('player2_name','')})\n")
            if exp.get("model"):
                f.write(f"Model    : {exp.get('model')} ({exp.get('api_type','')})\n")
            if exp.get("context_rounds") is not None:
                f.write(f"Context  : {exp.get('context_rounds')} rounds\n")
            if exp.get("simulate_rounds") is not None:
                f.write(f"Simulate : {exp.get('simulate_rounds')} rounds\n")
            f.write(f"Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            sim = exp.get("llm_simulation", {})
            raw = sim.get("raw_output", "") if isinstance(sim, dict) else ""
            if raw:
                f.write("-" * 80 + "\n")
                f.write("LLM RAW OUTPUT (FAILED CASE)\n")
                f.write("-" * 80 + "\n")
                f.write(raw)
                f.write("\n")
            return json_path

        f.write(f"Match    : {p1} ({exp.get('player1_name','')}) vs {p2} ({exp.get('player2_name','')})\n")
        f.write(f"Model    : {exp['model']} ({exp['api_type']})\n")
        f.write(f"Context  : {exp['context_rounds']} rounds\n")
        f.write(f"Simulate : {exp['simulate_rounds']} rounds\n")
        f.write(f"Time     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        def write_stats_line(label, s):
            f.write(f"{label}: Rock={s['rock']}({s['rock_pct']:.1f}%) "
                    f"Paper={s['paper']}({s['paper_pct']:.1f}%) "
                    f"Scissors={s['scissors']}({s['scissors_pct']:.1f}%)\n")

        f.write("-" * 80 + "\n")
        f.write("CONTEXT (given to LLM)\n")
        f.write("-" * 80 + "\n")
        ctx = exp["context"]
        f.write(f"P1: {ctx['p1_trajectory']}\n")
        f.write(f"P2: {ctx['p2_trajectory']}\n")
        write_stats_line("P1 dist", ctx["p1_stats"])
        write_stats_line("P2 dist", ctx["p2_stats"])
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("LLM SIMULATION\n")
        f.write("-" * 80 + "\n")
        sim = exp.get("llm_simulation", {})
        f.write(f"P1 inferred identity : {sim.get('p1_identity', 'N/A')}  "
                f"(ground truth: {p1})\n")
        f.write(f"P2 inferred identity : {sim.get('p2_identity', 'N/A')}  "
                f"(ground truth: {p2})\n")
        f.write(f"Parsed rounds        : {sim.get('parsed_rounds', 0)}\n\n")
        f.write(f"P1: {sim.get('p1_trajectory', '')}\n")
        f.write(f"P2: {sim.get('p2_trajectory', '')}\n")
        if sim.get("p1_stats") and sim["p1_stats"].get("total", 0) > 0:
            write_stats_line("P1 dist", sim["p1_stats"])
            write_stats_line("P2 dist", sim["p2_stats"])
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("LLM RAW OUTPUT\n")
        f.write("-" * 80 + "\n")
        f.write(sim["raw_output"])
        f.write("\n\n" + "=" * 80 + "\n")
        f.write("End of Report\n")
        f.write("=" * 80 + "\n")

    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODEL_MAP = {
    "qwen-api":           ("qwen-plus",            "qwen"),
    "gemini":             ("gemini-3-flash-preview","gemini"),
    "gpt-5-mini":         ("gpt-5-mini",            "openai"),
    "gpt-5":              ("gpt-5",                 "openai"),
    "deepseek-chat":      ("deepseek-chat",          "deepseek"),
    "deepseek-reasoner":  ("deepseek-reasoner",      "deepseek"),
    "jamba-mini":         ("jamba-mini",             "jamba"),
    "jamba-large":        ("jamba-large",            "jamba"),
}


def main():
    parser = argparse.ArgumentParser(
        description="LLM Trajectory Simulation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single game
  python tools/simulate_trajectory.py --p1 D --p2 X --context 100 --simulate 1000 --model gemini

  # Batch: 10 type1 + 5 type2 + 5 type3
  python tools/simulate_trajectory.py --type1 10 --type2 5 --type3 5 --context 100 --simulate 1000 --model gemini

  # Run all combinations
  python tools/simulate_trajectory.py --all --context 100 --simulate 1000 --model deepseek-chat

  # Without knowledge base (test pure trajectory inference)
  python tools/simulate_trajectory.py --p1 D --p2 D --context 100 --simulate 1000 --model gemini --no-kb
        """,
    )

    # Single-game mode
    parser.add_argument("--p1", type=str, help="Player 1 ID (A-Z), for single-game mode")
    parser.add_argument("--p2", type=str, help="Player 2 ID (A-Z), for single-game mode")

    # Batch mode
    parser.add_argument("--type1", type=int, default=0,
                        help="Number of Type-1 (non-Markov vs non-Markov) combinations")
    parser.add_argument("--type2", type=int, default=0,
                        help="Number of Type-2 (Markov as P1) combinations")
    parser.add_argument("--type3", type=int, default=0,
                        help="Number of Type-3 (Markov as P2) combinations")
    parser.add_argument("--all", action="store_true",
                        help="Run all valid combinations for all 3 types")

    parser.add_argument("--context",  type=int, default=100,
                        help="Number of history rounds given to LLM (default: 100)")
    parser.add_argument("--simulate", type=int, default=1000,
                        help="Number of rounds LLM must simulate (default: 1000)")
    parser.add_argument("--capture-rounds", type=int, default=None,
                        help="Number of rounds to keep from the model output (default: same as --simulate)")
    parser.add_argument("--model", type=str, default="gemini",
                        choices=list(MODEL_MAP.keys()),
                        help="LLM model (default: gemini)")
    parser.add_argument("--no-kb", action="store_true",
                        help="Omit knowledge base from prompt (test pure trajectory inference)")
    parser.add_argument("--save", type=int, default=1,
                        help="Save every N experiments (default: 1)")

    args = parser.parse_args()

    model_name, api_type = MODEL_MAP[args.model]
    use_kb = not args.no_kb
    capture_rounds = args.capture_rounds if args.capture_rounds is not None else args.simulate

    # -----------------------------------------------------------------------
    # Single-game mode
    # -----------------------------------------------------------------------
    if args.p1 and args.p2:
        args.p1 = args.p1.upper()
        args.p2 = args.p2.upper()
        if args.p1 not in PLAYER_CONFIGS:
            parser.error(f"Unknown player ID: {args.p1}")
        if args.p2 not in PLAYER_CONFIGS:
            parser.error(f"Unknown player ID: {args.p2}")

        print("\n" + "=" * 80)
        print("LLM Trajectory Simulation (single game)")
        print("=" * 80)
        print(f"  P1       : {args.p1} ({PLAYER_CONFIGS[args.p1][0]})")
        print(f"  P2       : {args.p2} ({PLAYER_CONFIGS[args.p2][0]})")
        print(f"  Context  : {args.context} rounds")
        print(f"  Simulate : {args.simulate} rounds")
        print(f"  Capture  : {capture_rounds} rounds")
        print(f"  Model    : {model_name} ({api_type})")
        print(f"  KB       : {'yes' if use_kb else 'no'}")

        exp = run_simulation_experiment(
            args.p1, args.p2, args.context, args.simulate, capture_rounds,
            api_type, model_name, use_kb,
        )

        # Determine combo type for folder
        p1_markov = args.p1 in MARKOV_PLAYERS
        p2_markov = args.p2 in MARKOV_PLAYERS
        if not p1_markov and not p2_markov:
            combo_type = 1
        elif p1_markov:
            combo_type = 2
        else:
            combo_type = 3

        clean_model = model_name.replace("/", "_")
        out_dir = os.path.join(
            OUTPUT_DIR, clean_model,
            f"ctx{args.context}_sim{args.simulate}",
            TYPE_FOLDERS[combo_type],
        )
        path = save_result(exp, out_dir)
        print(f"\n✓ Saved: {path}")
        return

    # -----------------------------------------------------------------------
    # Batch mode
    # -----------------------------------------------------------------------
    if not args.all and args.type1 == 0 and args.type2 == 0 and args.type3 == 0:
        parser.error("Specify --p1/--p2 for a single game, or --type1/--type2/--type3/--all for batch.")

    print("\n" + "=" * 80)
    print("LLM Trajectory Simulation (batch)")
    print("=" * 80)
    print(f"  Context  : {args.context} rounds")
    print(f"  Simulate : {args.simulate} rounds")
    print(f"  Capture  : {capture_rounds} rounds")
    print(f"  Model    : {model_name} ({api_type})")
    print(f"  KB       : {'yes' if use_kb else 'no'}")

    # Collect all combinations
    all_combinations: list[tuple[int, str, str]] = []

    if args.all:
        t1, t2, t3 = generate_valid_combinations()
        for pool, ctype in [(t1, 1), (t2, 2), (t3, 3)]:
            existing = get_existing_combinations(model_name, args.context, args.simulate, ctype)
            remaining = [c for c in pool if c not in existing]
            if remaining:
                print(f"  Type {ctype}: {len(remaining)} unseen (of {len(pool)} total)")
                all_combinations.extend((ctype, p1, p2) for p1, p2 in remaining)
            else:
                # All done — run the full pool again (new random draws)
                print(f"  Type {ctype}: all {len(pool)} seen, re-running full pool.")
                all_combinations.extend((ctype, p1, p2) for p1, p2 in pool)
    else:
        for ctype, count in [(1, args.type1), (2, args.type2), (3, args.type3)]:
            if count > 0:
                combos = select_combinations(ctype, count, model_name, args.context, args.simulate)
                all_combinations.extend((ctype, p1, p2) for p1, p2 in combos)

    total = len(all_combinations)
    print(f"\n  Total combinations to run: {total}")
    print(f"  Save every {args.save} experiment(s)")

    clean_model = model_name.replace("/", "_")
    saved_count = 0
    pending: list[tuple[int, dict]] = []

    for idx, (combo_type, p1, p2) in enumerate(all_combinations, 1):
        print(f"\n[{idx}/{total}] Type {combo_type}: {p1} vs {p2}")
        try:
            exp = run_simulation_experiment(
                p1, p2, args.context, args.simulate, capture_rounds,
                api_type, model_name, use_kb,
            )
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            exp = {
                "success": False, "error": str(e),
                "player1_id": p1, "player2_id": p2,
            }
        exp["combo_type"] = combo_type
        pending.append((combo_type, exp))

        if len(pending) >= args.save:
            for ct, e in pending:
                out_dir = os.path.join(
                    OUTPUT_DIR, clean_model,
                    f"ctx{args.context}_sim{args.simulate}",
                    TYPE_FOLDERS[ct],
                )
                try:
                    save_result(e, out_dir)
                    saved_count += 1
                except Exception as e2:
                    print(f"  ⚠️  Save failed: {e2}")
            print(f"  Saved {len(pending)} results (total saved: {saved_count})")
            pending = []

    # Save remainder
    if pending:
        for ct, e in pending:
            out_dir = os.path.join(
                OUTPUT_DIR, clean_model,
                f"ctx{args.context}_sim{args.simulate}",
                TYPE_FOLDERS[ct],
            )
            try:
                save_result(e, out_dir)
                saved_count += 1
            except Exception as e2:
                print(f"  ⚠️  Save failed: {e2}")
        print(f"  Saved {len(pending)} remaining results (total: {saved_count})")

    print(f"\n{'=' * 80}")
    print(f"Batch complete. {saved_count}/{total} experiments saved.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
