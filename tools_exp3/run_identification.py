"""
Exp3 — Identification Experiment
Given an observed trajectory, ask the LLM to identify both players.

  --markov-set xyz  : X / Y / Z  (first-order, opp-only)   — baseline
  --markov-set qrs  : Q / R / S  (first-order, joint-state)
  --markov-set tuv  : T / U / V  (second-order, joint-state)

Combo types:
  Type 1 — both non-Markov       : A-P vs A-P
  Type 2 — Markov is Player 1    : Markov vs A-P (excl. A/B/C)
  Type 3 — Markov is Player 2    : A-P (excl. A/B/C) vs Markov

Output dir: exp3(complex_markov)/identification/{markov_set}/{model}/rounds{N}/type{X}/
"""
import os
import sys
import json
import re
import argparse
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.game import Game
from core.players import PLAYER_CONFIGS

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP3_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
ID_ROOT   = os.path.join(EXP3_ROOT, 'identification')

MARKOV_SETS = {
    'xyz': {'X', 'Y', 'Z'},
    'qrs': {'Q', 'R', 'S'},
    'tuv': {'T', 'U', 'V'},
}

_ALL_MARKOV = {'X', 'Y', 'Z', 'Q', 'R', 'S', 'T', 'U', 'V'}
NON_MARKOV_PLAYERS  = {k for k in PLAYER_CONFIGS if k not in _ALL_MARKOV}
NON_MARKOV_EXCL_ABC = NON_MARKOV_PLAYERS - {'A', 'B', 'C'}

TYPE_FOLDERS = {
    1: "type1_non_markov",
    2: "type2_markov_p1",
    3: "type3_markov_p2",
}

MODEL_MAP = {
    "qwen-api":          ("qwen-plus",             "qwen"),
    "gemini":            ("gemini-3-flash-preview", "gemini"),
    "gpt-5-mini":        ("gpt-5-mini",             "openai"),
    "gpt-5":             ("gpt-5",                  "openai"),
    "deepseek-chat":     ("deepseek-chat",           "deepseek"),
    "deepseek-reasoner": ("deepseek-reasoner",       "deepseek"),
    "jamba-mini":        ("jamba-mini",              "jamba"),
    "jamba-large":       ("jamba-large",             "jamba"),
}

# ---------------------------------------------------------------------------
# Knowledge bases (imported from run_generation; duplicated here for portability)
# ---------------------------------------------------------------------------

_KB_DIST_PLAYERS = """# Rock-Paper-Scissors Player Behavior Knowledge Base

## Distribution Strategy Players
These players always choose the same action:

**A - Pure Scissors**: Always plays Scissors (0% Rock, 0% Paper, 100% Scissors)
**B - Pure Rock**: Always plays Rock (100% Rock, 0% Paper, 0% Scissors)
**C - Pure Paper**: Always plays Paper (0% Rock, 100% Paper, 0% Scissors)
**D - Uniform Random**: Equal probability for all actions (33.3% Rock, 33.3% Paper, 33.4% Scissors)
**E - Rock + Paper**: Only plays Rock or Paper (50% Rock, 50% Paper, 0% Scissors)
**F - Rock + Scissors**: Only plays Rock or Scissors (50% Rock, 0% Paper, 50% Scissors)
**G - Paper + Scissors**: Only plays Paper or Scissors (0% Rock, 50% Paper, 50% Scissors)
**H - Rock Biased**: Prefers Rock (50% Rock, 25% Paper, 25% Scissors)
**I - Paper Biased**: Prefers Paper (25% Rock, 50% Paper, 25% Scissors)
**J - Scissors Biased**: Prefers Scissors (25% Rock, 25% Paper, 50% Scissors)
**K - Rock > Paper**: Strong Rock preference (50% Rock, 33.3% Paper, 16.7% Scissors)
**L - Rock > Scissors**: Strong Rock preference (50% Rock, 16.7% Paper, 33.3% Scissors)
**M - Paper > Rock**: Strong Paper preference (33.3% Rock, 50% Paper, 16.7% Scissors)
**N - Paper > Scissors**: Strong Paper preference (16.7% Rock, 50% Paper, 33.3% Scissors)
**O - Scissors > Rock**: Strong Scissors preference (33.3% Rock, 16.7% Paper, 50% Scissors)
**P - Scissors > Paper**: Strong Scissors preference (16.7% Rock, 33.3% Paper, 50% Scissors)
"""

_KB_XYZ = """
## Markov/Reactive Players (First-Order, Opponent-Only)
These players react to opponent's PREVIOUS move only:

**X - Win-Last**: Plays the move that would have beaten opponent's last move.
- First move: Random
- Round t: plays beat(opponent_{t-1})

**Y - Lose-Last**: Plays the move that would have lost to opponent's last move.
- First move: Random
- Round t: plays lose(opponent_{t-1})

**Z - Copy-Last**: Copies opponent's last move.
- First move: Random
- Round t: plays opponent_{t-1}

## Key Insights:
- beat(Rock)=Paper, beat(Paper)=Scissors, beat(Scissors)=Rock
- lose(Rock)=Scissors, lose(Paper)=Rock, lose(Scissors)=Paper
- At most one player is a Markov/reactive player (X, Y, Z). The other is from A–P.
- Valid player identities: A B C D E F G H I J K L M N O P X Y Z
"""

_KB_QRS = """
## Joint-State Markov Players (First-Order, Both Players)
These players react to BOTH players' previous moves.
State: (self_prev, opp_prev). Find the action NEITHER player played last round (the "missing" action).

**Q - Missing-Action**: Plays the missing action directly.
- If self_prev ≠ opp_prev: exactly one action was missing → plays it.
- If self_prev == opp_prev (draw): two actions missing → picks one of the two at random (50/50).
- First move: Random.
- Example: last round (Rock, Paper) → missing = Scissors → Q plays Scissors.
- Example: last round (Rock, Rock) → missing = {Paper, Scissors} → Q plays Paper or Scissors (50/50).

**R - Beat-Missing**: Plays the action that BEATS the missing action.
- If self_prev ≠ opp_prev: plays beat(missing).
- If self_prev == opp_prev: 50/50 between beat(missing_1) and beat(missing_2).
- First move: Random.
- Example: last round (Rock, Paper) → missing = Scissors → beat(Scissors) = Rock → R plays Rock.

**S - Lose-Missing**: Plays the action that LOSES TO the missing action.
- If self_prev ≠ opp_prev: plays lose(missing).
- If self_prev == opp_prev: 50/50 between lose(missing_1) and lose(missing_2).
- First move: Random.
- Example: last round (Rock, Paper) → missing = Scissors → lose(Scissors) = Paper → S plays Paper.

## Key Insights:
- beat(Rock)=Paper, beat(Paper)=Scissors, beat(Scissors)=Rock
- lose(Rock)=Scissors, lose(Paper)=Rock, lose(Scissors)=Paper
- At most one player is a joint-state Markov player (Q, R, S). The other is from A–P.
- Valid player identities: A B C D E F G H I J K L M N O P Q R S
"""

_KB_TUV = """
## Second-Order Joint-State Markov Players
These players look at the last TWO rounds for BOTH players (4 moves total).
Find the action(s) NOT present among those 4 moves (the "missing" set).

**T - 2R-Missing**: Plays the missing action directly.
- Exactly 1 missing: plays it.
- 2 missing: picks one at random (50/50).
- 0 missing (all 3 appeared): plays randomly (1/3 each).
- Rounds 1–2 (insufficient history): plays randomly.
- Example: t-2=(Rock,Paper), t-1=(Rock,Paper) → appeared={Rock,Paper}, missing=Scissors → T plays Scissors.

**U - 2R-Beat-Missing**: Plays the action that BEATS the missing action.
- Exactly 1 missing m: plays beat(m).
- 2 missing: 50/50 between beat(m1) and beat(m2).
- 0 missing: plays randomly.
- Example: t-2=(Rock,Paper), t-1=(Rock,Paper) → missing=Scissors → beat(Scissors)=Rock → U plays Rock.

**V - 2R-Lose-Missing**: Plays the action that LOSES TO the missing action.
- Exactly 1 missing m: plays lose(m).
- 2 missing: 50/50 between lose(m1) and lose(m2).
- 0 missing: plays randomly.
- Example: t-2=(Rock,Paper), t-1=(Rock,Paper) → missing=Scissors → lose(Scissors)=Paper → V plays Paper.

## Key Insights:
- beat(Rock)=Paper, beat(Paper)=Scissors, beat(Scissors)=Rock
- lose(Rock)=Scissors, lose(Paper)=Rock, lose(Scissors)=Paper
- At most one player is a second-order Markov player (T, U, V). The other is from A–P.
- Valid player identities: A B C D E F G H I J K L M N O P T U V
"""


def get_knowledge_base(markov_set: str) -> str:
    if markov_set == 'xyz':
        return _KB_DIST_PLAYERS + _KB_XYZ
    elif markov_set == 'qrs':
        return _KB_DIST_PLAYERS + _KB_QRS
    else:  # tuv
        return _KB_DIST_PLAYERS + _KB_TUV


# ---------------------------------------------------------------------------
# Combination helpers
# ---------------------------------------------------------------------------

def generate_valid_combinations(markov_set: str) -> tuple:
    markov = MARKOV_SETS[markov_set]
    type1 = [(p1, p2) for p1 in NON_MARKOV_PLAYERS for p2 in NON_MARKOV_PLAYERS if p1 != p2]
    type2 = [(m, nm) for m in sorted(markov) for nm in sorted(NON_MARKOV_EXCL_ABC)]
    type3 = [(nm, m) for nm in sorted(NON_MARKOV_EXCL_ABC) for m in sorted(markov)]
    return type1, type2, type3


def get_existing_combinations(model_name: str, num_rounds: int,
                               combo_type: int, markov_set: str) -> set:
    clean = model_name.replace('/', '_').replace('\\', '_')
    folder = os.path.join(ID_ROOT, markov_set, clean,
                          f"rounds{num_rounds}", TYPE_FOLDERS[combo_type])
    existing = set()
    if not os.path.isdir(folder):
        return existing
    for fname in os.listdir(folder):
        if fname.startswith("id_") and fname.endswith(".json"):
            m = re.match(r"id_([A-Z])_vs_([A-Z])_", fname)
            if m:
                existing.add((m.group(1), m.group(2)))
    return existing


def select_combinations(combo_type: int, count: int, model_name: str,
                         num_rounds: int, markov_set: str) -> list:
    import random
    t1, t2, t3 = generate_valid_combinations(markov_set)
    pool = {1: t1, 2: t2, 3: t3}[combo_type]
    existing = get_existing_combinations(model_name, num_rounds, combo_type, markov_set)
    full_pool = list(pool)
    remaining = [c for c in full_pool if c not in existing]
    print(f"  Type {combo_type}: {len(remaining)} unseen / {len(full_pool)} total")
    if len(remaining) >= count:
        return random.sample(remaining, count)
    result = list(remaining)
    still_need = count - len(result)
    if still_need:
        result.extend(random.choices(full_pool, k=still_need))
        print(f"  All unseen used ({len(remaining)}), re-sampling {still_need} from full pool.")
    return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

VALID_IDS = {
    'xyz': 'A B C D E F G H I J K L M N O P X Y Z',
    'qrs': 'A B C D E F G H I J K L M N O P Q R S',
    'tuv': 'A B C D E F G H I J K L M N O P T U V',
}

MARKOV_DESC = {
    'xyz': 'at most one Markov/reactive player (X, Y, or Z)',
    'qrs': 'at most one joint-state Markov player (Q, R, or S)',
    'tuv': 'at most one second-order Markov player (T, U, or V)',
}


def build_identification_prompt(player1_trajectory: str, player2_trajectory: str,
                                 player1_wins: int, player2_wins: int, draws: int,
                                 num_rounds: int, markov_set: str) -> str:
    valid_ids = VALID_IDS[markov_set]
    markov_desc = MARKOV_DESC[markov_set]
    knowledge_base = get_knowledge_base(markov_set)

    p1_moves = player1_trajectory.split()
    p2_moves = player2_trajectory.split()
    history_lines = "\n".join(
        f"Round {i+1}: P1={p1_moves[i]}, P2={p2_moves[i]}"
        for i in range(len(p1_moves))
    )

    return f"""{knowledge_base}

## Rock-Paper-Scissors Identification Task

You are a strategy expert analyzing an anonymous Rock-Paper-Scissors game.

**Player pool rules:**
- There are {len(valid_ids.split())} possible strategies: {valid_ids}
- The game contains {markov_desc}. The other player is a distribution player (A–P).
- Both players may also be distribution players (A–P) with no Markov player present.
- You must identify both players from the trajectory alone.

**Game Results:**
- Total Rounds: {num_rounds}
- Player 1 wins: {player1_wins} | Player 2 wins: {player2_wins} | Draws: {draws}

**Full Trajectory ({num_rounds} rounds):**
{history_lines}

---

**Your Task — Identify both players:**

**Step 1 — Markov Check:**
Check round by round whether either player's moves follow a Markov/reactive pattern.
For QRS/TUV: look for the "missing action" relationship between consecutive rounds.
For XYZ: look for reactions to the opponent's previous move.

**Step 2 — Distribution Check:**
If no Markov player is found, identify each player's distribution strategy based on
their Rock/Paper/Scissors frequencies and any bias patterns.

**Step 3 — Count Verification:**
Count the actual Rock, Paper, Scissors plays for both players to confirm your identification.

**Output Format (place at the very end):**

Final Answer:
Player1: <single letter from: {valid_ids}>, Rock count=<int>, Paper count=<int>, Scissors count=<int>
Player2: <single letter from: {valid_ids}>, Rock count=<int>, Paper count=<int>, Scissors count=<int>

**Rules:**
- You MUST provide a single-letter identity for EACH player. No refusals.
- Use ONLY letters from: {valid_ids}
- Rock count + Paper count + Scissors count must equal {num_rounds} for each player.
- Even if uncertain, give your best single guess.
"""


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------

def parse_identification_output(raw_output: str) -> dict:
    final_match = re.search(r"Final Answer\s*:(.*?)$", raw_output, re.DOTALL | re.IGNORECASE)
    block = final_match.group(1) if final_match else raw_output

    p1_match = re.search(
        r"Player1\s*:\s*([A-Z])\s*,\s*Rock count\s*=\s*(\d+)\s*,\s*Paper count\s*=\s*(\d+)\s*,\s*Scissors count\s*=\s*(\d+)",
        block, re.IGNORECASE
    )
    p2_match = re.search(
        r"Player2\s*:\s*([A-Z])\s*,\s*Rock count\s*=\s*(\d+)\s*,\s*Paper count\s*=\s*(\d+)\s*,\s*Scissors count\s*=\s*(\d+)",
        block, re.IGNORECASE
    )

    result = {}
    if p1_match:
        result["pred_p1_id"]        = p1_match.group(1).upper()
        result["pred_p1_rock"]      = int(p1_match.group(2))
        result["pred_p1_paper"]     = int(p1_match.group(3))
        result["pred_p1_scissors"]  = int(p1_match.group(4))
    else:
        # Fallback: just try to find "Player1: <letter>"
        fb = re.search(r"Player1\s*:\s*([A-Z])", block, re.IGNORECASE)
        result["pred_p1_id"] = fb.group(1).upper() if fb else None

    if p2_match:
        result["pred_p2_id"]        = p2_match.group(1).upper()
        result["pred_p2_rock"]      = int(p2_match.group(2))
        result["pred_p2_paper"]     = int(p2_match.group(3))
        result["pred_p2_scissors"]  = int(p2_match.group(4))
    else:
        fb = re.search(r"Player2\s*:\s*([A-Z])", block, re.IGNORECASE)
        result["pred_p2_id"] = fb.group(1).upper() if fb else None

    result["parse_success"] = bool(result.get("pred_p1_id") and result.get("pred_p2_id"))
    return result


# ---------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------

def trajectory_stats(traj: str) -> dict:
    moves = traj.split()
    total = len(moves)
    rock     = sum(1 for m in moves if m.lower() == "rock")
    paper    = sum(1 for m in moves if m.lower() == "paper")
    scissors = sum(1 for m in moves if m.lower() == "scissors")
    return {"total": total, "rock": rock, "paper": paper, "scissors": scissors}


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------

def run_identification_experiment(player1_id: str, player2_id: str,
                                   num_rounds: int, api_type: str,
                                   model_name: str, markov_set: str) -> dict:
    print(f"  Simulating {num_rounds} rounds: {player1_id} vs {player2_id} ...")
    result = Game.simulate(player1_id, player2_id, num_rounds)

    p1_traj = " ".join(a.value for a in result.player1_trajectory)
    p2_traj = " ".join(a.value for a in result.player2_trajectory)

    prompt = build_identification_prompt(
        p1_traj, p2_traj,
        result.player1_wins, result.player2_wins, result.draws,
        num_rounds, markov_set,
    )

    print(f"  Calling {api_type.upper()} API ({model_name}) ...")
    from analysis.llm import (get_response, get_response_gemini,
                               get_response_openai, get_response_deepseek,
                               get_response_jamba)
    try:
        if api_type == "gemini":
            _, raw_output = get_response_gemini(prompt, model_name=model_name, max_tokens=8192)
        elif api_type == "openai":
            _, raw_output = get_response_openai(prompt, model_name=model_name, max_tokens=8192)
        elif api_type == "deepseek":
            _, raw_output = get_response_deepseek(prompt, model_name=model_name)
        elif api_type == "jamba":
            _, raw_output = get_response_jamba(prompt, model_name=model_name, max_tokens=4096)
        else:
            _, raw_output = get_response(prompt, model_name=model_name, max_tokens=8192)
    except Exception as e:
        return {
            "success": False, "error": str(e),
            "player1_id": player1_id, "player2_id": player2_id,
        }

    parsed = parse_identification_output(raw_output)
    p1_correct = parsed.get("pred_p1_id") == player1_id
    p2_correct = parsed.get("pred_p2_id") == player2_id

    print(f"  GT=({player1_id},{player2_id})  "
          f"Pred=({parsed.get('pred_p1_id')},{parsed.get('pred_p2_id')})  "
          f"P1={'✓' if p1_correct else '✗'}  P2={'✓' if p2_correct else '✗'}")

    return {
        "success": parsed["parse_success"],
        "markov_set": markov_set,
        "num_rounds": num_rounds,
        "model": model_name,
        "api_type": api_type,
        "player1_id": player1_id,
        "player2_id": player2_id,
        "player1_name": PLAYER_CONFIGS[player1_id][0],
        "player2_name": PLAYER_CONFIGS[player2_id][0],
        "game_stats": {
            "player1_wins": result.player1_wins,
            "player2_wins": result.player2_wins,
            "draws": result.draws,
        },
        "trajectories": {
            "player1": p1_traj,
            "player2": p2_traj,
            "player1_stats": trajectory_stats(p1_traj),
            "player2_stats": trajectory_stats(p2_traj),
        },
        "llm_output": {
            "raw_output": raw_output,
            **parsed,
        },
        "eval": {
            "p1_correct": p1_correct,
            "p2_correct": p2_correct,
            "both_correct": p1_correct and p2_correct,
        },
    }


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_result(exp: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p1 = exp.get("player1_id", "?")
    p2 = exp.get("player2_id", "?")
    path = os.path.join(output_dir, f"id_{p1}_vs_{p2}_{timestamp}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    return path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp3 Identification Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # QRS set, 10 type2 + 10 type3, 1000 rounds
  python tools_exp3/run_identification.py --markov-set qrs --type2 10 --type3 10 --rounds 1000 --model deepseek-chat

  # TUV set, all combinations, 1000 rounds
  python tools_exp3/run_identification.py --markov-set tuv --all --rounds 1000 --model deepseek-reasoner

  # XYZ baseline, 10 of each type
  python tools_exp3/run_identification.py --markov-set xyz --type1 10 --type2 10 --type3 10 --rounds 1000 --model gpt-5-mini

  # Single game
  python tools_exp3/run_identification.py --markov-set qrs --p1 Q --p2 D --rounds 200 --model deepseek-chat
        """,
    )
    parser.add_argument("--markov-set", type=str, required=True, choices=["xyz", "qrs", "tuv"],
                        help="Markov player set for this experiment")
    parser.add_argument("--p1",    type=str, help="Player 1 ID (single-game mode)")
    parser.add_argument("--p2",    type=str, help="Player 2 ID (single-game mode)")
    parser.add_argument("--type1", type=int, default=0, help="# type1 combos (non-markov vs non-markov)")
    parser.add_argument("--type2", type=int, default=0, help="# type2 combos (markov as P1)")
    parser.add_argument("--type3", type=int, default=0, help="# type3 combos (markov as P2)")
    parser.add_argument("--all",   action="store_true",  help="Run all unseen combinations")
    parser.add_argument("--rounds", type=int, default=1000, help="Rounds per game (default: 1000)")
    parser.add_argument("--model", type=str, default="deepseek-chat", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--save",  type=int, default=1, help="Save every N experiments")

    args = parser.parse_args()

    if not (args.p1 and args.p2) and not args.all and args.type1 == 0 and args.type2 == 0 and args.type3 == 0:
        parser.error("Specify --p1/--p2, --type1/--type2/--type3, or --all.")

    model_name, api_type = MODEL_MAP[args.model]
    markov_set = args.markov_set
    clean_model = model_name.replace("/", "_")

    print("\n" + "=" * 80)
    print(f"Exp3 Identification — markov-set={markov_set.upper()}")
    print("=" * 80)
    print(f"  Rounds : {args.rounds}")
    print(f"  Model  : {model_name} ({api_type})")

    out_base = os.path.join(ID_ROOT, markov_set, clean_model, f"rounds{args.rounds}")

    # -----------------------------------------------------------------------
    # Single-game mode
    # -----------------------------------------------------------------------
    if args.p1 and args.p2:
        p1, p2 = args.p1.upper(), args.p2.upper()
        exp = run_identification_experiment(p1, p2, args.rounds, api_type, model_name, markov_set)
        markov = MARKOV_SETS[markov_set]
        ct = 2 if p1 in markov else (3 if p2 in markov else 1)
        path = save_result(exp, os.path.join(out_base, TYPE_FOLDERS[ct]))
        print(f"\n✓ Saved: {path}")
        return

    # -----------------------------------------------------------------------
    # Batch mode
    # -----------------------------------------------------------------------
    all_combos: list = []

    if args.all:
        t1, t2, t3 = generate_valid_combinations(markov_set)
        for pool, ct in [(t1, 1), (t2, 2), (t3, 3)]:
            existing = get_existing_combinations(model_name, args.rounds, ct, markov_set)
            remaining = [c for c in pool if c not in existing]
            target = remaining if remaining else list(pool)
            all_combos.extend((ct, p1, p2) for p1, p2 in target)
            print(f"  Type {ct}: {len(target)} to run")
    else:
        for ct, count in [(1, args.type1), (2, args.type2), (3, args.type3)]:
            if count > 0:
                combos = select_combinations(ct, count, model_name, args.rounds, markov_set)
                all_combos.extend((ct, p1, p2) for p1, p2 in combos)

    total = len(all_combos)
    print(f"\n  Total: {total} experiments | Save every {args.save}")

    saved_count = 0
    pending = []

    p1_correct_total = 0
    p2_correct_total = 0
    both_correct_total = 0
    run_count = 0

    for idx, (combo_type, p1, p2) in enumerate(all_combos, 1):
        print(f"\n[{idx}/{total}] Type {combo_type}: {p1} vs {p2}")
        try:
            exp = run_identification_experiment(
                p1, p2, args.rounds, api_type, model_name, markov_set)
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            exp = {
                "success": False, "error": str(e),
                "player1_id": p1, "player2_id": p2,
                "markov_set": markov_set, "num_rounds": args.rounds,
                "model": model_name,
            }
        exp["combo_type"] = combo_type
        pending.append((combo_type, exp))

        if exp.get("eval"):
            run_count += 1
            p1_correct_total  += int(exp["eval"]["p1_correct"])
            p2_correct_total  += int(exp["eval"]["p2_correct"])
            both_correct_total += int(exp["eval"]["both_correct"])

        if len(pending) >= args.save:
            for ct, e in pending:
                out_dir = os.path.join(out_base, TYPE_FOLDERS[ct])
                try:
                    save_result(e, out_dir)
                    saved_count += 1
                except Exception as e2:
                    print(f"  ⚠️  Save failed: {e2}")
            print(f"  Saved {len(pending)} (total: {saved_count})")
            pending = []

    for ct, e in pending:
        out_dir = os.path.join(out_base, TYPE_FOLDERS[ct])
        try:
            save_result(e, out_dir)
            saved_count += 1
        except Exception as e2:
            print(f"  ⚠️  Save failed: {e2}")
    if pending:
        print(f"  Saved remaining (total: {saved_count})")

    print(f"\n{'=' * 80}")
    print(f"Done. {saved_count}/{total} saved.")
    if run_count:
        print(f"Accuracy — P1: {p1_correct_total}/{run_count} ({p1_correct_total/run_count*100:.1f}%)  "
              f"P2: {p2_correct_total}/{run_count} ({p2_correct_total/run_count*100:.1f}%)  "
              f"Both: {both_correct_total}/{run_count} ({both_correct_total/run_count*100:.1f}%)")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
