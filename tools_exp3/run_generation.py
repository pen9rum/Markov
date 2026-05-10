"""
Exp3 — Generation (Blind) Experiment
Same design as exp2 but supports three Markov player sets:
  --markov-set xyz  : X / Y / Z  (first-order, opp-only)   — baseline
  --markov-set qrs  : Q / R / S  (first-order, joint-state)
  --markov-set tuv  : T / U / V  (second-order, joint-state)

Output dir: exp3(complex_markov)/generation/{markov_set}/{model}/ctx{N}_sim{M}/{type}/
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

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXP3_ROOT = os.path.join(os.path.dirname(__file__), '..', 'exp3(complex_markov)')
GEN_ROOT  = os.path.join(EXP3_ROOT, 'generation')

MARKOV_SETS = {
    'xyz': {'X', 'Y', 'Z'},
    'qrs': {'Q', 'R', 'S'},
    'tuv': {'T', 'U', 'V'},
    'xyz_opp': {'x', 'y', 'z'},
}

# Non-Markov pool is always A-P (same regardless of markov-set)
_ALL_MARKOV = {'X', 'Y', 'Z', 'Q', 'R', 'S', 'T', 'U', 'V', 'x', 'y', 'z'}
NON_MARKOV_PLAYERS = {k for k in PLAYER_CONFIGS if k not in _ALL_MARKOV}
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
# Knowledge base
# ---------------------------------------------------------------------------

_KB_DIST_PLAYERS = """# Rock-Paper-Scissors Player Behavior Knowledge Base

## Distribution Strategy Players
These players choose actions according to a fixed probability distribution, independently each round.

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
- At most one player is a Markov/reactive player (X, Y, Z). The other is from A–P.
- Both can also be distribution players (A–P).
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
- Both can also be distribution players (A–P), with neither being Markov.
- Valid player identities: A B C D E F G H I J K L M N O P Q R S
"""

_KB_XYZ_OPP = """
## Second-Order Opponent-Only Markov Players
These players look at the opponent's last TWO moves only (2 moves total).
Find the action NOT present among those 2 opponent moves (the "missing" set).

**x - 2R-Opp-Missing**: Plays the missing action directly.
- Exactly 1 missing (opponent's two moves differ): plays it.
- 2 missing (opponent played same action twice): picks one at random (50/50).
- Rounds 1–2 (insufficient history): plays randomly.
- Example: opponent's last two moves = (Rock, Paper) → missing = Scissors → x plays Scissors.
- Example: opponent's last two moves = (Rock, Rock) → missing = {Paper, Scissors} → x plays Paper or Scissors (50/50).

**y - 2R-Opp-Beat-Missing**: Plays the action that BEATS the missing action.
- Exactly 1 missing m: plays beat(m).
- 2 missing: 50/50 between beat(m1) and beat(m2).
- Example: opponent's last two = (Rock, Paper) → missing = Scissors → beat(Scissors) = Rock → y plays Rock.

**z - 2R-Opp-Lose-Missing**: Plays the action that LOSES TO the missing action.
- Exactly 1 missing m: plays lose(m).
- 2 missing: 50/50 between lose(m1) and lose(m2).
- Example: opponent's last two = (Rock, Paper) → missing = Scissors → lose(Scissors) = Paper → z plays Paper.

## Key Insights:
- beat(Rock)=Paper, beat(Paper)=Scissors, beat(Scissors)=Rock
- lose(Rock)=Scissors, lose(Paper)=Rock, lose(Scissors)=Paper
- At most one player is a second-order opponent-only Markov player (x, y, z). The other is from A–P (uppercase).
- Valid player identities: D E F G H I J K L M N O P x y z
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
        from analysis.llm import get_player_knowledge_base
        return get_player_knowledge_base()
    elif markov_set == 'qrs':
        return _KB_DIST_PLAYERS + _KB_QRS
    elif markov_set == 'tuv':
        return _KB_DIST_PLAYERS + _KB_TUV
    else:  # xyz_opp
        return _KB_DIST_PLAYERS + _KB_XYZ_OPP


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_valid_combinations(markov_set: str) -> tuple:
    markov = MARKOV_SETS[markov_set]
    type1 = [(p1, p2) for p1 in NON_MARKOV_PLAYERS for p2 in NON_MARKOV_PLAYERS if p1 != p2]
    type2 = [(m, nm) for m in sorted(markov) for nm in sorted(NON_MARKOV_EXCL_ABC)]
    type3 = [(nm, m) for nm in sorted(NON_MARKOV_EXCL_ABC) for m in sorted(markov)]
    return type1, type2, type3


def get_existing_combinations(model_name: str, context: int, simulate: int,
                               combo_type: int, markov_set: str) -> set:
    clean = model_name.replace('/', '_').replace('\\', '_')
    folder = os.path.join(GEN_ROOT, markov_set, clean,
                          f"ctx{context}_sim{simulate}", TYPE_FOLDERS[combo_type])
    existing = set()
    if not os.path.isdir(folder):
        return existing
    for fname in os.listdir(folder):
        if fname.startswith("sim_") and fname.endswith(".json"):
            m = re.match(r"sim_([A-Za-z])_vs_([A-Za-z])_", fname)
            if m:
                existing.add((m.group(1), m.group(2)))
    return existing


def select_combinations(combo_type: int, count: int, model_name: str,
                         context: int, simulate: int, markov_set: str) -> list:
    import random
    t1, t2, t3 = generate_valid_combinations(markov_set)
    pool = {1: t1, 2: t2, 3: t3}[combo_type]
    existing = get_existing_combinations(model_name, context, simulate, combo_type, markov_set)
    full_pool = list(pool)
    remaining = [c for c in full_pool if c not in existing]
    print(f"  Type {combo_type}: {len(remaining)} unseen, {len(full_pool)} total")
    if not full_pool:
        return []
    if len(remaining) >= count:
        return random.sample(remaining, count)
    result = list(remaining)
    still_need = count - len(result)
    result.extend(random.choices(full_pool, k=still_need))
    print(f"  All unseen used ({len(remaining)}), re-sampling {still_need} from full pool.")
    return result


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_simulation_prompt(player1_trajectory: str, player2_trajectory: str,
                              context_rounds: int, simulate_rounds: int,
                              markov_set: str, include_knowledge_base: bool = True) -> str:
    knowledge_section = ""
    if include_knowledge_base:
        knowledge_section = get_knowledge_base(markov_set) + "\n\n"

    markov_desc = {
        'xyz': 'a Markov/reactive player (X, Y, or Z)',
        'qrs': 'a joint-state Markov player (Q, R, or S)',
        'tuv': 'a second-order Markov player (T, U, or V)',
        'xyz_opp': 'a second-order opponent-only Markov player (x, y, or z)',
    }
    markov_check_hint = {
        'xyz': "(1) Check if either player's moves react to the opponent's previous move (first-order Markov).",
        'qrs': "(1) Check if either player's moves react to the joint previous-round state (their own + opponent previous move).",
        'tuv': "(1) Check if either player's moves react to the previous TWO rounds (second-order Markov).",
        'xyz_opp': "(1) Check if either player's moves react to the opponent's previous TWO moves (second-order opponent-only Markov).",
    }
    valid_ids = {
        'xyz': 'A B C D E F G H I J K L M N O P X Y Z',
        'qrs': 'A B C D E F G H I J K L M N O P Q R S',
        'tuv': 'A B C D E F G H I J K L M N O P T U V',
        'xyz_opp': 'D E F G H I J K L M N O P x y z',
    }

    p1_moves = player1_trajectory.split()
    p2_moves = player2_trajectory.split()
    history_lines = "\n".join(
        f"Round {i+1}: P1={p1_moves[i]}, P2={p2_moves[i]}"
        for i in range(context_rounds)
    )

    prompt = f"""{knowledge_section}## Rock-Paper-Scissors Trajectory Simulation Task

You are observing a Rock-Paper-Scissors game between two **anonymous** players (P1 and P2).

**Player pool rules:**
- Both players are drawn from the strategy pool described above.
- At most ONE of them is {markov_desc[markov_set]}. The other is a distribution player (A–P).
- It is also possible that BOTH are distribution players (neither is Markov).
- You do NOT know which specific strategy each player uses — you must infer it from the trajectory.

### Previous Game History ({context_rounds} rounds):
{history_lines}

---

### Your Task
Simulate the next **{simulate_rounds} rounds** (rounds {context_rounds + 1} to {context_rounds + simulate_rounds}).

**MANDATORY: You MUST generate all {simulate_rounds} rounds. No refusals or partial outputs.**
Refusing, asking for clarification, or generating fewer rounds is not acceptable.
If perfect randomness is difficult, approximate it — a best-effort simulation is required.

**Step 1 — Strategy Inference:**
(1) {markov_check_hint[markov_set]}
(2) If a Markov player is detected, identify the other player's distribution strategy.
(3) If neither is Markov, identify both players' distribution strategies.

**Step 2 — Simulation:**
Generate exactly {simulate_rounds} rounds consistent with the inferred strategies.

**Anti-Periodic Pattern Constraint (CRITICAL):**
- Do NOT approximate randomness by repeating a fixed short pattern or block.
- Avoid periodic sequences such as "Scissors Paper Scissors Rock Scissors Paper Scissors Rock".
- Do NOT use deterministic cycles to satisfy the distribution constraint.
- The same short sequence (e.g., length 2–10) should not repeat regularly across the simulation.
**Avoid Repetitive Patterns (best effort):**
- Prefer varied sequences over mechanical cycles like "Rock Paper Rock Paper Rock Paper ...".
- Avoid long contiguous runs of the same action unless the inferred strategy strongly implies it.
- The sequence should resemble realistic random draws from inferred probabilities.

**Output Format (place at the very end):**

SIMULATION:
P1_identity: <single letter from: {valid_ids[markov_set]}>
P2_identity: <single letter from: {valid_ids[markov_set]}>
Round 1: <P1 action> <P2 action>
Round 2: <P1 action> <P2 action>
...
Round {simulate_rounds}: <P1 action> <P2 action>

Rules:
- Each <action> must be exactly: Rock, Paper, or Scissors (capitalize first letter only).
- Every round line has exactly two actions separated by a space.
- Exactly {simulate_rounds} Round lines. No extra text inside the SIMULATION block.
"""
    return prompt


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------

def parse_simulation_output(raw_output: str, capture_rounds: int) -> dict | None:
    sim_matches = list(re.finditer(r"SIMULATION\s*:(.*?)(?=SIMULATION\s*:|$)", raw_output, re.DOTALL | re.IGNORECASE))
    sim_block = sim_matches[-1].group(1) if sim_matches else raw_output

    p1_id_match = re.search(r"P1_identity\s*:\s*([A-Za-z])", sim_block)
    p2_id_match = re.search(r"P2_identity\s*:\s*([A-Za-z])", sim_block)
    p1_identity = p1_id_match.group(1) if p1_id_match else None
    p2_identity = p2_id_match.group(1) if p2_id_match else None
    if not p1_identity or not p2_identity:
        print("  ⚠️  Could not find P1_identity or P2_identity.")

    round_pattern = re.compile(
        r"Round\s+\d+\s*:\s*(Rock|Paper|Scissors)\s+(Rock|Paper|Scissors)", re.IGNORECASE)
    p1_moves, p2_moves = [], []
    for m in round_pattern.finditer(sim_block):
        p1_moves.append(m.group(1).capitalize())
        p2_moves.append(m.group(2).capitalize())

    if not p1_moves:
        print("  ⚠️  No round lines found in SIMULATION block.")
        return None

    got = len(p1_moves)
    complete = got >= capture_rounds
    if got < capture_rounds:
        print(f"  ⚠️  Incomplete: wanted {capture_rounds}, got {got}. Keeping partial.")
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
# Stats helpers
# ---------------------------------------------------------------------------

def compute_trajectory_stats(moves: list) -> dict:
    total = len(moves)
    rock = moves.count("Rock")
    paper = moves.count("Paper")
    scissors = moves.count("Scissors")
    return {
        "total": total, "rock": rock, "paper": paper, "scissors": scissors,
        "rock_pct": rock / total * 100 if total else 0,
        "paper_pct": paper / total * 100 if total else 0,
        "scissors_pct": scissors / total * 100 if total else 0,
    }


def compute_window_stats(moves: list, window: int = 100) -> list:
    return [compute_trajectory_stats(moves[s:s + window])
            for s in range(0, len(moves), window) if moves[s:s + window]]


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_simulation_experiment(player1_id: str, player2_id: str,
                               context_rounds: int, simulate_rounds: int,
                               capture_rounds: int, api_type: str, model_name: str,
                               markov_set: str, include_knowledge_base: bool = True) -> dict:
    total_rounds = context_rounds + simulate_rounds
    print(f"  Simulating {total_rounds} rounds ({context_rounds} context + {simulate_rounds} continuation)...")
    real_result = Game.simulate(player1_id, player2_id, total_rounds)
    context_p1 = [a.value for a in real_result.player1_trajectory[:context_rounds]]
    context_p2 = [a.value for a in real_result.player2_trajectory[:context_rounds]]
    cont_p1 = [a.value for a in real_result.player1_trajectory[context_rounds:context_rounds + capture_rounds]]
    cont_p2 = [a.value for a in real_result.player2_trajectory[context_rounds:context_rounds + capture_rounds]]

    prompt = build_simulation_prompt(
        " ".join(context_p1), " ".join(context_p2),
        context_rounds, simulate_rounds, markov_set, include_knowledge_base,
    )

    print(f"  Calling {api_type.upper()} API ({model_name})...")
    from analysis.llm import (get_response, get_response_gemini,
                               get_response_openai, get_response_deepseek,
                               get_response_jamba)
    try:
        if api_type == "gemini":
            _, raw_output = get_response_gemini(prompt, model_name=model_name, max_tokens=16384)
        elif api_type == "openai":
            _, raw_output = get_response_openai(prompt, model_name=model_name, max_tokens=16384)
        elif api_type == "deepseek":
            _, raw_output = get_response_deepseek(prompt, model_name=model_name)
        elif api_type == "jamba":
            _, raw_output = get_response_jamba(prompt, model_name=model_name, max_tokens=4096)
        else:
            _, raw_output = get_response(prompt, model_name=model_name, max_tokens=16384)
    except Exception as e:
        return {"success": False, "error": str(e),
                "player1_id": player1_id, "player2_id": player2_id}

    parsed = parse_simulation_output(raw_output, capture_rounds)
    result = {
        "success": parsed is not None,
        "markov_set": markov_set,
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
        "real_continuation": {
            "p1_trajectory": " ".join(cont_p1),
            "p2_trajectory": " ".join(cont_p2),
            "p1_stats": compute_trajectory_stats(cont_p1),
            "p2_stats": compute_trajectory_stats(cont_p2),
            "p1_window_stats": compute_window_stats(cont_p1),
            "p2_window_stats": compute_window_stats(cont_p2),
        },
    }

    if parsed:
        llm_p1, llm_p2 = parsed["p1_moves"], parsed["p2_moves"]
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
        print(f" Parsed {len(llm_p1)} rounds | "
              f"P1={parsed['p1_identity']} P2={parsed['p2_identity']}")
    else:
        result["llm_simulation"] = {"raw_output": raw_output, "parsed_rounds": 0, "complete": False}
        result["error"] = "Incomplete or unparseable output"
        print("  ✗ Parsing failed")

    return result


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_result(exp: dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    p1, p2 = exp.get("player1_id", "?"), exp.get("player2_id", "?")
    base = f"sim_{p1}_vs_{p2}_{timestamp}"
    json_path = os.path.join(output_dir, base + ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(exp, f, ensure_ascii=False, indent=2)
    return json_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Exp3 Generation (Blind) Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # QRS set, 10 type2 + 10 type3
  python tools_exp3/run_generation.py --markov-set qrs --type2 10 --type3 10 --context 1000 --simulate 1000 --model deepseek-reasoner

  # TUV set, all combinations
  python tools_exp3/run_generation.py --markov-set tuv --all --context 1000 --simulate 1000 --model deepseek-reasoner

  # XYZ baseline
  python tools_exp3/run_generation.py --markov-set xyz --type1 10 --type2 10 --type3 10 --context 1000 --simulate 1000 --model deepseek-reasoner
        """,
    )
    parser.add_argument("--markov-set", type=str, required=True, choices=["xyz", "qrs", "tuv", "xyz_opp"],
                        help="Which Markov player set to use")
    parser.add_argument("--p1", type=str, help="Player 1 ID, single-game mode")
    parser.add_argument("--p2", type=str, help="Player 2 ID, single-game mode")
    parser.add_argument("--type1", type=int, default=0)
    parser.add_argument("--type2", type=int, default=0)
    parser.add_argument("--type3", type=int, default=0)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--context",  type=int, default=1000)
    parser.add_argument("--simulate", type=int, default=1000)
    parser.add_argument("--capture-rounds", type=int, default=None)
    parser.add_argument("--model", type=str, default="deepseek-reasoner", choices=list(MODEL_MAP.keys()))
    parser.add_argument("--no-kb", action="store_true")
    parser.add_argument("--save", type=int, default=1)

    args = parser.parse_args()
    model_name, api_type = MODEL_MAP[args.model]
    use_kb = not args.no_kb
    capture_rounds = args.capture_rounds if args.capture_rounds is not None else args.simulate
    markov_set = args.markov_set

    print("\n" + "=" * 80)
    print(f"Exp3 Generation — markov-set={markov_set.upper()}")
    print("=" * 80)
    print(f"  Context  : {args.context} rounds")
    print(f"  Simulate : {args.simulate} rounds | Capture: {capture_rounds}")
    print(f"  Model    : {model_name} ({api_type})")

    clean_model = model_name.replace("/", "_")
    OUTPUT_DIR = os.path.join(GEN_ROOT, markov_set)

    # -----------------------------------------------------------------------
    # Single-game mode
    # -----------------------------------------------------------------------
    if args.p1 and args.p2:
        markov = MARKOV_SETS[markov_set]
        def _norm(s): return s if s.lower() in {m.lower() for m in markov} and any(m.islower() for m in markov) else s.upper()
        p1, p2 = _norm(args.p1), _norm(args.p2)
        exp = run_simulation_experiment(
            p1, p2, args.context, args.simulate, capture_rounds,
            api_type, model_name, markov_set, use_kb)
        markov = MARKOV_SETS[markov_set]
        if p1 in markov:
            ct = 2
        elif p2 in markov:
            ct = 3
        else:
            ct = 1
        out_dir = os.path.join(OUTPUT_DIR, clean_model,
                               f"ctx{args.context}_sim{args.simulate}", TYPE_FOLDERS[ct])
        path = save_result(exp, out_dir)
        print(f"\n✓ Saved: {path}")
        return

    # -----------------------------------------------------------------------
    # Batch mode
    # -----------------------------------------------------------------------
    if not args.all and args.type1 == 0 and args.type2 == 0 and args.type3 == 0:
        parser.error("Specify --p1/--p2, --type1/--type2/--type3, or --all.")

    all_combos: list = []
    if args.all:
        t1, t2, t3 = generate_valid_combinations(markov_set)
        for pool, ct in [(t1, 1), (t2, 2), (t3, 3)]:
            existing = get_existing_combinations(model_name, args.context, args.simulate, ct, markov_set)
            remaining = [c for c in pool if c not in existing]
            target = remaining if remaining else list(pool)
            all_combos.extend((ct, p1, p2) for p1, p2 in target)
            print(f"  Type {ct}: {len(target)} to run")
    else:
        for ct, count in [(1, args.type1), (2, args.type2), (3, args.type3)]:
            if count > 0:
                combos = select_combinations(ct, count, model_name,
                                             args.context, args.simulate, markov_set)
                all_combos.extend((ct, p1, p2) for p1, p2 in combos)

    total = len(all_combos)
    print(f"\n  Total: {total} experiments | Save every {args.save}")

    saved_count = 0
    pending = []

    for idx, (combo_type, p1, p2) in enumerate(all_combos, 1):
        print(f"\n[{idx}/{total}] Type {combo_type}: {p1} vs {p2}")
        try:
            exp = run_simulation_experiment(
                p1, p2, args.context, args.simulate, capture_rounds,
                api_type, model_name, markov_set, use_kb)
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            exp = {"success": False, "error": str(e), "player1_id": p1, "player2_id": p2}
        exp["combo_type"] = combo_type
        pending.append((combo_type, exp))

        if len(pending) >= args.save:
            for ct, e in pending:
                out_dir = os.path.join(OUTPUT_DIR, clean_model,
                                       f"ctx{args.context}_sim{args.simulate}", TYPE_FOLDERS[ct])
                try:
                    save_result(e, out_dir)
                    saved_count += 1
                except Exception as e2:
                    print(f"  ⚠️  Save failed: {e2}")
            print(f"  Saved {len(pending)} (total: {saved_count})")
            pending = []

    for ct, e in pending:
        out_dir = os.path.join(OUTPUT_DIR, clean_model,
                               f"ctx{args.context}_sim{args.simulate}", TYPE_FOLDERS[ct])
        try:
            save_result(e, out_dir)
            saved_count += 1
        except Exception as e2:
            print(f"  ⚠️  Save failed: {e2}")
    if pending:
        print(f"  Saved remaining (total: {saved_count})")

    print(f"\n{'=' * 80}")
    print(f"Done. {saved_count}/{total} saved.")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
