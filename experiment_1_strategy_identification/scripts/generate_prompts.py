"""
生成 prompt 專用工具
完全不修改 llm.py
直接複製原本 analyze_game_trajectory 的 prompt 組法
"""

import os
import sys
import json
import argparse
from datetime import datetime

# 專案根目錄
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from tools.batch_experiment import select_combinations
from core.game import Game


def get_player_knowledge_base() -> str:
    """
    直接複製自 llm.py
    """
    knowledge = """
# Rock-Paper-Scissors Player Behavior Knowledge Base

## Distribution Strategy Players
These players always choose the same action:

**A - Pure Scissors**: Always plays Scissors (0% Rock, 0% Paper, 100% Scissors)
- Completely predictable
- Will lose to Rock players, beat Paper players, tie with Scissors players

**B - Pure Rock**: Always plays Rock (100% Rock, 0% Paper, 0% Scissors)
- Completely predictable
- Will beat Scissors players, lose to Paper players, tie with Rock players

**C - Pure Paper**: Always plays Paper (0% Rock, 100% Paper, 0% Scissors)
- Completely predictable
- Will beat Rock players, lose to Scissors players, tie with Paper players

**D - Uniform Random**: Equal probability for all actions (33.3% Rock, 33.3% Paper, 33.4% Scissors)
- Unpredictable order but balanced distribution
- No exploitable pattern

**E - Rock + Paper**: Only plays Rock or Paper (50% Rock, 50% Paper, 0% Scissors)
- Never plays Scissors
- Vulnerable to Scissors-heavy strategies

**F - Rock + Scissors**: Only plays Rock or Scissors (50% Rock, 0% Paper, 50% Scissors)
- Never plays Paper
- Vulnerable to Paper-heavy strategies

**G - Paper + Scissors**: Only plays Paper or Scissors (0% Rock, 50% Paper, 50% Scissors)
- Never plays Rock
- Vulnerable to Rock-heavy strategies

**H - Rock Biased**: Prefers Rock (50% Rock, 25% Paper, 25% Scissors)
- Rock appears twice as often as other moves
- Can be exploited by Paper-biased strategies

**I - Paper Biased**: Prefers Paper (25% Rock, 50% Paper, 25% Scissors)
- Paper appears twice as often as other moves
- Can be exploited by Scissors-biased strategies

**J - Scissors Biased**: Prefers Scissors (25% Rock, 25% Paper, 50% Scissors)
- Scissors appears twice as often as other moves
- Can be exploited by Rock-biased strategies

**K - Rock > Paper**: Strong Rock preference (50% Rock, 33.3% Paper, 16.7% Scissors)
- Rock most common, Paper second, Scissors least
- Hierarchy: Rock > Paper > Scissors

**L - Rock > Scissors**: Strong Rock preference (50% Rock, 16.7% Paper, 33.3% Scissors)
- Rock most common, Scissors second, Paper least
- Hierarchy: Rock > Scissors > Paper

**M - Paper > Rock**: Strong Paper preference (33.3% Rock, 50% Paper, 16.7% Scissors)
- Paper most common, Rock second, Scissors least
- Hierarchy: Paper > Rock > Scissors

**N - Paper > Scissors**: Strong Paper preference (16.7% Rock, 50% Paper, 33.3% Scissors)
- Paper most common, Scissors second, Rock least
- Hierarchy: Paper > Scissors > Rock

**O - Scissors > Rock**: Strong Scissors preference (33.3% Rock, 16.7% Paper, 50% Scissors)
- Scissors most common, Rock second, Paper least
- Hierarchy: Scissors > Rock > Paper

**P - Scissors > Paper**: Strong Scissors preference (16.7% Rock, 33.3% Paper, 50% Scissors)
- Scissors most common, Paper second, Rock least
- Hierarchy: Scissors > Paper > Rock

## Markov Based Players 
These players adapt their strategy based on opponent's previous moves:

**X - Win-Last**: Plays the move that would have beaten opponent's last move
- First move: Random
- Subsequent moves: Counter opponent's previous move
- Against Pure Rock (always Rock): Will play Paper after first round
- Highly effective against predictable opponents

**Y - Lose-Last**: Plays the move that would have lost to opponent's last move
- First move: Random
- Subsequent moves: Intentionally lose to opponent's previous move
- Against Pure Rock: Will keep playing Scissors
- Intentionally losing strategy

**Z - Copy-Last**: Copies opponent's last move
- First move: Random
- Subsequent moves: Mirrors opponent's previous move
- Against Pure Rock: Will always play Rock (all ties)
- Creates mirror matches

## Key Insights:
1. Pure strategies are completely predictable and can be easily exploited
2. Distribution strategies maintain consistent ratios but unpredictable order
3. Reactive strategies are most effective against predictable opponents
4. Win-Last (X) tends to perform best overall
5. Strategies with heavy bias can be counter-strategized
"""
    return knowledge


def build_analysis_prompt(
    player1_id: str,
    player2_id: str,
    player1_trajectory: str,
    player2_trajectory: str,
    player1_wins: int,
    player2_wins: int,
    draws: int,
    num_rounds: int,
) -> str:
    """
    直接複製原本 llm.py 的 prompt 組法
    不呼叫 API，只回傳 prompt 字串
    """
    knowledge_base = get_player_knowledge_base()

    prompt = f"""{knowledge_base}

## Game Analysis Task

You are a Rock-Paper-Scissors strategy expert. Above is the background knowledge of 19 player strategies.

Now analyze a game where both players' identities are unknown based on their move trajectories:

**CRITICAL CONSTRAINTS**:
- Valid player identities are ONLY: A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P (distribution strategies) and X, Y, Z (Markov/reactive strategies)
- You MUST only use these 19 letters. Do NOT invent other letters like U, T, etc.
- Your final answer MUST contain both lines: "Player1: ..." and "Player2: ..." and they cannot be empty.
- Even if uncertain, you MUST provide your single best guess (no abstain / no refusal).
- You MUST provide concrete numeric counts for Rock, Paper, Scissors for BOTH players.
- Forbidden placeholders in final answer: "[count ...]", "cannot be computed", "same", "N/A", "unknown", "TBD".
At most, only one of the players will be Markov player (X, Y, Z). The other player will be from A-P. It is also possible that both players are from A-P.
The Markov player's strategy comes from last player. Therefore, you have to analyze the trajectory every round to determine which player is Markov and which is distribution. If both players are from A-P, then you can analyze them as two distribution players.
The best solution is to follow the following steps and think step by step:
(1) Find out is there any player is Markov player (X, Y, Z) by analyzing the trajectory round by round. If there is a Markov player, then you can determine the other player's identity as well.
(2) If there is no Markov player, then you can analyze the trajectory as two distribution players and find out the best matching identities for both players.
(3) Based on the identified player identities, count how many times rock, paper, and scissors appear in each player's {num_rounds}-round trajectory.
**Game Info**:
- Total Rounds: {num_rounds}
- Results: Player1 won {player1_wins}, Player2 won {player2_wins}, Draws {draws}

**Player1 Trajectory**:
{player1_trajectory}

**Player2 Trajectory**:
{player2_trajectory}
Please respond with a detailed analysis of the players' strategies, their most likely identities, and the predicted probabilities for their next moves. Be sure to justify your reasoning based on the trajectories and the knowledge base provided.
After your detailed analysis, start your final answer with "Final Answer:" and provide the identified player identities and the predicted probabilities in a clear format with:
Player1: Identity, Rock count, Paper count, Scissors count
Player2: Identity, Rock count, Paper count, Scissors count
where count means the actual number of times the player played that action in the whole trajectory, which can be used to verify the correctness of your analysis.
Use EXACTLY this output template at the end:
Final Answer:
Player1: <Identity>, Rock count=<int>, Paper count=<int>, Scissors count=<int>
Player2: <Identity>, Rock count=<int>, Paper count=<int>, Scissors count=<int>
"""
    return prompt


def generate_prompts(
    combo_type: int,
    count: int,
    num_rounds: int = 100,
    model_name: str = None,
):
    """
    抽組合 + simulate + 生成完整 prompt
    不呼叫 API
    """
    combos = select_combinations(
        combo_type=combo_type,
        count=count,
        model_name=model_name,
        num_rounds=num_rounds,
    )

    results = []

    for idx, (player1_id, player2_id) in enumerate(combos, 1):
        print(f"[{idx}/{len(combos)}] Generating prompt for {player1_id} vs {player2_id}")

        game_result = Game.simulate(player1_id, player2_id, num_rounds)

        player1_trajectory = game_result.get_trajectory_string(1)
        player2_trajectory = game_result.get_trajectory_string(2)

        prompt = build_analysis_prompt(
            player1_id=player1_id,
            player2_id=player2_id,
            player1_trajectory=player1_trajectory,
            player2_trajectory=player2_trajectory,
            player1_wins=game_result.player1_wins,
            player2_wins=game_result.player2_wins,
            draws=game_result.draws,
            num_rounds=num_rounds,
        )

        results.append(
            {
                "player1_id": player1_id,
                "player2_id": player2_id,
                "combo_type": combo_type,
                "num_rounds": num_rounds,
                "player1_wins": game_result.player1_wins,
                "player2_wins": game_result.player2_wins,
                "draws": game_result.draws,
                "player1_trajectory": player1_trajectory,
                "player2_trajectory": player2_trajectory,
                "prompt": prompt,
            }
        )

    return results


def save_results(results, output_path: str):
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate full analysis prompts without calling any API."
    )
    parser.add_argument(
        "--type",
        type=int,
        required=True,
        choices=[1, 2],
        help="1 = non-markov vs non-markov, 2 = one markov vs one non-markov",
    )
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="number of combinations to sample",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=100,
        help="number of rounds per match",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="optional model name, used only to exclude existing combinations",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="output json path",
    )
    parser.add_argument(
        "--print-only",
        action="store_true",
        help="print prompts to terminal instead of saving",
    )

    args = parser.parse_args()

    results = generate_prompts(
        combo_type=args.type,
        count=args.count,
        num_rounds=args.rounds,
        model_name=args.model_name,
    )

    if not results:
        print("No available combinations.")
        return

    if args.print_only:
        for i, item in enumerate(results, 1):
            print("=" * 100)
            print(f"Prompt #{i}: {item['player1_id']} vs {item['player2_id']}")
            print("=" * 100)
            print(item["prompt"])
            print()
        return

    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        type_name = "type1_non_markov" if args.type == 1 else "type2_with_markov"
        output_path = os.path.join(
            PROJECT_ROOT,
            "prompt_results",
            f"{type_name}_{args.rounds}r_{timestamp}.json",
        )

    save_results(results, output_path)


if __name__ == "__main__":
    main()