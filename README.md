# Supplementary Experiment Code

This anonymous-release branch consolidates runnable experiment code for the three experiments in the paper. Plotting and paper-figure generation scripts are intentionally excluded so reviewers can focus on the experimental pipelines.

## Contents

- `experiment_1_strategy_identification/`: Experiment 1 code for strategy identification from observed play histories.
- `experiment_2_generation_blind/`: Experiment 2 code for blind trajectory generation from context histories.
- `experiment_3_complex_markov/`: Experiment 3 code for complex Markov-player generation and identification experiments.

Each experiment folder contains:

- `src/`: game, player, and LLM API code for that experiment branch.
- `scripts/`: runnable experiment, parsing, checking, or metric-export scripts.
- `requirements.txt`: Python dependencies.

## Setup

Create and activate a Python environment, then install dependencies from the relevant experiment folder:

```bash
pip install -r experiment_1_strategy_identification/requirements.txt
```

API-based scripts expect the relevant API keys to be provided as environment variables, such as `OPENAI_API_KEY` and `DEEPSEEK_API_KEY`.

## Running Experiments

Experiment 1 example:

```bash
cd experiment_1_strategy_identification
python scripts/batch_experiment.py --all --rounds 100 --model gpt-5-mini
```

Experiment 2 example:

```bash
cd experiment_2_generation_blind
python scripts/simulate_trajectory.py --all --context 1000 --simulate 1500 --model deepseek-chat
```

Experiment 3 examples:

```bash
cd experiment_3_complex_markov
python scripts/run_generation.py --markov-set qrs --all --context 1000 --simulate 1500 --model deepseek-reasoner
python scripts/run_identification.py --markov-set qrs --all --rounds 1000 --model deepseek-reasoner
```

## Notes

This branch does not include raw model outputs, generated plots, or paper-figure scripts. It is intended as a compact reviewer-facing code release for reproducing the experimental procedures.
