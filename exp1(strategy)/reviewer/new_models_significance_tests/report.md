# Experiment 1 new models: reviewer-style significance tests

Input: `new_models_success_highlow_selected.csv`. qwen/200 uses the selected 50 Markov-player and 150 non-Markov-player observations.

## Markov versus non-Markov

| Model | Markov | Non-Markov | Difference (95% CI) | z p (Holm) | Fisher p (Holm) |
|---|---:|---:|---:|---:|---:|
| gpt-4.1 | 57.0% | 76.3% | -19.3% (-27.0%, -11.8%) | <0.001 (<0.001) | <0.001 (<0.001) |
| gemini | 52.0% | 77.3% | -25.3% (-32.9%, -17.7%) | <0.001 (<0.001) | <0.001 (<0.001) |
| qwen | 25.0% | 45.5% | -20.5% (-27.3%, -13.0%) | <0.001 (<0.001) | <0.001 (<0.001) |

## Context degradation of Markov identity accuracy

| Model | 100 | 200 | 500 | 1000 | CA p (Holm) | Logistic slope; p (Holm) | 1000-100 (95% CI) | Fisher p (Holm) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gpt-4.1 | 70.0% | 70.0% | 46.0% | 42.0% | <0.001 (0.001) | -0.412; <0.001 (0.002) | -28.0% (-44.7%, -8.5%) | 0.008 (0.025) |
| gemini | 70.0% | 60.0% | 36.0% | 42.0% | <0.001 (0.001) | -0.404; <0.001 (0.002) | -28.0% (-44.7%, -8.5%) | 0.008 (0.025) |
| qwen | 22.0% | 18.0% | 36.0% | 24.0% | 0.326 (0.326) | +0.128; 0.327 (0.327) | +2.0% (-14.4%, +18.3%) | 1.000 (1.000) |
