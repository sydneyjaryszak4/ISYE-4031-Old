# ISYE 4031 Homework 1

Sydney Jaryszak — statistical review, forecast baselines, and agent audit.

## Files

| File | Purpose |
|------|---------|
| `analysis.py` | Reproduces all numerical results for Parts 1–3 and saves figures |
| `test_analysis.py` | Automated checks required for Part 4c |
| `report.pdf` | Written answers (submit separately) |
| `AI_USAGE.md` | AI assistance audit record |

## Reproduce the analysis

From this directory in the course Codespace:

```bash
cd hw1_jaryszak_sydney
python analysis.py
```

This prints results for Parts 1b–3 and writes:

- `fig_1a_train_test_split.png`
- `fig_2a_baseline_forecasts.png`

## Run the tests

```bash
cd hw1_jaryszak_sydney
pytest -q test_hw1_analysis.py
```

For the Canvas ZIP, rename or copy the test file to `test_analysis.py` as required by the assignment:

```bash
cp test_hw1_analysis.py test_analysis.py
```

All four tests should pass.

## Submission archive

After `python analysis.py` and `pytest -q test_hw1_analysis.py` both succeed, create the Canvas upload from this directory:

```bash
cp test_hw1_analysis.py test_analysis.py
zip hw1_jaryszak_sydney.zip analysis.py test_analysis.py report.pdf AI_USAGE.md README.md
```

Do not include the Git repository, virtual environment, `.pytest_cache`, or the homework PDF prompt file.

## Analysis overview

- **Part 1:** Training-only descriptive summaries, 95% mean confidence interval, and 95% prediction interval for the 20-day temperature series (cutoff after day 14).
- **Part 2:** Fixed mean and last-value baseline forecasts for days 15–20, with MAE and RMSE on the held-out period.
- **Part 3:** One-sample z-test, p-value, and Type II error calculation for the melting-point example (known σ).

Held-out observations are never used to estimate models, intervals, or baselines.
