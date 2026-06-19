# World Cup 2026 Goal Prediction Challenge

A machine learning pipeline that predicts each team's **total goals scored** and **stage reached** at the FIFA World Cup 2026, built for [Zindi's World Cup 2026 Goal Prediction Challenge](https://zindi.africa/).

Trained entirely on historical World Cup data (1930–2022) — no use of 2026 match results, betting odds, or rankings, per competition rules.

## The Challenge

Predict, for each of the 48 qualified teams:

1. **`total_goals`** — total goals scored across the tournament (scored by RMSE)
2. **`Target`** — stage reached: `group`, `roundof32`, `roundof16`, `qf`, `sf`, `runnerup`, or `champion` (scored by F1)

Final leaderboard score: `0.60 × RMSE(goals) + 0.40 × F1(stage)`

## Approach

**Data sources** (from the [jfjelstul/worldcup](https://github.com/jfjelstul/worldcup) historical dataset):
- `team_appearances.csv` — match-level results per team per tournament
- `goals.csv` — individual goal events (minute, type, scorer)
- `penalty_kicks.csv` — penalty shootout history

**Feature engineering** (32 features):
- Career historical averages (goals, wins, goal difference, stage reached) — computed using only *prior* tournaments to avoid leakage
- Most recent tournament performance, weighted more heavily than older history
- Opponent strength and schedule difficulty
- Goal timing patterns (early/late/extra-time goals, penalty vs. open play)
- Penalty shootout conversion rate
- Tournament era (scoring patterns differ significantly between 1930s and modern football)

**Models** — weighted ensembles for each target:
- **Goals (regression):** XGBoost + LightGBM + Gradient Boosting + Ridge
- **Stage (classification):** XGBoost + LightGBM + HistGradientBoosting + Random Forest

**Validation:** held out the 2022 World Cup as a test set.

| Metric | Baseline | Final |
|---|---|---|
| Goals RMSE | 3.945 | **3.049** |
| Stage F1 | 0.554 | **0.566** |

## Usage

```bash
pip install pandas numpy scikit-learn xgboost lightgbm

# place team_appearances.csv, goals.csv, penalty_kicks.csv,
# and SampleSubmission.csv in the project folder

python worldcup2026_final.py
```

This trains the ensemble, validates against the 2022 holdout, retrains on the full dataset, and writes `submission.csv` in the exact format Zindi expects (`ID, total_goals, Target`), matched against the official `SampleSubmission.csv` IDs and row order.

## Notes & limitations

- 4 of the 48 qualified teams (Cape Verde, Jordan, Uzbekistan, Curaçao) are World Cup debutants with no historical data — they fall back to conservative defaults.
- Team matching is done via ISO3 `team_code`, which correctly handles historical name changes (e.g. Zaire → DR Congo).
- This is a baseline/exploratory model — there's clear room for improvement via Poisson regression for goal counts, stacking instead of simple ensembling, or incorporating squad/manager-level features.

## License

Historical data used under CC-BY-SA 4.0, per Zindi's data sharing terms for this challenge.
