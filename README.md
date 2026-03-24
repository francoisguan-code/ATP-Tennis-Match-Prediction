# ATP-Tennis-Match-Prediction
A machine learning pipeline for pre-match ATP tennis prediction, using data from 39,541 matches (2011-2024), built from scratch on the Jeff Sackmann dataset. 

**Final result:** AUC around 0.71 on a fixed 2023–2024 test set, which seems coherent against aggregated bookmakers closing odds (bookmaker AUC 0.74).

Data source: (https://github.com/JeffSackmann/tennis_atp)

---

## Results

We built a baseline score using only ATP-rank variables for prediction. 

| Model | AUC | Log Loss | Brier |
|---|---|---|---|
| Baseline (rank only) | 0.652 | — | — |
| Logistic Regression | 0.7160 | 0.624 | 0.215 |
| Random Forest | 0.7156 | 0.769 | 0.232 |
| XGBoost | 0.7164 | 0.635 | 0.216 |
| Pinnacle bookmaker | 0.742 | 0.593 | 0.205 |



---

## Project structure

```
atp-tennis-prediction/
│
├── src/                         
│   ├── 01_data_loading.py        # Load and clean ATP CSV files
│   ├── 02_elo_ratings.py         # Elo-538 rating system (FiveThirtyEight variant)
│   ├── 03_player_profiles.py     # Rolling stats: overall, surface-specific, EWM
│   ├── 04_h2h.py                 # Head-to-head win rate computation
│   ├── 05_features.py            # Feature definitions and dataset construction
│   ├── 06_preprocessing.py       # sklearn preprocessor (imputation + scaling + OHE)
│   ├── 07_training.py            # GridSearchCV + TimeSeriesSplit + calibration
│   └── 08_predict.py             # Matchup prediction by player name and date
│
├── notebooks/
│   ├── 01_Evaluation.ipynb # AUC, LogLoss, Brier, calibration curves, ROC
│   ├── 02_BettingOdds_Benchmark.ipynb# Comparison vs Pinnacle/Bet365 closing odds
│   ├── 03_Tennis_Insights.ipynb  # Elo trajectories, upsets, surface specialists
│   └── 04_Predict_Matchup.ipynb  # Interactive matchup predictor
│
├── config.py                     # All global parameters in one place

---
```
## Features

### Rating features
- **Elo-538** (`elo_diff`, `elo_surface_diff`): FiveThirtyEight variant of the Elo scoring system with decaying K-factor. Updates after every match. Computed separately per surface. Consistently the strongest predictor (correlation of ~0.34 with the outcome). This solution was drawn from @BrandoPolistirolo project (https://github.com/BrandoPolistirolo/Tennis-Betting-ML)
- **ATP rank** (`rank_diff`, `rank_pts_diff`): Official ranking and points. Correlated with Elo but captures different information (52-week accumulation vs dynamic updating).

### Rolling performance features (per player, last 20 matches)
- Overall: `won_roll`, `ace_rate_roll`, `df_rate_roll`, `first_in_rate_roll`, `first_won_rate_roll`, `second_won_rate_roll`, `bp_save_rate_roll`
- Surface-specific (last 15 matches on same surface): same stats with `_surface` suffix
All rolling features were lagged by 1, hence the current match is never included in its own features.

### Head-to-head
- `h2h_win_rate_diff`: player A's win rate vs player B in last 10 meetings
- `h2h_n`: number of past meetings (used as confidence measure)

### Static features
- `age_diff`, `ht_diff`, `best_of`, `surface`, `tourney_level`, `round`

---

## Key design decisions

### Data leakage from match stats
Early versions used in-match statistics (`w_ace`, `w_df`, etc.) as features to predict the outcome of that same match. This gave ~99% accuracy. All features are derived exclusively from matches played *before* the match being predicted. Rolling averages use `shift(1)` (lagged by 1) to guarantee no current-match information is included.

### Mirror problem
Generating two rows per match (winner perspective + loser perspective) creates perfectly symmetric pairs. `rank_diff > 0 → y=1` becomes an obvious identity that inflates AUC to 0.98. We solved it using one row per match. A fixed random seed assigns which player is "A" vs "B" for each match. The resulting dataset is 50/50 balanced without creating symmetric pairs.

### Temporal leakage in train/test split
A random `train_test_split` allows future matches to appear in the training set, which is again unrealistic. To fix this, we did a strict chronological split: train on all matches before 2022, and test on all matches ≥ 2023. `TimeSeriesSplit` is also used inside `GridSearchCV` for hyperparameter tuning.

### Elo rating inflation
We observed that the mean Elo drifted from the 1500 initialisation to ~1670. This could be explained by the fact that, when weaker players retire, their points remain in the active pool. Since `elo_diff` shows that (A − B) is correctly centred at 0, it should not significantly affect our performances.

### Multicollinearity between rank and Elo
We observed that `rank_diff` and `elo_diff` have strong correlation (0.35). Both are included because they capture slightly different signals (Elo is opponent-weighted and updates continuously; ATP rank accumulates over 52 weeks). XGBoost handles this wihtout problems.

---

## Betting validation

The model was compared against Bet365 and Pinnacle closing odds on 2,727 ATP matches in 2023–2024. A flat €1 betting simulation on matches where the model disagreed with the bookmaker by more than 3% surprising yielded a +2.1% ROI — but a bootstrap confidence interval crossed zero, meaning the result is not yet statistically significant. Additional matches data would be needed to confirm a genuine edge.

See BettingOdds_benchmark.ipynb. 

---

## Tennis insights

Beyond prediction, I used the model to reveal interesting historical patterns:

- **Biggest upsets**: matches where the actual winner had the lowest predicted probability
- **Player Elo trajectories**: career peaks, injury dips, surface-specific evolution
- **Surface specialists vs strugglers**: players whose surface Elo diverges from their overall level (filtered for players with ≥ 20 surface matches and overall Elo ≥ 1650)
- **Closest finals**: Grand Slam finals the model considered most evenly matched

See Tennis_Insights.ipynb.

---

## Limitations

- Dataset only covers for 2011–2024 period; predictions for 2025+ require data updates
- No in-match data, which should held strong predictive value. 


---

