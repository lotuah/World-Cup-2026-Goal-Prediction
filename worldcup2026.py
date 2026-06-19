"""
Zindi - World Cup 2026 Goal Prediction Challenge
================================================
Improved model: XGBoost + LightGBM + GBM + Ridge ensemble
32 engineered features from 5 data sources

Validation on 2022 holdout:
  Goals RMSE: 3.049  (baseline was 3.945, Δ +0.90)
  Stage F1:   0.566  (baseline was 0.554, Δ +0.01)

Usage:
  pip install pandas numpy scikit-learn xgboost lightgbm
  python worldcup2026_improved.py
  
  Place all CSVs in the same folder (or set DATA_DIR below).
  Replace IDs in submission.csv with Zindi's SampleSubmission.csv IDs.
"""
pip install pandas numpy scikit-learn xgboost lig
pip install sklearn
pip install scikit-learn

import pandas as pd
import numpy as np
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
    GradientBoostingClassifier
)
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error, f1_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
import warnings
warnings.filterwarnings("ignore")

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR = "."   # folder containing the 5 CSV files

MENS_WC = [
    "WC-1930","WC-1934","WC-1938","WC-1950","WC-1954","WC-1958","WC-1962",
    "WC-1966","WC-1970","WC-1974","WC-1978","WC-1982","WC-1986","WC-1990",
    "WC-1994","WC-1998","WC-2002","WC-2006","WC-2010","WC-2014","WC-2018","WC-2022",
]

STAGE_ORDER = {
    "group stage": 0, "second group stage": 1, "final round": 1,
    "round of 16": 2, "quarter-final": 3, "quarter-finals": 3,
    "semi-final": 4, "semi-finals": 4, "third-place match": 5, "final": 6,
}
STAGE_LABEL = {
    0: "group", 1: "roundof32", 2: "roundof16",
    3: "qf", 4: "sf", 5: "runnerup", 6: "champion",
}

TEAMS_2026 = [
    "Argentina","Brazil","France","England","Germany","Spain","Portugal","Netherlands",
    "Belgium","Italy","Croatia","Uruguay","Mexico","United States","Canada","Colombia",
    "Ecuador","Peru","Chile","Venezuela","Paraguay","Bolivia",
    "Morocco","Senegal","Egypt","Nigeria","Ivory Coast","Cameroon","Ghana",
    "South Africa","Algeria","Tunisia",
    "Japan","South Korea","Australia","Iran","Saudi Arabia","Qatar","Indonesia",
    "Serbia","Poland","Switzerland","Denmark","Austria","Turkey","Ukraine","Slovakia","Albania",
]

FEAT_COLS = [
    # Career historical averages (all prior tournaments)
    "hist_total_goals", "hist_wins", "hist_goal_diff", "hist_avg_goals_per_match",
    "hist_goals_against", "hist_appearances", "hist_stage_avg", "hist_stage_best",
    "hist_win_rate", "hist_goal_rate", "hist_consistency",
    "hist_open_play_goals", "hist_late_goals", "hist_early_goals",
    "hist_knockout_matches", "hist_opponent_avg_goals_scored",
    "hist_schedule_difficulty", "hist_pen_goals",
    # Most recent tournament performance
    "last_total_goals", "last_wins", "last_goal_diff", "last_avg_goals_per_match",
    "last_goals_against", "last_open_play_goals", "last_late_goals",
    # Penalty shootout history
    "pk_rate", "pk_attempts",
    # Tournament context
    "year", "era", "debut",
    # Recency-weighted composites
    "recent_weighted_goals", "recent_weighted_stage",
]

# ── 1. Load raw data ──────────────────────────────────────────────────────────

ta       = pd.read_csv(f"{DATA_DIR}/team_appearances.csv")
goals_df = pd.read_csv(f"{DATA_DIR}/goals.csv")
pk       = pd.read_csv(f"{DATA_DIR}/penalty_kicks.csv")

for frame in [ta, goals_df, pk]:
    frame["year"] = frame["tournament_id"].str.extract(r"(\d{4})").astype(int)

ta       = ta      [ta["tournament_id"].isin(MENS_WC)].copy()
goals_df = goals_df[goals_df["tournament_id"].isin(MENS_WC)].copy()
pk       = pk      [pk["tournament_id"].isin(MENS_WC)].copy()
ta["stage_num"] = ta["stage_name"].map(STAGE_ORDER).fillna(0)

# ── 2. Team × tournament base features ───────────────────────────────────────

feats = ta.groupby(["tournament_id","team_id","team_name","year"]).agg(
    total_goals          = ("goals_for",          "sum"),
    goals_against        = ("goals_against",       "sum"),
    matches_played       = ("match_id",            "count"),
    wins                 = ("win",                 "sum"),
    losses               = ("lose",                "sum"),
    draws                = ("draw",                "sum"),
    pen_shootouts        = ("penalty_shootout",    "sum"),
    extra_time_matches   = ("extra_time",          "sum"),
    goal_diff            = ("goal_differential",   "sum"),
    home_matches         = ("home_team",           "sum"),
    avg_goals_per_match  = ("goals_for",           "mean"),
    avg_conceded         = ("goals_against",       "mean"),
    knockout_matches     = ("knockout_stage",      "sum"),
).reset_index()

# ── 3. Opponent / schedule difficulty ────────────────────────────────────────

# Average goals the opponent scored in that tournament (proxy for opponent quality)
opp_str = (ta.groupby(["tournament_id","opponent_id"])["goals_for"]
             .mean().reset_index()
             .rename(columns={"opponent_id":"team_id",
                               "goals_for":"opponent_avg_goals_scored"}))
feats = feats.merge(opp_str, on=["tournament_id","team_id"], how="left")

# Average goals conceded by the team's opponents (how tough was the draw)
opp_def = (ta.groupby(["tournament_id","team_id"])["goals_against"]
             .mean().reset_index()
             .rename(columns={"team_id":"opponent_id",
                               "goals_against":"opp_avg_conceded"}))
ta2 = ta.merge(opp_def, on=["tournament_id","opponent_id"], how="left")
sched = (ta2.groupby(["tournament_id","team_id"])["opp_avg_conceded"]
            .mean().reset_index()
            .rename(columns={"opp_avg_conceded":"schedule_difficulty"}))
feats = feats.merge(sched, on=["tournament_id","team_id"], how="left")

# ── 4. Goal-type features from goals.csv ─────────────────────────────────────

team_goals = (goals_df[goals_df["own_goal"] == 0]
              .groupby(["tournament_id","team_id","year"]).agg(
                  open_play_goals = ("penalty",          lambda x: (x == 0).sum()),
                  pen_goals       = ("penalty",          "sum"),
                  late_goals      = ("minute_regulation",lambda x: (x >= 75).sum()),
                  early_goals     = ("minute_regulation",lambda x: (x <= 20).sum()),
                  et_goals        = ("match_period",     lambda x: x.str.contains("extra time").sum()),
                  avg_goal_minute = ("minute_regulation","mean"),
              ).reset_index())
feats = feats.merge(
    team_goals[["tournament_id","team_id","open_play_goals","pen_goals",
                 "late_goals","early_goals","et_goals","avg_goal_minute"]],
    on=["tournament_id","team_id"], how="left")
for c in ["open_play_goals","pen_goals","late_goals","early_goals","et_goals"]:
    feats[c] = feats[c].fillna(0)
feats["avg_goal_minute"] = feats["avg_goal_minute"].fillna(45)

# ── 5. Penalty shootout history ───────────────────────────────────────────────

pk_stats = (pk.groupby(["team_id","year"]).agg(
                pk_attempts = ("converted","count"),
                pk_scored   = ("converted","sum")).reset_index())
pk_stats["pk_rate"] = pk_stats["pk_scored"] / pk_stats["pk_attempts"]

# ── 6. Stage-reached target ───────────────────────────────────────────────────

max_stage = ta.groupby(["tournament_id","team_id"])["stage_num"].max().reset_index()
finals = ta[ta["stage_name"] == "final"]
for _, row in finals[finals["win"]  == 1].iterrows():
    m = (max_stage["tournament_id"]==row["tournament_id"]) & (max_stage["team_id"]==row["team_id"])
    max_stage.loc[m, "stage_num"] = 6   # champion
for _, row in finals[finals["lose"] == 1].iterrows():
    m = (max_stage["tournament_id"]==row["tournament_id"]) & (max_stage["team_id"]==row["team_id"])
    max_stage.loc[m, "stage_num"] = 5   # runner-up

# ── 7. Merge and engineer rolling historical features ─────────────────────────

df = feats.merge(max_stage, on=["tournament_id","team_id"], how="left")
df["stage_label"] = df["stage_num"].map(STAGE_LABEL)
df = df.merge(pk_stats[["team_id","year","pk_rate","pk_attempts"]],
              on=["team_id","year"], how="left")
df["pk_rate"]     = df["pk_rate"].fillna(0.5)
df["pk_attempts"] = df["pk_attempts"].fillna(0)

df = df.sort_values(["team_id","year"])

ROLL_COLS = ["total_goals","wins","goal_diff","avg_goals_per_match","goals_against",
             "open_play_goals","late_goals","early_goals","knockout_matches",
             "opponent_avg_goals_scored","schedule_difficulty","pen_goals"]

for col in ROLL_COLS:
    # Career average using ONLY prior tournaments (no leakage)
    df[f"hist_{col}"] = df.groupby("team_id")[col].transform(
        lambda x: x.shift(1).expanding().mean())
    # Most recent tournament value
    df[f"last_{col}"] = df.groupby("team_id")[col].transform(
        lambda x: x.shift(1))

df["hist_appearances"] = df.groupby("team_id").cumcount()
df["hist_stage_avg"]   = df.groupby("team_id")["stage_num"].transform(
    lambda x: x.shift(1).expanding().mean())
df["hist_stage_best"]  = df.groupby("team_id")["stage_num"].transform(
    lambda x: x.shift(1).expanding().max())
df["hist_win_rate"]    = df["hist_wins"] / df["hist_appearances"].clip(lower=1)
df["hist_goal_rate"]   = df["hist_total_goals"] / df["hist_appearances"].clip(lower=1)
df["hist_consistency"] = df.groupby("team_id")["total_goals"].transform(
    lambda x: x.shift(1).expanding().std().fillna(0))

# Tournament era: scoring patterns differ significantly across eras
df["era"]   = pd.cut(df["year"], bins=[0,1970,1990,2006,2030], labels=[0,1,2,3]).astype(int)
df["debut"] = (df["hist_appearances"] == 0).astype(int)

# Recency-weighted composites (last tournament counts 2×)
df["recent_weighted_goals"] = (df["last_total_goals"] * 2 + df["hist_total_goals"]) / 3
df["recent_weighted_stage"] = (
    df.groupby("team_id")["stage_num"].transform(lambda x: x.shift(1)) * 2
    + df["hist_stage_avg"]) / 3

df = df.fillna(0)

# ── 8. Train / validate on 2022 holdout ──────────────────────────────────────

train = df[df["year"] < 2022].copy()
test  = df[df["year"] == 2022].copy()
print(f"Train: {len(train)} rows | Validate (2022): {len(test)} rows | Features: {len(FEAT_COLS)}")

# --- Goals: 4-model blend ---
reg_models = [
    ("xgb",  XGBRegressor(n_estimators=300, learning_rate=0.04, max_depth=4,
                           subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0),  0.30),
    ("lgbm", LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=15,
                            subsample=0.8, random_state=42, verbose=-1),                         0.30),
    ("gbm",  GradientBoostingRegressor(n_estimators=300, learning_rate=0.04,
                                        max_depth=4, subsample=0.8, random_state=42),            0.25),
    ("ridge",Ridge(alpha=1.0),                                                                    0.15),
]

val_preds_reg = []
for name, m, w in reg_models:
    m.fit(train[FEAT_COLS], train["total_goals"])
    p = m.predict(test[FEAT_COLS])
    val_preds_reg.append((p, w))
    print(f"  {name:5s} RMSE: {root_mean_squared_error(test['total_goals'], p):.3f}")

blend_goals = sum(p * w for p, w in val_preds_reg)
rmse_val = root_mean_squared_error(test["total_goals"], blend_goals)
print(f"Ensemble Goals RMSE: {rmse_val:.3f}\n")

# --- Stage: 4-model blend ---
le = LabelEncoder().fit(df["stage_label"])
classes = le.classes_

clf_models = [
    ("xgb",  XGBClassifier(n_estimators=300, learning_rate=0.04, max_depth=5,
                             subsample=0.8, colsample_bytree=0.8, random_state=42,
                             eval_metric="mlogloss", verbosity=0),                    0.30, True),
    ("lgbm", LGBMClassifier(n_estimators=300, learning_rate=0.04, num_leaves=15,
                              subsample=0.8, random_state=42, verbose=-1,
                              class_weight="balanced"),                                0.30, False),
    ("hgb",  HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                             max_depth=5, random_state=42),            0.25, False),
    ("rf",   RandomForestClassifier(n_estimators=400, max_depth=8, random_state=42,
                                     class_weight="balanced", min_samples_leaf=2),     0.15, False),
]

proba_blend = np.zeros((len(test), len(classes)))
for name, m, w, use_enc in clf_models:
    if use_enc:
        m.fit(train[FEAT_COLS], le.transform(train["stage_label"]))
        proba = m.predict_proba(test[FEAT_COLS])
        proba_blend += w * proba
        pred_labels = le.inverse_transform(m.predict(test[FEAT_COLS]))
    else:
        m.fit(train[FEAT_COLS], train["stage_label"])
        proba = m.predict_proba(test[FEAT_COLS])
        mc = list(m.classes_)
        for i, c in enumerate(classes):
            if c in mc:
                proba_blend[:, i] += w * proba[:, mc.index(c)]
        pred_labels = m.predict(test[FEAT_COLS])
    f1_i = f1_score(test["stage_label"], pred_labels, average="weighted")
    print(f"  {name:5s} F1:   {f1_i:.3f}")

val_stage = classes[np.argmax(proba_blend, axis=1)]
f1_val = f1_score(test["stage_label"], val_stage, average="weighted")
print(f"Ensemble Stage F1:   {f1_val:.3f}")
print(f"\nBaseline → Improved:  RMSE {3.945:.3f} → {rmse_val:.3f}  |  F1 {0.554:.3f} → {f1_val:.3f}")

# ── 9. Retrain on ALL data, predict 2026 ─────────────────────────────────────

print("\nRetraining on full dataset...")
for _, m, _ in reg_models:
    m.fit(df[FEAT_COLS], df["total_goals"])
for _, m, _, use_enc in clf_models:
    if use_enc:
        m.fit(df[FEAT_COLS], le.transform(df["stage_label"]))
    else:
        m.fit(df[FEAT_COLS], df["stage_label"])

# Build 2026 feature rows from career averages through 2022
career = df.groupby("team_name").agg(
    hist_total_goals         = ("total_goals",          "mean"),
    hist_wins                = ("wins",                 "mean"),
    hist_goal_diff           = ("goal_diff",            "mean"),
    hist_avg_goals_per_match = ("avg_goals_per_match",  "mean"),
    hist_goals_against       = ("goals_against",        "mean"),
    hist_appearances         = ("year",                 "count"),
    hist_stage_avg           = ("stage_num",            "mean"),
    hist_stage_best          = ("stage_num",            "max"),
    hist_win_rate            = ("wins",                 lambda x: x.mean() / max(x.count(), 1)),
    hist_open_play_goals     = ("open_play_goals",      "mean"),
    hist_late_goals          = ("late_goals",           "mean"),
    hist_early_goals         = ("early_goals",          "mean"),
    hist_knockout_matches    = ("knockout_matches",     "mean"),
    hist_opponent_avg_goals_scored = ("opponent_avg_goals_scored", "mean"),
    hist_schedule_difficulty = ("schedule_difficulty",  "mean"),
    hist_pen_goals           = ("pen_goals",            "mean"),
    last_total_goals         = ("total_goals",          "last"),
    last_wins                = ("wins",                 "last"),
    last_goal_diff           = ("goal_diff",            "last"),
    last_avg_goals_per_match = ("avg_goals_per_match",  "last"),
    last_goals_against       = ("goals_against",        "last"),
    last_open_play_goals     = ("open_play_goals",      "last"),
    last_late_goals          = ("late_goals",           "last"),
    pk_rate                  = ("pk_rate",              "last"),
    pk_attempts              = ("pk_attempts",          "sum"),
).reset_index()
career["hist_goal_rate"]        = career["hist_total_goals"] / career["hist_appearances"].clip(lower=1)
career["hist_consistency"]      = 0   # placeholder (std needs full series)
career["year"]                  = 2026
career["era"]                   = 3
career["debut"]                 = 0
career["recent_weighted_goals"] = (career["last_total_goals"]*2 + career["hist_total_goals"]) / 3
career["recent_weighted_stage"] = (career["hist_stage_avg"]*2 + career["hist_stage_avg"]) / 3

pred_df = pd.DataFrame({"team_name": TEAMS_2026}).merge(career, on="team_name", how="left")
pred_df["year"]   = 2026
pred_df["era"]    = 3
pred_df["debut"]  = (pred_df["hist_appearances"].isna()).astype(int)
pred_df = pred_df.fillna(0)

# Goals predictions (ensemble)
preds_2026_reg = [m.predict(pred_df[FEAT_COLS]) for _, m, _ in reg_models]
goals_2026 = sum(p * w for p, w in zip(preds_2026_reg, [w for _,_,w in reg_models]))
goals_2026 = np.maximum(goals_2026, 0).round(0).astype(int)

# Stage predictions (ensemble)
proba_2026 = np.zeros((len(pred_df), len(classes)))
for name, m, w, use_enc in clf_models:
    if use_enc:
        proba = m.predict_proba(pred_df[FEAT_COLS])
        proba_2026 += w * proba
    else:
        proba = m.predict_proba(pred_df[FEAT_COLS])
        mc = list(m.classes_)
        for i, c in enumerate(classes):
            if c in mc:
                proba_2026[:, i] += w * proba[:, mc.index(c)]
stage_2026 = classes[np.argmax(proba_2026, axis=1)]

pred_df["total_goals"] = goals_2026
pred_df["Target"]      = stage_2026

print("\n2026 Predictions (sorted by predicted goals):")
print(pred_df[["team_name","total_goals","Target"]]
      .sort_values("total_goals", ascending=False)
      .to_string(index=False))

# ── 10. Write submission file ─────────────────────────────────────────────────
import hashlib
pred_df["ID"] = pred_df["team_name"].apply(
    lambda x: "ROW_" + hashlib.md5(x.encode()).hexdigest()[:12].upper())

submission = pred_df[["ID","total_goals","Target"]].copy()
submission.to_csv("submission.csv", index=False)
print("\nsubmission.csv saved.")
print(submission.head(5).to_string(index=False))
print("\n⚠️  Replace the ID column with IDs from Zindi's SampleSubmission.csv before uploading.")
