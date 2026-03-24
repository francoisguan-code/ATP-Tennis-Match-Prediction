import numpy as np
import pandas as pd
from features import ALL_FEATURES, ROLL_COLS, SURF_COLS, EWM_COLS


def predict_match(model, features: dict) -> dict:
    """
    Runs a single prediction given a feature dictionary.
    Missing keys are filled with NaN and handled by the preprocessor.
    Returns win probability for player A and a human-readable label.
    """
    X    = pd.DataFrame([{col: features.get(col, np.nan) for col in ALL_FEATURES}])
    prob = model.predict_proba(X)[0, 1]
    return {"P(A wins)": round(float(prob), 4),
            "prediction": "A wins" if prob >= 0.5 else "B wins"}


def get_player_profile(name: str, date: str,
                        raw: pd.DataFrame,
                        profiles: pd.DataFrame) -> dict:
    """
    Looks up a player's most recent rolling profile before a given date.

    Parameters
    ----------
    name     : partial or full player name (case-insensitive)
    date     : ISO date string, e.g. "2024-06-01"
    raw      : full match DataFrame from load_data()
    profiles : profiles DataFrame from compute_player_profiles()
    """
    date       = pd.Timestamp(date)
    name_lower = name.lower()
    mask_w     = raw["winner_name"].str.lower().str.contains(name_lower, na=False)
    mask_l     = raw["loser_name"].str.lower().str.contains(name_lower, na=False)

    candidates = pd.concat([
        raw[mask_w][["winner_id", "winner_name"]].rename(columns={"winner_id": "id", "winner_name": "name"}),
        raw[mask_l][["loser_id",  "loser_name" ]].rename(columns={"loser_id":  "id", "loser_name":  "name"}),
    ]).drop_duplicates()

    if candidates.empty:
        raise ValueError(f"Player '{name}' not found in dataset.")
    if len(candidates) > 1:
        print(f"Multiple matches for '{name}':")
        print(candidates.to_string(index=False))
        raise ValueError("use full name?")

    player_id   = candidates.iloc[0]["id"]
    player_name = candidates.iloc[0]["name"]

    past = raw[
        ((raw["winner_id"] == player_id) | (raw["loser_id"] == player_id)) &
        (raw["tourney_date"] < date)
    ].sort_values("tourney_date")

    if past.empty:
        raise ValueError(f"No matches before {date.date()} for {player_name}.")

    last    = past.iloc[-1]
    last_mid = f"{last['tourney_id']}_{last['match_num']}"

    try:
        row = profiles.loc[(player_id, last_mid)]
    except KeyError:
        raise ValueError(f"Profile not found for {player_name} at match {last_mid}.")

    print(f" {player_name}  |  last match: {last['tourney_date'].date()}"
          f"  |  {int(row['n_past'])} matches in window")

    return {"player_id": player_id, "player_name": player_name,
            "n_past": int(row["n_past"]),
            **{k: v for k, v in row.to_dict().items() if k != "n_past"}}


def predict_matchup(name_a: str, name_b: str, date: str,
                     surface: str, level: str, round_: str, best_of: int,
                     raw: pd.DataFrame, profiles: pd.DataFrame,
                     h2h: pd.DataFrame, trained: dict) -> None:
    """
    Predicts the outcome of a match between two players by name and date.

    Parameters
    ----------
    name_a, name_b : player names (partial match accepted)
    date           : match date, e.g. "2025-06-01"
    surface        : "Hard", "Clay", or "Grass"
    level          : "G" (Grand Slam), "M" (Masters), "A" (ATP 500/250)
    round_         : "F", "SF", "QF", "R16", "R32", "R64", "R128"
    best_of        : 3 or 5
    """
    print(f"\n{'='*55}\n  {name_a}  vs  {name_b}")
    print(f"  {surface} | {level} | {round_} | BO{best_of} | {date}\n{'='*55}")

    pa = get_player_profile(name_a, date, raw, profiles)
    pb = get_player_profile(name_b, date, raw, profiles)

    def _get_static(pid, field):
        sub = raw[(raw["winner_id"] == pid) | (raw["loser_id"] == pid)].sort_values("tourney_date")
        for _, r in sub.iloc[::-1].iterrows():
            for pref in ["winner", "loser"]:
                if r.get(f"{pref}_id") == pid and pd.notna(r.get(f"{pref}_{field}")):
                    return float(r[f"{pref}_{field}"])
        return np.nan

    aid, bid = pa["player_id"], pb["player_id"]
    pair     = raw[
        ((raw["winner_id"] == aid) & (raw["loser_id"] == bid)) |
        ((raw["winner_id"] == bid) & (raw["loser_id"] == aid))
    ].sort_values("tourney_date")
    past_pair = pair[pair["tourney_date"] < pd.Timestamp(date)]

    h2h_a, h2h_b, h2h_n = 0.5, 0.5, 0.0
    if not past_pair.empty:
        ref     = past_pair.iloc[-1]
        ref_mid = f"{ref['tourney_id']}_{ref['match_num']}"
        try:
            h2h_a = float(h2h.loc[(aid, bid, ref_mid)]["h2h_win_rate"])
            h2h_n = float(h2h.loc[(aid, bid, ref_mid)]["h2h_n"])
        except KeyError:
            pass
        try:
            h2h_b = float(h2h.loc[(bid, aid, ref_mid)]["h2h_win_rate"])
        except KeyError:
            pass

    all_roll_keys = ROLL_COLS + SURF_COLS + EWM_COLS
    features = {
        "rank_diff":         _get_static(aid, "rank")        - _get_static(bid, "rank"),
        "rank_pts_diff":     _get_static(aid, "rank_points") - _get_static(bid, "rank_points"),
        "age_diff":          _get_static(aid, "age")         - _get_static(bid, "age"),
        "ht_diff":           _get_static(aid, "ht")          - _get_static(bid, "ht"),
        "best_of":           best_of,
        "surface":           surface,
        "tourney_level":     level,
        "round":             round_,
        "h2h_win_rate_diff": h2h_a - h2h_b,
        "h2h_n":             h2h_n,
        "elo_diff":          np.nan,   # not available without elo_df lookup
        "elo_surface_diff":  np.nan,
        **{f"{k}_diff": pa.get(k, np.nan) - pb.get(k, np.nan) for k in all_roll_keys},
    }

    print(f"\n  {'Feature':<38} {'Value':>10}")
    print(f"  {'─'*38} {'─'*10}")
    for k, v in features.items():
        if isinstance(v, float):
            print(f"  {k:<38} {v:>10.4f}")
        else:
            print(f"  {k:<38} {str(v):>10}")

    print(f"\n  {'Model':<25} {'P('+name_a+' wins)':>18}  Prediction")
    print(f"  {'─'*25} {'─'*18}  {'─'*12}")
    for mname, model in trained.items():
        res  = predict_match(model, features)
        prob = res["P(A wins)"]
        pred = name_a if prob >= 0.5 else name_b
        print(f"  {mname:<25} {prob:>17.1%}  → {pred}")
