import numpy as np
import pandas as pd
from config import MIN_MATCHES, RANDOM_SEED
from player_profiles import STAT_COLS

ROLL_COLS = [f"{c}_roll"         for c in STAT_COLS]
SURF_COLS = [f"{c}_roll_surface" for c in STAT_COLS]
EWM_COLS  = [f"{c}_ewm"          for c in STAT_COLS]

NUMERIC_FEATURES = (
    ["rank_diff", "rank_pts_diff", "age_diff", "ht_diff", "best_of",
     "h2h_win_rate_diff", "h2h_n", "elo_diff", "elo_surface_diff"]
    + [f"{c}_diff" for c in ROLL_COLS]
    + [f"{c}_diff" for c in SURF_COLS]
    + [f"{c}_diff" for c in EWM_COLS]
)

CATEGORICAL_FEATURES = ["surface", "tourney_level", "round"]
ALL_FEATURES         = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_dataset(raw: pd.DataFrame,
                  profiles: pd.DataFrame,
                  h2h: pd.DataFrame,
                  elo_df: pd.DataFrame,
                  min_matches: int = MIN_MATCHES,
                  seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Builds the final modelling dataset: one row per match.

    A random coin flip (fixed seed) assigns which player is A vs B.
    y = 1 if A wins, 0 if B wins — guarantees ~50/50 class balance
    without the mirror-problem of doubling rows.

    Features are expressed as differences: profile_A - profile_B.
    """
    rng          = np.random.default_rng(seed)
    all_roll_keys = ROLL_COLS + SURF_COLS + EWM_COLS

    # Build a static cache of each player's most recent rank/age/ht
    static_cache = {}
    for _, r in raw.iterrows():
        for pref in ["winner", "loser"]:
            pid = r.get(f"{pref}_id")
            if pid is None or (isinstance(pid, float) and np.isnan(pid)):
                continue
            for field in ["rank", "rank_points", "age", "ht"]:
                val = r.get(f"{pref}_{field}")
                if pd.notna(val):
                    static_cache[(pid, field)] = float(val)

    def _static(pid, field):
        return static_cache.get((pid, field), np.nan)

    rows    = []
    skipped = 0

    for _, match in raw.iterrows():
        mid = f"{match['tourney_id']}_{match['match_num']}"
        wid = match["winner_id"]
        lid = match["loser_id"]

        try:
            wp = profiles.loc[(wid, mid)]
            lp = profiles.loc[(lid, mid)]
        except KeyError:
            skipped += 1
            continue

        if wp["n_past"] < min_matches or lp["n_past"] < min_matches:
            skipped += 1
            continue

        a_is_winner = bool(rng.integers(2))
        ap, bp      = (wp, lp) if a_is_winner else (lp, wp)
        aid, bid    = (wid, lid) if a_is_winner else (lid, wid)

        diffs = {}
        for col in all_roll_keys:
            av = float(ap.get(col, np.nan))
            bv = float(bp.get(col, np.nan))
            diffs[f"{col}_diff"] = (av - bv) if not (np.isnan(av) or np.isnan(bv)) else np.nan

        try:
            h2h_a = float(h2h.loc[(aid, bid, mid)]["h2h_win_rate"])
            h2h_n = float(h2h.loc[(aid, bid, mid)]["h2h_n"])
        except KeyError:
            h2h_a, h2h_n = 0.5, 0.0
        try:
            h2h_b = float(h2h.loc[(bid, aid, mid)]["h2h_win_rate"])
        except KeyError:
            h2h_b = 0.5

        try:
            elo_row       = elo_df.loc[mid]
            w_elo         = float(elo_row["winner_elo"])
            l_elo         = float(elo_row["loser_elo"])
            w_elo_s       = float(elo_row["winner_elo_surf"])
            l_elo_s       = float(elo_row["loser_elo_surf"])
            elo_diff      = (w_elo   - l_elo)   if a_is_winner else (l_elo   - w_elo)
            elo_surf_diff = (w_elo_s - l_elo_s) if a_is_winner else (l_elo_s - w_elo_s)
        except KeyError:
            elo_diff, elo_surf_diff = np.nan, np.nan

        def _d(field):
            va, vb = _static(aid, field), _static(bid, field)
            return (va - vb) if not (np.isnan(va) or np.isnan(vb)) else np.nan

        rows.append({
            "match_id":          mid,
            "tourney_date":      match["tourney_date"],
            "year":              match["year"],
            "y":                 1 if a_is_winner else 0,
            "rank_diff":         _d("rank"),
            "rank_pts_diff":     _d("rank_points"),
            "age_diff":          _d("age"),
            "ht_diff":           _d("ht"),
            "best_of":           match.get("best_of"),
            "surface":           match.get("surface"),
            "tourney_level":     match.get("tourney_level"),
            "round":             match.get("round"),
            "h2h_win_rate_diff": h2h_a - h2h_b,
            "h2h_n":             h2h_n,
            "elo_diff":          elo_diff,
            "elo_surface_diff":  elo_surf_diff,
            **diffs,
        })

    df = pd.DataFrame(rows).sort_values("tourney_date").reset_index(drop=True)
    print(f"✓ Dataset built — {len(df):,} matches | skipped: {skipped:,}")
    print(f"  y=1: {df['y'].sum():,}  |  y=0: {(1-df['y']).sum():,}")
    return df
