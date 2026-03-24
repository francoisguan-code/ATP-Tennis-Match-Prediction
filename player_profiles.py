import numpy as np
import pandas as pd
from config import ROLLING_WINDOW, SURFACE_WINDOW, EWM_HALFLIFE

STAT_COLS = ["won", "ace_rate", "df_rate", "first_in_rate",
             "first_won_rate", "second_won_rate", "bp_save_rate"]


def _ratio(num, den):
    try:
        num, den = float(num), float(den)
        return num / den if den > 0 else np.nan
    except (TypeError, ValueError):
        return np.nan


def compute_player_profiles(raw: pd.DataFrame,
                             window: int = ROLLING_WINDOW,
                             surface_window: int = SURFACE_WINDOW,
                             ewm_halflife: int = EWM_HALFLIFE) -> pd.DataFrame:
    """
    For each (player, match) computes three types of rolling stats:
      *_roll         : overall rolling mean (last `window` matches)
      *_roll_surface : surface-specific rolling mean (last `surface_window` matches)
      *_ewm          : exponentially weighted mean (half-life = `ewm_halflife`)

    All use shift(1) — the current match is never included in its own features.
    Returns a DataFrame indexed by (player_id, match_id).
    """
    records = []
    for _, row in raw.iterrows():
        mid  = f"{row['tourney_id']}_{row['match_num']}"
        surf = row.get("surface", np.nan)

        for role, prefix in [("winner", "w_"), ("loser", "l_")]:
            pid = row.get(f"{role}_id")
            if pd.isna(pid):
                continue

            svpt       = row.get(f"{prefix}svpt")
            first_in   = row.get(f"{prefix}1stIn")
            second_svpt = np.nan
            try:
                s, fi = float(svpt), float(first_in)
                if not (np.isnan(s) or np.isnan(fi)):
                    second_svpt = s - fi
            except (TypeError, ValueError):
                pass

            records.append({
                "player_id":       pid,
                "match_id":        mid,
                "tourney_date":    row["tourney_date"],
                "year":            row["year"],
                "surface":         surf,
                "won":             1 if role == "winner" else 0,
                "ace_rate":        _ratio(row.get(f"{prefix}ace"), svpt),
                "df_rate":         _ratio(row.get(f"{prefix}df"),  svpt),
                "first_in_rate":   _ratio(first_in, svpt),
                "first_won_rate":  _ratio(row.get(f"{prefix}1stWon"), first_in),
                "second_won_rate": _ratio(row.get(f"{prefix}2ndWon"), second_svpt),
                "bp_save_rate":    _ratio(row.get(f"{prefix}bpSaved"),
                                          row.get(f"{prefix}bpFaced")),
            })

    long = (pd.DataFrame(records)
              .sort_values(["player_id", "tourney_date"])
              .reset_index(drop=True))

    all_parts = []
    for _, grp in long.groupby("player_id"):
        grp = grp.copy().reset_index(drop=True)

        grp["n_past"] = grp["won"].shift(1).rolling(window, min_periods=1).count()
        for col in STAT_COLS:
            grp[f"{col}_roll"] = grp[col].shift(1).rolling(window, min_periods=1).mean()

        for col in STAT_COLS:
            grp[f"{col}_roll_surface"] = np.nan
        for surf_val in grp["surface"].dropna().unique():
            mask = grp["surface"] == surf_val
            for col in STAT_COLS:
                rolled = (grp[col].where(mask)
                                  .shift(1)
                                  .rolling(surface_window, min_periods=1)
                                  .mean())
                grp.loc[mask, f"{col}_roll_surface"] = rolled[mask]
        for col in STAT_COLS:
            grp[f"{col}_roll_surface"] = grp[f"{col}_roll_surface"].fillna(grp[f"{col}_roll"])

        for col in STAT_COLS:
            grp[f"{col}_ewm"] = grp[col].shift(1).ewm(halflife=ewm_halflife, min_periods=1).mean()

        all_parts.append(grp)

    profiles = pd.concat(all_parts, ignore_index=True).set_index(["player_id", "match_id"])

    keep = ([f"{c}_roll" for c in STAT_COLS]
            + [f"{c}_roll_surface" for c in STAT_COLS]
            + [f"{c}_ewm" for c in STAT_COLS]
            + ["n_past"])

    print(f" Profiles computed — {profiles.index.get_level_values('player_id').nunique():,} players")
    return profiles[keep]
