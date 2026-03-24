import pandas as pd
from config import H2H_WINDOW


def compute_h2h(raw: pd.DataFrame, h2h_window: int = H2H_WINDOW) -> pd.DataFrame:
    """
    For each (player_A, player_B, match) computes:
      h2h_win_rate : A's win rate vs B over their last h2h_window meetings
                     before this match (shift(1)). Defaults to 0.5 on first meeting.
      h2h_n        : number of past meetings used.

    Returns a DataFrame indexed by (p1, p2, match_id).
    """
    records = []
    for _, row in raw.iterrows():
        mid  = f"{row['tourney_id']}_{row['match_num']}"
        date = row["tourney_date"]
        wid  = row["winner_id"]
        lid  = row["loser_id"]
        records.append({"match_id": mid, "tourney_date": date, "p1": wid, "p2": lid, "p1_won": 1})
        records.append({"match_id": mid, "tourney_date": date, "p1": lid, "p2": wid, "p1_won": 0})

    h2h_df = (pd.DataFrame(records)
                .sort_values(["p1", "p2", "tourney_date"])
                .reset_index(drop=True))

    parts = []
    for _, grp in h2h_df.groupby(["p1", "p2"]):
        grp = grp.copy().reset_index(drop=True)
        grp["h2h_win_rate"] = (grp["p1_won"].shift(1)
                                             .rolling(h2h_window, min_periods=1)
                                             .mean()
                                             .fillna(0.5))
        grp["h2h_n"] = (grp["p1_won"].shift(1)
                                      .rolling(h2h_window, min_periods=1)
                                      .count()
                                      .fillna(0))
        parts.append(grp[["match_id", "p1", "p2", "h2h_win_rate", "h2h_n"]])

    h2h_out = pd.concat(parts, ignore_index=True).set_index(["p1", "p2", "match_id"])
    print(f" H2H computed — {len(h2h_out):,} player-pair-match entries")
    return h2h_out
