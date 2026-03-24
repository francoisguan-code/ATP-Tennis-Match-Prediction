import os
import numpy as np
import pandas as pd
from config import DATA_FOLDER, YEARS


NUM_COLS = [
    "winner_rank", "winner_rank_points", "winner_age", "winner_ht",
    "loser_rank",  "loser_rank_points",  "loser_age",  "loser_ht",
    "w_ace", "w_df", "w_svpt", "w_1stIn", "w_1stWon", "w_2ndWon",
    "w_SvGms", "w_bpSaved", "w_bpFaced",
    "l_ace", "l_df", "l_svpt", "l_1stIn", "l_1stWon", "l_2ndWon",
    "l_SvGms", "l_bpSaved", "l_bpFaced",
]


def load_data(folder: str = DATA_FOLDER, years=YEARS) -> pd.DataFrame:
    frames = []
    for y in years:
        path = os.path.join(folder, f"atp_matches_{y}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, low_memory=False)
            df["year"] = y
            frames.append(df)
            print(f" {y} : {len(df):,} matches")
        else:
            print(f" {y} : not found")

    if not frames:
        raise FileNotFoundError(f"No files found in: {folder}")

    raw = pd.concat(frames, ignore_index=True)
    raw["tourney_date"] = pd.to_datetime(
        raw["tourney_date"].astype(str), format="%Y%m%d", errors="coerce"
    )
    raw = raw.dropna(subset=["tourney_date"])
    raw = raw.sort_values("tourney_date").reset_index(drop=True)

    for col in NUM_COLS:
        if col in raw.columns:
            raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.dropna(subset=["winner_id", "loser_id"])
    print(f"\n total: {len(raw):,} matches\n")
    return raw
