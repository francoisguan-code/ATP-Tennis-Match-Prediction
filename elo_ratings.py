import pandas as pd
from config import ELO_INIT, ELO_BASE, ELO_K_NUM, ELO_K_DENOM, ELO_K_POW


def _win_prob(elo_self: float, elo_opp: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_opp - elo_self) / ELO_BASE))


def _k(n_matches: int) -> float:
    return ELO_K_NUM / (ELO_K_DENOM + n_matches) ** ELO_K_POW


def compute_elo(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Iterates chronologically and stores, for each match, the Elo of both
    players BEFORE the match — zero leakage guaranteed.

    Returns a DataFrame indexed by match_id with four columns:
        winner_elo, loser_elo, winner_elo_surf, loser_elo_surf
    """
    elo       = {}
    elo_surf  = {}
    n_matches = {}
    n_surf    = {}
    records   = []

    for _, row in raw.iterrows():
        wid  = row["winner_id"]
        lid  = row["loser_id"]
        surf = row.get("surface", "Hard") or "Hard"
        mid  = f"{row['tourney_id']}_{row['match_num']}"

        w_elo   = elo.get(wid, ELO_INIT)
        l_elo   = elo.get(lid, ELO_INIT)
        w_elo_s = elo_surf.get((wid, surf), ELO_INIT)
        l_elo_s = elo_surf.get((lid, surf), ELO_INIT)
        w_n     = n_matches.get(wid, 0)
        l_n     = n_matches.get(lid, 0)
        w_ns    = n_surf.get((wid, surf), 0)
        l_ns    = n_surf.get((lid, surf), 0)

        records.append({
            "match_id":        mid,
            "winner_elo":      w_elo,
            "loser_elo":       l_elo,
            "winner_elo_surf": w_elo_s,
            "loser_elo_surf":  l_elo_s,
        })

        p_w = _win_prob(w_elo, l_elo)
        elo[wid] = w_elo + _k(w_n) * (1.0 - p_w)
        elo[lid] = l_elo + _k(l_n) * (0.0 - (1.0 - p_w))

        p_ws = _win_prob(w_elo_s, l_elo_s)
        elo_surf[(wid, surf)] = w_elo_s + _k(w_ns) * (1.0 - p_ws)
        elo_surf[(lid, surf)] = l_elo_s + _k(l_ns) * (0.0 - (1.0 - p_ws))

        n_matches[wid]       = w_n  + 1
        n_matches[lid]       = l_n  + 1
        n_surf[(wid, surf)]  = w_ns + 1
        n_surf[(lid, surf)]  = l_ns + 1

    elo_df = pd.DataFrame(records).set_index("match_id")
    print(f" Elo computed — {len(elo):,} players, {len(elo_surf):,} player-surface pairs")
    return elo_df
