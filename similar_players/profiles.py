"""Turn match rows into comparable style profiles."""

from __future__ import annotations

import numpy as np
import pandas as pd

RATE_FEATURES = [
    "passes", "progressive_passes", "long_passes", "key_passes", "crosses", "progressive_carries",
    "dribbles", "shots", "xg", "box_touches", "pressures", "tackles", "interceptions", "recoveries",
    "clearances", "blocks", "fouls_committed",
]
SHARE_FEATURES = ["pass_completion", "dribble_success", "avg_x", "avg_width"]
FEATURES = [f"{name}_p90" for name in RATE_FEATURES] + SHARE_FEATURES


def build_profile(rows: pd.DataFrame) -> pd.Series:
    """Add up match rows for one player and express them per 90 minutes."""
    total = rows.drop(columns=["match_id", "player_id", "player", "team", "position"], errors="ignore").sum()
    per_90 = 90.0 / total["minutes"]
    profile = {f"{name}_p90": total[name] * per_90 for name in RATE_FEATURES}
    profile["pass_completion"] = total["passes_completed"] / total["passes"] if total["passes"] else 0.0
    profile["dribble_success"] = total["dribbles_completed"] / total["dribbles"] if total["dribbles"] else 0.0
    profile["avg_x"] = total["x_total"] / total["actions"] if total["actions"] else 0.0
    profile["avg_width"] = total["width_total"] / total["actions"] if total["actions"] else 0.0
    profile["minutes"] = total["minutes"]
    return pd.Series(profile)


def build_profiles(rows: pd.DataFrame, min_minutes: float = 270) -> pd.DataFrame:
    """Build one profile per player, using their most common position."""
    records = []
    for player_id, group in rows.groupby("player_id"):
        if group["minutes"].sum() < min_minutes:
            continue
        profile = build_profile(group)
        profile["player_id"] = player_id
        profile["player"] = group["player"].iloc[-1]
        profile["team"] = group["team"].iloc[-1]
        profile["position"] = group.groupby("position")["minutes"].sum().idxmax()
        records.append(profile)
    frame = pd.DataFrame(records)
    return frame[frame["position"].isin(["Unknown", "Goalkeeper"]) == False].reset_index(drop=True)  # noqa: E712


def split_rows(rows: pd.DataFrame, min_half_minutes: float = 135) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split each player's matches into two halves that share no match."""
    first, second = [], []
    for _, group in rows.sort_values("match_id").groupby("player_id"):
        a, b = group.iloc[::2], group.iloc[1::2]
        if a["minutes"].sum() >= min_half_minutes and b["minutes"].sum() >= min_half_minutes:
            first.append(a)
            second.append(b)
    return pd.concat(first), pd.concat(second)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
    b = b / np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)
    return a @ b.T
